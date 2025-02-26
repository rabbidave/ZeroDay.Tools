import gradio as gr
import json
import logging
import time
import pandas as pd
import os
import re
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from tenacity import retry, stop_after_attempt, wait_exponential
import csv
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model_tester")


@dataclass
class ModelEndpoint:
    """Simple model endpoint configuration."""
    name: str
    api_url: str
    api_key: str
    model_id: str
    max_tokens: int = 1024
    temperature: float = 0.0
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "name": self.name,
            "api_url": self.api_url,
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


@dataclass
class TestCase:
    """Test case containing a key to query and actual value for evaluation."""
    key: str
    value: str
    id: Optional[str] = None


@dataclass
class ModelResponse:
    """Model response for a test case."""
    test_id: str
    model_name: str
    output: str
    latency: float


@dataclass
class EvaluationResult:
    """Evaluation result from the LM judge."""
    test_id: str
    champion_output: str
    challenger_output: str
    winner: str  # "CHAMPION", "CHALLENGER", or "TIE"
    confidence: float
    reasoning: str


# Global preprocessing settings (can be updated through UI)
PREPROCESS_ENABLED = True
MAX_LENGTH = 8000
REMOVE_SPECIAL_CHARS = True
NORMALIZE_WHITESPACE = True

# New CSV preprocessing settings
DETECT_DELIMITER = True
FIX_QUOTES = True
REMOVE_CONTROL_CHARS = True
NORMALIZE_NEWLINES = True
SKIP_BAD_LINES = True
SHOW_SAMPLE = True

def preprocess_text(text, max_length=None, remove_special_chars=None, normalize_whitespace=None):
    """
    Preprocess text to make it more suitable for model prompts:
    - Truncate to prevent token limits
    - Remove problematic characters
    - Normalize whitespace
    - Handle potential HTML/XML
    """
    # Use global settings if not specified
    if max_length is None:
        max_length = MAX_LENGTH
    if remove_special_chars is None:
        remove_special_chars = REMOVE_SPECIAL_CHARS
    if normalize_whitespace is None:
        normalize_whitespace = NORMALIZE_WHITESPACE
    
    # Skip preprocessing if disabled
    if not PREPROCESS_ENABLED:
        return text
    
    if text is None:
        return ""
    
    text = str(text)
    
    # Truncate to prevent excessive token usage
    if len(text) > max_length:
        text = text[:max_length] + "... [truncated]"
    
    if remove_special_chars:
        # Remove control characters and other potentially problematic characters
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        # Remove any XML/HTML-like tags that might interfere with prompts
        text = re.sub(r'<[^>]+>', '', text)
    
    if normalize_whitespace:
        # Normalize whitespace (convert multiple spaces, tabs, newlines to single space)
        text = re.sub(r'\s+', ' ', text)
        # But preserve paragraph breaks for readability
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
    
    return text


class ModelRunner:
    """Handles model API calls."""
    
    def __init__(self, endpoint: ModelEndpoint, prompt_template: str):
        self.endpoint = endpoint
        self.prompt_template = prompt_template
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate(self, test_case: TestCase) -> ModelResponse:
        """Call the model API with the test case."""
        start_time = time.time()
        
        try:
            # Preprocess the input key
            preprocessed_key = preprocess_text(test_case.key)
            
            # Format prompt using the preprocessed key
            # Handle potential formatting issues by escaping curly braces
            try:
                # For judge prompts, the "key" is already the full prompt
                if test_case.id and test_case.id.startswith("judge_"):
                    prompt = preprocessed_key  # The key already contains the full prompt for judge
                else:
                    # Use a try-except block for string formatting
                    prompt = self.prompt_template.replace("{key}", preprocessed_key)
            except Exception as e:
                # Fallback for other formatting issues
                logger.warning(f"Error formatting prompt template with replace: {str(e)}")
                try:
                    # Try with string formatting if replacement fails
                    prompt = self.prompt_template.format(key=preprocessed_key)
                except Exception as e2:
                    # Second fallback - just concatenate the template and input
                    logger.warning(f"Error formatting prompt template: {str(e2)}")
                    prompt = f"{self.prompt_template}\n\nINPUT: {preprocessed_key}"
            
            # Determine API type and make appropriate call
            response = ""
            api_url_lower = self.endpoint.api_url.lower() if self.endpoint.api_url else ""
            
            try:
                if "openai" in api_url_lower:
                    response = self._call_openai_api(prompt)
                elif "anthropic" in api_url_lower:
                    response = self._call_anthropic_api(prompt)
                elif "ollama" in api_url_lower or ":11434" in api_url_lower:
                    response = self._call_ollama_api(prompt)
                elif "generativelanguage.googleapis.com" in api_url_lower:
                    response = self._call_gemini_api(prompt)
                else:
                    response = self._call_generic_api(prompt)
            except AttributeError as ae:
                logger.error(f"AttributeError when trying to call API method: {str(ae)}")
                # Try the generic API as fallback
                try:
                    response = self._call_generic_api(prompt)
                except Exception as generic_e:
                    logger.error(f"Generic API call also failed: {str(generic_e)}")
                    # Last resort - return error message as response
                    response = f"Error: Failed to call model API. Please check API configuration and try again. Details: {str(ae)}"
            
            end_time = time.time()
            
            return ModelResponse(
                test_id=test_case.id or test_case.key[:10],
                model_name=self.endpoint.name,
                output=response,
                latency=end_time - start_time,
            )
        except Exception as e:
            logger.error(f"Unexpected error in generate method: {str(e)}")
            # Re-raise to trigger retry
            raise
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.endpoint.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.endpoint.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.endpoint.max_tokens,
            "temperature": self.endpoint.temperature,
        }
        
        response = requests.post(
            self.endpoint.api_url, 
            headers=headers, 
            json=data
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API."""
        headers = {
            "x-api-key": self.endpoint.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"  # Added required version header
        }
        
        data = {
            "model": self.endpoint.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.endpoint.max_tokens,
            "temperature": self.endpoint.temperature,
        }
        
        response = requests.post(
            self.endpoint.api_url, 
            headers=headers, 
            json=data
        )
        response.raise_for_status()
        
        result = response.json()
        return result["content"][0]["text"]
    
    def _call_ollama_api(self, prompt: str) -> str:
        """Call Ollama API."""
        data = {
            "model": self.endpoint.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.endpoint.temperature,
                "num_predict": self.endpoint.max_tokens
            }
        }
        
        response = requests.post(
            self.endpoint.api_url, 
            json=data
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API."""
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.endpoint.api_key:
            url = f"{self.endpoint.api_url}?key={self.endpoint.api_key}"
        else:
            url = self.endpoint.api_url
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": self.endpoint.max_tokens,
                "temperature": self.endpoint.temperature
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    
    def _call_generic_api(self, prompt: str) -> str:
        """Call a generic API endpoint."""
        try:
            headers = {
                "Authorization": f"Bearer {self.endpoint.api_key}",
                "Content-Type": "application/json",
            }
            
            data = {
                "prompt": prompt,
                "model": self.endpoint.model_id,
                "max_tokens": self.endpoint.max_tokens,
                "temperature": self.endpoint.temperature,
            }
            
            response = requests.post(
                self.endpoint.api_url, 
                headers=headers, 
                json=data
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("output", str(result))
        except Exception as e:
            logger.error(f"Error in generic API call: {str(e)}")
            return f"API Error: {str(e)}"


class LMJudge:
    """Uses a language model to judge between champion and challenger outputs."""
    
    def __init__(
        self, 
        endpoint: ModelEndpoint,
        evaluation_prompt_template: str,
    ):
        self.endpoint = endpoint
        self.evaluation_prompt_template = evaluation_prompt_template
        # Initialize the model runner with a pass-through prompt template
        # for judge prompts directly
        self.model_runner = ModelRunner(endpoint, "{key}")
    
    def evaluate(
        self, 
        test_case: TestCase, 
        champion_response: ModelResponse, 
        challenger_response: ModelResponse
    ) -> EvaluationResult:
        """Evaluate champion vs challenger outputs against the true value."""
        # Preprocess all inputs to ensure they're clean and properly formatted
        preprocessed_key = preprocess_text(test_case.key)
        preprocessed_value = preprocess_text(test_case.value)
        preprocessed_champion = preprocess_text(champion_response.output)
        preprocessed_challenger = preprocess_text(challenger_response.output)
        
        # Build the evaluation prompt directly
        evaluation_prompt = f"""
# Model Response Evaluation

You are evaluating two AI model responses to determine which one is better.

## Input Query
{preprocessed_key}

## Reference Value
{preprocessed_value}

## Model A (Champion) Response
```
{preprocessed_champion}
```

## Model B (Challenger) Response
```
{preprocessed_challenger}
```

## Evaluation Instructions
Please compare both responses carefully based on:
1. Factual accuracy compared to the reference value
2. Comprehensiveness of information
3. Clarity and conciseness
4. Overall quality and usefulness

## Required Response Format
You MUST start your response with EXACTLY ONE of these verdicts:
- "VERDICT: MODEL_A_WINS" (if Model A is better)
- "VERDICT: MODEL_B_WINS" (if Model B is better) 
- "VERDICT: TIE" (if both are equally good)

Then, on the next line, include:
"CONFIDENCE: X/5" (where X is a number from 1-5, with 5 being highest confidence)

After these two required lines, explain your reasoning in detail.
"""
        
        # Log the prompt for debugging
        logger.info(f"Judge evaluation prompt (truncated): {evaluation_prompt[:500]}...")
        
        # Get judge's response
        judge_response = self.model_runner.generate(
            TestCase(key=evaluation_prompt, value="", id=f"judge_{test_case.id}")
        )
        
        # Log the response for debugging
        logger.info(f"Judge response (truncated): {judge_response.output[:500]}...")
        
        # Parse the judge's decision
        parsed_result = self.parse_judge_response(judge_response.output)
        
        return EvaluationResult(
            test_id=test_case.id or test_case.key[:10],
            champion_output=champion_response.output,
            challenger_output=challenger_response.output,
            winner=parsed_result["winner"],
            confidence=parsed_result["confidence"],
            reasoning=parsed_result["reasoning"],
        )
    
    def parse_judge_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the judge's response to extract verdict, confidence, and reasoning.
        """
        # Initialize default values
        verdict = "UNDETERMINED"
        confidence = 0.0
        reasoning = response
        
        # Extract verdict using regex
        verdict_match = re.search(r"VERDICT:\s*(MODEL_A_WINS|MODEL_B_WINS|TIE)", response)
        if verdict_match:
            verdict = verdict_match.group(1)
        
        # Extract confidence using regex
        confidence_match = re.search(r"CONFIDENCE:\s*(\d+)/5", response)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1)) / 5.0  # Normalize to 0-1 range
            except ValueError:
                # If conversion fails, keep default
                pass
        
        return {
            "winner": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
        }


class ResultAggregator:
    """Collects raw evaluation results for Excel post-processing."""
    
    def __init__(self, confidence_threshold: float = 0.8):
        # Keep parameter for backward compatibility, though not used
        self.confidence_threshold = confidence_threshold
    
    def aggregate(self, evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Simply collect all results for Excel post-processing.
        No filtering or counting is done here.
        """
        return {
            "total_evaluations": len(evaluation_results),
            "raw_evaluations": [
                {
                    "test_id": result.test_id,
                    "champion_output": result.champion_output,
                    "challenger_output": result.challenger_output,
                    "judge_response": result.reasoning,
                    # The following fields will be filled in Excel
                    "winner": "TBD IN EXCEL",
                    "confidence": "TBD IN EXCEL"
                } 
                for result in evaluation_results
            ],
            "note": "Results need post-processing in Excel to determine winners and calculate statistics."
        }


class ModelTester:
    """Main class that orchestrates the A/B testing process."""
    
    def __init__(
        self,
        champion_endpoint: ModelEndpoint,
        challenger_endpoint: ModelEndpoint,
        judge_endpoint: ModelEndpoint,
        model_prompt_template: str,
        evaluation_prompt_template: str,
        confidence_threshold: float = 0.8,
    ):
        self.champion_runner = ModelRunner(champion_endpoint, model_prompt_template)
        self.challenger_runner = ModelRunner(challenger_endpoint, model_prompt_template)
        self.judge = LMJudge(judge_endpoint, evaluation_prompt_template)
        self.aggregator = ResultAggregator(confidence_threshold)
        self.champion_endpoint = champion_endpoint
        self.challenger_endpoint = challenger_endpoint
        self.judge_endpoint = judge_endpoint
    
    def run_test(self, test_cases: List[TestCase], batch_size: int = 10, progress=None) -> Dict[str, Any]:
        """Run the complete test process."""
        all_evaluation_results = []
        champion_metrics = {"total_latency": 0, "total_tokens": 0}
        challenger_metrics = {"total_latency": 0, "total_tokens": 0}
        judge_metrics = {"total_latency": 0, "total_tokens": 0}
        
        total_batches = (len(test_cases) + batch_size - 1) // batch_size
        
        # Process in batches
        for i in range(0, len(test_cases), batch_size):
            batch = test_cases[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            if progress is not None:
                progress(batch_num / total_batches, f"Processing batch {batch_num}/{total_batches}")
            
            # 1. Get responses from both models
            champion_responses = []
            challenger_responses = []
            
            for test_case in batch:
                # Assign an ID if not present
                if not test_case.id:
                    test_case.id = test_case.key[:10]
                
                # Get champion response
                champion_response = self.champion_runner.generate(test_case)
                champion_responses.append(champion_response)
                champion_metrics["total_latency"] += champion_response.latency
                champion_metrics["total_tokens"] += len(champion_response.output.split())
                
                # Get challenger response
                challenger_response = self.challenger_runner.generate(test_case)
                challenger_responses.append(challenger_response)
                challenger_metrics["total_latency"] += challenger_response.latency
                challenger_metrics["total_tokens"] += len(challenger_response.output.split())
            
            # 2. Evaluate with LM judge
            batch_evaluation_results = []
            for idx, test_case in enumerate(batch):
                start_time = time.time()
                evaluation_result = self.judge.evaluate(
                    test_case,
                    champion_responses[idx],
                    challenger_responses[idx],
                )
                judge_latency = time.time() - start_time
                judge_metrics["total_latency"] += judge_latency
                judge_metrics["total_tokens"] += len(evaluation_result.reasoning.split())
                
                batch_evaluation_results.append(evaluation_result)
            
            all_evaluation_results.extend(batch_evaluation_results)
            
            # Extract and log verdicts from this batch
            batch_counts = {"MODEL_A_WINS": 0, "MODEL_B_WINS": 0, "TIE": 0, "UNKNOWN": 0}
            for result in batch_evaluation_results:
                verdict_match = re.search(r"VERDICT:\s*(MODEL_A_WINS|MODEL_B_WINS|TIE)", result.reasoning)
                if verdict_match:
                    verdict = verdict_match.group(1)
                    batch_counts[verdict] += 1
                else:
                    batch_counts["UNKNOWN"] += 1
            
            logger.info(
                f"Batch results: A wins: {batch_counts['MODEL_A_WINS']}, "
                f"B wins: {batch_counts['MODEL_B_WINS']}, "
                f"Ties: {batch_counts['TIE']}, "
                f"Unknown: {batch_counts['UNKNOWN']}"
            )
        
        # Calculate average metrics
        num_cases = len(test_cases)
        champion_metrics["avg_latency"] = champion_metrics["total_latency"] / num_cases
        champion_metrics["avg_tokens"] = champion_metrics["total_tokens"] / num_cases
        challenger_metrics["avg_latency"] = challenger_metrics["total_latency"] / num_cases
        challenger_metrics["avg_tokens"] = challenger_metrics["total_tokens"] / num_cases
        judge_metrics["avg_latency"] = judge_metrics["total_latency"] / num_cases
        judge_metrics["avg_tokens"] = judge_metrics["total_tokens"] / num_cases
        
        # 3. Aggregate results
        aggregated_results = self.aggregator.aggregate(all_evaluation_results)
        
        # 4. Summarize results by counting verdicts
        verdicts = {"MODEL_A_WINS": 0, "MODEL_B_WINS": 0, "TIE": 0, "UNKNOWN": 0}
        confidences = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        
        for result in all_evaluation_results:
            verdict_match = re.search(r"VERDICT:\s*(MODEL_A_WINS|MODEL_B_WINS|TIE)", result.reasoning)
            if verdict_match:
                verdict = verdict_match.group(1)
                verdicts[verdict] += 1
            else:
                verdicts["UNKNOWN"] += 1
            
            confidence_match = re.search(r"CONFIDENCE:\s*(\d+)/5", result.reasoning)
            if confidence_match:
                confidence = confidence_match.group(1)
                confidences[confidence] = confidences.get(confidence, 0) + 1
        
        # Calculate percentages
        total_verdicts = sum(verdicts.values())
        verdict_percentages = {
            k: round(v / total_verdicts * 100, 2) if total_verdicts > 0 else 0
            for k, v in verdicts.items() if k != "UNKNOWN"
        }
        
        # Determine overall winner
        decision = "MAINTAIN_CHAMPION"
        reason = "Challenger did not demonstrate significant improvement"
        
        # Challenger needs to win by a margin to replace champion
        if verdict_percentages.get("MODEL_B_WINS", 0) > 55 and verdict_percentages.get("MODEL_B_WINS", 0) > verdict_percentages.get("MODEL_A_WINS", 0):
            decision = "REPLACE_WITH_CHALLENGER"
            reason = f"Challenger outperformed champion ({verdict_percentages.get('MODEL_B_WINS', 0)}% vs {verdict_percentages.get('MODEL_A_WINS', 0)}%)"
        
        # Log final results
        logger.info(
            f"Final verdict counts: A wins: {verdicts['MODEL_A_WINS']} ({verdict_percentages.get('MODEL_A_WINS', 0)}%), "
            f"B wins: {verdicts['MODEL_B_WINS']} ({verdict_percentages.get('MODEL_B_WINS', 0)}%), "
            f"Ties: {verdicts['TIE']} ({verdict_percentages.get('TIE', 0)}%)"
        )
        logger.info(f"Decision: {decision} - {reason}")
        
        if progress is not None:
            progress(1.0, "Testing completed")
        
        return {
            "evaluations": [result.__dict__ for result in all_evaluation_results],
            "summary": {
                "total_test_cases": num_cases,
                "verdicts": verdicts,
                "verdict_percentages": verdict_percentages,
                "confidences": confidences,
                "decision": decision,
                "reason": reason,
                "champion_metrics": champion_metrics,
                "challenger_metrics": challenger_metrics,
                "judge_metrics": judge_metrics,
                "champion_name": self.champion_endpoint.name,
                "challenger_name": self.challenger_endpoint.name,
                "judge_name": self.judge_endpoint.name,
            }
        }


def save_results_to_json(results: Dict[str, Any], file_path: str):
    """Save results to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)


def save_results_to_csv(results: Dict[str, Any], file_path: str):
    """
    Save evaluation results to a CSV file for Excel processing.
    Uses the standardized verdict format for easier sorting and filtering.
    """
    try:
        with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:  # Use utf-8-sig for Excel compatibility
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)  # Quote all fields for better compatibility
            
            # Write header
            writer.writerow([
                "test_id", 
                "query", 
                "reference",
                "champion_model",
                "challenger_model",
                "verdict",  # Automatically extracted from judge response
                "confidence",  # Automatically extracted from judge response
                "explanation",
                "champion_output", 
                "challenger_output", 
                "full_judge_response"
            ])
            
            # Write data rows
            for eval_item in results.get("evaluations", []):
                # Extract test case data if available
                test_id = eval_item.get("test_id", "")
                
                # Try to find the original test case
                query = ""
                reference = ""
                for test_case in results.get("metadata", {}).get("test_cases", []):
                    if test_case.get("id") == test_id:
                        query = test_case.get("key", "")
                        reference = test_case.get("value", "")
                        break
                
                # Extract verdict and confidence using regex
                verdict = "UNKNOWN"
                confidence = "0"
                judge_response = eval_item.get("reasoning", "")
                
                # Extract verdict
                verdict_match = re.search(r"VERDICT:\s*(MODEL_A_WINS|MODEL_B_WINS|TIE)", judge_response)
                if verdict_match:
                    verdict = verdict_match.group(1)
                
                # Extract confidence
                confidence_match = re.search(r"CONFIDENCE:\s*(\d+)/5", judge_response)
                if confidence_match:
                    confidence = confidence_match.group(1)
                
                # Extract explanation (everything after the confidence line)
                explanation = ""
                lines = judge_response.split('\n')
                capture = False
                for line in lines:
                    if capture:
                        explanation += line + " "
                    if "CONFIDENCE:" in line:
                        capture = True
                
                # Clean output for CSV - remove line breaks that might break the CSV format
                champion_output = eval_item.get("champion_output", "").replace("\n", " ")
                challenger_output = eval_item.get("challenger_output", "").replace("\n", " ")
                judge_response_clean = judge_response.replace("\n", " ")
                
                # Truncate long text fields to prevent Excel issues
                max_cell_length = 32700  # Excel's limit is ~32,767 characters
                
                writer.writerow([
                    test_id,
                    query[:max_cell_length] if query else "",
                    reference[:max_cell_length] if reference else "",
                    results.get("summary", {}).get("champion_name", ""),
                    results.get("summary", {}).get("challenger_name", ""),
                    verdict,
                    confidence,
                    explanation.strip()[:max_cell_length],
                    champion_output[:max_cell_length],
                    challenger_output[:max_cell_length],
                    judge_response_clean[:max_cell_length]
                ])
        
        logger.info(f"Successfully saved results to CSV: {file_path}")
    except Exception as e:
        logger.error(f"Error saving results to CSV: {str(e)}")
        # If saving to CSV fails, try with a more robust approach
        try:
            logger.info("Attempting to save CSV with a more robust approach...")
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\', 
                                  doublequote=True, lineterminator='\n')
                
                # Write just the critical data if the full save failed
                writer.writerow(["test_id", "verdict", "confidence", "champion_model", "challenger_model"])
                
                for eval_item in results.get("evaluations", []):
                    test_id = eval_item.get("test_id", "")
                    judge_response = eval_item.get("reasoning", "")
                    
                    # Extract verdict
                    verdict = "UNKNOWN"
                    verdict_match = re.search(r"VERDICT:\s*(MODEL_A_WINS|MODEL_B_WINS|TIE)", judge_response)
                    if verdict_match:
                        verdict = verdict_match.group(1)
                    
                    # Extract confidence
                    confidence = "0"
                    confidence_match = re.search(r"CONFIDENCE:\s*(\d+)/5", judge_response)
                    if confidence_match:
                        confidence = confidence_match.group(1)
                    
                    writer.writerow([
                        test_id,
                        verdict,
                        confidence,
                        results.get("summary", {}).get("champion_name", ""),
                        results.get("summary", {}).get("challenger_name", "")
                    ])
                
                logger.info(f"Saved simplified results to CSV: {file_path}")
        except Exception as inner_e:
            logger.error(f"Failed to save even simplified CSV: {str(inner_e)}")


def load_test_cases_from_csv_url(csv_url: str, key_column: str = "text", value_column: str = "label", 
                              limit: int = None) -> List[TestCase]:
    """Load test cases from a CSV URL with robust preprocessing and error handling."""
    try:
        # Convert GitHub URL to raw format if needed
        if "github.com" in csv_url and "/blob/" in csv_url and not "raw.githubusercontent.com" in csv_url:
            csv_url = csv_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            logger.info(f"Converted GitHub URL to raw format: {csv_url}")
        
        # Download the CSV file
        logger.info(f"Downloading CSV from: {csv_url}")
        response = requests.get(csv_url)
        response.raise_for_status()
        
        # Get raw content
        csv_content = response.content.decode('utf-8', errors='replace')
        
        # Log first few lines for debugging
        lines = csv_content.split('\n')
        logger.info(f"CSV first few lines (raw):")
        for i, line in enumerate(lines[:3]):
            if i == 0:
                logger.info(f"Header: {line}")
            else:
                logger.info(f"Data row {i}: {line[:100]}...")
        
        # --- PREPROCESSING SECTION START ---
        if REMOVE_CONTROL_CHARS:
            # Remove control characters except for legitimate newlines and tabs
            csv_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', csv_content)
        
        if FIX_QUOTES:
            # Handle unescaped quotes within fields
            lines = csv_content.split('\n')
            processed_lines = []
            
            in_quoted_field = False
            for line in lines:
                if in_quoted_field:
                    # This line continues a quoted field from previous line
                    processed_lines[-1] += "\\n" + line  # Add escaped newline
                    if line.count('"') % 2 == 1:  # Odd number of quotes means end of quoted field
                        in_quoted_field = False
                else:
                    # Count quotes to see if we have an open quoted field
                    if line.count('"') % 2 == 1:
                        in_quoted_field = True
                    processed_lines.append(line)
            
            csv_content = '\n'.join(processed_lines)
        
        if NORMALIZE_NEWLINES:
            # Normalize newlines within quoted fields
            def replace_newlines_in_quotes(match):
                return match.group(0).replace('\n', '\\n').replace('\r', '')
                
            csv_content = re.sub(r'"[^"]*"', replace_newlines_in_quotes, csv_content)
        
        detected_delimiter = ','  # Default
        
        if DETECT_DELIMITER:
            # Try to detect the delimiter by sampling first few lines
            sample_lines = csv_content.split('\n')[:5]
            candidate_delimiters = [',', '\t', ';', '|']
            delimiter_counts = {}
            
            for delimiter in candidate_delimiters:
                delimiter_counts[delimiter] = 0
                for line in sample_lines:
                    # Count delimiters in unquoted text
                    in_quotes = False
                    unquoted_delimiter_count = 0
                    for char in line:
                        if char == '"':
                            in_quotes = not in_quotes
                        elif char == delimiter and not in_quotes:
                            unquoted_delimiter_count += 1
                    delimiter_counts[delimiter] += unquoted_delimiter_count
            
            # Choose the delimiter with the most consistent count
            max_count = 0
            for delimiter, count in delimiter_counts.items():
                if count > max_count:
                    max_count = count
                    detected_delimiter = delimiter
                
            logger.info(f"Detected delimiter: '{detected_delimiter}'")
        # --- PREPROCESSING SECTION END ---
        
        # Try multiple parsing approaches with the preprocessed content
        parsing_methods = [
            # Method 1: Using detected delimiter
            lambda: pd.read_csv(io.StringIO(csv_content), 
                               sep=detected_delimiter,
                               quoting=csv.QUOTE_MINIMAL,
                               escapechar='\\',
                               on_bad_lines='warn'),
            
            # Method 2: Standard CSV parsing
            lambda: pd.read_csv(io.StringIO(csv_content)),
            
            # Method 3: Try with different quoting options
            lambda: pd.read_csv(io.StringIO(csv_content), 
                               quoting=csv.QUOTE_NONE, 
                               escapechar='\\'),
            
            # Method 4: Skip bad lines if enabled
            lambda: pd.read_csv(io.StringIO(csv_content), 
                               on_bad_lines='skip'),
            
            # Method 5: Try with Python's csv module directly
            lambda: pd.DataFrame(list(csv.reader(io.StringIO(csv_content))))
        ]
        
        df = None
        parsing_error = None
        successful_method = None
        
        for i, method in enumerate(parsing_methods):
            try:
                df = method()
                if len(df) > 0:  # Check if we got valid data
                    # If we got here, parsing succeeded
                    successful_method = i + 1
                    break
            except Exception as e:
                parsing_error = e
                # Skip to next method if SKIP_BAD_LINES is enabled
                if SKIP_BAD_LINES:
                    logger.warning(f"Method {i+1} failed: {str(e)}. Trying next method.")
                    continue
                else:
                    # If SKIP_BAD_LINES is disabled and this was the first method,
                    # we'll continue to try other methods
                    if i == 0:
                        logger.warning(f"Method {i+1} failed: {str(e)}. Trying next method even though SKIP_BAD_LINES is disabled.")
                        continue
                    else:
                        # For other methods, respect the SKIP_BAD_LINES setting
                        raise
        
        if df is None:
            # All methods failed
            raise ValueError(f"Failed to parse CSV after preprocessing. Last error: {str(parsing_error)}")
        
        logger.info(f"Successfully parsed CSV using method {successful_method}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        
        # If dataframe has no column names (from method 5), use first row as header
        if all(isinstance(col, int) for col in df.columns):
            df.columns = df.iloc[0]
            df = df[1:]
        
        # More robust column selection - try to match case-insensitive
        # and strip whitespace from column names
        clean_columns = {col: col.strip() for col in df.columns}
        df.columns = [col.strip() for col in df.columns]
        
        # Try to infer column names if they don't exist
        if key_column not in df.columns:
            # Try case-insensitive match
            key_match = next((col for col in df.columns if col.lower() == key_column.lower()), None)
            if key_match:
                key_column = key_match
                logger.info(f"Using case-insensitive match for key column: '{key_column}'")
            # If we have at least two columns, use the first one as key
            elif len(df.columns) >= 2:
                key_column = df.columns[0]
                logger.warning(f"Key column '{key_column}' not found, using first column: {key_column}")
            else:
                raise ValueError(f"Key column '{key_column}' not found and couldn't infer a suitable column")
        
        if value_column not in df.columns:
            # Try case-insensitive match
            value_match = next((col for col in df.columns if col.lower() == value_column.lower()), None)
            if value_match:
                value_column = value_match
                logger.info(f"Using case-insensitive match for value column: '{value_column}'")
            # If we have at least two columns, use the second one as value
            elif len(df.columns) >= 2:
                value_column = df.columns[1]
                logger.warning(f"Value column '{value_column}' not found, using second column: {value_column}")
            else:
                # If only one column exists, use it for both key and value
                value_column = key_column
                logger.warning(f"Value column '{value_column}' not found, using key column for both")
        
        # Create test cases
        test_cases = []
        for i, row in df.iterrows():
            if limit is not None and i >= limit:
                break
            
            # Handle missing values
            key_value = row[key_column] if pd.notna(row[key_column]) else ""
            val_value = row[value_column] if pd.notna(row[value_column]) else ""
            
            # Convert to string and ensure they don't contain problematic formatting characters
            key_str = str(key_value).replace("{", "{{").replace("}", "}}")
            val_str = str(val_value).replace("{", "{{").replace("}", "}}")
            
            test_cases.append(TestCase(
                key=key_str,
                value=val_str,
                id=f"case_{i}",
            ))
        
        if not test_cases:
            raise ValueError("No valid test cases found in CSV")
            
        logger.info(f"Successfully loaded {len(test_cases)} test cases from CSV")
        
        # Print sample of preprocessed data
        if SHOW_SAMPLE and test_cases:
            logger.info("----- Sample Test Case After Preprocessing -----")
            sample_case = test_cases[0]
            logger.info(f"ID: {sample_case.id}")
            logger.info(f"Key (truncated): {sample_case.key[:100]}...")
            logger.info(f"Value (truncated): {sample_case.value[:100]}...")
            logger.info("-------------------------------------------")
            
        return test_cases
    
    except Exception as e:
        logger.error(f"Error loading CSV from URL {csv_url}: {str(e)}")
        raise ValueError(f"CSV parsing error: {str(e)}")
        
def create_model_endpoint(model_type, model_name, api_url, api_key, model_id, max_tokens, temperature):
    """Create a model endpoint configuration based on provided parameters."""
    # Default URL for common models if not specified
    if not api_url:
        if model_type == "OpenAI":
            api_url = "https://api.openai.com/v1/chat/completions"
        elif model_type == "Anthropic":
            api_url = "https://api.anthropic.com/v1/messages"
        elif model_type == "Gemini":
            api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        elif model_type == "Ollama":
            api_url = "http://localhost:11434/api/generate"
    
    # Sanitize inputs
    try:
        max_tokens = int(max_tokens)
    except (ValueError, TypeError):
        logger.warning(f"Invalid max_tokens value: {max_tokens}, using default 1024")
        max_tokens = 1024
    
    try:
        temperature = float(temperature)
    except (ValueError, TypeError):
        logger.warning(f"Invalid temperature value: {temperature}, using default 0.0")
        temperature = 0.0
    
    # Create and return the endpoint
    return ModelEndpoint(
        name=model_name,
        api_url=api_url,
        api_key=api_key,
        model_id=model_id,
        max_tokens=max_tokens,
        temperature=temperature
    )


def run_model_test(
    champion_type, champion_name, champion_api_url, champion_api_key, champion_model_id, champion_max_tokens, champion_temperature,
    challenger_type, challenger_name, challenger_api_url, challenger_api_key, challenger_model_id, challenger_max_tokens, challenger_temperature,
    judge_type, judge_name, judge_api_url, judge_api_key, judge_model_id, judge_max_tokens, judge_temperature,
    csv_url, key_column, value_column, limit, output_dir, prompt_template, evaluation_template,
    progress=gr.Progress()
):
    """Run a model test with the provided parameters and return results."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Log current preprocessing settings
        logger.info("CSV Preprocessing settings:")
        logger.info(f"- Detect Delimiter: {DETECT_DELIMITER}")
        logger.info(f"- Fix Unescaped Quotes: {FIX_QUOTES}")
        logger.info(f"- Remove Control Chars: {REMOVE_CONTROL_CHARS}")
        logger.info(f"- Normalize Newlines: {NORMALIZE_NEWLINES}")
        logger.info(f"- Skip Bad Lines: {SKIP_BAD_LINES}")
        logger.info(f"- Show Sample: {SHOW_SAMPLE}")
        
        # Create model endpoints
        champion_endpoint = create_model_endpoint(
            champion_type, champion_name, champion_api_url, champion_api_key, champion_model_id,
            champion_max_tokens, champion_temperature
        )
        
        challenger_endpoint = create_model_endpoint(
            challenger_type, challenger_name, challenger_api_url, challenger_api_key, challenger_model_id,
            challenger_max_tokens, challenger_temperature
        )
        
        judge_endpoint = create_model_endpoint(
            judge_type, judge_name, judge_api_url, judge_api_key, judge_model_id,
            judge_max_tokens, judge_temperature
        )
        
        # Load test cases
        progress(0.1, "Loading and preprocessing test cases from CSV")
        try:
            test_cases = load_test_cases_from_csv_url(
                csv_url, 
                key_column, 
                value_column, 
                int(limit) if limit and int(limit) > 0 else None
            )
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return f"""
## Error Loading CSV

**Error Message:**
```
{str(e)}
```

**Troubleshooting Tips:**
1. Verify the CSV URL is accessible and properly formatted
2. Check that column names exist in the file
3. Try enabling more preprocessing options
4. Set a small limit to test with a subset of data first
""", None, None
        
        # Log preprocessing settings being used
        logger.info(f"Model Input Preprocessing: Enabled={PREPROCESS_ENABLED}, MaxLength={MAX_LENGTH}, " +
                    f"RemoveSpecialChars={REMOVE_SPECIAL_CHARS}, NormalizeWhitespace={NORMALIZE_WHITESPACE}")
        
        # Preview sample test cases after preprocessing
        if test_cases and SHOW_SAMPLE:
            preview_case = test_cases[0]
            logger.info("----- Sample Test Case -----")
            logger.info(f"ID: {preview_case.id}")
            logger.info(f"Original Key: {preview_case.key[:200]}...")
            logger.info(f"Original Value: {preview_case.value[:200]}...")
            
            processed_key = preprocess_text(preview_case.key)
            processed_value = preprocess_text(preview_case.value)
            logger.info(f"Processed Key: {processed_key[:200]}...")
            logger.info(f"Processed Value: {processed_value[:200]}...")
            logger.info("--------------------------")
        
        # Set up the model tester
        tester = ModelTester(
            champion_endpoint=champion_endpoint,
            challenger_endpoint=challenger_endpoint,
            judge_endpoint=judge_endpoint,
            model_prompt_template=prompt_template,
            evaluation_prompt_template=evaluation_template,
        )
        
        # Run the test
        progress(0.2, f"Starting test with {len(test_cases)} test cases")
        results = tester.run_test(test_cases, batch_size=5, progress=progress)
        
        # Save results
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        json_path = os.path.join(output_dir, f"results-{timestamp}.json")
        csv_path = os.path.join(output_dir, f"results-{timestamp}.csv")
        
        # Add processing settings to results metadata
        results["metadata"] = {
            "timestamp": timestamp,
            "preprocessing": {
                "enabled": PREPROCESS_ENABLED,
                "max_length": MAX_LENGTH,
                "remove_special_chars": REMOVE_SPECIAL_CHARS,
                "normalize_whitespace": NORMALIZE_WHITESPACE,
                "csv_preprocessing": {
                    "detect_delimiter": DETECT_DELIMITER,
                    "fix_quotes": FIX_QUOTES,
                    "remove_control_chars": REMOVE_CONTROL_CHARS,
                    "normalize_newlines": NORMALIZE_NEWLINES,
                    "skip_bad_lines": SKIP_BAD_LINES,
                }
            },
            "csv_source": {
                "url": csv_url,
                "key_column": key_column,
                "value_column": value_column,
                "limit": limit
            },
            "test_cases": [
                {"id": case.id, "key": case.key, "value": case.value}
                for case in test_cases
            ]
        }
        
        save_results_to_json(results, json_path)
        save_results_to_csv(results, csv_path)
        
        # Format summary for display
        summary = results["summary"]
        verdicts = summary["verdicts"]
        verdict_percentages = summary["verdict_percentages"]
        
        summary_text = f"""
## Test Results Summary

**Models:**
- Champion: {summary['champion_name']}
- Challenger: {summary['challenger_name']}
- Judge: {summary['judge_name']}

**Verdict Counts:**
- MODEL A WINS: {verdicts['MODEL_A_WINS']} ({verdict_percentages.get('MODEL_A_WINS', 0)}%)
- MODEL B WINS: {verdicts['MODEL_B_WINS']} ({verdict_percentages.get('MODEL_B_WINS', 0)}%)
- TIE: {verdicts['TIE']} ({verdict_percentages.get('TIE', 0)}%)
- UNKNOWN: {verdicts['UNKNOWN']}

**Decision:** {summary['decision']}
**Reason:** {summary['reason']}

**Performance Metrics:**
- Champion Avg Latency: {summary['champion_metrics']['avg_latency']:.2f}s
- Challenger Avg Latency: {summary['challenger_metrics']['avg_latency']:.2f}s
- Judge Avg Latency: {summary['judge_metrics']['avg_latency']:.2f}s

**Preprocessing Settings:**
- Model Inputs: Enabled={PREPROCESS_ENABLED}, MaxLength={MAX_LENGTH}
- CSV: DetectDelimiter={DETECT_DELIMITER}, FixQuotes={FIX_QUOTES}, SkipBadLines={SKIP_BAD_LINES}

**Files Saved:**
- JSON: {json_path}
- CSV: {csv_path}
"""
        
        return summary_text, json_path, csv_path
    
    except Exception as e:
        logger.exception("Error running model test")
        error_message = f"""
## Error Running Test

**Error Message:**
```
{str(e)}
```

**Troubleshooting Tips:**
1. Check that the CSV URL is accessible and properly formatted
2. Verify that the column names are correct
3. Ensure API keys and endpoints are valid
4. Try adjusting CSV preprocessing options to handle special characters, quotes, and delimiters
5. Set a small limit (e.g., 5) to test with a subset of data first
"""
        return error_message, None, None


# Default templates
DEFAULT_PROMPT_TEMPLATE = """
Please provide information about the following topic: {key}

Be concise yet comprehensive and factually accurate.
"""

DEFAULT_EVALUATION_TEMPLATE = """
I need you to evaluate two model responses about a topic, comparing them with the reference value.

Topic: {key}

Reference value: {value}

Response from Model A (Champion):
```
{champion_output}
```

Response from Model B (Challenger):
```
{challenger_output}
```

Please carefully evaluate both responses in terms of:
1. Factual accuracy compared to the reference value
2. Comprehensiveness of information
3. Clarity and conciseness

IMPORTANT: Your response MUST start with exactly ONE of these specific phrases:
- "VERDICT: MODEL_A_WINS" (if Model A is better)
- "VERDICT: MODEL_B_WINS" (if Model B is better) 
- "VERDICT: TIE" (if both are equally good)

Then on a new line, add:
- "CONFIDENCE: X/5" (where X is a number from 1-5, with 5 being highest confidence)

After these two required lines, explain your reasoning briefly.
"""

# Create Gradio interface
def create_gradio_interface():
    """Create a Gradio interface for the model tester."""
    # CSS for better layout
    css = """
    .container { max-width: 1200px; margin: auto; }
    .model-group { border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px; }
    .model-group h3 { margin-top: 0; }
    """
    
    with gr.Blocks(css=css) as app:
        gr.Markdown("# Model A/B Testing Tool")
        gr.Markdown("Compare two models against a reference value using a judge model")
        
        with gr.Tab("Test Configuration"):
            with gr.Group(elem_classes="model-group"):
                gr.Markdown("### Data Source & Test Settings")
                with gr.Row():
                    csv_url = gr.Textbox(
                        label="CSV URL", 
                        value="https://example.com/data.csv", 
                        info="URL to CSV file with test cases"
                    )
                
                with gr.Row():
                    key_column = gr.Textbox(
                        label="Key Column", 
                        value="text", 
                        info="Column name for input data (queries)"
                    )
                    value_column = gr.Textbox(
                        label="Value Column", 
                        value="label", 
                        info="Column name for reference/ground truth"
                    )
                    limit = gr.Number(
                        label="Limit", 
                        value=10, 
                        info="Maximum number of test cases to use (0 for all)"
                    )
                
                with gr.Group():
                    gr.Markdown("#### Preprocessing Options")
                    with gr.Row():
                        preprocess_inputs = gr.Checkbox(
                            label="Preprocess Inputs", 
                            value=True,
                            info="Clean and format inputs before sending to models"
                        )
                        max_length = gr.Slider(
                            label="Max Input Length", 
                            minimum=1000, 
                            maximum=50000, 
                            value=8000, 
                            step=1000,
                            info="Maximum length of inputs before truncation"
                        )
                    
                    with gr.Row():
                        remove_special_chars = gr.Checkbox(
                            label="Remove Special Characters", 
                            value=True,
                            info="Remove control characters and HTML/XML tags"
                        )
                        normalize_whitespace = gr.Checkbox(
                            label="Normalize Whitespace", 
                            value=True,
                            info="Standardize spaces, tabs, and newlines"
                        )
                
                # Add CSV Preprocessing options
                with gr.Group():
                    gr.Markdown("#### CSV Preprocessing Options")
                    with gr.Row():
                        detect_delimiter = gr.Checkbox(
                            label="Auto-detect Delimiter", 
                            value=True,
                            info="Try to detect the most likely delimiter (comma, tab, semicolon, etc.)"
                        )
                        fix_quotes = gr.Checkbox(
                            label="Fix Unescaped Quotes", 
                            value=True,
                            info="Attempt to fix unescaped quotes within text fields that might break parsing"
                        )
                    
                    with gr.Row():
                        remove_control_chars = gr.Checkbox(
                            label="Remove Control Characters", 
                            value=True,
                            info="Remove special control characters that can break CSV parsing"
                        )
                        normalize_newlines = gr.Checkbox(
                            label="Normalize Newlines in Fields", 
                            value=True,
                            info="Convert newlines within fields to escaped newlines (\\n)"
                        )
                    
                    with gr.Row():
                        skip_bad_lines = gr.Checkbox(
                            label="Skip Bad Lines", 
                            value=True,
                            info="Skip lines that can't be parsed correctly instead of failing"
                        )
                        show_sample = gr.Checkbox(
                            label="Show Sample After Preprocessing", 
                            value=True,
                            info="Display a sample of the preprocessed data in the results"
                        )
                
                with gr.Row():
                    output_dir = gr.Textbox(
                        label="Output Directory", 
                        value="./results", 
                        info="Directory to save results"
                    )
            
            with gr.Group(elem_classes="model-group"):
                gr.Markdown("### Champion Model (A)")
                with gr.Row():
                    champion_type = gr.Dropdown(
                        choices=["OpenAI", "Anthropic", "Gemini", "Ollama", "Generic"], 
                        value="OpenAI", 
                        label="Provider"
                    )
                    champion_name = gr.Textbox(label="Model Name", value="Champion-GPT-3.5")
                
                with gr.Row():
                    champion_api_url = gr.Textbox(
                        label="API URL", 
                        value="https://api.openai.com/v1/chat/completions"
                    )
                    champion_api_key = gr.Textbox(
                        label="API Key", 
                        value="", 
                        type="password"
                    )
                
                with gr.Row():
                    champion_model_id = gr.Textbox(label="Model ID", value="gpt-3.5-turbo")
                    champion_max_tokens = gr.Number(label="Max Tokens", value=1024, minimum=1)
                    champion_temperature = gr.Slider(
                        label="Temperature", 
                        minimum=0.0, 
                        maximum=1.0, 
                        value=0.0, 
                        step=0.1
                    )
            
            with gr.Group(elem_classes="model-group"):
                gr.Markdown("### Challenger Model (B)")
                with gr.Row():
                    challenger_type = gr.Dropdown(
                        choices=["OpenAI", "Anthropic", "Gemini", "Ollama", "Generic"], 
                        value="Anthropic", 
                        label="Provider"
                    )
                    challenger_name = gr.Textbox(label="Model Name", value="Challenger-Claude")
                
                with gr.Row():
                    challenger_api_url = gr.Textbox(
                        label="API URL", 
                        value="https://api.anthropic.com/v1/messages"
                    )
                    challenger_api_key = gr.Textbox(
                        label="API Key", 
                        value="", 
                        type="password"
                    )
                
                with gr.Row():
                    challenger_model_id = gr.Textbox(label="Model ID", value="claude-3-haiku-20240307")
                    challenger_max_tokens = gr.Number(label="Max Tokens", value=1024, minimum=1)
                    challenger_temperature = gr.Slider(
                        label="Temperature", 
                        minimum=0.0, 
                        maximum=1.0, 
                        value=0.0, 
                        step=0.1
                    )
            
            with gr.Group(elem_classes="model-group"):
                gr.Markdown("### Judge Model (C)")
                with gr.Row():
                    judge_type = gr.Dropdown(
                        choices=["OpenAI", "Anthropic", "Gemini", "Ollama", "Generic"], 
                        value="OpenAI", 
                        label="Provider"
                    )
                    judge_name = gr.Textbox(label="Model Name", value="Judge-GPT-4")
                
                with gr.Row():
                    judge_api_url = gr.Textbox(
                        label="API URL", 
                        value="https://api.openai.com/v1/chat/completions"
                    )
                    judge_api_key = gr.Textbox(
                        label="API Key", 
                        value="", 
                        type="password"
                    )
                
                with gr.Row():
                    judge_model_id = gr.Textbox(label="Model ID", value="gpt-4")
                    judge_max_tokens = gr.Number(label="Max Tokens", value=2048, minimum=1)
                    judge_temperature = gr.Slider(
                        label="Temperature", 
                        minimum=0.0, 
                        maximum=1.0, 
                        value=0.0, 
                        step=0.1
                    )
            
            with gr.Group(elem_classes="model-group"):
                gr.Markdown("### Prompt Templates")
                with gr.Tabs():
                    with gr.TabItem("Model Prompt"):
                        prompt_template = gr.Textbox(
                            label="Model Prompt Template", 
                            value=DEFAULT_PROMPT_TEMPLATE,
                            lines=10,
                            info="Use {key} as placeholder for the input"
                        )
                    
                    with gr.TabItem("Evaluation Prompt"):
                        evaluation_template = gr.Textbox(
                            label="Evaluation Prompt Template", 
                            value=DEFAULT_EVALUATION_TEMPLATE,
                            lines=15,
                            info="Use {key}, {value}, {champion_output}, and {challenger_output} as placeholders"
                        )
            
            run_button = gr.Button("Run Test", variant="primary")
        
        with gr.Tab("Results"):
            with gr.Group():
                results_markdown = gr.Markdown("Results will appear here after running the test")
                
                with gr.Row():
                    json_file = gr.File(label="JSON Results", interactive=False)
                    csv_file = gr.File(label="CSV Results", interactive=False)
        
        # Define the function to run when the button is clicked
        def run_test_with_preprocessing(
            champion_type, champion_name, champion_api_url, champion_api_key, champion_model_id, champion_max_tokens, champion_temperature,
            challenger_type, challenger_name, challenger_api_url, challenger_api_key, challenger_model_id, challenger_max_tokens, challenger_temperature,
            judge_type, judge_name, judge_api_url, judge_api_key, judge_model_id, judge_max_tokens, judge_temperature,
            csv_url, key_column, value_column, limit, output_dir, prompt_template, evaluation_template,
            preprocess_inputs, max_length, remove_special_chars, normalize_whitespace,
            detect_delimiter, fix_quotes, remove_control_chars, normalize_newlines, skip_bad_lines, show_sample,
            progress=gr.Progress()
        ):
            # Set preprocessing globals based on user selections
            global PREPROCESS_ENABLED, MAX_LENGTH, REMOVE_SPECIAL_CHARS, NORMALIZE_WHITESPACE
            global DETECT_DELIMITER, FIX_QUOTES, REMOVE_CONTROL_CHARS, NORMALIZE_NEWLINES, SKIP_BAD_LINES, SHOW_SAMPLE
            
            PREPROCESS_ENABLED = preprocess_inputs
            MAX_LENGTH = max_length
            REMOVE_SPECIAL_CHARS = remove_special_chars
            NORMALIZE_WHITESPACE = normalize_whitespace
            
            # Set CSV preprocessing globals
            DETECT_DELIMITER = detect_delimiter
            FIX_QUOTES = fix_quotes
            REMOVE_CONTROL_CHARS = remove_control_chars
            NORMALIZE_NEWLINES = normalize_newlines
            SKIP_BAD_LINES = skip_bad_lines
            SHOW_SAMPLE = show_sample
            
            return run_model_test(
                champion_type, champion_name, champion_api_url, champion_api_key, champion_model_id, champion_max_tokens, champion_temperature,
                challenger_type, challenger_name, challenger_api_url, challenger_api_key, challenger_model_id, challenger_max_tokens, challenger_temperature,
                judge_type, judge_name, judge_api_url, judge_api_key, judge_model_id, judge_max_tokens, judge_temperature,
                csv_url, key_column, value_column, limit, output_dir, prompt_template, evaluation_template,
                progress
            )
        
        # Connect function to run button
        run_button.click(
            fn=run_test_with_preprocessing,
            inputs=[
                champion_type, champion_name, champion_api_url, champion_api_key, champion_model_id, champion_max_tokens, champion_temperature,
                challenger_type, challenger_name, challenger_api_url, challenger_api_key, challenger_model_id, challenger_max_tokens, challenger_temperature,
                judge_type, judge_name, judge_api_url, judge_api_key, judge_model_id, judge_max_tokens, judge_temperature,
                csv_url, key_column, value_column, limit, output_dir, prompt_template, evaluation_template,
                preprocess_inputs, max_length, remove_special_chars, normalize_whitespace,
                detect_delimiter, fix_quotes, remove_control_chars, normalize_newlines, skip_bad_lines, show_sample
            ],
            outputs=[results_markdown, json_file, csv_file]
        )
        
        # Add example data
        gr.Examples(
            examples=[
                ["OpenAI", "GPT-3.5", "https://api.openai.com/v1/chat/completions", "YOUR_KEY", "gpt-3.5-turbo", 1024, 0.0,
                 "Anthropic", "Claude-Haiku", "https://api.anthropic.com/v1/messages", "YOUR_KEY", "claude-3-haiku-20240307", 1024, 0.0,
                 "OpenAI", "GPT-4", "https://api.openai.com/v1/chat/completions", "YOUR_KEY", "gpt-4", 2048, 0.0,
                 "https://example.com/data.csv", "text", "label", 10, "./results"],
                ["Ollama", "Llama3-8B", "http://localhost:11434/api/generate", "", "llama3:8b", 1024, 0.0,
                 "Gemini", "Gemini-Pro", "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent", "YOUR_KEY", "gemini-pro", 1024, 0.0,
                 "Anthropic", "Claude-Opus", "https://api.anthropic.com/v1/messages", "YOUR_KEY", "claude-3-opus-20240229", 2048, 0.0,
                 "https://example.com/data.csv", "text", "label", 5, "./results"]
            ],
            inputs=[
                champion_type, champion_name, champion_api_url, champion_api_key, champion_model_id, champion_max_tokens, champion_temperature,
                challenger_type, challenger_name, challenger_api_url, challenger_api_key, challenger_model_id, challenger_max_tokens, challenger_temperature,
                judge_type, judge_name, judge_api_url, judge_api_key, judge_model_id, judge_max_tokens, judge_temperature,
                csv_url, key_column, value_column, limit, output_dir
            ]
        )
    
    return app


# Run the Gradio app
if __name__ == "__main__":
    try:
        print("Starting Model A/B Testing Tool...")
        app = create_gradio_interface()
        print("Launching Gradio interface at http://127.0.0.1:7860")
        print("If your browser doesn't open automatically, please visit that URL manually.")
        app.launch(inbrowser=True)  # Try to force browser to open
    except Exception as e:
        print(f"Error launching Gradio interface: {str(e)}")
        print("Check that gradio is installed: pip install gradio")
