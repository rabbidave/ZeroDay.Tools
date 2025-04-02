import sys
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
from urllib.parse import urlparse

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
    key: str # The input/prompt for the model
    value: str # The reference value/ground truth
    id: Optional[str] = None # Unique ID for the test case


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
    winner: str  # "MODEL_A_WINS", "MODEL_B_WINS", or "TIE" (extracted from reasoning)
    confidence: float # Extracted confidence score (e.g., 4/5 -> 0.8)
    reasoning: str # Full response from the judge model


# Global preprocessing settings (can be updated through UI)
PREPROCESS_ENABLED = True
MAX_LENGTH = 8000
REMOVE_SPECIAL_CHARS = True
NORMALIZE_WHITESPACE = True

# CSV preprocessing settings (specific to CSV format)
DETECT_DELIMITER = True
FIX_QUOTES = True
REMOVE_CONTROL_CHARS = True
NORMALIZE_NEWLINES = True
SKIP_BAD_LINES = True
SHOW_SAMPLE = True # Show sample data after loading & preprocessing

# --- Text Preprocessing Function ---
def preprocess_text(text, max_length=None, remove_special_chars=None, normalize_whitespace=None):
    """
    Preprocess text (key or value) before using in prompts or comparisons.
    - Truncate to prevent token limits
    - Remove problematic characters
    - Normalize whitespace
    """
    # Use global settings if not specified
    if max_length is None: max_length = MAX_LENGTH
    if remove_special_chars is None: remove_special_chars = REMOVE_SPECIAL_CHARS
    if normalize_whitespace is None: normalize_whitespace = NORMALIZE_WHITESPACE

    # Skip preprocessing if disabled globally
    if not PREPROCESS_ENABLED:
        return str(text) if text is not None else ""

    if text is None: return ""
    text = str(text) # Ensure it's a string

    # Truncate
    if len(text) > max_length:
        text = text[:max_length] + "... [truncated]"

    if remove_special_chars:
        # Remove control characters and other potentially problematic characters
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        # Remove any XML/HTML-like tags that might interfere
        text = re.sub(r'<[^>]+>', '', text)

    if normalize_whitespace:
        # Normalize whitespace (multiple spaces, tabs, newlines to single space)
        text = re.sub(r'\s+', ' ', text)
        # But preserve paragraph breaks for readability (optional, maybe confusing)
        # text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()

    return text

# --- Model Runner Class ---
# (No changes needed in ModelRunner, LMJudge, ResultAggregator, ModelTester core logic)
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
            # Preprocess the input key using the global settings
            preprocessed_key = preprocess_text(test_case.key)

            # Format prompt using the preprocessed key
            prompt = ""
            try:
                # For judge prompts, the "key" is already the full prompt
                if test_case.id and test_case.id.startswith("judge_"):
                    prompt = preprocessed_key # Use directly, assume already preprocessed if needed
                else:
                    # Use simple replacement first, escaping existing braces in the key
                    safe_key = preprocessed_key.replace("{", "{{").replace("}", "}}")
                    prompt = self.prompt_template.replace("{key}", safe_key)
            except Exception as e:
                 logger.warning(f"Error formatting prompt template with replace: {str(e)}. Falling back.")
                 try:
                      prompt = self.prompt_template.format(key=preprocessed_key)
                 except Exception as e2:
                      logger.error(f"Error formatting prompt template: {str(e2)}. Using concatenation.")
                      prompt = f"{self.prompt_template}\n\nINPUT: {preprocessed_key}"

            # Determine API type and make appropriate call
            response_text = ""
            api_url_lower = self.endpoint.api_url.lower() if self.endpoint.api_url else ""

            try:
                # Add more specific checks based on common API structures
                is_openai_compatible = "/v1/chat/completions" in api_url_lower or \
                                        "openai" in api_url_lower or \
                                        "openrouter.ai" in api_url_lower or \
                                        "lmstudio.ai" in api_url_lower or \
                                        ":1234/v1" in api_url_lower # Common LM Studio port

                is_anthropic_compatible = "/v1/messages" in api_url_lower or "anthropic" in api_url_lower
                is_gemini = "generativelanguage.googleapis.com" in api_url_lower
                is_ollama = "ollama" in api_url_lower and "/api/generate" in api_url_lower

                if is_openai_compatible:
                    response_text = self._call_openai_compatible_api(prompt)
                elif is_anthropic_compatible:
                    response_text = self._call_anthropic_api(prompt)
                elif is_gemini:
                     response_text = self._call_gemini_api(prompt)
                elif is_ollama:
                    response_text = self._call_ollama_api(prompt)
                else:
                    # Fallback to generic or attempt intelligent guess
                    logger.warning(f"Could not determine API type for {self.endpoint.api_url}. Attempting generic call.")
                    response_text = self._call_generic_api(prompt) # Or try OpenAI as a default?

            except requests.exceptions.RequestException as req_err:
                 logger.error(f"API request failed for {self.endpoint.name}: {req_err}")
                 response_text = f"Error: API request failed. Details: {str(req_err)}"
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as parse_err:
                 logger.error(f"Failed to parse response from {self.endpoint.name}: {parse_err}")
                 response_text = f"Error: Failed to parse API response. Details: {str(parse_err)}"
            except Exception as e:
                logger.error(f"Unexpected error calling API for {self.endpoint.name}: {str(e)}")
                response_text = f"Error: An unexpected error occurred. Details: {str(e)}"


            end_time = time.time()

            return ModelResponse(
                test_id=test_case.id or "unknown", # Ensure test_id is never None
                model_name=self.endpoint.name,
                output=str(response_text), # Ensure output is always string
                latency=end_time - start_time,
            )
        except Exception as e:
            logger.error(f"Unexpected error in generate method for {self.endpoint.name}: {str(e)}")
            # Re-raise to trigger tenacity retry
            raise

    def _prepare_headers(self):
        """Prepares common headers, including Authorization if API key exists."""
        headers = {"Content-Type": "application/json"}
        if self.endpoint.api_key and self.endpoint.api_key.strip():
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"

        # Add OpenRouter specific headers if applicable
        if "openrouter.ai" in self.endpoint.api_url.lower():
             # These might be optional now, but good practice
             headers["HTTP-Referer"] = "http://localhost" # Can be anything, localhost is common
             headers["X-Title"] = "Model A/B Testing Tool"
        return headers

    def _call_openai_compatible_api(self, prompt: str) -> str:
        """Calls APIs following the OpenAI chat completions format."""
        logger.info(f"Calling OpenAI-compatible API: {self.endpoint.api_url} for model {self.endpoint.model_id}")
        headers = self._prepare_headers()
        data = {
            "model": self.endpoint.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.endpoint.max_tokens,
            "temperature": self.endpoint.temperature,
        }
        try:
            response = requests.post(self.endpoint.api_url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            if not result.get("choices"):
                 raise ValueError(f"Invalid response format: 'choices' key missing or empty. Response: {result}")
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI-compatible request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None: logger.error(f"Response content: {e.response.text}")
            raise
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Failed to parse OpenAI-compatible response: {str(e)}")
            logger.error(f"Full response: {result if 'result' in locals() else 'Response not available'}")
            raise

    def _call_anthropic_api(self, prompt: str) -> str:
        """Calls the Anthropic messages API."""
        logger.info(f"Calling Anthropic API: {self.endpoint.api_url} for model {self.endpoint.model_id}")
        headers = self._prepare_headers()
        # Anthropic requires API key via header, not Bearer token
        if "Authorization" in headers: del headers["Authorization"]
        if self.endpoint.api_key: headers["x-api-key"] = self.endpoint.api_key
        headers["anthropic-version"] = "2023-06-01" # Required header

        data = {
            "model": self.endpoint.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.endpoint.max_tokens,
            "temperature": self.endpoint.temperature,
        }
        try:
            response = requests.post(self.endpoint.api_url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            if not result.get("content"):
                 raise ValueError(f"Invalid response format: 'content' key missing or empty. Response: {result}")
            # Assuming the first content block is the text response
            return result["content"][0]["text"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Anthropic request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None: logger.error(f"Response content: {e.response.text}")
            raise
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Failed to parse Anthropic response: {str(e)}")
            logger.error(f"Full response: {result if 'result' in locals() else 'Response not available'}")
            raise

    def _call_ollama_api(self, prompt: str) -> str:
        """Calls the Ollama generate API."""
        logger.info(f"Calling Ollama API: {self.endpoint.api_url} for model {self.endpoint.model_id}")
        # Ollama doesn't use API keys or standard headers
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.endpoint.model_id,
            "prompt": prompt,
            "stream": False, # Ensure we get the full response at once
            "options": {
                "temperature": self.endpoint.temperature,
                # Ollama might not support max_tokens directly in generate,
                # it's often part of model parameters or uses context window limits.
                # "num_predict": self.endpoint.max_tokens # Common parameter name for max tokens
            }
        }
        try:
            response = requests.post(self.endpoint.api_url, headers=headers, json=data, timeout=180) # Longer timeout for local models
            response.raise_for_status()
            result = response.json()
            if "response" not in result:
                 raise ValueError(f"Invalid response format: 'response' key missing. Response: {result}")
            return result["response"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None: logger.error(f"Response content: {e.response.text}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse Ollama response: {str(e)}")
            logger.error(f"Full response: {result if 'result' in locals() else 'Response not available'}")
            raise

    def _call_gemini_api(self, prompt: str) -> str:
        """Calls the Google Gemini API."""
        logger.info(f"Calling Gemini API for model {self.endpoint.model_id}")
        # Gemini uses API key in the URL usually
        api_url = f"{self.endpoint.api_url}?key={self.endpoint.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.endpoint.temperature,
                "maxOutputTokens": self.endpoint.max_tokens,
            }
        }
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            # Navigate the Gemini response structure
            if not result.get("candidates") or not result["candidates"][0].get("content") or not result["candidates"][0]["content"].get("parts"):
                 raise ValueError(f"Invalid response format. Response: {result}")
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None: logger.error(f"Response content: {e.response.text}")
            raise
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Failed to parse Gemini response: {str(e)}")
            logger.error(f"Full response: {result if 'result' in locals() else 'Response not available'}")
            raise

    def _call_generic_api(self, prompt: str) -> str:
        """Attempts a generic POST request, assuming OpenAI-like structure as a guess."""
        logger.warning(f"Attempting generic API call (assuming OpenAI format) to: {self.endpoint.api_url}")
        try:
            # Try OpenAI format as the most common fallback
            return self._call_openai_compatible_api(prompt)
        except Exception as e:
            logger.error(f"Generic API call failed: {str(e)}. Returning error message.")
            return f"Error: Failed to call or parse response from generic API endpoint {self.endpoint.api_url}. Please check configuration and API documentation. Details: {str(e)}"


class LMJudge:
    """Uses a language model to judge between champion and challenger outputs."""

    DEFAULT_EVALUATION_PROMPT = """
# Model Response Evaluation

You are evaluating two AI model responses based on the input query and potentially a reference value.

## Input Query
```
{key}
```
{reference_section}

## Model A (Champion: {champion_name}) Response
```
{champion_output}
```

## Model B (Challenger: {challenger_name}) Response
```
{challenger_output}
```

## Evaluation Instructions
Compare Model A and Model B based on the Input Query{reference_value_instruction}. Consider:
1. Relevance and accuracy in addressing the Input Query.
{reference_value_criteria}
{clarity_criteria_number}. Clarity, conciseness, and quality of the response.
{overall_criteria_number}. Overall usefulness.

## Required Response Format
You MUST start your response with a clear verdict and confidence rating:

VERDICT: [Choose ONE: MODEL_A_WINS, MODEL_B_WINS, or TIE]
CONFIDENCE: [Number]/5 (where 1=low confidence, 5=high confidence)

Then provide a detailed explanation of your reasoning. Be explicit about which model performed better and why, or why they were tied. Include specific examples from each response that influenced your decision.

Example format:
VERDICT: MODEL_A_WINS
CONFIDENCE: 4/5

[Your detailed reasoning here...]
"""

    def __init__(
        self,
        endpoint: ModelEndpoint,
        evaluation_prompt_template: str = DEFAULT_EVALUATION_PROMPT,
    ):
        self.endpoint = endpoint
        self.evaluation_prompt_template = evaluation_prompt_template
        # The judge runner uses a simple placeholder template, as the full prompt
        # is formatted within the evaluate method before being passed as the 'key'.
        self.model_runner = ModelRunner(endpoint, "{key}") # Pass-through template

    def evaluate(
        self,
        test_case: TestCase,
        champion_response: ModelResponse,
        challenger_response: ModelResponse
    ) -> EvaluationResult:
        """Evaluate champion vs challenger outputs using a dynamically built prompt."""
        # Preprocess all inputs to ensure they're clean strings
        # Use the same preprocess_text function for consistency
        preprocessed_key = preprocess_text(test_case.key)
        preprocessed_value = preprocess_text(test_case.value) # Preprocess reference value too
        preprocessed_champion = preprocess_text(champion_response.output)
        preprocessed_challenger = preprocess_text(challenger_response.output)

        # Prepare context for the evaluation prompt template
        has_reference = bool(preprocessed_value)
        reference_section_text = f"\n## Reference Value\n```\n{preprocessed_value}\n```" if has_reference else "\n## Reference Value\nN/A"
        reference_value_instruction_text = ' and Reference Value' if has_reference else ''
        reference_value_criteria_text = '2. Factual correctness compared to the Reference Value (if provided).' if has_reference else ''
        clarity_criteria_number_text = '3' if has_reference else '2'
        overall_criteria_number_text = '4' if has_reference else '3'

        # Format the evaluation prompt using the template and context
        try:
            evaluation_prompt = self.evaluation_prompt_template.format(
                key=preprocessed_key,
                reference_section=reference_section_text,
                champion_name=champion_response.model_name,
                champion_output=preprocessed_champion,
                challenger_name=challenger_response.model_name,
                challenger_output=preprocessed_challenger,
                reference_value_instruction=reference_value_instruction_text,
                reference_value_criteria=reference_value_criteria_text,
                clarity_criteria_number=clarity_criteria_number_text,
                overall_criteria_number=overall_criteria_number_text
            )
        except KeyError as e:
             logger.error(f"Missing key in judge prompt template: {e}. Using default prompt structure.")
             # Fallback to a basic structure if formatting fails
             evaluation_prompt = f"Evaluate Model A vs Model B.\nInput: {preprocessed_key}\nRef: {preprocessed_value}\nA: {preprocessed_champion}\nB: {preprocessed_challenger}\nFormat: VERDICT: [MODEL_A_WINS/MODEL_B_WINS/TIE]\nCONFIDENCE: [1-5]/5\nReasoning: ..."
        except Exception as e:
             logger.error(f"Error formatting judge prompt template: {e}. Using basic prompt.")
             evaluation_prompt = f"Evaluate Model A vs Model B.\nInput: {preprocessed_key}\nRef: {preprocessed_value}\nA: {preprocessed_champion}\nB: {preprocessed_challenger}\nFormat: VERDICT: [MODEL_A_WINS/MODEL_B_WINS/TIE]\nCONFIDENCE: [1-5]/5\nReasoning: ..."


        # Log the prompt for debugging (truncated)
        logger.info(f"Using Judge evaluation prompt (truncated): {evaluation_prompt[:500]}...")

        # Get judge's response using the constructed prompt as the 'key'
        judge_test_case = TestCase(
            key=evaluation_prompt,
            value="", # No value needed for judge call itself
            id=f"judge_{test_case.id or 'unknown'}"
        )
        judge_response_obj = self.model_runner.generate(judge_test_case)

        # Log the response for debugging (truncated)
        logger.info(f"Judge raw response (truncated): {judge_response_obj.output[:500]}...")

        # Parse the judge's decision from the raw output string
        parsed_result = self.parse_judge_response(judge_response_obj.output)

        return EvaluationResult(
            test_id=test_case.id or "unknown",
            champion_output=champion_response.output, # Store original, not preprocessed
            challenger_output=challenger_response.output, # Store original, not preprocessed
            winner=parsed_result["winner"],
            confidence=parsed_result["confidence"],
            reasoning=judge_response_obj.output, # Store the full raw response as reasoning
        )

    def parse_judge_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the judge's raw response string to extract verdict and confidence.
        Uses more flexible regex patterns to handle various response formats.
        """
        verdict = "UNDETERMINED"
        confidence = 0.0

        # Log the first part of the response for debugging
        logger.info(f"Parsing judge response (first 100 chars): {response_text[:100]}")

        # Extract verdict from anywhere in the text - more flexible approach
        # First check for explicit verdict statements
        verdict_patterns = [
            # Direct verdict statements
            r"VERDICT:\s*(MODEL_A_WINS|MODEL_B_WINS|TIE)",
            r"verdict[:\s]*\s*(MODEL_A_WINS|MODEL_B_WINS|TIE)",
            # Look for "winner" statements in the reasoning
            r"winner[:\s]*\s*(MODEL_A_WINS|MODEL_B_WINS|TIE)",
            # Look for bracketed model indicators like [[A]] or [[MODEL_A]]
            r"\[\[\s*([AB]|MODEL_[AB]|TIE)\s*\]\]",
            # Look for double-bracketed verdict at start of response (common pattern)
            r"^\s*\[\[\s*([AB]|MODEL_[AB]|TIE)\s*\]\]"
        ]

        for pattern in verdict_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                # Extract the verdict
                verdict_text = match.group(1).upper()
                logger.info(f"Found verdict pattern match: {verdict_text}")
                if verdict_text == "A" or verdict_text == "MODEL_A":
                    verdict = "MODEL_A_WINS"
                    break
                elif verdict_text == "B" or verdict_text == "MODEL_B":
                    verdict = "MODEL_B_WINS"
                    break
                elif verdict_text == "TIE":
                    verdict = "TIE"
                    break
                else:
                    # Direct match for full verdict strings
                    verdict = verdict_text
                    break

        # If no explicit verdict found, look for descriptive statements
        if verdict == "UNDETERMINED":
            # Check for statements about Model A being better
            if re.search(r"(Model A|Champion).*?(better|superior|wins|outperforms|delivers a superior response)", response_text, re.IGNORECASE) and not re.search(r"(Model B|Challenger).*?(better|superior|wins|outperforms)", response_text, re.IGNORECASE):
                verdict = "MODEL_A_WINS"
                logger.info("Detected Model A wins from descriptive text")
            # Check for statements about Model B being better
            elif re.search(r"(Model B|Challenger).*?(better|superior|wins|outperforms|delivers a superior response)", response_text, re.IGNORECASE) and not re.search(r"(Model A|Champion).*?(better|superior|wins|outperforms)", response_text, re.IGNORECASE):
                verdict = "MODEL_B_WINS"
                logger.info("Detected Model B wins from descriptive text")
            # Check for tie statements
            elif re.search(r"(both models are comparable|neither model is clearly better|tie)", response_text, re.IGNORECASE):
                verdict = "TIE"
                logger.info("Detected TIE from descriptive text")
            else:
                # Look for conclusion statements
                conclusion_match = re.search(r"(conclusion|in summary|overall).*?(model [ab]|champion|challenger)", response_text, re.IGNORECASE)
                if conclusion_match:
                    conclusion = conclusion_match.group(0).lower()
                    if ("model a" in conclusion or "champion" in conclusion) and not ("model b" in conclusion or "challenger" in conclusion):
                        verdict = "MODEL_A_WINS"
                        logger.info("Detected Model A wins from conclusion")
                    elif ("model b" in conclusion or "challenger" in conclusion) and not ("model a" in conclusion or "champion" in conclusion):
                        verdict = "MODEL_B_WINS"
                        logger.info("Detected Model B wins from conclusion")

                # Check for "superior response" pattern
                if verdict == "UNDETERMINED":
                    if "Model A" in response_text and "superior response" in response_text:
                        verdict = "MODEL_A_WINS"
                        logger.info("Detected Model A wins from 'superior response' pattern")
                    elif "Model B" in response_text and "superior response" in response_text:
                        verdict = "MODEL_B_WINS"
                        logger.info("Detected Model B wins from 'superior response' pattern")

                # If still undetermined, log the issue
                if verdict == "UNDETERMINED":
                    logger.warning(f"Could not parse VERDICT from judge response: {response_text[:200]}...")

        # Try to extract CONFIDENCE from anywhere in the text
        confidence_patterns = [
            r"CONFIDENCE:\s*(\d+)\s*/\s*5",
            r"confidence[:\s]*\s*(\d+)\s*/\s*5",
            r"confidence[:\s]*\s*(\d+)\.(\d+)",  # For decimal confidence like 4.5/5
            r"confidence[:\s]*\s*(\d+)"  # For just a number
        ]

        for pattern in confidence_patterns:
            confidence_match = re.search(pattern, response_text, re.IGNORECASE)
            if confidence_match:
                try:
                    if '.' in confidence_match.group(0):
                        # Handle decimal confidence
                        confidence_parts = re.findall(r'\d+\.\d+', confidence_match.group(0))
                        if confidence_parts:
                            confidence_score = float(confidence_parts[0])
                            # Normalize to 0-1 range assuming it's out of 5
                            confidence = max(0.2, min(1.0, confidence_score / 5.0))
                            logger.info(f"Parsed decimal confidence: {confidence_score} -> {confidence}")
                    else:
                        # Handle integer confidence
                        confidence_score = int(confidence_match.group(1))
                        # Clamp confidence between 1 and 5, then normalize
                        confidence = max(0.2, min(1.0, confidence_score / 5.0))
                        logger.info(f"Parsed integer confidence: {confidence_score} -> {confidence}")
                    break
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse CONFIDENCE value: {confidence_match.group(0)}")

        # Look for numeric scores like "9.5/10" or "4.0/10"
        if confidence == 0.0:
            score_match = re.search(r"(\d+\.?\d*)\s*/\s*10", response_text)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    # Convert from 10-point scale to 0-1 range
                    confidence = max(0.2, min(1.0, score / 10.0))
                    logger.info(f"Parsed 10-scale confidence: {score} -> {confidence}")
                except (ValueError, IndexError):
                    pass

        # Look for final scores like "9.5" and "4.0" in the conclusion
        if confidence == 0.0:
            # Look for "Final Scores" section with numeric values
            score_section_match = re.search(r"Final Scores.*?(\d+\.?\d*)", response_text, re.IGNORECASE | re.DOTALL)
            if score_section_match:
                try:
                    score = float(score_section_match.group(1))
                    # Normalize to 0-1 range assuming it's out of 10
                    confidence = max(0.2, min(1.0, score / 10.0))
                    logger.info(f"Parsed score from 'Final Scores' section: {score} -> {confidence}")
                except (ValueError, IndexError):
                    pass

        # If still no confidence, estimate based on reasoning length and language
        if confidence == 0.0:
            # Check for strong confidence language
            if re.search(r"(clearly|significantly|substantially|definitely|without doubt|strong)", response_text, re.IGNORECASE):
                confidence = 0.8
                logger.info("Using strong confidence language estimate: 0.8")
            # Check for moderate confidence language
            elif re.search(r"(somewhat|moderately|reasonably|relatively)", response_text, re.IGNORECASE):
                confidence = 0.6
                logger.info("Using moderate confidence language estimate: 0.6")
            # Default based on reasoning length
            else:
                reasoning_length = len(response_text)
                if reasoning_length > 1000:
                    confidence = 0.7  # High confidence for detailed reasoning
                    logger.info("Using reasoning length estimate (>1000 chars): 0.7")
                elif reasoning_length > 500:
                    confidence = 0.5  # Medium confidence
                    logger.info("Using reasoning length estimate (>500 chars): 0.5")
                else:
                    confidence = 0.3  # Lower confidence for brief reasoning
                    logger.info("Using reasoning length estimate (<500 chars): 0.3")

        # If we have a verdict but no confidence, set a default confidence
        if verdict != "UNDETERMINED" and confidence == 0.0:
            confidence = 0.6  # Default confidence when we have a verdict but couldn't parse confidence
            logger.info(f"Setting default confidence for verdict {verdict}: 0.6")

        # Log the final verdict and confidence
        logger.info(f"Final verdict: {verdict}, confidence: {confidence}")

        return {
            "winner": verdict,
            "confidence": confidence,
            # Reasoning is the full response text, handled in evaluate method
        }


class ResultAggregator:
    """Collects evaluation results and calculates summary statistics."""

    def aggregate(self, evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Aggregates results, calculating counts and percentages."""
        total_evaluations = len(evaluation_results)
        verdict_counts = {"MODEL_A_WINS": 0, "MODEL_B_WINS": 0, "TIE": 0, "UNDETERMINED": 0, "JUDGE_ERROR": 0}
        confidence_sum = 0
        valid_verdicts = 0

        # Track which test cases had undetermined verdicts for logging
        undetermined_cases = []

        for result in evaluation_results:
            verdict = result.winner # Use the pre-parsed winner
            if verdict in verdict_counts:
                verdict_counts[verdict] += 1
                if verdict != "UNDETERMINED" and verdict != "JUDGE_ERROR":
                    confidence_sum += result.confidence
                    valid_verdicts += 1
            else:
                # Should not happen if parsing is correct, but handle defensively
                verdict_counts["UNDETERMINED"] += 1

            # Log undetermined cases for debugging
            if verdict == "UNDETERMINED":
                undetermined_cases.append(result.test_id)

        # Log summary of undetermined cases
        if undetermined_cases:
            logger.warning(f"Found {len(undetermined_cases)} undetermined verdicts: {undetermined_cases[:5]}" +
                          (f"... and {len(undetermined_cases)-5} more" if len(undetermined_cases) > 5 else ""))


        # Calculate percentages based on determined verdicts only
        determined_verdicts = total_evaluations - verdict_counts["UNDETERMINED"] - verdict_counts["JUDGE_ERROR"]
        verdict_percentages = {}
        if determined_verdicts > 0:
            verdict_percentages["MODEL_A_WINS"] = round(
                (verdict_counts["MODEL_A_WINS"] / determined_verdicts) * 100, 2
            )
            verdict_percentages["MODEL_B_WINS"] = round(
                (verdict_counts["MODEL_B_WINS"] / determined_verdicts) * 100, 2
            )
            verdict_percentages["TIE"] = round(
                (verdict_counts["TIE"] / determined_verdicts) * 100, 2
            )
        else:
            verdict_percentages = {"MODEL_A_WINS": 0, "MODEL_B_WINS": 0, "TIE": 0}

        average_confidence = (confidence_sum / valid_verdicts) if valid_verdicts > 0 else 0

        return {
            "total_evaluations": total_evaluations,
            "verdict_counts": verdict_counts,
            "verdict_percentages": verdict_percentages, # Based on determined verdicts
            "average_confidence": round(average_confidence, 3), # Avg confidence for non-undetermined
            "raw_evaluations": [res.__dict__ for res in evaluation_results] # Keep raw for output
        }


class ModelTester:
    """Main class that orchestrates the A/B testing process."""

    def __init__(
        self,
        champion_endpoint: ModelEndpoint,
        challenger_endpoint: ModelEndpoint,
        judge_endpoint: ModelEndpoint,
        model_prompt_template: str,
        judge_prompt_template: str = LMJudge.DEFAULT_EVALUATION_PROMPT # Add judge template param
    ):
        self.champion_runner = ModelRunner(champion_endpoint, model_prompt_template)
        self.challenger_runner = ModelRunner(challenger_endpoint, model_prompt_template)
        # Pass the judge prompt template to the LMJudge constructor
        self.judge = LMJudge(judge_endpoint, evaluation_prompt_template=judge_prompt_template)
        self.aggregator = ResultAggregator() # Aggregator just collects/counts
        self.champion_endpoint = champion_endpoint
        self.challenger_endpoint = challenger_endpoint
        self.judge_endpoint = judge_endpoint

    def run_test(self, test_cases: List[TestCase], batch_size: int = 5, progress=None) -> Dict[str, Any]:
        """Run the complete test process: generate responses, evaluate, aggregate."""
        all_evaluation_results: List[EvaluationResult] = []
        champion_metrics = {"total_latency": 0.0, "total_output_chars": 0, "success_count": 0, "error_count": 0}
        challenger_metrics = {"total_latency": 0.0, "total_output_chars": 0, "success_count": 0, "error_count": 0}
        judge_metrics = {"total_latency": 0.0, "total_output_chars": 0, "success_count": 0, "error_count": 0}

        num_cases = len(test_cases)
        if num_cases == 0:
             logger.warning("No test cases provided to run_test.")
             return {"evaluations": [], "summary": {"error": "No test cases loaded."}}

        total_batches = (num_cases + batch_size - 1) // batch_size
        current_case_index = 0

        # Process in batches
        for i in range(0, num_cases, batch_size):
            batch = test_cases[i:min(i + batch_size, num_cases)]
            batch_num = i // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches} (Cases {i+1}-{min(i+batch_size, num_cases)})")

            if progress is not None:
                progress(i / num_cases, f"Running Batch {batch_num}/{total_batches}")

            # Store responses and evaluation results for the batch
            batch_champ_responses: Dict[str, ModelResponse] = {}
            batch_chall_responses: Dict[str, ModelResponse] = {}
            batch_eval_results: List[EvaluationResult] = []

            # 1. Get responses from Champion and Challenger models
            for test_case in batch:
                 current_case_index += 1
                 case_id = test_case.id or f"case_{current_case_index}" # Ensure ID
                 test_case.id = case_id # Update test case object with ID

                 # Champion
                 try:
                      champ_resp = self.champion_runner.generate(test_case)
                      batch_champ_responses[case_id] = champ_resp
                      champion_metrics["total_latency"] += champ_resp.latency
                      champion_metrics["total_output_chars"] += len(champ_resp.output)
                      if not champ_resp.output.startswith("Error:"): champion_metrics["success_count"] += 1
                      else: champion_metrics["error_count"] += 1
                 except Exception as e:
                      logger.error(f"Error generating champion response for case {case_id}: {e}")
                      batch_champ_responses[case_id] = ModelResponse(case_id, self.champion_endpoint.name, f"Error: Generation failed - {e}", 0)
                      champion_metrics["error_count"] += 1

                 # Challenger
                 try:
                      chall_resp = self.challenger_runner.generate(test_case)
                      batch_chall_responses[case_id] = chall_resp
                      challenger_metrics["total_latency"] += chall_resp.latency
                      challenger_metrics["total_output_chars"] += len(chall_resp.output)
                      if not chall_resp.output.startswith("Error:"): challenger_metrics["success_count"] += 1
                      else: challenger_metrics["error_count"] += 1
                 except Exception as e:
                      logger.error(f"Error generating challenger response for case {case_id}: {e}")
                      batch_chall_responses[case_id] = ModelResponse(case_id, self.challenger_endpoint.name, f"Error: Generation failed - {e}", 0)
                      challenger_metrics["error_count"] += 1


            # 2. Evaluate with LM Judge
            if progress is not None:
                progress((i + batch_size * 0.5) / num_cases, f"Evaluating Batch {batch_num}/{total_batches}")

            for test_case in batch:
                 case_id = test_case.id # Should have been set above
                 champ_response = batch_champ_responses.get(case_id)
                 chall_response = batch_chall_responses.get(case_id)

                 # Skip evaluation if either model failed catastrophically
                 if not champ_response or not chall_response:
                      logger.warning(f"Skipping evaluation for case {case_id} due to missing model response.")
                      # Create a dummy evaluation result indicating failure? Or just skip? Let's skip.
                      continue

                 # If one model produced an error string but the other didn't, judge might still work
                 # If both produced errors, evaluation is pointless

                 try:
                      start_time = time.time()
                      evaluation_result = self.judge.evaluate(
                          test_case,
                          champ_response,
                          chall_response,
                      )
                      judge_latency = time.time() - start_time
                      judge_metrics["total_latency"] += judge_latency
                      judge_metrics["total_output_chars"] += len(evaluation_result.reasoning) # Judge output length
                      batch_eval_results.append(evaluation_result)
                      # Check if judge produced a valid verdict (not undetermined)
                      if evaluation_result.winner != "UNDETERMINED": judge_metrics["success_count"] += 1
                      else: judge_metrics["error_count"] += 1

                 except Exception as e:
                      logger.error(f"Error during judge evaluation for case {case_id}: {e}")
                      # Create a placeholder eval result indicating judge failure
                      batch_eval_results.append(EvaluationResult(
                           test_id=case_id,
                           champion_output=champ_response.output,
                           challenger_output=chall_response.output,
                           winner="JUDGE_ERROR",
                           confidence=0.0,
                           reasoning=f"Error: Judge evaluation failed - {e}"
                      ))
                      judge_metrics["error_count"] += 1


            all_evaluation_results.extend(batch_eval_results)

            # Log batch summary (optional)
            batch_summary = self.aggregator.aggregate(batch_eval_results)
            logger.info(f"Batch {batch_num} summary: {batch_summary['verdict_counts']}")


        # 3. Aggregate final results
        aggregated_summary = self.aggregator.aggregate(all_evaluation_results)

        # 4. Calculate final metrics
        def calculate_avg_metrics(metrics, count):
            if count == 0: return {"avg_latency": 0, "avg_chars": 0, "success_rate": 0, "errors": metrics.get("error_count", 0)}
            total_runs = metrics["success_count"] + metrics["error_count"]
            return {
                "avg_latency": round(metrics["total_latency"] / total_runs, 2) if total_runs > 0 else 0,
                "avg_chars": int(metrics["total_output_chars"] / metrics["success_count"]) if metrics["success_count"] > 0 else 0,
                "success_rate": round((metrics["success_count"] / total_runs) * 100, 1) if total_runs > 0 else 0,
                "errors": metrics["error_count"]
            }

        champion_avg_metrics = calculate_avg_metrics(champion_metrics, num_cases)
        challenger_avg_metrics = calculate_avg_metrics(challenger_metrics, num_cases)
        judge_avg_metrics = calculate_avg_metrics(judge_metrics, num_cases) # Judge runs once per case


        # 5. Determine overall decision based on aggregated results
        decision = "MAINTAIN_CHAMPION" # Default
        reason = "Insufficient data or challenger did not significantly outperform."
        win_margin_threshold = 5 # Challenger needs to win by at least 5% points
        min_determined_verdicts = 3 # Minimum number of determined verdicts needed for a reliable decision

        percentages = aggregated_summary["verdict_percentages"]
        determined_verdicts = num_cases - aggregated_summary["verdict_counts"].get("UNDETERMINED", 0) - aggregated_summary["verdict_counts"].get("JUDGE_ERROR", 0)

        if determined_verdicts >= min_determined_verdicts:
            champ_wins_pct = percentages.get("MODEL_A_WINS", 0)
            chall_wins_pct = percentages.get("MODEL_B_WINS", 0)
            ties_pct = percentages.get("TIE", 0)

            # Calculate confidence-weighted percentages if we have confidence scores
            avg_confidence = aggregated_summary["average_confidence"]
            confidence_factor = f" with {avg_confidence:.2f} average confidence" if avg_confidence > 0 else ""

            if chall_wins_pct > champ_wins_pct + win_margin_threshold:
                 decision = "REPLACE_WITH_CHALLENGER"
                 reason = f"Challenger won {chall_wins_pct}% vs Champion's {champ_wins_pct}%{confidence_factor} (>{win_margin_threshold}% margin)."
            elif champ_wins_pct > chall_wins_pct + win_margin_threshold:
                 decision = "MAINTAIN_CHAMPION"
                 reason = f"Champion won {champ_wins_pct}% vs Challenger's {chall_wins_pct}%{confidence_factor}."
            else:
                 # Closer results, consider ties or maintain status quo
                 decision = "MAINTAIN_CHAMPION"
                 reason = f"Results close ({champ_wins_pct}% vs {chall_wins_pct}%, {ties_pct}% ties){confidence_factor}. Challenger did not show clear superiority."
        else:
            # Not enough determined verdicts for a reliable decision
            decision = "MAINTAIN_CHAMPION"
            reason = f"Insufficient determined verdicts ({determined_verdicts}/{num_cases}) to make a reliable decision. Defaulting to maintaining champion."


        # Log final summary
        logger.info(f"Final Aggregated Results: {aggregated_summary['verdict_counts']}")
        logger.info(f"Final Percentages: {aggregated_summary['verdict_percentages']}")
        logger.info(f"Decision: {decision} - {reason}")

        if progress is not None:
            progress(1.0, "Testing completed")

        return {
            # Keep raw evaluations separate from summary
            "evaluations": aggregated_summary["raw_evaluations"],
            "summary": {
                "total_test_cases": num_cases,
                "verdicts": aggregated_summary["verdict_counts"],
                "verdict_percentages": aggregated_summary["verdict_percentages"],
                "average_confidence": aggregated_summary["average_confidence"],
                "decision": decision,
                "reason": reason,
                "champion_metrics": champion_avg_metrics,
                "challenger_metrics": challenger_avg_metrics,
                "judge_metrics": judge_avg_metrics,
                "champion_name": self.champion_endpoint.name,
                "challenger_name": self.challenger_endpoint.name,
                "judge_name": self.judge_endpoint.name,
            }
        }


# --- Gradio UI Components & Logic ---

def parse_test_data(file_obj, text_data, key_field_name: str = "key", value_field_name: str = "value") -> List[TestCase]:
    """
    Parses test data from Gradio file upload or text input.
    Uses specified field names for key and value.
    """
    test_cases = []
    raw_data = None

    if file_obj is not None:
        logger.info(f"Loading test data from uploaded file: {file_obj.name}")
        file_path = file_obj.name
        try:
            if file_path.lower().endswith(".json"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
            elif file_path.lower().endswith(".csv"):
                # Read CSV into pandas DataFrame first for easier handling
                try:
                     # Try detecting delimiter, handle potential bad lines
                     df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='warn')
                     logger.info(f"CSV loaded successfully. Columns: {df.columns.tolist()}")
                     # Convert DataFrame rows to list of dictionaries
                     raw_data = df.to_dict(orient='records')
                except Exception as e:
                     logger.error(f"Error reading CSV file '{file_path}': {e}")
                     raise ValueError(f"Error reading CSV: {e}")
            elif file_path.lower().endswith((".jsonl", ".ndjson")):
                # Handle JSONL (newline-delimited JSON)
                raw_data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            raw_data.append(item)
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid JSON line: {line.strip()}")
                if not raw_data:
                    raise ValueError("No valid JSON objects found in JSONL file.")
            else:
                # Check against allowed extensions explicitly
                allowed_extensions = ['.csv', '.json', '.jsonl', '.ndjson']
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext not in allowed_extensions:
                    raise ValueError(f"Invalid file type. Please upload a file that is one of these formats: {allowed_extensions}")
                else:
                    # This case should ideally not be reached if all supported types are handled above
                    raise ValueError(f"Unhandled supported file type: {file_ext}. Please report this bug.")

        except Exception as e:
            logger.error(f"Error processing uploaded file {file_path}: {e}")
            raise ValueError(f"Failed to process file: {e}")

    elif text_data and text_data.strip():
        logger.info("Loading test data from text input.")
        try:
            raw_data = json.loads(text_data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in text input: {e}")
            raise ValueError(f"Invalid JSON format in text input: {e}")
        except Exception as e:
            logger.error(f"Error processing text input data: {e}")
            raise ValueError(f"Failed to process text data: {e}")

    else:
        raise ValueError("No test data provided. Please upload a file or paste JSON.")

    # Convert raw_data (list of dicts) to TestCase objects
    if isinstance(raw_data, list):
        for i, item in enumerate(raw_data):
            if isinstance(item, dict):
                try:
                    # Ensure the specified key field exists, value field is optional
                    # 'id' is optional (defaults to None, ModelTester assigns later if needed)
                    key = item.get(key_field_name)
                    if key is None:
                         logger.warning(f"Skipping item {i+1} due to missing '{key_field_name}' field. Data: {item}")
                         continue
                    test_cases.append(TestCase(
                        id=str(item.get('id', f"item_{i+1}")), # Ensure ID is string
                        key=str(key), # Ensure key is string
                        value=str(item.get(value_field_name, '')) # Ensure value is string, default empty
                    ))
                except Exception as e:
                    logger.warning(f"Skipping item {i+1} due to error during TestCase creation: {e}. Data: {item}")
            else:
                 logger.warning(f"Skipping item {i+1} as it is not a dictionary. Data: {item}")
    else:
        raise ValueError("Parsed data is not a list of test cases (expected list of dictionaries).")

    if not test_cases:
         raise ValueError("No valid test cases could be loaded from the provided data.")

    logger.info(f"Successfully loaded {len(test_cases)} test cases.")
    return test_cases


def format_summary_output(summary_data: Dict[str, Any]) -> str:
    """Formats the summary dictionary into a readable string."""
    if not summary_data or summary_data.get("error"):
        return f"Error generating summary: {summary_data.get('error', 'Unknown error')}"

    output = f"--- Test Summary ---\n"
    output += f"Champion: {summary_data.get('champion_name', 'N/A')}\n"
    output += f"Challenger: {summary_data.get('challenger_name', 'N/A')}\n"
    output += f"Judge: {summary_data.get('judge_name', 'N/A')}\n"
    output += f"Total Test Cases: {summary_data.get('total_test_cases', 'N/A')}\n"

    output += "\nVerdicts:\n"
    for verdict, count in summary_data.get('verdicts', {}).items():
        output += f"  {verdict}: {count}\n"

    output += "\nVerdict Percentages (Determined Only):\n"
    for verdict, pct in summary_data.get('verdict_percentages', {}).items():
        output += f"  {verdict}: {pct}%\n"

    avg_conf = summary_data.get('average_confidence', 0)
    output += f"\nAverage Confidence: {avg_conf:.3f}\n"

    output += "\nMetrics (Avg Latency / Avg Chars / Success Rate / Errors):\n"
    champ_metrics = summary_data.get('champion_metrics', {})
    chall_metrics = summary_data.get('challenger_metrics', {})
    judge_metrics = summary_data.get('judge_metrics', {})
    output += f"  Champion:   {champ_metrics.get('avg_latency', 0):.2f}s / {champ_metrics.get('avg_chars', 0)} / {champ_metrics.get('success_rate', 0):.1f}% / {champ_metrics.get('errors', 0)}\n"
    output += f"  Challenger: {chall_metrics.get('avg_latency', 0):.2f}s / {chall_metrics.get('avg_chars', 0)} / {chall_metrics.get('success_rate', 0):.1f}% / {chall_metrics.get('errors', 0)}\n"
    output += f"  Judge:      {judge_metrics.get('avg_latency', 0):.2f}s / {judge_metrics.get('avg_chars', 0)} / {judge_metrics.get('success_rate', 0):.1f}% / {judge_metrics.get('errors', 0)}\n"

    output += f"\nDecision: {summary_data.get('decision', 'N/A')}\n"
    output += f"Reason: {summary_data.get('reason', 'N/A')}\n"

    return output


def run_test_from_ui(
    # Model Configs (15 inputs)
    champ_name, champ_api_url, champ_model_id, champ_temp, champ_max_tokens,
    chall_name, chall_api_url, chall_model_id, chall_temp, chall_max_tokens,
    judge_name, judge_api_url, judge_model_id, judge_temp, judge_max_tokens,
    # API Key (1 input) - Added
    api_key_input,
    # Prompts (2 inputs)
    model_prompt_template_input,
    judge_prompt_template_input,
    # Test Data (2 inputs)
    test_data_file,
    test_data_text,
    # Parameters (1 input)
    batch_size_input,
    # Data Field Names (2 inputs) - Added
    key_field_name_input,
    value_field_name_input,
    # Gradio progress object
    progress=gr.Progress(track_tqdm=True)
):
    """
    Handles the logic for running the A/B test triggered by the Gradio UI button.
    """
    logger.info("Starting test run from Gradio UI...")
    progress(0, desc="Initializing...")

    try:
        # 1. Get API Key from UI input
        api_key = str(api_key_input).strip() if api_key_input else None
        if not api_key:
            # Optionally check environment variable as a fallback
            # api_key = os.getenv("OPENROUTER_API_KEY")
            # if not api_key:
            logger.error("API Key not provided in the UI.")
            raise gr.Error("Error: OpenRouter API Key is required. Please enter it in the Configuration tab.")

        progress(0.1, desc="Loading test data...")
        # 2. Load Test Cases (Pass field names from UI)
        try:
            key_field = str(key_field_name_input).strip() or "key" # Default to "key" if empty
            value_field = str(value_field_name_input).strip() or "value" # Default to "value" if empty
            logger.info(f"Using key field: '{key_field}', value field: '{value_field}'")
            test_cases = parse_test_data(test_data_file, test_data_text, key_field, value_field)
            logger.info(f"Loaded {len(test_cases)} test cases.")
        except ValueError as e:
            logger.error(f"Failed to load test data: {e}")
            raise gr.Error(f"Test Data Error: {e}")
        except Exception as e:
            logger.exception("Unexpected error loading test data.")
            raise gr.Error(f"Unexpected error loading test data: {e}")

        if not test_cases:
             raise gr.Error("No valid test cases were loaded.")

        progress(0.2, desc="Configuring models...")
        # 3. Create Model Endpoints (Use the api_key from UI)
        try:
            champion_endpoint = ModelEndpoint(
                name=str(champ_name), api_url=str(champ_api_url), api_key=api_key, model_id=str(champ_model_id),
                temperature=float(champ_temp), max_tokens=int(champ_max_tokens)
            )
            challenger_endpoint = ModelEndpoint(
                name=str(chall_name), api_url=str(chall_api_url), api_key=api_key, model_id=str(chall_model_id),
                temperature=float(chall_temp), max_tokens=int(chall_max_tokens)
            )
            judge_endpoint = ModelEndpoint(
                name=str(judge_name), api_url=str(judge_api_url), api_key=api_key, model_id=str(judge_model_id),
                temperature=float(judge_temp), max_tokens=int(judge_max_tokens)
            )
        except Exception as e:
             logger.error(f"Error creating ModelEndpoint objects: {e}")
             raise gr.Error(f"Model Configuration Error: {e}")

        # 4. Instantiate ModelTester
        try:
            tester = ModelTester(
                champion_endpoint=champion_endpoint,
                challenger_endpoint=challenger_endpoint,
                judge_endpoint=judge_endpoint,
                model_prompt_template=str(model_prompt_template_input),
                judge_prompt_template=str(judge_prompt_template_input)
            )
        except Exception as e:
             logger.error(f"Error instantiating ModelTester: {e}")
             raise gr.Error(f"Tester Initialization Error: {e}")

        # 5. Run the Test
        logger.info(f"Running test with {len(test_cases)} cases, batch size {int(batch_size_input)}...")
        progress(0.3, desc="Running A/B test...")
        try:
            results = tester.run_test(test_cases, batch_size=int(batch_size_input), progress=progress)
        except Exception as e:
            logger.exception("An error occurred during ModelTester.run_test()")
            raise gr.Error(f"Test Execution Error: {e}")

        progress(0.9, desc="Formatting results...")
        # 6. Format Results
        summary_output = format_summary_output(results.get("summary", {}))
        # Convert raw evaluations (list of dicts) to DataFrame
        raw_evals = results.get("evaluations", [])
        if raw_evals:
             # Select and reorder columns for better display
             display_columns = ['test_id', 'winner', 'confidence', 'champion_output', 'challenger_output', 'reasoning']
             details_df = pd.DataFrame(raw_evals)
             # Ensure all expected columns exist, add if missing
             for col in display_columns:
                  if col not in details_df.columns:
                       details_df[col] = None
             details_df = details_df[display_columns] # Reorder/select
        else:
             details_df = pd.DataFrame(columns=['test_id', 'winner', 'confidence', 'reasoning']) # Empty DF with headers

        logger.info("Test run completed successfully.")
        return summary_output, details_df

    except gr.Error as e: # Catch Gradio-specific errors to display them directly
        logger.error(f"Gradio Error: {e}")
        # Return error message to both outputs
        error_df = pd.DataFrame([{"Error": str(e)}])
        return str(e), error_df
    except Exception as e:
        logger.exception("An unexpected error occurred in run_test_from_ui.")
        error_message = f"An unexpected error occurred: {e}"
        error_df = pd.DataFrame([{"Error": error_message}])
        return error_message, error_df


def create_ui():
    """Creates the Gradio web interface for the A/B testing tool."""
    logger.info("Creating Gradio UI...")

    # Default values for UI components
    default_api_url = "https://openrouter.ai/api/v1/chat/completions"
    default_model_prompt = "Answer the following question: {key}"
    # Use the default judge prompt from the LMJudge class
    default_judge_prompt = LMJudge.DEFAULT_EVALUATION_PROMPT

    css = """
    .model-config-group .gr-form { background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    .model-config-group .gr-form > :first-child { font-weight: bold; margin-bottom: 5px; } /* Style the label */
    .results-box { border: 1px solid #ccc; padding: 15px; border-radius: 5px; margin-top: 15px; }
    .api-key-warning { color: orange; font-weight: bold; margin-bottom: 15px; }
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft()) as iface:
        gr.Markdown("# Model A/B Testing & Evaluation Tool")
        gr.Markdown(
            "Configure champion, challenger, and judge models, provide test data, "
            "and run evaluations to compare model performance."
        )
        gr.Markdown(
            "**Note:** The `OPENROUTER_API_KEY` environment variable must be set in the environment "
            "where this Gradio app is running for API calls to succeed.",
            elem_classes="api-key-warning"
        )

        with gr.Tabs():
            with gr.TabItem("Configuration"):
                # Add API Key input field
                with gr.Row():
                     api_key_input = gr.Textbox(
                          label="OpenRouter API Key",
                          type="password",
                          placeholder="Enter key here (required to run tests)",
                          info="Overrides OPENROUTER_API_KEY environment variable if set."
                     )
                with gr.Row():
                    # Champion Model Configuration
                    with gr.Column(scale=1):
                         with gr.Group(elem_classes="model-config-group"):
                              gr.Label("Champion Model (Model A)")
                              champ_name = gr.Textbox(label="Display Name", value="Champion (Gemini Flash)")
                              champ_api_url = gr.Textbox(label="API URL", value=default_api_url)
                              champ_model_id = gr.Textbox(label="Model ID", value="google/gemini-2.0-flash-exp:free")
                              champ_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.1, value=0.1)
                              champ_max_tokens = gr.Number(label="Max Tokens", value=1024, precision=0)
                    # Challenger Model Configuration
                    with gr.Column(scale=1):
                         with gr.Group(elem_classes="model-config-group"):
                              gr.Label("Challenger Model (Model B)")
                              chall_name = gr.Textbox(label="Display Name", value="Challenger (Gemma 3 27B)")
                              chall_api_url = gr.Textbox(label="API URL", value=default_api_url)
                              chall_model_id = gr.Textbox(label="Model ID", value="google/gemma-3-27b-it:free")
                              chall_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.1, value=0.1)
                              chall_max_tokens = gr.Number(label="Max Tokens", value=1024, precision=0)
                    # Judge Model Configuration
                    with gr.Column(scale=1):
                         with gr.Group(elem_classes="model-config-group"):
                              gr.Label("Judge Model")
                              judge_name = gr.Textbox(label="Display Name", value="Judge (Gemini 2.5 Pro)")
                              judge_api_url = gr.Textbox(label="API URL", value=default_api_url)
                              judge_model_id = gr.Textbox(label="Model ID", value="google/gemini-2.5-pro-exp-03-25:free")
                              judge_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.0) # Judge usually deterministic
                              judge_max_tokens = gr.Number(label="Max Tokens", value=2048, precision=0) # Judge might need more tokens

                with gr.Row():
                    # Model Prompt Template
                    with gr.Column(scale=1):
                        gr.Markdown("### Model Prompt Template")
                        model_prompt_template_input = gr.Textbox(
                            label="Template for Champion/Challenger (use {key} for input)",
                            value=default_model_prompt,
                            lines=5,
                            show_copy_button=True
                        )
                    # Judge Prompt Template
                    with gr.Column(scale=1):
                        gr.Markdown("### Judge Prompt Template")
                        judge_prompt_template_input = gr.Textbox(
                            label="Template for Judge (see code/docs for available placeholders like {key}, {champion_output}, etc.)",
                            value=default_judge_prompt,
                            lines=15,
                            show_copy_button=True
                        )

                with gr.Row():
                    # Test Data Input
                    with gr.Column(scale=1):
                        gr.Markdown("### Test Data")
                        gr.Markdown("Upload a CSV/JSON/JSONL file or paste data below. Specify the field names containing the model input (key) and optional reference answer (value). Add an `id` field for stable identification.")
                        test_data_file = gr.File(label="Upload Test Data (CSV, JSON, JSONL/NDJSON)", file_types=[".csv", ".json", ".jsonl", ".ndjson"])
                        test_data_text = gr.Textbox(label="Or Paste Test Data (JSON list format)", lines=8, placeholder='[{"id": "t1", "question": "Input 1", "reference_answer": "Expected 1"}, ...]')
                        # Add options for CSV parsing later if needed (delimiter, quote char)
                        with gr.Row():
                             key_field_name_input = gr.Textbox(label="Key Field Name", value="question", info="Field containing the main input/prompt.") # Default updated
                             value_field_name_input = gr.Textbox(label="Value Field Name", value="reference_answer", info="Field containing the reference/ground truth (optional).") # Default updated

                    # Test Run Parameters
                    with gr.Column(scale=1):
                        gr.Markdown("### Test Parameters")
                        batch_size_input = gr.Number(label="Batch Size", value=5, precision=0)
                        # Add preprocessing options later if needed

            with gr.TabItem("Results"):
                gr.Markdown("### Test Execution & Results")
                run_button = gr.Button("Run A/B Test", variant="primary")
                with gr.Group(elem_classes="results-box"):
                     gr.Markdown("#### Summary")
                     summary_output = gr.Textbox(label="Overall Results", lines=10, show_copy_button=True)
                with gr.Group(elem_classes="results-box"):
                     gr.Markdown("#### Detailed Evaluations")
                     details_output = gr.DataFrame(label="Individual Case Results", wrap=True) # Use DataFrame for better display

        # Define interactions
        run_button.click(
            fn=run_test_from_ui,
            inputs=[
                # Model Configs
                champ_name, champ_api_url, champ_model_id, champ_temp, champ_max_tokens,
                chall_name, chall_api_url, chall_model_id, chall_temp, chall_max_tokens,
                judge_name, judge_api_url, judge_model_id, judge_temp, judge_max_tokens,
                # API Key - Added
                api_key_input,
                # Prompts
                model_prompt_template_input,
                judge_prompt_template_input,
                # Test Data
                test_data_file,
                test_data_text,
                # Parameters
                batch_size_input,
                # Data Field Names - Added
                key_field_name_input,
                value_field_name_input
            ],
            outputs=[summary_output, details_output]
        )

    return iface


def run_cli_test():
    """Runs the A/B test from the command line using hardcoded examples."""
    logger.info("Starting CLI execution of ModelTester...")

     # --- Configuration (Requires OPENROUTER_API_KEY environment variable) ---
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        logger.error("Error: OPENROUTER_API_KEY environment variable not set.")
        print("\nError: Please set the OPENROUTER_API_KEY environment variable before running.")
        sys.exit(1) # Exit if the key is missing
    else:
        logger.info("OPENROUTER_API_KEY environment variable found.")


    # Define Model Endpoints (Example using OpenRouter)
    champion_model = ModelEndpoint(
        name="Champion (GPT-3.5)",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key=OPENROUTER_API_KEY,
        model_id="openai/gpt-3.5-turbo", # Example model
        temperature=0.1
    )

    challenger_model = ModelEndpoint(
        name="Challenger (Claude)",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key=OPENROUTER_API_KEY,
        model_id="anthropic/claude-3-haiku-20240307", # Example model
        temperature=0.1
    )

    judge_model = ModelEndpoint(
        name="Judge (GPT-4)",
        api_url="https://openrouter.ai/api/v1/chat/completions",
        api_key=OPENROUTER_API_KEY,
        model_id="openai/gpt-4-turbo", # Example judge model
        temperature=0.0
    )

    # Define Model Prompt Template (Simple example)
    model_prompt = "Answer the following question: {key}"

    # Define Sample Test Cases
    test_cases = [
        TestCase(id="q1", key="What is the capital of France?", value="Paris"),
        TestCase(id="q2", key="Summarize the main points of the article about AI.", value="AI advancements, ethical concerns, future outlook."),
        # Add more test cases or load from a file
    ]

    # --- Execution ---
    try:
        # Instantiate the tester
        tester = ModelTester(
            champion_endpoint=champion_model,
            challenger_endpoint=challenger_model,
            judge_endpoint=judge_model,
            model_prompt_template=model_prompt,
            # judge_prompt_template=LMJudge.DEFAULT_EVALUATION_PROMPT # Use default judge prompt for CLI
        )

        logger.info(f"Running CLI test with {len(test_cases)} test cases...")
        # Run the test (batch size can be adjusted)
        results = tester.run_test(test_cases, batch_size=2) # Using a small batch size for testing

        # --- Output Results ---
        logger.info("Test completed. Final Results:")

        # Pretty print the summary
        summary = results.get("summary", {})
        print("\n--- Test Summary ---")
        print(f"Champion: {summary.get('champion_name', 'N/A')}")
        print(f"Challenger: {summary.get('challenger_name', 'N/A')}")
        print(f"Judge: {summary.get('judge_name', 'N/A')}")
        print(f"Total Test Cases: {summary.get('total_test_cases', 'N/A')}")
        print("\nVerdicts:")
        for verdict, count in summary.get('verdicts', {}).items():
            print(f"  {verdict}: {count}")
        print("\nVerdict Percentages (Determined Only):")
        for verdict, pct in summary.get('verdict_percentages', {}).items():
            print(f"  {verdict}: {pct}%")
        print(f"\nAverage Confidence: {summary.get('average_confidence', 'N/A'):.3f}")

        print("\nMetrics (Avg Latency / Avg Chars / Success Rate / Errors):")
        champ_metrics = summary.get('champion_metrics', {})
        chall_metrics = summary.get('challenger_metrics', {})
        judge_metrics = summary.get('judge_metrics', {})
        print(f"  Champion:   {champ_metrics.get('avg_latency', 0):.2f}s / {champ_metrics.get('avg_chars', 0)} / {champ_metrics.get('success_rate', 0):.1f}% / {champ_metrics.get('errors', 0)}")
        print(f"  Challenger: {chall_metrics.get('avg_latency', 0):.2f}s / {chall_metrics.get('avg_chars', 0)} / {chall_metrics.get('success_rate', 0):.1f}% / {chall_metrics.get('errors', 0)}")
        print(f"  Judge:      {judge_metrics.get('avg_latency', 0):.2f}s / {judge_metrics.get('avg_chars', 0)} / {judge_metrics.get('success_rate', 0):.1f}% / {judge_metrics.get('errors', 0)}")


        print(f"\nDecision: {summary.get('decision', 'N/A')}")
        print(f"Reason: {summary.get('reason', 'N/A')}")

        # Optionally save full results to JSON
        results_filename = f"results_{time.strftime('%Y%m%d-%H%M%S')}.json"
        try:
            with open(results_filename, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Full results saved to {results_filename}")
            print(f"\nFull results saved to: {results_filename}")
        except Exception as e:
            logger.error(f"Failed to save results to JSON: {e}")

    except Exception as e:
        logger.exception("An error occurred during the CLI execution.")
        print(f"\nAn error occurred during CLI execution: {e}")


# --- Main Entry Point ---
if __name__ == "__main__":
    # Basic argument parsing: run CLI test by default, or launch UI with --ui flag
    import argparse
    parser = argparse.ArgumentParser(description="Model A/B Testing Tool")
    parser.add_argument("--ui", action="store_true", help="Launch the Gradio web UI instead of running the CLI test.")
    args = parser.parse_args()

    if args.ui:
        logger.info("Launching Gradio UI...")
        # Placeholder for launching the UI
        iface = create_ui()
        if iface:
             # Add authentication later if needed: auth=("username", "password")
             iface.launch()
        else:
             logger.error("Failed to create Gradio UI.")
             print("Error: Could not create the Gradio UI.")
    else:
        # Run the command-line test
        run_cli_test()
