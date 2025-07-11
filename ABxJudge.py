import sys
import os
import subprocess
import signal # Added for CLI stop handling
import itertools # Added for pairwise combinations
from pathlib import Path
import shutil

# --- Standard Library Imports (Safe to be global) ---
import json
import logging
import time
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
import base64
import mimetypes

# --- Venv Setup ---
# Determine if we need to set up or reactivate the virtual environment

VENV_DIR = ".venv"
REQUIRED_PACKAGES = [
    "gradio",
    "pandas",
    "requests",
    "tenacity",
    "Pillow", # For image handling (needed for dummy image in CLI test)
    "python-dotenv", # Added for easy API key management in CLI
    "numpy", # Added for potential aggregation/matrix operations
    "tqdm", # For CLI progress bars
    "llama-cpp-python" # For local server, if used
]

# Configure logging early so it can be used by the setup function
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("abx_judge_tool")

def ensure_environment_with_uv():
    """
    Ensures a virtual environment exists and has dependencies installed using uv.
    If uv is not found, it attempts to install it.
    If the script is not running in the venv, it re-executes itself within it.
    """
    try:
        import gradio, pandas, requests, tenacity, PIL, dotenv, numpy, tqdm
        return True # Dependencies are already present
    except ImportError:
        logger.info("One or more dependencies are missing, proceeding with uv setup.")

    venv_path = Path(__file__).parent.resolve() / VENV_DIR
    # Use os.path.realpath to handle symlinks correctly
    is_in_venv = os.path.realpath(sys.prefix) == os.path.realpath(str(venv_path))

    if is_in_venv:
        return True

    logger.info("--- Environment Setup: Bootstrapping with uv ---")

    uv_executable = shutil.which("uv")
    if not uv_executable:
        logger.warning("`uv` not found in PATH. Attempting to install it with pip...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True, capture_output=True, text=True)
            uv_executable = shutil.which("uv")
            if not uv_executable:
                raise RuntimeError("`uv` was not found in PATH after pip install.")
            logger.info(f"Using `uv` installed at: {uv_executable}")
        except (subprocess.CalledProcessError, RuntimeError) as e:
            logger.error(f"Failed to automatically install `uv`. Please install it manually: `pip install uv`")
            logger.error(f"Error details: {e}")
            sys.exit(1)

    python_in_venv = venv_path / "bin" / "python" if sys.platform != "win32" else venv_path / "Scripts" / "python.exe"
    if not venv_path.exists():
        logger.info(f"Creating virtual environment with `uv` at: {venv_path}")
        subprocess.run([uv_executable, "venv", str(venv_path), "--python", sys.executable], check=True, capture_output=True, text=True)

    logger.info(f"Installing dependencies into '{venv_path}' with `uv pip install`...")
    install_command = [uv_executable, "pip", "install", "-p", str(python_in_venv)] + REQUIRED_PACKAGES
    try:
        process = subprocess.Popen(install_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1)
        for line in iter(process.stdout.readline, ''):
            logger.info(f"[uv pip install]: {line.strip()}")
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
             raise subprocess.CalledProcessError(return_code, install_command)
        logger.info("All dependencies installed successfully.")
    except (subprocess.CalledProcessError, Exception) as e:
        logger.error(f"Failed to install dependencies with `uv`: {e}")
        sys.exit(1)

    logger.info(f"Restarting script inside the '{VENV_DIR}' environment...")
    try:
        os.execv(str(python_in_venv), [str(python_in_venv), __file__] + sys.argv[1:])
    except Exception as e:
        logger.error(f"FATAL: Failed to re-execute script in virtual environment: {e}")
        sys.exit(1)
    return False  # Properly indented under function scope

# --- Dependency-based Imports (for after venv is ensured) ---
import gradio as gr
import pandas as pd
import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from collections import Counter
import tempfile
import csv
import io
import argparse

# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class ModelEndpoint:
    """Configuration for a single model API endpoint."""
    name: str
    api_url: str
    api_key: Optional[str]
    model_id: str
    max_tokens: int = 1024
    temperature: float = 0.0
    file_upload_method: str = "JSON (Embedded Data)"
    is_active: bool = True

@dataclass
class TestCase:
    """A single test case from the input corpus."""
    key: str
    value: str
    image_path_or_url: Optional[str] = None
    id: Optional[str] = None

@dataclass
class ModelResponse:
    """The output from a model for a single test case."""
    test_id: str
    model_name: str
    output: str
    latency: float
    is_error: bool = False

@dataclass
class EvaluationResult:
    """The result of a single pairwise evaluation by the judge model."""
    test_id: str
    model_a_name: str
    model_b_name: str
    model_a_output: str
    model_b_output: str
    winner: str
    confidence: float
    reasoning: str

# Global flag to signal stopping the test run from the UI or Ctrl+C
STOP_REQUESTED = False

# ==============================================================================
# Core Logic: Analysis, API Communication, and Evaluation
# ==============================================================================

def preprocess_text(text: str, max_length: int = 8000) -> str:
    """Cleans and truncates text for safe processing."""
    if text is None: return ""
    text = str(text)
    if max_length and len(text) > max_length: text = text[:max_length] + "... [truncated]"
    # Remove control characters and HTML tags
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_ngrams(text: str, n: int) -> List[str]:
    """Generates a list of n-grams from a given text."""
    if not isinstance(text, str) or not text.strip(): return []
    # Simple tokenization: lowercase, remove punctuation, split by space.
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    if len(words) < n: return []
    ngrams = zip(*[words[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def run_post_processing_analysis(results_json: Dict, enable_ngram: bool, n_val: int, top_k: int) -> Dict:
    """
    Performs post-hoc analysis on the results, like n-gram frequency.
    # --- CLARIFICATION: This function modifies the results_json dictionary in-place for efficiency.
    """
    if not enable_ngram or "summary" not in results_json:
        return results_json

    logger.info(f"Running n-gram analysis (n={n_val}, top_k={top_k})...")
    summary = results_json["summary"]
    model_names = summary.get("model_names", [])
    if not model_names:
        return results_json

    ngram_counters = {name: Counter() for name in model_names}
    seen_outputs = set()

    # --- CLARIFICATION:
    # We iterate through the 'evaluations' (pairwise results) but need to count each model's
    # output for a given test case only once. `seen_outputs` tracks `test_id-model_name`
    # pairs to prevent double-counting when a model appears in multiple pairs (e.g., A vs B, A vs C).
    for eval_item in results_json.get("evaluations", []):
        for model_key, output_key in [("model_a_name", "model_a_output"), ("model_b_name", "model_b_output")]:
            model_name = eval_item.get(model_key)
            output_id = f"{eval_item.get('test_id')}-{model_name}"
            if model_name in ngram_counters and eval_item.get(output_key) and output_id not in seen_outputs:
                ngrams = generate_ngrams(eval_item[output_key], n_val)
                ngram_counters[model_name].update(ngrams)
                seen_outputs.add(output_id)

    analysis_results = {
        "ngram_frequency": {
            "n_value": n_val,
            "top_k_reported": top_k,
            "results_by_model": {
                name: {"most_frequent": [{"ngram": item, "count": count} for item, count in counts.most_common(top_k)]}
                for name, counts in ngram_counters.items()
            }
        }
    }

    summary["post_processing_analysis"] = analysis_results
    results_json["summary"] = summary
    return results_json


class ModelRunner:
    """Handles all API communication for a single model endpoint."""
    def __init__(self, endpoint: ModelEndpoint, prompt_template: str):
        self.endpoint = endpoint
        self.prompt_template = prompt_template
        self._temp_files: List[str] = []

    def _load_and_encode_file(self, file_path_or_url: str) -> Tuple[Optional[str], Optional[str]]:
        """Loads a file from a local path or URL and returns its base64-encoded string and MIME type."""
        try:
            if urlparse(file_path_or_url).scheme in ['http', 'https']:
                logger.info(f"Downloading file from URL: {file_path_or_url} for model {self.endpoint.name}")
                response = requests.get(file_path_or_url, stream=True, timeout=30)
                response.raise_for_status()
                file_bytes = response.content
                mime_type = response.headers.get('content-type')
            else:
                logger.info(f"Reading file from local path: {file_path_or_url} for model {self.endpoint.name}")
                if not os.path.exists(file_path_or_url):
                    raise FileNotFoundError(f"File not found at path: {file_path_or_url}")
                with open(file_path_or_url, "rb") as f:
                    file_bytes = f.read()
                mime_type, _ = mimetypes.guess_type(file_path_or_url)

            if not file_bytes: raise ValueError("Failed to load file bytes.")
            mime_type = mime_type or 'application/octet-stream'
            base64_data = base64.b64encode(file_bytes).decode('utf-8')
            return base64_data, mime_type
        except Exception as e:
            logger.error(f"Failed to load or encode file {file_path_or_url} for model {self.endpoint.name}: {e}")
            return None, None

    def _prepare_local_file_path(self, file_path_or_url: str) -> Optional[str]:
        """Ensures a local file path exists, downloading from a URL if necessary. Used for multipart uploads."""
        try:
            parsed_url = urlparse(file_path_or_url)
            if parsed_url.scheme in ['http', 'https']:
                response = requests.get(file_path_or_url, stream=True, timeout=60)
                response.raise_for_status()
                suffix = os.path.splitext(parsed_url.path)[1]
                # Create a temporary file that is not deleted on close, so we can get its path.
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                with temp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)
                local_path = temp_file.name
                self._temp_files.append(local_path) # Track for later cleanup.
                return local_path
            else:
                return file_path_or_url if os.path.exists(file_path_or_url) else None
        except Exception as e:
            logger.error(f"Error preparing local file path {file_path_or_url} for model {self.endpoint.name}: {e}")
            return None

    def _cleanup_temp_files(self):
        """Removes any temporary files created by this runner instance during a test run."""
        for temp_path in self._temp_files:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError as e:
                logger.warning(f"Failed to clean up temporary file {temp_path}: {e}")
        self._temp_files = []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, test_case: TestCase) -> ModelResponse:
        """
        Generates a model response for a given test case, handling file uploads and API logic.
        Wrapped in a retry decorator to handle transient network issues.
        """
        start_time = time.time()
        is_error = False
        response_text = ""
        try:
            prompt = self.prompt_template.format(key=preprocess_text(test_case.key))
            base64_data, mime_type, local_file_path = None, None, None

            if test_case.image_path_or_url:
                if self.endpoint.file_upload_method == "JSON (Embedded Data)":
                    base64_data, mime_type = self._load_and_encode_file(test_case.image_path_or_url)
                    if not base64_data: is_error, response_text = True, f"Error: Failed to load/encode file for JSON method."
                elif self.endpoint.file_upload_method == "Multipart Form Data":
                    local_file_path = self._prepare_local_file_path(test_case.image_path_or_url)
                    if not local_file_path: is_error, response_text = True, f"Error: Failed to prepare file for Multipart method."
            
            if not is_error:
                try:
                    if self.endpoint.file_upload_method == "JSON (Embedded Data)":
                        response_text = self._call_json_api(prompt, base64_data, mime_type)
                    else: # Multipart Form Data
                        response_text = self._call_multipart_api(prompt, local_file_path)
                except Exception as api_err:
                    is_error, response_text = True, f"Error: API call failed. Details: {str(api_err)}"
                    raise # Re-raise to trigger tenacity retry

        except Exception as e:
            is_error, response_text = True, f"Error: Generation failed critically. Details: {str(e)}"
            logger.error(f"Critical failure in generate for {self.endpoint.name}: {e}", exc_info=True)
        finally:
            self._cleanup_temp_files()

        return ModelResponse(
            test_id=str(test_case.id), model_name=self.endpoint.name,
            output=str(response_text), latency=time.time() - start_time, is_error=is_error
        )

    def _prepare_headers(self, is_json_request=True):
        """Prepares HTTP headers for the API request."""
        headers = {}
        if self.endpoint.api_key: headers["Authorization"] = f"Bearer {self.endpoint.api_key}"
        if is_json_request: headers["Content-Type"] = "application/json"
        if "openrouter.ai" in self.endpoint.api_url:
            headers["HTTP-Referer"] = "http://localhost"
            headers["X-Title"] = "ABxJudge Tool"
        return headers

    def _call_json_api(self, prompt: str, base64_data: Optional[str], mime_type: Optional[str]) -> str:
        """Makes a POST request to a JSON-based API (e.g., OpenAI, Anthropic, Ollama)."""
        headers = self._prepare_headers(is_json_request=True)
        api_url_lower = self.endpoint.api_url.lower()

        # --- CLARIFICATION:
        # This block dynamically determines the correct JSON payload structure by inspecting the API URL.
        # This is what makes the tool "endpoint-agnostic" for major API standards.
        if "/v1/chat/completions" in api_url_lower or "openrouter.ai" in api_url_lower:
            payload = self._format_openai_json(prompt, base64_data, mime_type)
        elif "anthropic" in api_url_lower:
            payload = self._format_anthropic_json(prompt, base64_data, mime_type)
        elif "generativelanguage.googleapis.com" in api_url_lower:
            payload = self._format_gemini_json(prompt, base64_data, mime_type)
        elif "/api/generate" in api_url_lower: # Ollama-specific endpoint
            payload = self._format_ollama_json(prompt, base64_data)
        else:
            logger.warning(f"Unknown API type for {self.endpoint.name}. Defaulting to OpenAI format.")
            payload = self._format_openai_json(prompt, base64_data, mime_type)

        response = requests.post(self.endpoint.api_url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        
        # --- CLARIFICATION: This block parses the response based on the same URL detection.
        if "choices" in result: # OpenAI-like
            return result["choices"][0]["message"].get("content", "")
        elif "content" in result and isinstance(result["content"], list): # Anthropic-like
            return next((block.get("text", "") for block in result["content"] if block.get("type") == "text"), "")
        elif "candidates" in result: # Gemini-like
            return result["candidates"][0]["content"]["parts"][0].get("text", "")
        elif "response" in result: # Ollama-like
            return result["response"]
        raise ValueError("Could not parse response from API.")

    def _call_multipart_api(self, prompt: str, local_file_path: Optional[str]) -> str:
        """Makes a POST request using multipart/form-data, typically for file-based models."""
        headers = self._prepare_headers(is_json_request=False)
        data = {'model': self.endpoint.model_id, 'prompt': prompt}
        files = {}
        if not local_file_path: return "Error: No file path for multipart upload."
        try:
            with open(local_file_path, 'rb') as f:
                files['file'] = (os.path.basename(local_file_path), f)
                response = requests.post(self.endpoint.api_url, headers=headers, data=data, files=files, timeout=180)
            response.raise_for_status()
            result = response.json()
            return result.get('text', json.dumps(result)) # Generic parsing
        except Exception as e:
            logger.error(f"Multipart API request failed for {self.endpoint.name}: {e}")
            raise

    # --- API-Specific Payload Formatters ---

    def _format_openai_json(self, prompt: str, base64_data: Optional[str], mime_type: Optional[str]) -> Dict:
        # --- FIX: Uncommented debug logs for better troubleshooting.
        logger.debug(f"Formatting payload for OpenAI JSON ({self.endpoint.name})")
        content_list = [{"type": "text", "text": prompt}]
        if base64_data:
            content_list.append({"type": "image_url", "image_url": {"url": f"data:{mime_type or 'image/jpeg'};base64,{base64_data}"}})
        return {
            "model": self.endpoint.model_id,
            "messages": [{"role": "user", "content": content_list}],
            "max_tokens": self.endpoint.max_tokens,
            "temperature": self.endpoint.temperature,
        }

    def _format_anthropic_json(self, prompt: str, base64_data: Optional[str], mime_type: Optional[str]) -> Dict:
        content = [{"type": "text", "text": prompt}]
        if base64_data:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type or "image/jpeg",
                    "data": base64_data,
                },
            })
        return {
            "model": self.endpoint.model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.endpoint.max_tokens,
            "temperature": self.endpoint.temperature,
        }

    def _format_gemini_json(self, prompt: str, base64_data: Optional[str], mime_type: Optional[str]) -> Dict:
        parts = [{"text": prompt}]
        if base64_data:
            parts.append({"inline_data": {"mime_type": mime_type or "application/octet-stream", "data": base64_data}})
        return {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": self.endpoint.temperature,
                "maxOutputTokens": self.endpoint.max_tokens,
            },
        }

    def _format_ollama_json(self, prompt: str, base64_data: Optional[str]) -> Dict:
        data = {
            "model": self.endpoint.model_id,
            "prompt": prompt,
            "stream": False,
        }
        if base64_data:
            data["images"] = [base64_data]
        return data



class LMJudge:
    """Uses a language model to judge between TWO model outputs."""

    DEFAULT_EVALUATION_PROMPT = """# Model Response Evaluation (Pairwise)

You are evaluating two AI model responses (Model A and Model B) based on the input query, potentially an accompanying image, and potentially a reference value.

## Input Query

{key}

{image_context_section}
{reference_section}

## Model A ({model_a_name}) Response

{model_a_output}

## Model B ({model_b_name}) Response

{model_b_output}

## Evaluation Instructions
Compare Model A ({model_a_name}) and Model B ({model_b_name}) based on the Input Query{reference_value_instruction}. Consider:
1. Relevance and accuracy in addressing the Input Query.
{reference_value_criteria}
{clarity_criteria_number}. Clarity, conciseness, and quality of the response.
{overall_criteria_number}. Overall usefulness.

Which model provided the better response *in this specific comparison*?

## Required Response Format
You MUST start your response with a clear verdict and confidence rating FOR THIS PAIR:

VERDICT: [Choose ONE: MODEL_A_WINS, MODEL_B_WINS, or TIE]
CONFIDENCE: [Number]/5 (where 1=low confidence, 5=high confidence)

Then provide a detailed explanation of your reasoning. Be explicit about which model performed better **in this pair** and why, or why they were tied. Reference {model_a_name} as Model A and {model_b_name} as Model B.

Example format:
VERDICT: MODEL_A_WINS
CONFIDENCE: 4/5

[Your detailed reasoning comparing Model A ({model_a_name}) and Model B ({model_b_name}) here...]
"""

    def __init__(
        self,
        endpoint: ModelEndpoint,
        evaluation_prompt_template: str = DEFAULT_EVALUATION_PROMPT,
    ):
        self.endpoint = endpoint
        self.evaluation_prompt_template = evaluation_prompt_template
        # Judge runner needs its own ModelRunner instance
        self.model_runner = ModelRunner(endpoint, "{key}") # Pass-through template

    def evaluate(
        self,
        test_case: TestCase,
        response_a: ModelResponse, # Changed parameter name
        response_b: ModelResponse, # Changed parameter name
    ) -> EvaluationResult:
        """Evaluate response_a vs response_b using a dynamically built prompt."""
        # Preprocess text inputs
        preprocessed_key = preprocess_text(test_case.key)
        preprocessed_value = preprocess_text(test_case.value)
        preprocessed_output_a = preprocess_text(response_a.output)
        preprocessed_output_b = preprocess_text(response_b.output)

        # Prepare context for the evaluation prompt template
        has_reference = bool(preprocessed_value)
        reference_section_text = f"\n## Reference Value\n\n{preprocessed_value}\n" if has_reference else "\n## Reference Value\nN/A"
        reference_value_instruction_text = ' and Reference Value' if has_reference else ''
        reference_value_criteria_text = '2. Factual correctness compared to the Reference Value (if provided).' if has_reference else ''
        clarity_criteria_number_text = '3' if has_reference else '2'
        overall_criteria_number_text = '4' if has_reference else '3'
        has_image = bool(test_case.image_path_or_url)
        image_context_section_text = "\n## Input Image\nAn image was provided with the input query. Consider it as context when evaluating the responses.\n" if has_image else ""

        # Format the evaluation prompt using the template and specific model names
        try:
            evaluation_prompt = self.evaluation_prompt_template.format(
                key=preprocessed_key,
                image_context_section=image_context_section_text,
                reference_section=reference_section_text,
                model_a_name=response_a.model_name, # Use actual model name
                model_a_output=preprocessed_output_a,
                model_b_name=response_b.model_name, # Use actual model name
                model_b_output=preprocessed_output_b,
                reference_value_instruction=reference_value_instruction_text,
                reference_value_criteria=reference_value_criteria_text,
                clarity_criteria_number=clarity_criteria_number_text,
                overall_criteria_number=overall_criteria_number_text
            )
        except KeyError as e:
            logger.error(f"Missing key in judge prompt template: {e}. Using basic prompt structure.")
            evaluation_prompt = f"Evaluate Model A ({response_a.model_name}) vs Model B ({response_b.model_name}).\nInput: {preprocessed_key}\nRef: {preprocessed_value}\nA: {preprocessed_output_a}\nB: {preprocessed_output_b}\nFormat: VERDICT: [MODEL_A_WINS/MODEL_B_WINS/TIE]\nCONFIDENCE: [1-5]/5\nReasoning: ..."
        except Exception as e:
            logger.error(f"Error formatting judge prompt template: {e}. Using basic prompt.")
            evaluation_prompt = f"Evaluate Model A ({response_a.model_name}) vs Model B ({response_b.model_name}).\nInput: {preprocessed_key}\nRef: {preprocessed_value}\nA: {preprocessed_output_a}\nB: {preprocessed_output_b}\nFormat: VERDICT: [MODEL_A_WINS/MODEL_B_WINS/TIE]\nCONFIDENCE: [1-5]/5\nReasoning: ..."

        # Log the prompt for debugging
        logger.info(f"Judge prompt for pair ({response_a.model_name} vs {response_b.model_name}) on test ID {test_case.id} (truncated): {evaluation_prompt[:500]}...")

        # Create a TestCase for the judge call, passing the original image ref
        judge_test_case = TestCase(
            key=evaluation_prompt,
            value="",
            image_path_or_url=test_case.image_path_or_url,
            id=f"judge-{test_case.id}-{response_a.model_name}-vs-{response_b.model_name}" # More specific ID
        )

        # Call the judge's generate method
        judge_response_obj = self.model_runner.generate(
            test_case=judge_test_case
        )

        # Log the response
        logger.info(f"Judge raw response for pair ({response_a.model_name} vs {response_b.model_name}) on test ID {test_case.id} (truncated): {judge_response_obj.output[:500]}...")

        # Parse the judge's decision
        parsed_result = self.parse_judge_response(judge_response_obj.output)

        # Return the EvaluationResult with model names
        return EvaluationResult(
            test_id=str(test_case.id or "unknown"),
            model_a_name=response_a.model_name,
            model_b_name=response_b.model_name,
            model_a_output=response_a.output, # Store original output
            model_b_output=response_b.output, # Store original output
            winner=parsed_result["winner"],
            confidence=parsed_result["confidence"],
            reasoning=judge_response_obj.output, # Store full raw response
        )

    def parse_judge_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the judge's raw response string (unchanged)."""
        verdict = "UNDETERMINED"
        confidence = 0.0
        logger.debug(f"Parsing judge response (first 100 chars): {response_text[:100]}")
        verdict_match = re.search(r"^\s*VERDICT:\s*(MODEL_A_WINS|MODEL_B_WINS|TIE)\s*$", response_text, re.IGNORECASE | re.MULTILINE)
        if verdict_match: verdict = verdict_match.group(1).upper()
        else:
            bracket_match = re.search(r"\[\[\s*(MODEL_A_WINS|MODEL_B_WINS|TIE)\s*\]\]", response_text, re.IGNORECASE)
            if bracket_match: verdict = bracket_match.group(1).upper()
            else:
                simple_bracket_match = re.search(r"\[\[\s*([AB]|TIE)\s*\]\]", response_text, re.IGNORECASE)
                if simple_bracket_match:
                    verdict_text = simple_bracket_match.group(1).upper()
                    if verdict_text == "A": verdict = "MODEL_A_WINS"
                    elif verdict_text == "B": verdict = "MODEL_B_WINS"
                    else: verdict = "TIE"

        confidence_match = re.search(r"^\s*CONFIDENCE:\s*(\d(?:\.\d)?)\s*/\s*5\s*$", response_text, re.IGNORECASE | re.MULTILINE)
        if confidence_match:
            try:
                confidence_score = float(confidence_match.group(1))
                confidence = max(0.2, min(1.0, confidence_score / 5.0))
            except ValueError: logger.warning(f"Could not parse CONFIDENCE value: {confidence_match.group(1)}")
        else:
            score_match = re.search(r"(?:rating|score)[:\s]*(\d(?:\.\d)?)\s*/\s*(\d+)", response_text, re.IGNORECASE)
            if score_match:
                try:
                    score, scale = float(score_match.group(1)), float(score_match.group(2))
                    if scale > 0: confidence = max(0.2, min(1.0, score / scale))
                except ValueError: pass

        if verdict == "UNDETERMINED":
            logger.warning(f"Could not reliably parse VERDICT from judge response: {response_text[:200]}...")
            # Fallback keyword check (less reliable)
            resp_lower = response_text.lower()
            # Check for explicit model names if possible, otherwise fallback to A/B
            if "model a wins" in resp_lower and "model b wins" not in resp_lower: verdict = "MODEL_A_WINS"
            elif "model b wins" in resp_lower and "model a wins" not in resp_lower: verdict = "MODEL_B_WINS"
            elif "tie" in resp_lower or "comparable" in resp_lower or "similar" in resp_lower: verdict = "TIE"

        if verdict != "UNDETERMINED" and confidence == 0.0:
            confidence = 0.6
            logger.info(f"Could not parse CONFIDENCE, assigning default {confidence} for verdict {verdict}")

        logger.info(f"Final parsed judge result - Winner (A vs B): {verdict}, Confidence: {confidence:.2f}")
        return {"winner": verdict, "confidence": confidence}


class ResultAggregator:
    """Collects pairwise evaluation results and calculates N-way summary statistics."""

    def aggregate(self, evaluation_results: List[EvaluationResult], model_names: List[str]) -> Dict[str, Any]:
        """Aggregates pairwise results into N-way stats."""
        total_pairwise_evaluations = len(evaluation_results)
        if not total_pairwise_evaluations:
            return {
                "total_pairwise_evaluations": 0,
                "pairwise_stats": {}, "model_summary": {}, "win_matrix": None,
                "average_confidence": 0.0, "raw_evaluations": []
            }

        # Structure: { (sorted_model_a_name, sorted_model_b_name): { "MODEL_A_WINS": 0, "MODEL_B_WINS": 0, "TIE": 0, ... } }
        pairwise_stats: Dict[Tuple[str, str], Dict[str, int]] = {}
        # Structure: { model_name: { "wins": 0, "losses": 0, "ties": 0, "comparisons": 0, "avg_confidence": 0.0, "confidence_sum": 0.0 } }
        model_summary: Dict[str, Dict[str, Any]] = {name: {"wins": 0, "losses": 0, "ties": 0, "comparisons": 0, "avg_confidence": 0.0, "confidence_sum": 0.0, "undetermined": 0, "judge_errors": 0 } for name in model_names}

        confidence_sum_total = 0
        valid_verdicts_total = 0
        processed_test_ids = set() # Track unique test cases evaluated

        for result in evaluation_results:
            model_a = result.model_a_name
            model_b = result.model_b_name
            winner = result.winner
            confidence = result.confidence
            test_id = result.test_id
            processed_test_ids.add(test_id)

            # Ensure consistent pair ordering (alphabetical)
            ordered_pair = tuple(sorted((model_a, model_b)))
            first_model, second_model = ordered_pair # first_model is alphabetically first

            if ordered_pair not in pairwise_stats:
                pairwise_stats[ordered_pair] = {"MODEL_A_WINS": 0, "MODEL_B_WINS": 0, "TIE": 0, "UNDETERMINED": 0, "JUDGE_ERROR": 0, "total_confidence": 0.0, "valid_comps": 0}

            stats = pairwise_stats[ordered_pair]

            # Determine who won in the context of the ordered pair
            if winner == "MODEL_A_WINS":
                if model_a == first_model: # first_model won
                    stats["MODEL_A_WINS"] += 1
                    model_summary[first_model]["wins"] += 1
                    model_summary[second_model]["losses"] += 1
                else: # second_model won (model_a was the second model)
                    stats["MODEL_B_WINS"] += 1
                    model_summary[second_model]["wins"] += 1
                    model_summary[first_model]["losses"] += 1
            elif winner == "MODEL_B_WINS":
                if model_b == first_model: # first_model won (model_b was the first model)
                    # This case shouldn't happen if model_a/b map correctly, but handle defensively
                    stats["MODEL_A_WINS"] += 1
                    model_summary[first_model]["wins"] += 1
                    model_summary[second_model]["losses"] += 1
                else: # second_model won
                    stats["MODEL_B_WINS"] += 1
                    model_summary[second_model]["wins"] += 1
                    model_summary[first_model]["losses"] += 1
            elif winner == "TIE":
                stats["TIE"] += 1
                model_summary[first_model]["ties"] += 1
                model_summary[second_model]["ties"] += 1
            elif winner == "UNDETERMINED":
                stats["UNDETERMINED"] += 1
                model_summary[first_model]["undetermined"] += 1
                model_summary[second_model]["undetermined"] += 1
            elif winner == "JUDGE_ERROR":
                stats["JUDGE_ERROR"] += 1
                model_summary[first_model]["judge_errors"] += 1
                model_summary[second_model]["judge_errors"] += 1

            # Update comparison counts and confidence sums
            model_summary[first_model]["comparisons"] += 1
            model_summary[second_model]["comparisons"] += 1
            if winner not in ["UNDETERMINED", "JUDGE_ERROR"]:
                stats["total_confidence"] += confidence
                stats["valid_comps"] += 1
                model_summary[first_model]["confidence_sum"] += confidence
                model_summary[second_model]["confidence_sum"] += confidence
                confidence_sum_total += confidence
                valid_verdicts_total += 1


        # Calculate average confidences
        average_confidence_overall = (confidence_sum_total / valid_verdicts_total) if valid_verdicts_total > 0 else 0.0
        for name in model_names:
            valid_comps_model = model_summary[name]["wins"] + model_summary[name]["losses"] + model_summary[name]["ties"]
            if valid_comps_model > 0:
                model_summary[name]["avg_confidence"] = model_summary[name]["confidence_sum"] / valid_comps_model

        # Create Win Matrix (using pandas for nice formatting)
        win_matrix_df = None
        if model_names:
            matrix_data = pd.DataFrame(index=model_names, columns=model_names, data=0.0) # Initialize with floats for percentages
            np.fill_diagonal(matrix_data.values, np.nan) # Fill diagonal with NaN

            for pair, stats in pairwise_stats.items():
                model1, model2 = pair # Already sorted alphabetically
                total_determined_pairwise = stats["MODEL_A_WINS"] + stats["MODEL_B_WINS"] + stats["TIE"]
                if total_determined_pairwise > 0:
                    # Win % for model1 against model2
                    win_pct_m1 = (stats["MODEL_A_WINS"] / total_determined_pairwise) * 100
                    # Win % for model2 against model1
                    win_pct_m2 = (stats["MODEL_B_WINS"] / total_determined_pairwise) * 100
                    matrix_data.loc[model1, model2] = win_pct_m1
                    matrix_data.loc[model2, model1] = win_pct_m2
                else: # Handle case with no determined verdicts for the pair
                    matrix_data.loc[model1, model2] = np.nan
                    matrix_data.loc[model2, model1] = np.nan

            win_matrix_df = matrix_data.round(1) # Round percentages


        # Convert EvaluationResult objects to dictionaries for JSON serialization
        raw_eval_dicts = []
        for res in evaluation_results:
            try:
                # Convert dataclass to dict
                raw_eval_dicts.append({
                    "test_id": res.test_id,
                    "model_a_name": res.model_a_name,
                    "model_b_name": res.model_b_name,
                    "model_a_output": res.model_a_output,
                    "model_b_output": res.model_b_output,
                    "winner": res.winner,
                    "confidence": res.confidence,
                    "reasoning": res.reasoning,
                })
            except AttributeError as e:
                logger.error(f"Error converting EvaluationResult to dict for test_id {res.test_id}: {e}")
                raw_eval_dicts.append({"test_id": getattr(res, 'test_id', 'unknown'), "error": "Failed to serialize result"})

        return {
            "total_pairwise_evaluations": total_pairwise_evaluations,
            "total_test_cases_processed": len(processed_test_ids),
            "pairwise_stats": pairwise_stats, # Detailed counts per pair
            "model_summary": model_summary, # Aggregated wins/losses/ties per model
            "win_matrix_df": win_matrix_df, # Pandas DataFrame for win matrix
            "average_confidence": round(average_confidence_overall, 3),
            "raw_evaluations": raw_eval_dicts # List of all pairwise evaluation dicts
        }

class ModelTester:
    """Orchestrates N-way pairwise testing."""

    def __init__(
        self,
        endpoints: List[ModelEndpoint], # Changed from champion/challenger
        judge_endpoint: ModelEndpoint,
        model_prompt_template: str,
        judge_prompt_template: str = LMJudge.DEFAULT_EVALUATION_PROMPT
    ):
        if not endpoints:
            raise ValueError("At least one model endpoint is required.")
        if not judge_endpoint:
            raise ValueError("A judge endpoint is required.")

        self.endpoints = [ep for ep in endpoints if ep.is_active] # Only consider active endpoints
        self.endpoint_names = [ep.name for ep in self.endpoints]
        if len(self.endpoints) < 2:
            logger.warning(f"Only {len(self.endpoints)} active endpoints provided. Pairwise comparison requires at least 2.")
            # Proceed anyway, maybe for single model testing? Or raise error? Let's allow it for now.

        self.judge_endpoint = judge_endpoint
        self.model_prompt_template = model_prompt_template

        # Create runners for active models
        self.model_runners: Dict[str, ModelRunner] = {
            ep.name: ModelRunner(ep, model_prompt_template) for ep in self.endpoints
        }
        # Create judge runner
        self.judge = LMJudge(judge_endpoint, evaluation_prompt_template=judge_prompt_template)

        self.aggregator = ResultAggregator()

    def run_test(
        self,
        test_cases: List[TestCase],
        batch_size: int = 5,
        progress=None,
        batch_retry_attempts: int = 0,
        batch_backoff_factor: float = 2.0,
        batch_max_wait: int = 60,
        batch_retry_trigger_strings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run N-way pairwise test: generate responses, evaluate pairs, aggregate.
        """
        all_pairwise_evaluation_results: List[EvaluationResult] = []
        # Metrics per model
        model_metrics: Dict[str, Dict[str, Any]] = {
            name: {"total_latency": 0.0, "total_output_chars": 0, "success_count": 0, "error_count": 0, "image_load_errors": 0}
            for name in self.endpoint_names
        }
        judge_metrics = {"total_latency": 0.0, "total_output_chars": 0, "success_count": 0, "error_count": 0, "pairs_evaluated": 0}

        num_cases = len(test_cases)
        if num_cases == 0:
            logger.warning("No test cases provided.")
            yield {"type": "final", "evaluations": [], "summary": {"error": "No test cases loaded."}}
            return

        if len(self.endpoints) < 2:
            logger.warning("Cannot perform pairwise evaluation with fewer than 2 active models.")
            yield {"type": "final", "evaluations": [], "summary": {"error": "Need at least 2 active models for pairwise comparison."}}
            return

        total_batches = (num_cases + batch_size - 1) // batch_size
        processed_case_count = 0
        global STOP_REQUESTED

        last_update_payload = None # Cache last update to yield before next case

        for i in range(0, num_cases, batch_size):
            if STOP_REQUESTED:
                logger.warning(f"Stop requested. Finishing early after processing {processed_case_count} cases.")
                if progress is not None: progress(processed_case_count / num_cases, f"Stopping early after {processed_case_count} cases...")
                break
            current_batch = test_cases[i:min(i + batch_size, num_cases)]
            batch_num = i // batch_size + 1
            logger.info(f"--- Processing Batch {batch_num}/{total_batches} (Cases {i+1}-{min(i+batch_size, num_cases)}) ---")

            retry_count = 0
            batch_success = False
            batch_pairwise_eval_results: List[EvaluationResult] = [] # Store results for this successful batch attempt

            while not batch_success and retry_count <= batch_retry_attempts:
                if retry_count > 0:
                    delay = min(batch_backoff_factor ** (retry_count - 1), batch_max_wait)
                    logger.info(f"Retrying batch {batch_num} (attempt {retry_count}/{batch_retry_attempts}) after {delay:.2f}s delay")
                    if progress is not None: progress(processed_case_count / num_cases, f"Retrying Batch {batch_num} ({retry_count}/{batch_retry_attempts})")
                    time.sleep(delay)
                elif progress is not None:
                    progress(processed_case_count / num_cases, f"Running Batch {batch_num}/{total_batches}")
                # Stores for this attempt
                current_attempt_model_responses: Dict[str, Dict[str, ModelResponse]] = {tc.id: {} for tc in current_batch} # {case_id: {model_name: response}}
                current_attempt_pairwise_results: List[EvaluationResult] = []
                has_trigger_string_in_attempt = False

                # --- 1. Generate responses for all models for all cases in the batch ---
                for batch_idx, test_case in enumerate(current_batch):
                    if STOP_REQUESTED: break
                    case_id = test_case.id or f"case-{i + batch_idx + 1}"
                    test_case.id = case_id # Ensure ID is set

                    # Yield previous update before processing next case
                    if last_update_payload:
                        if STOP_REQUESTED: break
                        yield last_update_payload
                        last_update_payload = None # Clear after yielding

                    logger.info(f"Generating responses for test case {case_id} across {len(self.endpoints)} models...")
                    for endpoint in self.endpoints:
                        if STOP_REQUESTED: break
                        model_name = endpoint.name
                        runner = self.model_runners[model_name]
                        model_resp = None
                        try:
                            model_resp = runner.generate(test_case)
                            current_attempt_model_responses[case_id][model_name] = model_resp
                            m_metrics = model_metrics[model_name] # Get metrics dict for this model
                            if not model_resp.is_error:
                                m_metrics["total_latency"] += model_resp.latency
                                m_metrics["total_output_chars"] += len(model_resp.output)
                                m_metrics["success_count"] += 1
                            else:
                                m_metrics["error_count"] += 1
                                if "Error: Failed to load" in model_resp.output or "Error: Failed to prepare" in model_resp.output:
                                    m_metrics["image_load_errors"] += 1
                        except Exception as e:
                            logger.error(f"Critical error generating response for model {model_name}, case {case_id}: {e}", exc_info=True)
                            model_resp = ModelResponse(case_id, model_name, f"Error: Generation failed critically - {e}", 0, is_error=True)
                            current_attempt_model_responses[case_id][model_name] = model_resp
                            model_metrics[model_name]["error_count"] += 1

                        # Check for trigger strings in model responses *during* generation
                        if model_resp and not model_resp.is_error and batch_retry_attempts > 0 and batch_retry_trigger_strings:
                            for trigger in batch_retry_trigger_strings:
                                if trigger in model_resp.output.lower(): # Case-insensitive check
                                    logger.warning(f"Trigger string '{trigger}' found in {model_name}'s response for case {case_id}. Batch retry likely.")
                                    has_trigger_string_in_attempt = True
                                    # No need to break inner loop, let other models generate
                    if STOP_REQUESTED: break
                if STOP_REQUESTED: break # Break outer generation loop

                # --- 2. Evaluate pairs for each case in the batch ---
                if progress is not None: progress((processed_case_count + len(current_batch) * 0.5) / num_cases, f"Evaluating Pairs for Batch {batch_num}")
                for test_case in current_batch:
                    if STOP_REQUESTED: break
                    case_id = test_case.id
                    case_responses = current_attempt_model_responses.get(case_id, {})
                    if len(case_responses) < len(self.endpoints):
                        logger.warning(f"Missing some model responses for case {case_id}, skipping pairwise evaluation.")
                        continue # Skip if not all models responded (or errors occurred)

                    # Generate pairs of model names
                    model_pairs = list(itertools.combinations(self.endpoint_names, 2))
                    logger.info(f"Evaluating {len(model_pairs)} pairs for test case {case_id}...")

                    # Store interim evaluation details for UI update
                    evaluations_for_update = []
                    combined_latency = sum(resp.latency for resp in case_responses.values() if not resp.is_error)
                    avg_latency = combined_latency / len(case_responses) if case_responses else 0

                    for model_a_name, model_b_name in model_pairs:
                        if STOP_REQUESTED: break
                        response_a = case_responses.get(model_a_name)
                        response_b = case_responses.get(model_b_name)

                        # Skip evaluation if either model failed critically
                        if not response_a or response_a.is_error or not response_b or response_b.is_error:
                            logger.warning(f"Skipping evaluation for pair ({model_a_name}, {model_b_name}) on case {case_id} due to generation error.")
                            eval_reason = f"Skipped: Generation Error (A: {response_a.output[:50] if response_a else 'N/A'}... B: {response_b.output[:50] if response_b else 'N/A'}...)"
                            eval_result = EvaluationResult(
                                test_id=case_id, model_a_name=model_a_name, model_b_name=model_b_name,
                                model_a_output=response_a.output if response_a else "GENERATION FAILED",
                                model_b_output=response_b.output if response_b else "GENERATION FAILED",
                                winner="JUDGE_ERROR", confidence=0.0, reasoning=eval_reason
                            )
                            current_attempt_pairwise_results.append(eval_result)
                            judge_metrics["error_count"] += 1 # Count as judge error if models failed
                            evaluations_for_update.append(eval_result.__dict__) # Add for UI update
                            continue

                        # If retry triggered by model response earlier, mark pair as undetermined
                        if has_trigger_string_in_attempt:
                            eval_result = EvaluationResult(
                                test_id=case_id, model_a_name=model_a_name, model_b_name=model_b_name,
                                model_a_output=response_a.output, model_b_output=response_b.output,
                                winner="UNDETERMINED", confidence=0.0, reasoning="Retry triggered by model output string."
                            )
                            current_attempt_pairwise_results.append(eval_result)
                            judge_metrics["error_count"] += 1 # Count as error for judge metrics
                            evaluations_for_update.append(eval_result.__dict__) # Add for UI update
                            continue

                        # --- Call Judge ---
                        try:
                            start_time_judge = time.time()
                            evaluation_result = self.judge.evaluate(
                                test_case, response_a, response_b
                            )
                            judge_latency = time.time() - start_time_judge
                            judge_metrics["total_latency"] += judge_latency
                            judge_metrics["total_output_chars"] += len(evaluation_result.reasoning)
                            judge_metrics["pairs_evaluated"] += 1

                            # Check for trigger strings in judge reasoning
                            if batch_retry_attempts > 0 and batch_retry_trigger_strings:
                                for trigger in batch_retry_trigger_strings:
                                    if trigger in evaluation_result.reasoning.lower(): # Case-insensitive
                                        logger.warning(f"Trigger string '{trigger}' found in judge reasoning for pair ({model_a_name}, {model_b_name}), case {case_id}. Batch retry likely.")
                                        has_trigger_string_in_attempt = True
                                        evaluation_result.winner = "UNDETERMINED"
                                        evaluation_result.reasoning += "\n[Retry triggered by judge reasoning]"
                                        break

                            current_attempt_pairwise_results.append(evaluation_result)
                            evaluations_for_update.append(evaluation_result.__dict__) # Add for UI update

                            # Update judge success/error counts based on final verdict for this pair
                            if evaluation_result.winner not in ["UNDETERMINED", "JUDGE_ERROR"]: judge_metrics["success_count"] += 1
                            else: judge_metrics["error_count"] += 1

                        except Exception as e:
                            logger.error(f"Error during judge evaluation for pair ({model_a_name}, {model_b_name}), case {case_id}: {e}", exc_info=True)
                            eval_result = EvaluationResult(
                                test_id=case_id, model_a_name=model_a_name, model_b_name=model_b_name,
                                model_a_output=response_a.output, model_b_output=response_b.output,
                                winner="JUDGE_ERROR", confidence=0.0,
                                reasoning=f"Error: Judge evaluation failed critically - {e}"
                            )
                            current_attempt_pairwise_results.append(eval_result)
                            judge_metrics["error_count"] += 1
                            evaluations_for_update.append(eval_result.__dict__) # Add for UI update
                            # Optionally trigger retry on critical judge error:
                            # has_trigger_string_in_attempt = True

                    if STOP_REQUESTED: break # Break pair loop

                    # --- Yield intermediate update AFTER evaluating all pairs for this case ---
                    if evaluations_for_update: # Only yield if evaluations happened
                        if STOP_REQUESTED: break
                        # Prepare payload for UI update
                        # Include avg model latency, judge latency (maybe avg per pair?), and list of pair results
                        avg_judge_latency_case = judge_metrics["total_latency"] / judge_metrics["pairs_evaluated"] if judge_metrics["pairs_evaluated"] > 0 else 0
                        update_payload = {
                            "type": "update",
                            "case_id": case_id,
                            "image_path": test_case.image_path_or_url,
                            "avg_model_latency": round(avg_latency, 3),
                            "avg_judge_latency_pairs": round(avg_judge_latency_case, 3), # Avg judge latency for pairs in this case
                            "pairwise_evaluations": evaluations_for_update # List of dicts for pairs in this case
                        }
                        last_update_payload = update_payload # Cache to yield before next case
                        # yield update_payload # Yield immediately? Or cache? Let's cache.

                if STOP_REQUESTED: break # Break case loop in batch

                # --- Batch Retry Logic ---
                if has_trigger_string_in_attempt and retry_count < batch_retry_attempts:
                    logger.warning(f"Batch {batch_num} attempt {retry_count+1} failed due to trigger strings. Retrying...")
                    retry_count += 1
                    current_attempt_pairwise_results = [] # Discard results from failed attempt
                    continue # Go to next retry iteration
                else:
                    batch_success = True
                    batch_pairwise_eval_results = current_attempt_pairwise_results # Store results
                    if has_trigger_string_in_attempt and retry_count >= batch_retry_attempts:
                        logger.warning(f"Accepting batch {batch_num} results despite trigger strings after exhausting retries.")
                    # Log summary for the completed batch attempt
                    # Aggregating just for logging here might be excessive, just log count
                    logger.info(f"Batch {batch_num} completed with {len(batch_pairwise_eval_results)} pairwise evaluations.")

            # --- End of Batch Processing ---
            all_pairwise_evaluation_results.extend(batch_pairwise_eval_results)
            processed_case_count += len(current_batch) # Update count *after* successful batch

        # --- Final Aggregation and Results ---
        if STOP_REQUESTED and last_update_payload:
            yield last_update_payload # Yield the last cached update if stopped

        logger.info(f"--- Aggregating Final Results ({processed_case_count} cases processed) ---")
        aggregated_summary = self.aggregator.aggregate(all_pairwise_evaluation_results, self.endpoint_names)

        # --- Calculate final metrics ---
        final_model_metrics = {}
        for name in self.endpoint_names:
            metrics = model_metrics[name]
            total_attempts = metrics["success_count"] + metrics["error_count"] # Total attempts for this model
            valid_latency_runs = total_attempts - metrics.get("image_load_errors", 0) # Exclude image load errors from latency avg
            avg_latency = round(metrics["total_latency"] / valid_latency_runs, 2) if valid_latency_runs > 0 else 0
            avg_chars = int(metrics["total_output_chars"] / metrics["success_count"]) if metrics["success_count"] > 0 else 0
            # Success rate based on total cases *attempted* by this model
            success_rate = round((metrics["success_count"] / total_attempts) * 100, 1) if total_attempts > 0 else 0
            final_model_metrics[name] = {
                "avg_latency_s": avg_latency, "avg_output_chars": avg_chars,
                "success_rate_pct": success_rate, "errors": metrics["error_count"],
                "image_load_errors": metrics.get("image_load_errors", 0)
            }

        # Judge metrics based on pairs evaluated
        judge_attempts = judge_metrics["success_count"] + judge_metrics["error_count"]
        judge_avg_latency = round(judge_metrics["total_latency"] / judge_metrics["pairs_evaluated"], 2) if judge_metrics["pairs_evaluated"] > 0 else 0
        judge_avg_chars = int(judge_metrics["total_output_chars"] / judge_metrics["pairs_evaluated"]) if judge_metrics["pairs_evaluated"] > 0 else 0
        # Judge success rate: pairs successfully evaluated / pairs attempted evaluation
        judge_success_rate = round((judge_metrics["success_count"] / judge_attempts) * 100, 1) if judge_attempts > 0 else 0
        final_judge_metrics = {
            "avg_latency_s": judge_avg_latency, "avg_output_chars": judge_avg_chars,
            "success_rate_pct": judge_success_rate, "errors": judge_metrics["error_count"], # Includes undetermined, judge errors, skipped pairs
            "pairs_evaluated": judge_metrics["pairs_evaluated"]
        }

        # --- Combine Aggregated Results and Metrics ---
        final_summary_data = {
            "total_test_cases_processed": aggregated_summary["total_test_cases_processed"],
            "total_test_cases_loaded": num_cases,
            "model_names": self.endpoint_names,
            "judge_name": self.judge_endpoint.name,
            "pairwise_stats": aggregated_summary["pairwise_stats"], # Raw pair counts
            "model_summary_stats": aggregated_summary["model_summary"], # Overall model W/L/T
            "win_matrix_df": aggregated_summary["win_matrix_df"], # Win % matrix
            "average_confidence": aggregated_summary["average_confidence"],
            "model_performance_metrics": final_model_metrics, # Latency, errors per model
            "judge_performance_metrics": final_judge_metrics,
        }

        if progress is not None:
            final_status = "Testing completed" if not STOP_REQUESTED else "Testing stopped early"
            progress(1.0, final_status)

        logger.info("--- N-Way Test Finished ---")
        # Log summary stats here? (e.g., win matrix)
        if final_summary_data["win_matrix_df"] is not None:
            logger.info("Pairwise Win Matrix (%):\n" + final_summary_data["win_matrix_df"].to_string())
        logger.info("Model Summary Stats: " + json.dumps(final_summary_data["model_summary_stats"], indent=2))


        # Yield final results
        yield {
            "type": "final",
            "evaluations": aggregated_summary["raw_evaluations"], # List of all pairwise eval dicts
            "summary": final_summary_data # Combined summary and metrics
        }


# --- Gradio UI Components & Logic ---

def parse_test_data(
    file_obj, text_data, key_field_name: str = "key", value_field_name: str = "value", image_field_name: str = "image_url"
) -> List[TestCase]:
    """ Parses test data (unchanged from A/B version). """
    test_cases = []
    raw_data = None
    # ... (rest of parsing logic is identical to the original script) ...
    if file_obj is not None:
        file_path = file_obj.name
        logger.info(f"Loading test data from uploaded file: {file_path}")
        try:
            _, file_ext = os.path.splitext(file_path)
            file_ext = file_ext.lower()
            if file_ext == ".json":
                with open(file_path, 'r', encoding='utf-8') as f: raw_data = json.load(f)
            elif file_ext == ".csv":
                try:
                    df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='warn', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
                    logger.info(f"CSV loaded successfully. Columns: {df.columns.tolist()}")
                    df = df.fillna('')
                    raw_data = df.to_dict(orient='records')
                except Exception as e: raise ValueError(f"Error reading CSV: {e}")
            elif file_ext in (".jsonl", ".ndjson"):
                raw_data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line: continue
                        try: raw_data.append(json.loads(line))
                        except json.JSONDecodeError: logger.warning(f"Skipping invalid JSON line #{line_num + 1} in file '{file_path}': {line[:100]}...")
                if not raw_data: raise ValueError("No valid JSON objects found in JSONL file.")
            else: raise ValueError(f"Invalid file type ({file_ext}). Allowed: .csv, .json, .jsonl, .ndjson")
        except Exception as e: raise ValueError(f"Failed to process file: {e}")
    elif text_data and text_data.strip():
        logger.info("Loading test data from text input.")
        try:
            raw_data = json.loads(text_data)
            if not isinstance(raw_data, list): raise ValueError("Pasted text is valid JSON, but not a list of objects.")
        except json.JSONDecodeError:
            try:
                raw_data = [json.loads(line) for line in text_data.strip().splitlines() if line.strip()]
                if not raw_data: raise ValueError("No valid JSON objects found in text input lines.")
            except json.JSONDecodeError as line_err: raise ValueError(f"Invalid JSON format (list or line-by-line): {line_err}")
        except Exception as e: raise ValueError(f"Failed to process text data: {e}")
    else: raise ValueError("No test data provided.")

    if isinstance(raw_data, list):
        for i, item in enumerate(raw_data):
            if isinstance(item, dict):
                try:
                    key = item.get(key_field_name)
                    if key is None:
                        logger.warning(f"Skipping item {i+1} due to missing '{key_field_name}' field. Data: {item}")
                        continue
                    image_val = item.get(image_field_name) if image_field_name else None
                    image_path_or_url = str(image_val).strip() if image_val and str(image_val).strip() else None
                    test_cases.append(TestCase(
                        id=str(item.get('id', f"item-{i+1}")), key=str(key), value=str(item.get(value_field_name, '')), image_path_or_url=image_path_or_url
                    ))
                except Exception as e: logger.warning(f"Skipping item {i+1} due to error during TestCase creation: {e}. Data: {item}")
            else: logger.warning(f"Skipping item {i+1} as it is not a dictionary. Data: {item}")
    else: raise ValueError("Parsed data is not a list of test cases.")
    if not test_cases: raise ValueError("No valid test cases could be loaded.")
    logger.info(f"Successfully loaded {len(test_cases)} test cases.")
    return test_cases

def format_summary_output(summary_data: Dict[str, Any]) -> str:
    """Formats the N-way summary dictionary into a readable string."""
    if not summary_data or summary_data.get("error"):
        return f"Error generating summary: {summary_data.get('error', 'Unknown error')}"

    model_names = summary_data.get('model_names', [])
    n_models = len(model_names)
    output = f"--- N-Way Pairwise Test Summary ---\n"
    output += f"Models Tested ({n_models}): {', '.join(model_names)}\n"
    output += f"Judge: {summary_data.get('judge_name', 'N/A')}\n"
    output += f"Test Cases Loaded: {summary_data.get('total_test_cases_loaded', 'N/A')}\n"
    output += f"Test Cases Processed: {summary_data.get('total_test_cases_processed', 'N/A')}\n"

    # Model Summary Stats (W/L/T)
    output += "\n--- Model Overall Pairwise Performance ---\n"
    model_stats = summary_data.get('model_summary_stats', {})
    if model_stats:
        header = f"{'Model':<20} | {'Wins':>6} | {'Losses':>6} | {'Ties':>6} | {'Undet.':>6} | {'Errors':>6} | {'Avg Conf':>8} | {'Comps':>6}"
        output += header + "\n" + "-"*len(header) + "\n"
        for name in model_names:
            stats = model_stats.get(name, {})
            output += (f"{name:<20} | {stats.get('wins', 0):>6} | {stats.get('losses', 0):>6} | "
                      f"{stats.get('ties', 0):>6} | {stats.get('undetermined', 0):>6} | {stats.get('judge_errors', 0):>6} | "
                      f"{stats.get('avg_confidence', 0.0):>8.3f} | {stats.get('comparisons', 0):>6}\n")
    else:
        output += "No model summary statistics available.\n"

    # Win Matrix
    output += "\n--- Pairwise Win Matrix (%) ---\n"
    win_matrix_df = summary_data.get('win_matrix_df')
    if win_matrix_df is not None and not win_matrix_df.empty:
        output += "(Row model's win percentage against Column model)\n"
        # Use pandas to_markdown for better formatting if complex, or simple string conversion
        try:
            output += win_matrix_df.to_string(na_rep='-') + "\n"
        except Exception as e:
            logger.error(f"Error formatting win matrix: {e}")
            output += "Error displaying win matrix.\n"
    else:
        output += "Win matrix not available (requires at least 2 models and successful evaluations).\n"

    avg_conf = summary_data.get('average_confidence', 0)
    output += f"\nOverall Average Confidence (Determined Pairs): {avg_conf:.3f}\n"

    # Model Performance Metrics
    output += "\n--- Model Performance Metrics ---\n"
    perf_metrics = summary_data.get('model_performance_metrics', {})
    if perf_metrics:
        header = f"{'Model':<20} | {'Avg Lat (s)':>11} | {'Avg Chars':>9} | {'Success %':>10} | {'Errors':>6} | {'Img Err':>7}"
        output += header + "\n" + "-" * len(header) + "\n"
        for name in model_names:
            metrics = perf_metrics.get(name, {})
            output += (f"{name:<20} | {metrics.get('avg_latency_s', 0.0):>11.2f} | {metrics.get('avg_output_chars', 0):>9} | "
                      f"{metrics.get('success_rate_pct', 0.0):>9.1f}% | {metrics.get('errors', 0):>6} | {metrics.get('image_load_errors', 0):>7}\n")
    else:
        output += "No performance metrics available.\n"

    # Judge Performance Metrics
    output += "\n--- Judge Performance Metrics ---\n"
    judge_metrics = summary_data.get('judge_performance_metrics', {})
    if judge_metrics:
        output += (f" Avg Latency/Pair (s): {judge_metrics.get('avg_latency_s', 0.0):.2f}\n"
                  f" Avg Output Chars/Pair: {judge_metrics.get('avg_output_chars', 0)}\n"
                  f" Success Rate (% Pairs): {judge_metrics.get('success_rate_pct', 0.0):.1f}%\n"
                  f" Evaluation Errors/Skipped Pairs: {judge_metrics.get('errors', 0)}\n"
                  f" Total Pairs Evaluated: {judge_metrics.get('pairs_evaluated', 0)}\n")
    else:
        output += "No judge metrics available.\n"

    if "post_processing_analysis" in summary_data:
        ngram_info = summary_data["post_processing_analysis"]["ngram_frequency"]
        output += f"\n--- Post-Processing: Top {ngram_info['top_k_reported']} {ngram_info['n_value']}-grams ---\n"
        for name, data in ngram_info.get("results_by_model", {}).items():
            output += f" Model: {name}\n"
            if not data["most_frequent"]:
                output += "  - (No n-grams generated)\n"
            else:
                for item in data["most_frequent"]:
                    output += f"  - '{item['ngram']}' (Count: {item['count']})\n"

    return output


def run_test_from_ui(
    # Need to gather N model configs now. Use *args or pass a list/dict if possible.
    # For fixed N=4 approach: list all inputs explicitly.
    model_configs_state: List[Dict], # Get configs from state
    judge_name, judge_api_url, judge_model_id, judge_temp, judge_max_tokens, judge_file_upload_method, judge_is_active, # Judge Config
    api_key_input, # API Key
    model_prompt_template_input, # Prompts
    judge_prompt_template_input,
    test_data_file, test_data_text, # Test Data
    batch_size_input, batch_retry_attempts_input, batch_backoff_factor_input, batch_max_wait_input, batch_retry_trigger_strings_input, # Test Params
    enable_ngram_input, n_val_input, top_k_input, # Post-processing Params
    key_field_name_input, value_field_name_input, image_field_name_input, # Field Names
    progress=gr.Progress(track_tqdm=True)
):
    """
    Handles the logic for running the N-way test triggered by the Gradio UI.
    Uses model configurations passed via the state.
    """
    global STOP_REQUESTED
    STOP_REQUESTED = False
    logger.info("Starting N-way test run from Gradio UI...")
    progress(0, desc="Initializing...")

    # --- Intermediate Update Storage ---
    # We need to store results per case to update the UI incrementally.
    # Store the list of pairwise results dicts for the *last processed case*.
    last_case_pairwise_results = []
    last_case_id = ""
    last_image_path = None
    last_avg_model_latency = ""
    last_avg_judge_latency_pairs = ""

    try:
        # 1. Get API Key
        api_key_env = os.getenv("OPENROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GOOGLE_API_KEY")
        api_key_ui = str(api_key_input).strip() if api_key_input else None
        api_key = api_key_ui if api_key_ui else api_key_env
        if api_key: logger.info(f"API Key found ({'UI input' if api_key_ui else 'environment variable'}).")
        else: logger.info("API Key not provided. Only local/keyless endpoints will work.")

        progress(0.1, desc="Loading test data...")
        # 2. Load Test Cases
        try:
            key_field = str(key_field_name_input).strip() or "key"
            value_field = str(value_field_name_input).strip() or "value"
            image_field = str(image_field_name_input).strip() or "image_url"
            logger.info(f"Using data fields - Key: '{key_field}', Value: '{value_field}', Image: '{image_field}'")
            test_cases = parse_test_data(test_data_file, test_data_text, key_field, value_field, image_field)
            logger.info(f"Loaded {len(test_cases)} test cases.")
        except ValueError as e: raise gr.Error(f"Test Data Error: {e}")
        except Exception as e: raise gr.Error(f"Unexpected error loading test data: {e}")
        if not test_cases: raise gr.Error("No valid test cases were loaded.")

        progress(0.2, desc="Configuring models...")
        # 3. Create Model Endpoints from UI State and Judge Inputs
        try:
            # Helper to create endpoint (includes is_active)
            def create_ep(config: Dict, ep_type: str = "Model"):
                name = config.get("name", "").strip()
                url = config.get("api_url", "").strip()
                model_id = config.get("model_id", "").strip()
                upload_method = config.get("file_upload_method", "JSON (Embedded Data)")
                is_active = config.get("is_active", True) # Default to active

                # Only validate thoroughly if active
                if is_active:
                    if not name: raise ValueError(f"{ep_type} Display Name cannot be empty.")
                    # Allow empty URL/model for inactive models
                    if not url: raise ValueError(f"API URL cannot be empty for active {ep_type} '{name}'.")
                    if not model_id: raise ValueError(f"Model ID cannot be empty for active {ep_type} '{name}'.")
                    if upload_method not in ["JSON (Embedded Data)", "Multipart Form Data"]:
                        raise ValueError(f"Invalid file upload method '{upload_method}' for active {ep_type} '{name}'.")

                return ModelEndpoint(
                    name=name, api_url=url, api_key=api_key, model_id=model_id,
                    temperature=float(config.get("temperature", 0.1)),
                    max_tokens=int(config.get("max_tokens", 1024)),
                    file_upload_method=upload_method,
                    is_active=is_active
                )

            # Create endpoints from the state list
            model_endpoints: List[ModelEndpoint] = []
            for i, config_dict in enumerate(model_configs_state):
                # Only create if the config seems populated (has a name)
                if config_dict and config_dict.get("name"):
                    try:
                        model_endpoints.append(create_ep(config_dict, f"Model {i+1}"))
                    except ValueError as ve:
                        # If validation fails for an active model, raise error
                        if config_dict.get("is_active", True):
                            raise gr.Error(f"Configuration Error for Model {i+1}: {ve}")
                        else:
                            logger.warning(f"Skipping inactive Model {i+1} with incomplete config.")


            # Create Judge Endpoint (ensure is_active is handled if added)
            judge_config_dict = {
                "name": judge_name, "api_url": judge_api_url, "model_id": judge_model_id,
                "temperature": judge_temp, "max_tokens": judge_max_tokens,
                "file_upload_method": judge_file_upload_method, "is_active": True # Judge must be active
            }
            judge_endpoint = create_ep(judge_config_dict, "Judge")

            active_model_endpoints = [ep for ep in model_endpoints if ep.is_active]
            if not active_model_endpoints: raise gr.Error("No active models configured. Please check the 'Active' checkboxes.")
            if len(active_model_endpoints) < 2: logger.warning("Fewer than 2 active models selected. Pairwise evaluation will not occur.")

            logger.info(f"Active Model Endpoints ({len(active_model_endpoints)}):")
            for ep in active_model_endpoints: logger.info(f" - {ep.name}, URL: {ep.api_url}, Model: {ep.model_id}, Upload: {ep.file_upload_method}, Key Provided: {'Yes' if ep.api_key else 'No'}")
            logger.info(f"Judge Endpoint: {judge_endpoint.name}, URL: {judge_endpoint.api_url}, Model: {judge_endpoint.model_id}, Upload: {judge_endpoint.file_upload_method}, Key Provided: {'Yes' if judge_endpoint.api_key else 'No'}")

        except ValueError as ve: raise gr.Error(f"Model Configuration Error: {ve}")
        except Exception as e: raise gr.Error(f"Model Configuration Error: {e}")

        # 4. Instantiate ModelTester
        try:
            tester = ModelTester(
                endpoints=active_model_endpoints, # Pass list of active model endpoints
                judge_endpoint=judge_endpoint,
                model_prompt_template=str(model_prompt_template_input),
                judge_prompt_template=str(judge_prompt_template_input)
            )
        except Exception as e: raise gr.Error(f"Tester Initialization Error: {e}")

        # 5. Run the Test
        batch_size = int(batch_size_input) if batch_size_input is not None and batch_size_input > 0 else 1
        logger.info(f"Running N-way test with {len(test_cases)} cases, batch size {batch_size}...")
        progress(0.3, desc=f"Running test ({len(active_model_endpoints)} models)...")
        try:
            batch_retry_attempts = int(batch_retry_attempts_input) if batch_retry_attempts_input is not None else 0
            batch_backoff_factor = float(batch_backoff_factor_input) if batch_backoff_factor_input is not None else 2.0
            batch_max_wait = int(batch_max_wait_input) if batch_max_wait_input is not None else 60
            batch_retry_trigger_strings = None
            if batch_retry_trigger_strings_input and batch_retry_trigger_strings_input.strip():
                batch_retry_trigger_strings = [s.strip().lower() for s in batch_retry_trigger_strings_input.split(',') if s.strip()]
                logger.info(f"Using batch retry trigger strings: {batch_retry_trigger_strings}")

            final_results = None
            running_eval_results_all_pairs = [] # Store all pairwise dicts incrementally

            try: # Inner try for the loop
                for result_update in tester.run_test(
                    test_cases, batch_size=batch_size, progress=progress,
                    batch_retry_attempts=batch_retry_attempts, batch_backoff_factor=batch_backoff_factor,
                    batch_max_wait=batch_max_wait, batch_retry_trigger_strings=batch_retry_trigger_strings
                ):
                    if STOP_REQUESTED: break

                    if result_update.get("type") == "update":
                        # Store info from the last processed case for UI update
                        last_case_id = result_update.get("case_id", "N/A")
                        last_image_path = result_update.get("image_path")
                        last_avg_model_latency = str(result_update.get("avg_model_latency", ""))
                        last_avg_judge_latency_pairs = str(result_update.get("avg_judge_latency_pairs", ""))
                        last_case_pairwise_results = result_update.get("pairwise_evaluations", []) # List of dicts

                        # Add these results to the running total
                        running_eval_results_all_pairs.extend(last_case_pairwise_results)

                        # Create summary for the last case (e.g., list winners of pairs)
                        last_case_summary = f"--- Last Case: {last_case_id} ---\n"
                        if last_case_pairwise_results:
                            last_case_summary += f"Pairs Evaluated: {len(last_case_pairwise_results)}\n"
                            for pair_res in last_case_pairwise_results[:5]: # Show first few pairs
                                pair_winner = "Draw/Error"
                                if pair_res['winner'] == "MODEL_A_WINS": pair_winner = f"{pair_res['model_a_name']} Wins"
                                elif pair_res['winner'] == "MODEL_B_WINS": pair_winner = f"{pair_res['model_b_name']} Wins"
                                elif pair_res['winner'] == "TIE": pair_winner = "Tie"
                                last_case_summary += f" - {pair_res['model_a_name']} vs {pair_res['model_b_name']}: {pair_winner} (Conf: {pair_res.get('confidence', 0.0):.2f})\n"
                            if len(last_case_pairwise_results) > 5: last_case_summary += "...\n"
                        else:
                            last_case_summary += "(No pairs evaluated for this case)\n"

                        # Update DataFrame with all results so far
                        current_details_df = pd.DataFrame()
                        if running_eval_results_all_pairs:
                            try:
                                display_cols = ['test_id', 'model_a_name', 'model_b_name', 'winner', 'confidence', 'model_a_output', 'model_b_output', 'reasoning']
                                current_details_df = pd.DataFrame(running_eval_results_all_pairs)
                                # Ensure all display columns exist
                                for col in display_cols:
                                    if col not in current_details_df.columns:
                                        current_details_df[col] = None
                                current_details_df = current_details_df[display_cols] # Reorder/select
                            except Exception as df_err:
                                logger.error(f"Error creating incremental DataFrame: {df_err}")
                                current_details_df = pd.DataFrame([{"Error": "Failed to update details"}])


                        # Yield 8 values for incremental update
                        yield (
                            last_case_summary, # 1. Summary Text (last case details)
                            current_details_df, # 2. Detailed DataFrame (all pairs so far)
                            last_image_path, # 3. Last image path
                            last_avg_model_latency, # 4. Last avg model latency
                            last_avg_judge_latency_pairs,# 5. Last avg judge latency for pairs
                            "", # 6. Placeholder (was challenger latency) - maybe overall progress?
                            "", # 7. Placeholder (was last winner) - maybe pairs evaluated count?
                            running_eval_results_all_pairs # 8. Hidden state (all raw pair dicts)
                        )

                    elif result_update.get("type") == "final":
                        final_results = result_update
                    else:
                        logger.warning(f"Received unexpected update type: {result_update.get('type')}")

            except Exception as loop_err:
                if STOP_REQUESTED: logger.warning(f"Caught exception during test loop after stop request: {loop_err}")
                else:
                    logger.exception("An unexpected error occurred during the test execution loop.")
                    raise loop_err

            finally: # --- Post-Loop Processing ---
                # Use locals() to check if variables were defined in the loop
                final_summary_output = ""
                final_details_df = pd.DataFrame()
                final_raw_evals = []

                if STOP_REQUESTED:
                    logger.info("Test run stopped by user.")
                    final_summary_output = "Test run stopped by user."
                    if 'current_details_df' in locals() and not current_details_df.empty:
                        final_details_df = current_details_df
                    if 'running_eval_results_all_pairs' in locals():
                        final_raw_evals = running_eval_results_all_pairs

                elif 'final_results' not in locals() or final_results is None:
                    logger.error("Test run finished, but no final results structure was received.")
                    final_summary_output = "Test Execution Error: No final results generated."

                elif final_results:
                    # Run Post-Processing before formatting output
                    final_results = run_post_processing_analysis(final_results, enable_ngram_input, int(n_val_input), int(top_k_input))
                else: # Completed normally
                    logger.info("Test run completed normally.")
                    summary_data = final_results.get("summary", {})
                    final_raw_evals = final_results.get("evaluations", []) # List of pair dicts
                    final_summary_output = format_summary_output(summary_data)
                    display_columns = ['test_id', 'model_a_name', 'model_b_name', 'winner', 'confidence', 'model_a_output', 'model_b_output', 'reasoning']
                    try:
                        if final_raw_evals:
                            final_details_df = pd.DataFrame(final_raw_evals)
                            for col in display_columns:
                                if col not in final_details_df.columns: final_details_df[col] = None
                            final_details_df = final_details_df[display_columns]
                        else: final_summary_output += "\n\nNote: No evaluation results were generated."
                    except Exception as df_err:
                        logger.error(f"Error creating DataFrame from final results: {df_err}")
                        final_summary_output += f"\n\nError displaying detailed results: {df_err}"

                # Ensure monitoring variables exist before final yield
                last_image_path = last_image_path if 'last_image_path' in locals() else None
                last_avg_model_latency = last_avg_model_latency if 'last_avg_model_latency' in locals() else ""
                last_avg_judge_latency_pairs = last_avg_judge_latency_pairs if 'last_avg_judge_latency_pairs' in locals() else ""

                # Final yield
                yield (
                    final_summary_output, # 1. Final summary text
                    final_details_df, # 2. Final details dataframe
                    last_image_path, # 3. Last image path
                    last_avg_model_latency, # 4. Last avg model latency
                    last_avg_judge_latency_pairs, # 5. Last avg judge latency
                    "", # 6. Placeholder
                    "", # 7. Placeholder
                    final_raw_evals # 8. Final raw evaluations (list of dicts)
                )

        except Exception as test_exec_err:
            logger.exception("An error occurred during the main test execution phase.")
            raise test_exec_err # Re-raise to be caught by outer handler

    except gr.Error as e:
        logger.error(f"Gradio Error: {e}")
        error_message = str(e)
        error_df = pd.DataFrame([{"Error": error_message}])
        yield error_message, error_df, None, None, None, None, None, None # Yield 8 error values
    except Exception as e:
        logger.exception("An unexpected error occurred in run_test_from_ui.")
        error_message = f"An unexpected error occurred: {e}"
        error_df = pd.DataFrame([{"Error": error_message}])
        yield error_message, error_df, None, None, None, None, None, None # Yield 8 error values
    finally:
        STOP_REQUESTED = False


# --- Helper Function for Downloads ---
def generate_jsonl_download(results_list: Optional[List[Dict[str, Any]]]) -> Optional[gr.File]:
    """ Takes list of PAIRWISE evaluation dicts, saves as JSONL. (Unchanged) """
    if results_list is None: # Handle case where run was stopped before completion
        logger.warning("generate_jsonl_download called with None results_list (run stopped or failed early). Returning None.")
        return None
    if not results_list:
        logger.warning("generate_jsonl_download called with empty results list.")
        # Create an empty file anyway? Or return None? Let's return None.
        # return None
        # Let's create empty file for consistency
        results_list = []

    logger.info(f"generate_jsonl_download received results_list: type={type(results_list)}, len={len(results_list)}")
    try:
        jsonl_content = io.StringIO()
        for result in results_list:
            if isinstance(result, dict): jsonl_content.write(json.dumps(result) + '\n')
            else: logger.warning(f"Skipping non-dict item in results: {type(result)}")

        jsonl_string = jsonl_content.getvalue()
        jsonl_content.close()
        logger.info(f"Generated JSONL string length: {len(jsonl_string)}")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        temp_dir = tempfile.gettempdir()
        # Use a more descriptive name
        file_path = os.path.join(temp_dir, f"nway_pairwise_results_{timestamp}.jsonl")

        with open(file_path, "w", encoding="utf-8") as f: f.write(jsonl_string)
        logger.info(f"Generated JSONL file for download at: {file_path}")
        return gr.File(value=file_path, label="Download Results (JSONL)")

    except Exception as e:
        logger.error(f"Error generating JSONL file: {e}", exc_info=True)
        raise gr.Error(f"Failed to generate JSONL download: {e}")


def _generate_download_wrapper(results_state, *args):
    """Wrapper to call generate_jsonl_download. (Unchanged)"""
    logger.info(f"Download wrapper called. results_state type: {type(results_state)}, len: {len(results_state) if isinstance(results_state, list) else 'N/A'}. Ignoring {len(args)} extra args.")
    # results_state should contain the list of raw evaluation dicts (pairwise)
    return generate_jsonl_download(results_state)


# --- Stop Request Handling (Unchanged) ---
def request_stop():
    """Sets the global STOP_REQUESTED flag."""
    global STOP_REQUESTED
    status_message = ""
    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        logger.warning("Stop requested via UI button.")
        status_message = "Stop request received. Finishing current batch..."
    else:
        logger.warning("Stop already requested.")
        status_message = "Stop already requested. Please wait..."
    # No need to return status message if output isn't used
    # return status_message

# --- UI Creation ---

MAX_MODELS = 4 # Define max models for the fixed UI

def create_model_config_ui(index: int, defaults: Dict = None):
    """Helper to create a UI block for one model config."""
    defaults = defaults or {}
    prefix = f"model_{index+1}"
    with gr.Group(elem_classes="model-config-group"):
        gr.Label(f"Model {index+1} Configuration")
        # Checkbox to enable/disable this model
        is_active = gr.Checkbox(label="Active", value=defaults.get('is_active', index < 2), elem_id=f"{prefix}_active") # Default first 2 active
        name = gr.Textbox(label="Display Name", value=defaults.get('name', f"Model {index+1}"), elem_id=f"{prefix}_name")
        api_url = gr.Textbox(label="API URL", value=defaults.get('api_url', ""), elem_id=f"{prefix}_api_url")
        model_id = gr.Textbox(label="Model ID", value=defaults.get('model_id', ""), elem_id=f"{prefix}_model_id")
        temp = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.1, value=defaults.get('temperature', 0.1), elem_id=f"{prefix}_temp")
        max_tokens = gr.Number(label="Max Tokens", value=defaults.get('max_tokens', 1024), precision=0, elem_id=f"{prefix}_max_tokens")
        file_upload_method = gr.Dropdown(
            label="File Upload Method", choices=["JSON (Embedded Data)", "Multipart Form Data"],
            value=defaults.get('file_upload_method', "JSON (Embedded Data)"),
            info="How to send file data.", elem_id=f"{prefix}_upload"
        )
        # Return handles to the components for use in callbacks/state
        return is_active, name, api_url, model_id, temp, max_tokens, file_upload_method

def create_ui():
    """Creates the Gradio web interface for the N-Way testing tool."""
    logger.info("Creating Gradio UI for N-Way Tester...")

    default_judge_prompt = LMJudge.DEFAULT_EVALUATION_PROMPT
    css = """.model-config-group .gr-form { background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
.model-config-group .gr-form > :first-child { font-weight: bold; margin-bottom: 5px; } /* Style the label */
.results-box { border: 1px solid #ccc; padding: 15px; border-radius: 5px; margin-top: 15px; }
.api-key-warning { color: #cc5500; font-weight: bold; margin-bottom: 15px; }
"""

    # Default configurations (Example)
    default_configs = [
        {"name": "MiniCPM-o-2.6-Q4_K_M", "api_url": "http://localhost:8001/v1/chat/completions", "model_id": "openbmb/MiniCPM-o-2_6-gguf", "temperature": 0.1, "max_tokens": 8192, "is_active": True},
        {"name": "Model 2 (LM Studio Gemma 3 4B)", "api_url": "http://localhost:1234/v1/chat/completions", "model_id": "gemma-3-4b-it", "temperature": 0.1, "max_tokens": 8192, "is_active": True},
        {"name": "Model 3 (LM Studio Gemma 3 27B QAT)", "api_url": "http://localhost:1234/v1/chat/completions", "model_id": "gemma-3-27b-it-qat", "temperature": 0.1, "max_tokens": 8192, "is_active": False},
        {"name": "Model 4 (LM Studio Gemma 3 4B QAT)", "api_url": "http://localhost:1234/v1/chat/completions", "model_id": "gemma-3-4b-it-qat", "temperature": 0.1, "max_tokens": 8192, "is_active": False},
    ]


    with gr.Blocks(css=css, theme=gr.themes.Soft()) as iface:
        gr.Markdown("# N-Endpoint Pairwise AI Testing & Auto-Evaluation")
        gr.Markdown(
            "1) Configure up to N Models & Judge.\n"
            "2) Provide Test Data & Reference Input.\n"
            "3) Run Pairwise Evaluations & Compare Performance."
        )
        # API Key and Multimodal Input warnings (same as before)
        gr.Markdown("""**API Key**: Provide below if needed for cloud endpoints. Overrides environment variables.""", elem_classes="api-key-warning")
        gr.Markdown("""**Multimodal Input**: Ensure test data has path/URL field, specify field name, ensure models support it, and prompt accordingly.""", elem_classes="api-key-warning")

        # State to hold model configurations (list of dicts)
        # Initialize state with default configurations
        model_configs_state = gr.State(default_configs[:MAX_MODELS])

        with gr.Tabs():
            with gr.TabItem("Configuration"):
                with gr.Row():
                    api_key_input = gr.Textbox(label="API Key (Optional)", type="password", placeholder="Enter key if needed")

                gr.Markdown("### Model Configurations (Activate models to include in the test)")
                # Create UI blocks for MAX_MODELS
                model_ui_inputs = [] # Collect all input components
                with gr.Row():
                    for i in range(MAX_MODELS):
                        with gr.Column(scale=1):
                            components = create_model_config_ui(i, default_configs[i] if i < len(default_configs) else {})
                            model_ui_inputs.extend(list(components)) # Add tuple of components

                gr.Markdown("### Judge Configuration")
                with gr.Row():
                    # Use similar layout as models for consistency
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="model-config-group"):
                            gr.Label("Judge Model")
                            judge_name = gr.Textbox(label="Display Name", value="Judge")
                            judge_api_url = gr.Textbox(label="API URL", value="http://localhost:1234/v1/chat/completions")
                            judge_model_id = gr.Textbox(label="Model ID", value="gemma-3-27b-it-qat")
                            judge_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.0)
                            judge_max_tokens = gr.Number(label="Max Tokens", value=8192, precision=0)
                            judge_file_upload_method = gr.Dropdown(
                                label="File Upload Method", choices=["JSON (Embedded Data)", "Multipart Form Data"],
                                value="JSON (Embedded Data)", info="How to send file data."
                            )
                            # Judge doesn't need is_active checkbox, it's always active if provided
                            judge_is_active = gr.Checkbox(value=True, visible=False) # Hidden, always true


                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Model Prompt Template")
                        model_prompt_template_input = gr.Textbox(label="Template for Models (use {key})", value="{key}\nUser: Provide a detailed description\nAssistant:", lines=5, show_copy_button=True)
                    with gr.Column(scale=1):
                        gr.Markdown("### Judge Prompt Template")
                        judge_prompt_template_input = gr.Textbox(label="Template for Judge (use {model_a_name}, {model_b_name}, etc.)", value=default_judge_prompt, lines=15, show_copy_button=True)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Test Data")
                        gr.Markdown("Upload CSV/JSON/JSONL or paste data. Specify Key, Value (optional), and Image (optional) field names.")
                        test_data_file = gr.File(label="Upload Test Data", file_types=[".csv", ".json", ".jsonl", ".ndjson"])
                        test_data_text = gr.Textbox(label="Or Paste Test Data (JSON list or JSONL)", lines=8)
                        with gr.Row():
                            key_field_name_input = gr.Textbox(label="Key Field", value="name")
                            value_field_name_input = gr.Textbox(label="Value Field", value="caption")
                            image_field_name_input = gr.Textbox(label="Image Field", value="image_url")
                    with gr.Column(scale=1):
                        gr.Markdown("### Test Parameters")
                        batch_size_input = gr.Number(label="Batch Size", value=5, minimum=1, precision=0)
                        gr.Markdown("#### Batch Retry Settings")
                        batch_retry_attempts_input = gr.Number(label="Batch Retry Attempts", value=1, precision=0, minimum=0)
                        batch_backoff_factor_input = gr.Slider(label="Backoff Factor", value=2.0, minimum=1.0, maximum=5.0, step=0.1)
                        batch_max_wait_input = gr.Number(label="Maximum Wait Time (s)", value=60, precision=0, minimum=1)
                        batch_retry_trigger_strings_input = gr.Textbox(label="Retry Trigger Strings (Comma-separated)", placeholder="e.g., rate limit,error,timeout")
                with gr.Column(scale=1):
                    gr.Markdown("### Post-Processing")
                    with gr.Accordion("N-gram Analysis", open=False):
                        enable_ngram_input = gr.Checkbox(label="Enable N-gram Frequency Analysis", value=True)
                        n_val_input = gr.Slider(label="N-gram Size (n)", minimum=1, maximum=7, value=3, step=1)
                        top_k_input = gr.Number(label="Top K N-grams to Report", value=10, precision=0)

            with gr.TabItem("Monitoring & Results"):
                gr.Markdown("### Test Execution & Results")
                with gr.Row():
                    run_button = gr.Button("Run N-Way Test", variant="primary", scale=3)
                    stop_button = gr.Button("Stop Test", variant="stop", scale=1)
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="results-box"):
                            gr.Markdown("#### Last Processed Case")
                            last_case_id_display = gr.Textbox(label="Case ID", interactive=False) # Added
                            last_image_display = gr.Image(label="Image", type="filepath", interactive=False, height=150)
                            last_avg_model_latency_display = gr.Textbox(label="Avg Model Latency (s)", interactive=False)
                            last_avg_judge_latency_pairs_display = gr.Textbox(label="Avg Judge Latency/Pair (s)", interactive=False)
                            # Removed single winner output

                    with gr.Column(scale=2):
                        with gr.Group(elem_classes="results-box"):
                            gr.Markdown("#### Download Results")
                            results_state = gr.State([]) # State for raw pairwise results dicts
                            download_button = gr.DownloadButton(label="Download Pairwise Evals (JSONL)", value=None)

                        with gr.Group(elem_classes="results-box"):
                            gr.Markdown("#### Overall Results")
                            summary_output = gr.Textbox(label="Summary", lines=15, interactive=False) # Increased lines
                            detailed_evaluations_output = gr.DataFrame(label="Individual Pairwise Results", interactive=False, wrap=True) # Enable wrapping


        # --- Define UI Interactions ---

        # Function to gather model configs from UI components into the state dict list
        def update_model_configs_state(*ui_inputs):
            num_models = MAX_MODELS
            inputs_per_model = 7 # is_active, name, url, id, temp, tokens, upload
            configs = []
            for i in range(num_models):
                start_idx = i * inputs_per_model
                model_inputs = ui_inputs[start_idx : start_idx + inputs_per_model]
                config_dict = {
                    "is_active": model_inputs[0],
                    "name": model_inputs[1],
                    "api_url": model_inputs[2],
                    "model_id": model_inputs[3],
                    "temperature": model_inputs[4],
                    "max_tokens": model_inputs[5],
                    "file_upload_method": model_inputs[6],
                }
                # Only add if name is present, avoids adding empty placeholders
                if config_dict["name"] and config_dict["name"].strip():
                    configs.append(config_dict)
                else: # Add an empty dict as placeholder if no name
                    configs.append({}) # Important to keep state length consistent with UI elements if needed later
            # Filter out empty placeholder dicts before returning the final state
            active_configs = [cfg for cfg in configs if cfg.get("name")]
            logger.info(f"Updating model_configs_state with {len(active_configs)} configurations.")
            return active_configs # Return the list of dicts for the state

        # Trigger state update whenever any model config input changes
        config_inputs_to_watch = model_ui_inputs # List of all model input components
        # Add listeners to update the state
        # This might cause performance issues if many models. Consider updating state only on Run.
        # For now, update on change:
        # for input_comp in config_inputs_to_watch:
        # input_comp.change(
        # fn=update_model_configs_state,
        # inputs=config_inputs_to_watch,
        # outputs=model_configs_state
        # )
        # ---> Decision: Update state *just before* running the test for simplicity/performance.

        # Define the run event listener
        run_event = run_button.click(
            # First, update the state based on current UI values
            fn=update_model_configs_state,
            inputs=config_inputs_to_watch,
            outputs=model_configs_state
        ).then(
            # Then, call the main test function using the updated state
            fn=run_test_from_ui,
            inputs=[ # Pass state and other UI inputs
                model_configs_state, # The updated state
                # Judge inputs
                judge_name, judge_api_url, judge_model_id, judge_temp, judge_max_tokens, judge_file_upload_method, judge_is_active,
                # Other inputs
                api_key_input, model_prompt_template_input, judge_prompt_template_input,
                test_data_file, test_data_text,
                batch_size_input, batch_retry_attempts_input, batch_backoff_factor_input, batch_max_wait_input, batch_retry_trigger_strings_input,
                enable_ngram_input, n_val_input, top_k_input, # New n-gram parameters
                key_field_name_input, value_field_name_input, image_field_name_input
            ],
            outputs=[
                # Map the 8 yielded values from run_test_from_ui
                summary_output, # 1. Summary text (interim/final)
                detailed_evaluations_output, # 2. Details DataFrame (interim/final)
                last_image_display, # 3. Last image
                last_avg_model_latency_display, # 4. Last avg model latency
                last_avg_judge_latency_pairs_display,# 5. Last avg judge latency/pair
                last_case_id_display, # 6. Last Case ID (using placeholder 6)
                gr.Textbox(visible=False), # 7. Placeholder 7 (unused)
                results_state # 8. Hidden state for raw pairwise results
            ]
        )

        # Trigger download file generation *after* the run completes
        run_event.then(
            fn=_generate_download_wrapper,
            inputs=[results_state],
            outputs=[download_button]
        )

        # Stop button interaction
        stop_event = stop_button.click(
            fn=request_stop,
            inputs=None, outputs=None,
            cancels=[run_event] # Cancel the main run
        )

        return iface


def run_cli_test():
    """Runs the N-way test from the command line with 3 models."""
    logger.info("Starting CLI execution of N-Way ModelTester...")
    print("\n   MANUAL SETUP EXAMPLE (for MiniCPM-o-2.6):")
    print("   1. Download the model (approx. 1.7 GB):")
    print("      wget https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf/resolve/main/minicpm-o-2_6-q4_k_m.gguf")
    print("\n   2. Run the server in a separate terminal:")
    print("      python -m llama_cpp.server --model ./minicpm-o-2_6-q4_k_m.gguf --port 8001 --n_gpu_layers -1")

    # --- Configuration ---
    try: from dotenv import load_dotenv; load_dotenv(); logger.info("Loaded .env if present.")
    except ImportError: logger.warning("python-dotenv not installed, cannot load .env file.")

    API_KEY = os.getenv("OPENROUTER_API_KEY") # Example key needed for cloud model
    OLLAMA_API_URL_GEN = "http://localhost:11434/api/generate"
    OLLAMA_API_URL_CHAT = "http://localhost:11434/v1/chat/completions" # If using Ollama OpenAI endpoint
    LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions" # Example LM Studio

    # Define 3 Model Endpoints
    endpoints = [
        ModelEndpoint(
            name="MiniCPM-o-2.6-Q4_K_M",
            api_url="http://localhost:8001/v1/chat/completions",
            api_key=None,
            model_id="openbmb/MiniCPM-o-2_6-gguf",
            temperature=0.1,
            max_tokens=1000,
            is_active=True
        ),
        ModelEndpoint(
            name="M2_LMStudio_Gemma4B", api_url=LMSTUDIO_API_URL, api_key=None, model_id="gemma-3-4b-it", temperature=0.1, max_tokens=1000, is_active=True
        ),
        # ModelEndpoint(
        # name="M3_Ollama_Mistral", api_url=OLLAMA_API_URL_GEN, api_key=None, model_id="mistral:latest", temperature=0.1, max_tokens=1000, is_active=True
        # ),
        # Optional 4th model (cloud, requires key)
        ModelEndpoint(
            name="M4_OpenRouter_4oMini", api_url="https://openrouter.ai/api/v1/chat/completions", api_key=API_KEY,
            model_id="openai/gpt-4o-mini", temperature=0.1, max_tokens=1000, is_active=bool(API_KEY) # Only active if key exists
        ),
    ]

    # Judge Model (using LM Studio Gemma 27B)
    judge_model = ModelEndpoint(
        name="Judge_LMStudio_Gemma27B", api_url=LMSTUDIO_API_URL, api_key=None, model_id="gemma-3-27b-it", temperature=0.0, max_tokens=2048, is_active=True
    )

    active_endpoints = [ep for ep in endpoints if ep.is_active]
    if len(active_endpoints) < 2:
        print("Error: Need at least 2 active models with valid configuration for CLI test.")
        # Optionally check judge model validity too
        return

    logger.info(f"CLI Test: Using {len(active_endpoints)} active models: {[ep.name for ep in active_endpoints]}")
    logger.info(f"CLI Test: Using Judge: {judge_model.name}")


    model_prompt = "User: {key}\nAssistant:"

    # Sample Test Cases (same as before)
    dummy_image_path = "dummy_test_image.png"
    if not os.path.exists(dummy_image_path):
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (100, 50), color = (73, 109, 137)); d = ImageDraw.Draw(img); d.text((10,10), "Test Img", fill=(255,255,0)); img.save(dummy_image_path)
            logger.info(f"Created dummy image file: {dummy_image_path}")
        except Exception as e: logger.error(f"Failed to create dummy image: {e}"); dummy_image_path = None

    test_cases = [
        TestCase(id="q1", key="What is the capital of France?", value="Paris", image_path_or_url=None),
        TestCase(id="q2", key="Summarize the plot of 'Inception'.", value="Dream heist.", image_path_or_url=None),
        TestCase(id="q3", key="Write a short poem about a rainy day.", value="", image_path_or_url=None),
    ]
    if dummy_image_path:
        test_cases.append(TestCase(id="img1", key=f"Describe this image.", value="Blue rectangle with yellow text", image_path_or_url=dummy_image_path))

    # --- Execution ---
    try:
        tester = ModelTester(
            endpoints=active_endpoints,
            judge_endpoint=judge_model,
            model_prompt_template=model_prompt,
            judge_prompt_template=LMJudge.DEFAULT_EVALUATION_PROMPT
        )

        logger.info(f"Running CLI N-way test with {len(test_cases)} test cases...")
        results_generator = tester.run_test(
            test_cases, batch_size=2,
            batch_retry_attempts=1, batch_retry_trigger_strings=["rate limit", "error"]
        )

        final_results_dict = None
        for res in results_generator: final_results_dict = res # Get the last yielded item

        if final_results_dict is None or final_results_dict.get("type") != "final":
            logger.error("Test run generator did not yield a final result structure.")
            print("\nError: Test run did not complete successfully.")
            return

        # --- Output Results ---
        logger.info("N-Way test completed. Final Results:")
        summary_output = format_summary_output(final_results_dict.get("summary", {}))
        print("\n" + summary_output)

        # Save full results (list of pairwise dicts)
        results_filename = f"cli_nway_results_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        try:
            raw_evals = final_results_dict.get("evaluations", [])
            if raw_evals:
                with open(results_filename, 'w', encoding='utf-8') as f:
                    for eval_dict in raw_evals:
                        f.write(json.dumps(eval_dict) + '\n')
                logger.info(f"Full pairwise results saved to {results_filename}")
                print(f"\nFull pairwise results saved to: {results_filename}")
            else:
                logger.info("No pairwise evaluations to save.")
                print("\nNo pairwise evaluations to save.")
        except Exception as e:
            logger.error(f"Failed to save results to JSONL: {e}")
            print(f"\nWarning: Could not save full results to JSONL: {e}")

    except Exception as e:
        logger.exception("An error occurred during the CLI execution.")
        print(f"\nAn error occurred during CLI execution: {e}")
    finally:
        if dummy_image_path and os.path.exists(dummy_image_path):
            try: os.remove(dummy_image_path); logger.info("Removed dummy image file.")
            except Exception as e: logger.warning(f"Could not remove dummy image file: {e}")

# ==============================================================================
# Main Execution Logic
# ==============================================================================


def main():
    """Main function to parse arguments and run either CLI or UI."""
    import argparse
    parser = argparse.ArgumentParser(description="N-Way Pairwise Model Testing Tool")
    parser.add_argument("--ui", action="store_true", help="Launch the Gradio web UI.")
    args = parser.parse_args()

    if args.ui:
        logger.info("Launching Gradio UI...")
        iface = create_ui()
        if iface: iface.launch(share=True) # Add share=True for public link if needed
        else: logger.error("Failed to create Gradio UI."); print("Error: Could not create UI.")
    else:
        def signal_handler(sig, frame):
            global STOP_REQUESTED
            if not STOP_REQUESTED: print("\nCtrl+C detected. Requesting stop..."); logger.warning("Stop requested via Ctrl+C."); STOP_REQUESTED = True
            else: print("\nCtrl+C detected again. Forcing exit."); logger.error("Forced exit via second Ctrl+C."); sys.exit(1)
        signal.signal(signal.SIGINT, signal_handler)
        run_cli_test()

if __name__ == "__main__":
    # This block runs first. It ensures the environment is set up and all
    # dependencies are available before attempting to run the main application.
    if ensure_environment_with_uv():
        main()
def main():
    """Main function to parse arguments and run either CLI or UI."""
    parser = argparse.ArgumentParser(description="N-Way Pairwise Model Testing Tool")
    parser.add_argument("--ui", action="store_true", help="Launch the Gradio web UI.")
    args = parser.parse_args()

    def signal_handler(sig, frame):
        global STOP_REQUESTED
        if not STOP_REQUESTED:
            print("\nCtrl+C detected. Requesting stop...")
            logger.warning("Stop requested via Ctrl+C.")
            STOP_REQUESTED = True
        else:
            print("\nCtrl+C detected again. Forcing exit.")
            sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)

    if args.ui:
        logger.info("Launching Gradio UI...")
        iface = create_ui()
        iface.launch()
    else:
        run_cli_test()

if __name__ == "__main__":
    # This block runs first. It ensures the environment is set up and all
    # dependencies are available before attempting to run the main application.
    if ensure_environment_with_uv():
        main()
