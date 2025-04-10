
import sys
import os
import venv
import subprocess
import signal # Added for CLI stop handling

# --- Venv Setup ---
# Determine if we need to set up or reactivate the virtual environment

VENV_DIR = ".venv"
REQUIRED_PACKAGES = [
    "gradio",
    "pandas",
    "requests",
    "tenacity",
    "Pillow", # For image handling (needed for dummy image in CLI test)
    "python-dotenv", # Often useful, though not strictly required by current code
]

def ensure_venv():
    """Checks for venv, creates/installs if needed, and re-executes if not active."""
    venv_path = os.path.abspath(VENV_DIR)
    # Check if the current Python executable is from the target venv
    is_in_venv = sys.prefix == venv_path
    venv_exists = os.path.isdir(venv_path)

    if is_in_venv:
        # Already running in the correct venv, proceed
        print(f"Running inside the '{VENV_DIR}' virtual environment.")
        return True # Indicate we are ready to proceed

    print(f"Not running inside the target '{VENV_DIR}' virtual environment.")

    if not venv_exists:
        print(f"Creating virtual environment in '{venv_path}'...")
        try:
            venv.create(venv_path, with_pip=True)
            print("Virtual environment created successfully.")
        except Exception as e:
            print(f"Error creating virtual environment: {e}", file=sys.stderr)
            sys.exit(1) # Exit if creation fails

    # Determine the Python executable path within the venv
    if sys.platform == "win32":
        python_executable = os.path.join(venv_path, "Scripts", "python.exe")
        pip_executable = os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        python_executable = os.path.join(venv_path, "bin", "python")
        pip_executable = os.path.join(venv_path, "bin", "pip")

    if not os.path.exists(python_executable):
        print(f"Error: Python executable not found at '{python_executable}'. Venv creation might have failed.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(pip_executable):
        print(f"Error: Pip executable not found at '{pip_executable}'. Venv creation might have failed.", file=sys.stderr)
        sys.exit(1)


    # Install requirements into the venv using pip from the venv
    print(f"Installing/checking required packages in '{venv_path}'...")
    install_command = [pip_executable, "install"] + REQUIRED_PACKAGES
    try:
        # Run pip install, capture output for clarity/debugging
        result = subprocess.run(install_command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("Packages installed/verified successfully.")
        # print(result.stdout) # Uncomment to see pip output
        if result.stderr:
            # Show pip's stderr for warnings etc.
            print("--- pip stderr ---\n", result.stderr, "\n--- end pip stderr ---")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages using command: {' '.join(e.cmd)}", file=sys.stderr)
        print(f"Pip stdout:\n{e.stdout}", file=sys.stderr)
        print(f"Pip stderr:\n{e.stderr}", file=sys.stderr)
        sys.exit(1) # Exit if installation fails
    except Exception as e:
        print(f"An unexpected error occurred during package installation: {e}", file=sys.stderr)
        sys.exit(1)


    # Re-execute the script using the venv's Python interpreter
    print(f"\nRestarting script using Python from '{venv_path}'...\n{'='*20}\n")
    script_path = os.path.abspath(sys.argv[0])
    # os.execv replaces the current process, inheriting stdio etc.
    # Arguments must include the executable name as argv[0] for the new process
    try:
        os.execv(python_executable, [python_executable, script_path] + sys.argv[1:])
        # If execv is successful, this line is never reached
    except OSError as e:
        print(f"Error restarting script with '{python_executable}': {e}", file=sys.stderr)
        # Fallback attempt with subprocess if execv fails (less ideal)
        print("Attempting restart with subprocess as fallback...")
        try:
            subprocess.check_call([python_executable, script_path] + sys.argv[1:])
            sys.exit(0) # Exit cleanly if subprocess worked
        except Exception as sub_e:
            print(f"Subprocess restart also failed: {sub_e}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during script restart: {e}", file=sys.stderr)
        sys.exit(1)

    # This should not be reached if re-execution happens
    return False # Indicate re-execution was attempted

# --- Original Script Imports (ensure they are accessible after venv check) ---
# It's generally okay to keep imports here, as the script restarts if not in venv
import gradio as gr
import json
import logging
import time
import pandas as pd
# import os # Already imported above
import re
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from tenacity import retry, stop_after_attempt, wait_exponential
import csv
import io
from urllib.parse import urlparse
import base64
import mimetypes
# import signal # Already imported above

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
    api_key: Optional[str] # API key can be optional (e.g., for local Ollama)
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
    image_path_or_url: Optional[str] = None # Path or URL to an image for multimodal input
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

# Global flag to signal stopping the test run
STOP_REQUESTED = False

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
class ModelRunner:
    """Handles model API calls."""

    def __init__(self, endpoint: ModelEndpoint, prompt_template: str):
        self.endpoint = endpoint
        self.prompt_template = prompt_template

    def _load_and_encode_image(self, image_path_or_url: str) -> Tuple[Optional[str], Optional[str]]:
        """Loads image from path/URL and returns (base64_string, mime_type) or (None, None)."""
        try:
            image_bytes = None
            if urlparse(image_path_or_url).scheme in ['http', 'https']:
                logger.info(f"Downloading image from URL: {image_path_or_url}")
                response = requests.get(image_path_or_url, stream=True, timeout=20) # Increased timeout
                response.raise_for_status()
                image_bytes = response.content
                logger.info(f"Successfully downloaded {len(image_bytes)} bytes from URL.")
                # Try to get mime type from headers first
                mime_type = response.headers.get('content-type')
            else:
                logger.info(f"Reading image from local path: {image_path_or_url}")
                if not os.path.exists(image_path_or_url):
                    raise FileNotFoundError(f"Image file not found at path: {image_path_or_url}")
                with open(image_path_or_url, "rb") as f:
                    image_bytes = f.read()
                mime_type, _ = mimetypes.guess_type(image_path_or_url)

            if not image_bytes:
                 raise ValueError("Failed to load image bytes.")

            mime_type = mime_type or 'image/jpeg' # Default if guess fails or not available
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            logger.info(f"Successfully encoded image to base64. Mime type: {mime_type}")
            logger.info(f"Successfully loaded and encoded image from {image_path_or_url[:50]}... Type: {mime_type}, Size: {len(base64_image)} chars base64")
            return base64_image, mime_type
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path_or_url}")
            return None, None
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image from URL {image_path_or_url}: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Failed to load or encode image {image_path_or_url}: {e}")
            return None, None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate(self, test_case: TestCase) -> ModelResponse:
        """Call the model API with the test case, potentially including an image."""
        start_time = time.time()
        base64_image = None
        mime_type = None

        logger.info(f"Inside generate for model '{self.endpoint.name}', test_id: {test_case.id}. Image path/URL from test case: {test_case.image_path_or_url}")
        try:
            # Preprocess the input key using the global settings
            preprocessed_key = preprocess_text(test_case.key)

            # Format prompt using the preprocessed key
            prompt = ""
            try:
                # For judge prompts, the "key" is already the full prompt
                if test_case.id and test_case.id.startswith("judge"):
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

            # --- Multimodal Input Handling ---
            if test_case.image_path_or_url:
                logger.info(f"Test case {test_case.id} includes image: {test_case.image_path_or_url}")
                base64_image, mime_type = self._load_and_encode_image(test_case.image_path_or_url)
                logger.info(f"Result from _load_and_encode_image - Has image: {bool(base64_image)}, Mime type: {mime_type}")
                if base64_image is None:
                    # Handle image loading failure - return an error response
                    logger.error(f"Failed to load image for test case {test_case.id}. Returning error.")
                    return ModelResponse(test_id=test_case.id or "unknown", model_name=self.endpoint.name, output=f"Error: Failed to load image {test_case.image_path_or_url}", latency=time.time() - start_time)

            # Determine API type and make appropriate call
            response_text = ""
            api_url_lower = self.endpoint.api_url.lower() if self.endpoint.api_url else ""

            try:
                # Add more specific checks based on common API structures
                is_openai_compatible = "/v1/chat/completions" in api_url_lower or \
                                        "openai" in api_url_lower or \
                                        "openrouter.ai" in api_url_lower or \
                                        "mistral" in api_url_lower or \
                                        "together.ai" in api_url_lower or \
                                        "groq.com" in api_url_lower or \
                                        "fireworks.ai" in api_url_lower or \
                                        "deepinfra.com" in api_url_lower or \
                                        "lmstudio.ai" in api_url_lower or \
                                        ":1234/v1" in api_url_lower # Common LM Studio port

                is_anthropic_compatible = "/v1/messages" in api_url_lower or "anthropic" in api_url_lower
                is_gemini = "generativelanguage.googleapis.com" in api_url_lower
                # Check for local Ollama or URLs containing 'ollama' with the generate path
                is_ollama = ("/api/generate" in api_url_lower and \
                             ("localhost:11434" in api_url_lower or "127.0.0.1:11434" in api_url_lower)) or \
                            ("ollama" in api_url_lower and "/api/generate" in api_url_lower)

                if is_openai_compatible:
                    response_text = self._call_openai_compatible_api(prompt, base64_image, mime_type)
                elif is_anthropic_compatible:
                    response_text = self._call_anthropic_api(prompt, base64_image, mime_type)
                elif is_gemini:
                     response_text = self._call_gemini_api(prompt, base64_image, mime_type)
                elif is_ollama:
                    # Ollama's generate endpoint currently only supports images via 'images' field
                    response_text = self._call_ollama_api(prompt, base64_image) # Ollama doesn't need mime type separately in payload
                else:
                    # Fallback to generic or attempt intelligent guess (assume text-only for fallback)
                    if base64_image:
                         logger.warning(f"Could not determine API type for {self.endpoint.api_url} with multimodal input. Attempting generic text-only call.")
                         response_text = self._call_generic_api(prompt) # Fallback without image
                    else:
                         logger.warning(f"Could not determine API type for {self.endpoint.api_url}. Attempting generic call.")
                         response_text = self._call_generic_api(prompt)

            except requests.exceptions.RequestException as req_err:
                 logger.error(f"API request failed for {self.endpoint.name}: {req_err}")
                 if hasattr(req_err, 'response') and req_err.response is not None:
                     logger.error(f"Response status: {req_err.response.status_code}, Response text: {req_err.response.text[:500]}")
                 response_text = f"Error: API request failed. Details: {str(req_err)}"
            except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as parse_err:
                 logger.error(f"Failed to parse response or invalid response structure from {self.endpoint.name}: {parse_err}")
                 response_text = f"Error: Failed to parse API response. Details: {str(parse_err)}"
            except Exception as e:
                logger.error(f"Unexpected error calling API for {self.endpoint.name}: {str(e)}", exc_info=True)
                response_text = f"Error: An unexpected error occurred. Details: {str(e)}"

            end_time = time.time()

            return ModelResponse(
                test_id=test_case.id or "unknown", # Ensure test_id is never None
                model_name=self.endpoint.name,
                output=str(response_text), # Ensure output is always string
                latency=end_time - start_time,
            )
        except Exception as e:
            logger.error(f"Unexpected error in generate method for {self.endpoint.name}: {str(e)}", exc_info=True)
            # Re-raise to trigger tenacity retry
            raise

    def _prepare_headers(self):
        """Prepares common headers, including Authorization if API key exists."""
        headers = {"Content-Type": "application/json"}
        # Only add Authorization header if api_key is present and not empty
        if self.endpoint.api_key and self.endpoint.api_key.strip():
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"

        # Add OpenRouter specific headers if applicable
        if self.endpoint.api_url and "openrouter.ai" in self.endpoint.api_url.lower():
             # These might be optional now, but good practice
             headers["HTTP-Referer"] = "http://localhost" # Can be anything, localhost is common
             headers["X-Title"] = "Model A/B Testing Tool"
        return headers

    def _call_openai_compatible_api(self, prompt: str, image_base64: Optional[str] = None, mime_type: Optional[str] = None) -> str:
        """Calls APIs following the OpenAI chat completions format (including multimodal)."""
        logger.info(f"Calling OpenAI-compatible API: {self.endpoint.api_url} for model {self.endpoint.model_id}")
        headers = self._prepare_headers()

        # Construct messages payload - handling multimodal input
        messages = []
        logger.info(f"Checking for image data before constructing payload. Has image: {bool(image_base64)}, Mime type: {mime_type}")
        if image_base64:
            logger.info("Constructing OpenAI multimodal payload.")
            logger.info("Image data found. Proceeding with multimodal payload construction.") # Indentation should be correct relative to if block
            # Ensure mime_type is set, default if necessary
            mime_type = mime_type or 'image/jpeg'
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
                    }
                ]
            })
        else:
            # Text-only payload
            logger.info("Constructing OpenAI text-only payload.")
            logger.info("No image data found or image loading failed. Proceeding with text-only payload construction.") # Indentation should be correct relative to else block
            messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.endpoint.model_id,
            "messages": messages,
            "max_tokens": self.endpoint.max_tokens,
            "temperature": self.endpoint.temperature,
        }
        try:
            # Log payload size for debugging potential issues
            payload_size_kb = len(json.dumps(data)) / 1024
            logger.info(f"Sending OpenAI-compatible request. Payload size: {payload_size_kb:.2f} KB")
            if payload_size_kb > 4000: # Log warning for potentially large payloads
                logger.warning(f"Payload size ({payload_size_kb:.2f} KB) is large, may exceed limits.")

            response = requests.post(self.endpoint.api_url, headers=headers, json=data, timeout=180) # Increased timeout for multimodal
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            result = response.json()
            if not result.get("choices"):
                 raise ValueError(f"Invalid response format: 'choices' key missing or empty. Response: {result}")
            if not result["choices"][0].get("message"):
                 raise ValueError(f"Invalid response format: 'message' key missing in first choice. Response: {result}")
            if result["choices"][0]["message"].get("content") is None:
                 logger.warning(f"Response content is null for model {self.endpoint.model_id}. Check finish reason: {result['choices'][0].get('finish_reason')}")
                 return f"Error: Response content was null (Finish Reason: {result['choices'][0].get('finish_reason')})"
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI-compatible request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None: logger.error(f"Response content: {e.response.text[:500]}")
            raise
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Failed to parse OpenAI-compatible response: {str(e)}")
            logger.error(f"Full response: {result if 'result' in locals() else 'Response not available'}")
            raise

    def _call_anthropic_api(self, prompt: str, image_base64: Optional[str] = None, mime_type: Optional[str] = None) -> str:
        """Calls the Anthropic messages API (including multimodal)."""
        logger.info(f"Calling Anthropic API: {self.endpoint.api_url} for model {self.endpoint.model_id}")
        headers = self._prepare_headers()
        # Anthropic requires API key via header, not Bearer token
        if "Authorization" in headers: del headers["Authorization"]
        if self.endpoint.api_key: headers["x-api-key"] = self.endpoint.api_key
        else: raise ValueError("Anthropic API key is required but not provided.")
        headers["anthropic-version"] = "2023-06-01" # Required header

        # Construct messages payload - handling multimodal input
        messages = []
        content = []
        content.append({"type": "text", "text": prompt})

        if image_base64:
            logger.info("Constructing Anthropic multimodal payload.")
            # Ensure mime_type is set, default if necessary
            mime_type = mime_type or 'image/jpeg'
            # Check common image types supported by Claude
            supported_mime_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
            if mime_type not in supported_mime_types:
                 logger.warning(f"MIME type '{mime_type}' may not be directly supported by Claude. Using it anyway.")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": image_base64
                }
            })
        else:
            logger.info("Constructing Anthropic text-only payload.")

        messages.append({
            "role": "user",
            "content": content
        })

        data = {
            "model": self.endpoint.model_id,
            "messages": messages,
            "max_tokens": self.endpoint.max_tokens,
            "temperature": self.endpoint.temperature,
        }
        try:
            # Log payload size
            payload_size_kb = len(json.dumps(data)) / 1024
            logger.info(f"Sending Anthropic request. Payload size: {payload_size_kb:.2f} KB")

            response = requests.post(self.endpoint.api_url, headers=headers, json=data, timeout=180) # Increased timeout
            response.raise_for_status()
            result = response.json()
            if not result.get("content"):
                 raise ValueError(f"Invalid response format: 'content' key missing or empty. Response: {result}")
            # Find the first text block in the response content array
            text_content = ""
            for block in result.get("content", []):
                 if block.get("type") == "text":
                      text_content = block.get("text", "")
                      break
            if not text_content and result.get("stop_reason") == "max_tokens":
                 logger.warning("Anthropic response had no text content and hit max_tokens.")
                 return "[Reached Max Tokens - No text content returned]"
            elif not text_content:
                 logger.warning(f"Anthropic response had no text block. Full response: {result}")
                 return "[No text content found in response]"
            return text_content
        except requests.exceptions.RequestException as e:
            logger.error(f"Anthropic request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None: logger.error(f"Response content: {e.response.text[:500]}")
            raise
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Failed to parse Anthropic response: {str(e)}")
            logger.error(f"Full response: {result if 'result' in locals() else 'Response not available'}")
            raise

    def _call_ollama_api(self, prompt: str, image_base64: Optional[str] = None) -> str:
        """Calls the Ollama generate API (including multimodal image support)."""
        logger.info(f"Calling Ollama API: {self.endpoint.api_url} for model {self.endpoint.model_id}")
        # Ollama doesn't use API keys or standard Bearer tokens via this endpoint
        headers = {"Content-Type": "application/json"}

        data = {
            "model": self.endpoint.model_id,
            "prompt": prompt,
            "stream": False, # Ensure we get the full response at once
            # "options": {
            #     "temperature": self.endpoint.temperature,
            #     # Add num_predict if max_tokens is set and > 0
            #     # Ollama might ignore it if the model doesn't support it well
            #      **({"num_predict": self.endpoint.max_tokens} if self.endpoint.max_tokens > 0 else {})
            # }
        }

        # Add image data if provided
        if image_base64:
            logger.info(f"Adding base64 image to Ollama request for model {self.endpoint.model_id}")
            # Ollama expects a list of base64 encoded images
            data["images"] = [image_base64]
            logger.info("Constructing Ollama multimodal payload.")
        else:
             logger.info("Constructing Ollama text-only payload.")

        try:
            # Log payload size
            payload_size_kb = len(json.dumps(data)) / 1024 # Approximate size
            logger.info(f"Sending Ollama request. Payload size: {payload_size_kb:.2f} KB")

            response = requests.post(self.endpoint.api_url, headers=headers, json=data, timeout=300) # Longer timeout for local/potentially slow models
            response.raise_for_status()
            result = response.json()

            # Check for standard response field
            if "response" in result:
                return result["response"]
            # Check for potential error field (Ollama might return error this way)
            elif "error" in result:
                logger.error(f"Ollama API returned an error: {result['error']}")
                raise ValueError(f"Ollama API Error: {result['error']}")
            else:
                 # Handle cases where the response might be different (e.g., streaming was accidentally left on)
                 # For non-streaming, the expected key is 'response'. If it's missing, it's an issue.
                 raise ValueError(f"Invalid response format: 'response' key missing and no 'error' key. Response: {result}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None: logger.error(f"Response content: {e.response.text[:500]}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse Ollama response or invalid response structure: {str(e)}")
            logger.error(f"Full response: {result if 'result' in locals() else 'Response not available'}")
            raise

    def _call_gemini_api(self, prompt: str, image_base64: Optional[str] = None, mime_type: Optional[str] = None) -> str:
        """Calls the Google Gemini API (including multimodal)."""
        logger.info(f"Calling Gemini API for model {self.endpoint.model_id}")
        if not self.endpoint.api_key:
             raise ValueError("Gemini API key is required but not provided.")
        # Gemini uses API key in the URL usually
        api_url = f"{self.endpoint.api_url}?key={self.endpoint.api_key}"
        headers = {"Content-Type": "application/json"}

        # Construct parts payload - handling multimodal input
        parts = []
        parts.append({"text": prompt})

        if image_base64:
            logger.info("Constructing Gemini multimodal payload.")
            # Ensure mime_type is set, default if necessary
            mime_type = mime_type or 'image/jpeg'
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_base64
                }
            })
        else:
             logger.info("Constructing Gemini text-only payload.")

        data = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": self.endpoint.temperature,
                "maxOutputTokens": self.endpoint.max_tokens,
                # Add safety settings if needed, e.g., "safetySettings": [{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}]
            }
        }
        try:
            # Log payload size
            payload_size_kb = len(json.dumps(data)) / 1024
            logger.info(f"Sending Gemini request. Payload size: {payload_size_kb:.2f} KB")

            response = requests.post(api_url, headers=headers, json=data, timeout=180) # Increased timeout
            response.raise_for_status()
            result = response.json()

            # Navigate the Gemini response structure carefully
            if not result.get("candidates"):
                # Check for promptFeedback for block reasons
                if result.get("promptFeedback", {}).get("blockReason"):
                    block_reason = result["promptFeedback"]["blockReason"]
                    logger.error(f"Gemini API blocked the prompt. Reason: {block_reason}")
                    return f"Error: Prompt blocked by API - {block_reason}"
                raise ValueError(f"Invalid response format: 'candidates' key missing or empty. Response: {result}")

            candidate = result["candidates"][0]
            if not candidate.get("content") or not candidate["content"].get("parts"):
                 # Check finishReason
                 finish_reason = candidate.get("finishReason")
                 if finish_reason and finish_reason != "STOP":
                     logger.error(f"Gemini generation stopped unexpectedly. Reason: {finish_reason}")
                     # Check safetyRatings if stopped for safety
                     safety_ratings = candidate.get("safetyRatings")
                     if safety_ratings:
                         logger.error(f"Safety Ratings: {safety_ratings}")
                     return f"Error: Generation stopped - {finish_reason}"
                 raise ValueError(f"Invalid response format: 'content' or 'parts' missing. Finish Reason: {finish_reason}. Response: {result}")

            # Find the text part in the response
            text_response = ""
            for part in candidate["content"]["parts"]:
                if "text" in part:
                    text_response += part["text"] # Concatenate if multiple text parts exist

            if not text_response and candidate.get("finishReason") != "STOP":
                 logger.warning(f"Gemini response had no text part. Finish Reason: {candidate.get('finishReason')}. Full response: {result}")
                 return f"[No text content found in response - Finish Reason: {candidate.get('finishReason')}]"

            return text_response

        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None: logger.error(f"Response content: {e.response.text[:500]}")
            raise
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Failed to parse Gemini response: {str(e)}")
            logger.error(f"Full response: {result if 'result' in locals() else 'Response not available'}")
            raise

    def _call_generic_api(self, prompt: str) -> str:
        """Attempts a generic POST request, assuming OpenAI-like structure as a guess (text-only)."""
        logger.warning(f"Attempting generic API call (assuming OpenAI text-only format) to: {self.endpoint.api_url}")
        try:
            # Try OpenAI text-only format as the most common fallback
            return self._call_openai_compatible_api(prompt) # Will call with image=None
        except Exception as e:
            logger.error(f"Generic API call failed: {str(e)}. Returning error message.")
            return f"Error: Failed to call or parse response from generic API endpoint {self.endpoint.api_url}. Please check configuration and API documentation. Details: {str(e)}"

class LMJudge:
    """Uses a language model to judge between champion and challenger outputs."""

    DEFAULT_EVALUATION_PROMPT = """
# Model Response Evaluation

You are evaluating two AI model responses based on the input query, potentially an accompanying image, and potentially a reference value.

## Input Query

{key}

{image_context_section}
{reference_section}

## Model A (Champion: {champion_name}) Response

{champion_output}

## Model B (Challenger: {challenger_name}) Response

{challenger_output}

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
        # Judge model is assumed to be text-only for evaluation.
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
        # Note: We don't pass the image to the judge, only the text inputs/outputs.
        preprocessed_key = preprocess_text(test_case.key)
        preprocessed_value = preprocess_text(test_case.value) # Preprocess reference value too
        preprocessed_champion = preprocess_text(champion_response.output)
        preprocessed_challenger = preprocess_text(challenger_response.output)

        # Prepare context for the evaluation prompt template
        has_reference = bool(preprocessed_value)
        reference_section_text = f"\n## Reference Value\n\n{preprocessed_value}\n" if has_reference else "\n## Reference Value\nN/A"
        reference_value_instruction_text = ' and Reference Value' if has_reference else ''
        reference_value_criteria_text = '2. Factual correctness compared to the Reference Value (if provided).' if has_reference else ''
        clarity_criteria_number_text = '3' if has_reference else '2'
        overall_criteria_number_text = '4' if has_reference else '3'

        # Add image context section if an image was provided in the original test case
        has_image = bool(test_case.image_path_or_url)
        image_context_section_text = "\n## Input Image\nAn image was provided with the input query. Consider it as context when evaluating the responses.\n" if has_image else ""

        # Format the evaluation prompt using the template and context
        try:
            evaluation_prompt = self.evaluation_prompt_template.format(
                key=preprocessed_key,
                image_context_section=image_context_section_text, # Added image context
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
        # Judge does not receive the image.
        judge_test_case = TestCase(
            key=evaluation_prompt,
            value="", # No value needed for judge call itself
            # Pass the original image path/URL to the judge if it exists
            image_path_or_url=test_case.image_path_or_url,
            id=f"judge-{test_case.id or 'unknown'}"
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
        logger.debug(f"Parsing judge response (first 100 chars): {response_text[:100]}")

        # 1. Extract VERDICT (Case-insensitive search for the explicit line)
        verdict_match = re.search(r"^\s*VERDICT:\s*(MODEL_A_WINS|MODEL_B_WINS|TIE)\s*$", response_text, re.IGNORECASE | re.MULTILINE)
        if verdict_match:
            verdict = verdict_match.group(1).upper()
            logger.info(f"Parsed VERDICT line: {verdict}")
        else:
            # Fallback: Look for bracketed verdicts (common LLM-as-judge pattern)
            bracket_match = re.search(r"\[\[\s*(MODEL_A_WINS|MODEL_B_WINS|TIE)\s*\]\]", response_text, re.IGNORECASE)
            if bracket_match:
                verdict = bracket_match.group(1).upper()
                logger.info(f"Parsed bracketed verdict: {verdict}")
            else:
                # Fallback: Look for simpler A/B/TIE in brackets
                simple_bracket_match = re.search(r"\[\[\s*([AB]|TIE)\s*\]\]", response_text, re.IGNORECASE)
                if simple_bracket_match:
                     verdict_text = simple_bracket_match.group(1).upper()
                     if verdict_text == "A": verdict = "MODEL_A_WINS"
                     elif verdict_text == "B": verdict = "MODEL_B_WINS"
                     else: verdict = "TIE"
                     logger.info(f"Parsed simple bracketed verdict: {verdict}")


        # 2. Extract CONFIDENCE (Case-insensitive search for the explicit line)
        confidence_match = re.search(r"^\s*CONFIDENCE:\s*(\d(?:\.\d)?)\s*/\s*5\s*$", response_text, re.IGNORECASE | re.MULTILINE)
        if confidence_match:
            try:
                confidence_score = float(confidence_match.group(1))
                # Clamp confidence between 1 and 5, then normalize to 0.2-1.0 range
                confidence = max(0.2, min(1.0, confidence_score / 5.0))
                logger.info(f"Parsed CONFIDENCE line: {confidence_score}/5 -> {confidence}")
            except ValueError:
                logger.warning(f"Could not parse CONFIDENCE value: {confidence_match.group(1)}")
        else:
            # Fallback: Look for rating/score patterns if confidence line missing
            score_match = re.search(r"(?:rating|score)[:\s]*(\d(?:\.\d)?)\s*/\s*(\d+)", response_text, re.IGNORECASE)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    scale = float(score_match.group(2))
                    if scale > 0:
                         # Normalize to 0-1 range, clamping between 0.2 and 1.0
                         confidence = max(0.2, min(1.0, score / scale))
                         logger.info(f"Parsed score/rating: {score}/{scale} -> {confidence}")
                except ValueError:
                     pass # Ignore if parsing fails

        # 3. Final checks and fallbacks if parsing failed
        if verdict == "UNDETERMINED":
            logger.warning(f"Could not reliably parse VERDICT from judge response: {response_text[:200]}...")
            # Simple keyword check as a last resort (less reliable)
            if "model a wins" in response_text.lower() and "model b wins" not in response_text.lower():
                verdict = "MODEL_A_WINS"
            elif "model b wins" in response_text.lower() and "model a wins" not in response_text.lower():
                 verdict = "MODEL_B_WINS"
            elif "tie" in response_text.lower() or "comparable" in response_text.lower():
                 verdict = "TIE"

        # If we have a verdict but no confidence, assign a default moderate confidence
        if verdict != "UNDETERMINED" and confidence == 0.0:
            confidence = 0.6 # Default confidence when parsing fails but verdict is found
            logger.info(f"Could not parse CONFIDENCE, assigning default {confidence} for verdict {verdict}")

        # Log the final parsed values
        logger.info(f"Final parsed judge result - Winner: {verdict}, Confidence: {confidence:.2f}")

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
        judge_error_cases = []

        for result in evaluation_results:
            verdict = result.winner # Use the pre-parsed winner
            if verdict in verdict_counts:
                verdict_counts[verdict] += 1
                if verdict != "UNDETERMINED" and verdict != "JUDGE_ERROR":
                    confidence_sum += result.confidence
                    valid_verdicts += 1
                elif verdict == "UNDETERMINED":
                     undetermined_cases.append(result.test_id)
                elif verdict == "JUDGE_ERROR":
                     judge_error_cases.append(result.test_id)
            else:
                # Should not happen if parsing is robust, but handle defensively
                logger.warning(f"Unexpected verdict '{verdict}' encountered for test_id {result.test_id}. Counting as UNDETERMINED.")
                verdict_counts["UNDETERMINED"] += 1
                undetermined_cases.append(result.test_id)


        # Log summary of problematic cases
        if undetermined_cases:
            logger.warning(f"Found {len(undetermined_cases)} undetermined verdicts: {undetermined_cases[:5]}" +
                          (f"... and {len(undetermined_cases)-5} more" if len(undetermined_cases) > 5 else ""))
        if judge_error_cases:
             logger.warning(f"Found {len(judge_error_cases)} judge errors: {judge_error_cases[:5]}" +
                          (f"... and {len(judge_error_cases)-5} more" if len(judge_error_cases) > 5 else ""))

        # Calculate percentages based on determined verdicts only (excluding UNDETERMINED and JUDGE_ERROR)
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

        # Convert EvaluationResult objects to dictionaries for JSON serialization
        raw_eval_dicts = []
        for res in evaluation_results:
             try:
                 # Assuming EvaluationResult is a dataclass or has a simple structure
                 raw_eval_dicts.append({
                     "test_id": res.test_id,
                     "winner": res.winner,
                     "confidence": res.confidence,
                     "champion_output": res.champion_output,
                     "challenger_output": res.challenger_output,
                     "reasoning": res.reasoning,
                 })
             except AttributeError as e:
                 logger.error(f"Error converting EvaluationResult to dict for test_id {res.test_id}: {e}")
                 # Add a placeholder or skip
                 raw_eval_dicts.append({"test_id": getattr(res, 'test_id', 'unknown'), "error": "Failed to serialize result"})


        return {
            "total_evaluations": total_evaluations,
            "verdict_counts": verdict_counts,
            "verdict_percentages": verdict_percentages, # Based on determined verdicts
            "average_confidence": round(average_confidence, 3), # Avg confidence for non-undetermined/error
            "raw_evaluations": raw_eval_dicts # Keep raw for output (as dicts)
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

    def run_test(
        self,
        test_cases: List[TestCase],
        batch_size: int = 5,
        progress=None,
        batch_retry_attempts: int = 0,  # Number of retry attempts for batches
        batch_backoff_factor: float = 2.0,  # Exponential backoff factor
        batch_max_wait: int = 60,  # Maximum wait time between retries in seconds
        batch_retry_trigger_strings: Optional[List[str]] = None  # Strings that trigger a retry
    ) -> Dict[str, Any]:
        """
        Run the complete test process: generate responses, evaluate, aggregate.

        Includes batch retry mechanism for transient errors or problematic responses.
        Args:
            test_cases: List of test cases (potentially including image paths/URLs)
            batch_size: Number of test cases per batch
            progress: Gradio progress callback
            batch_retry_attempts: Max retries per batch
            batch_backoff_factor: Exponential backoff factor
            batch_max_wait: Max wait time between retries
            batch_retry_trigger_strings: List of strings triggering retry if found in outputs/reasoning
        """
        all_evaluation_results: List[EvaluationResult] = []
        champion_metrics = {"total_latency": 0.0, "total_output_chars": 0, "success_count": 0, "error_count": 0, "image_load_errors": 0}
        challenger_metrics = {"total_latency": 0.0, "total_output_chars": 0, "success_count": 0, "error_count": 0, "image_load_errors": 0}
        judge_metrics = {"total_latency": 0.0, "total_output_chars": 0, "success_count": 0, "error_count": 0}

        num_cases = len(test_cases)
        if num_cases == 0:
             logger.warning("No test cases provided to run_test.")
             return {"evaluations": [], "summary": {"error": "No test cases loaded."}}

        total_batches = (num_cases + batch_size - 1) // batch_size
        processed_case_count = 0 # Track actual processed cases for progress
        global STOP_REQUESTED # Access the global flag

        # Process in batches
        for i in range(0, num_cases, batch_size):
            if STOP_REQUESTED:
                logger.warning(f"Stop requested. Finishing early after processing {processed_case_count} cases.")
                if progress:
                    progress(processed_case_count / num_cases, f"Stopping early after {processed_case_count} cases...")
                break # Exit the batch loop

            current_batch = test_cases[i:min(i + batch_size, num_cases)]
            batch_num = i // batch_size + 1
            logger.info(f"--- Processing Batch {batch_num}/{total_batches} (Cases {i+1}-{min(i+batch_size, num_cases)}) ---")

            # Initialize retry counter and success flag for this batch
            retry_count = 0
            batch_success = False
            batch_eval_results: List[EvaluationResult] = [] # Store results for *this successful batch attempt*

            # Process this batch with retries if configured
            while not batch_success and retry_count <= batch_retry_attempts:
                if retry_count > 0:
                    # Calculate backoff delay with exponential increase, capped at max_wait
                    delay = min(batch_backoff_factor ** (retry_count - 1), batch_max_wait)
                    logger.info(f"Retrying batch {batch_num} (attempt {retry_count}/{batch_retry_attempts}) after {delay:.2f}s delay")
                    if progress is not None:
                        # Update progress based on already processed cases, not 'i'
                        progress(processed_case_count / num_cases, f"Retrying Batch {batch_num} ({retry_count}/{batch_retry_attempts})")
                    time.sleep(delay)
                else:
                    if progress is not None:
                         progress(processed_case_count / num_cases, f"Running Batch {batch_num}/{total_batches}")

                # Reset batch-specific stores for this attempt
                current_attempt_champ_responses: Dict[str, ModelResponse] = {}
                current_attempt_chall_responses: Dict[str, ModelResponse] = {}
                current_attempt_eval_results: List[EvaluationResult] = []
                has_trigger_string_in_attempt = False

                # 1. Get responses from Champion and Challenger models for the current batch attempt
                for batch_idx, test_case in enumerate(current_batch):
                     # Check if stop requested before processing this case
                     if STOP_REQUESTED:
                         logger.warning(f"Stop requested. Skipping remaining cases in batch {batch_num}.")
                         break # Exit the inner loop for this batch

                     # Generate a consistent ID if not present, using overall index 'i' + batch_idx
                     case_id = test_case.id or f"case-{i + batch_idx + 1}"
                     test_case.id = case_id # Ensure the test case object has the ID

                     # --- Champion ---
                     try:
                         champ_resp = self.champion_runner.generate(test_case)
                         current_attempt_champ_responses[case_id] = champ_resp
                         # Only count metrics if not an image loading error generated by our code
                         if not champ_resp.output.startswith("Error: Failed to load image"):
                              champion_metrics["total_latency"] += champ_resp.latency
                              champion_metrics["total_output_chars"] += len(champ_resp.output)
                              if not champ_resp.output.startswith("Error:"): champion_metrics["success_count"] += 1
                              else: champion_metrics["error_count"] += 1
                         else:
                              champion_metrics["image_load_errors"] += 1
                              champion_metrics["error_count"] += 1 # Count as an error

                     except Exception as e:
                         logger.error(f"Critical error generating champion response for case {case_id}: {e}", exc_info=True)
                         current_attempt_champ_responses[case_id] = ModelResponse(case_id, self.champion_endpoint.name, f"Error: Generation failed critically - {e}", 0)
                         champion_metrics["error_count"] += 1

                     # --- Challenger ---
                     try:
                         chall_resp = self.challenger_runner.generate(test_case)
                         current_attempt_chall_responses[case_id] = chall_resp
                         if not chall_resp.output.startswith("Error: Failed to load image"):
                             challenger_metrics["total_latency"] += chall_resp.latency
                             challenger_metrics["total_output_chars"] += len(chall_resp.output)
                             if not chall_resp.output.startswith("Error:"): challenger_metrics["success_count"] += 1
                             else: challenger_metrics["error_count"] += 1
                         else:
                             challenger_metrics["image_load_errors"] += 1
                             challenger_metrics["error_count"] += 1
                     except Exception as e:
                         logger.error(f"Critical error generating challenger response for case {case_id}: {e}", exc_info=True)
                         current_attempt_chall_responses[case_id] = ModelResponse(case_id, self.challenger_endpoint.name, f"Error: Generation failed critically - {e}", 0)
                         challenger_metrics["error_count"] += 1

                # --- Yield intermediate update ---
                # Check if the test case actually had an image associated
                image_to_display = test_case.image_path_or_url if test_case.image_path_or_url else None # Use None if no image
                # Calculate combined latency (handle potential errors where response might be missing)
                champ_resp = current_attempt_champ_responses.get(case_id)
                chall_resp = current_attempt_chall_responses.get(case_id)
                champ_lat = champ_resp.latency if champ_resp else 0
                chall_lat = chall_resp.latency if chall_resp else 0
                combined_latency = round(champ_lat + chall_lat, 3)

                # Removed the intermediate yield from here. It will be moved inside the evaluation loop below.
                # 2. Evaluate with LM Judge for the current batch attempt
                if progress is not None:
                    progress((processed_case_count + len(current_batch) * 0.5) / num_cases, f"Evaluating Batch {batch_num}")

                for test_case in current_batch:
                    case_id = test_case.id # Should have been set above
                    champ_response = current_attempt_champ_responses.get(case_id)
                    chall_response = current_attempt_chall_responses.get(case_id)

                    # Skip evaluation if either model failed critically or had image load error
                    if not champ_response or not chall_response or \
                       champ_response.output.startswith("Error:") or \
                       chall_response.output.startswith("Error:"):
                        logger.warning(f"Skipping evaluation for case {case_id} due to generation error in one or both models.")
                        # Create a dummy eval result indicating skip? Or just don't add? Let's not add.
                        # We need a placeholder if retry depends on judge output, otherwise skip.
                        # For simplicity now, we'll create an error result if a model failed.
                        eval_reason = f"Skipped: Champion Error: {champ_response.output[:100]}... Challenger Error: {chall_response.output[:100]}..." if champ_response and chall_response else "Skipped: Model generation failed."
                        current_attempt_eval_results.append(EvaluationResult(
                              test_id=case_id,
                              champion_output=champ_response.output if champ_response else "GENERATION FAILED",
                              challenger_output=chall_response.output if chall_response else "GENERATION FAILED",
                              winner="JUDGE_ERROR", # Count as judge error if models failed
                              confidence=0.0,
                              reasoning=eval_reason
                         ))
                        judge_metrics["error_count"] += 1
                        continue # Skip to next test case in batch

                    # Check for trigger strings in model responses *before* calling judge if retry is enabled
                    if batch_retry_attempts > 0 and batch_retry_trigger_strings:
                        for trigger in batch_retry_trigger_strings:
                            if trigger in champ_response.output or trigger in chall_response.output:
                                logger.warning(f"Trigger string '{trigger}' found in model responses for case {case_id}. Batch will be retried.")
                                has_trigger_string_in_attempt = True
                                break # No need to check other triggers for this case
                        if has_trigger_string_in_attempt:
                            # Add a placeholder result indicating retry trigger
                            current_attempt_eval_results.append(EvaluationResult(
                                test_id=case_id, champion_output=champ_response.output, challenger_output=chall_response.output,
                                winner="UNDETERMINED", confidence=0.0, reasoning=f"Retry triggered by model output string."
                            ))
                            continue # Skip judge call for this case if retry is triggered by models

                    # If no model trigger, proceed to judge evaluation
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
                        current_attempt_eval_results.append(evaluation_result)
                        # --- Yield intermediate update AFTER judge evaluation for this case ---
                        image_to_display = test_case.image_path_or_url if test_case.image_path_or_url else None
                        champ_lat = round(champ_response.latency, 3) if champ_response else 0.0
                        chall_lat = round(chall_response.latency, 3) if chall_response else 0.0
                        # Use the latency calculated just before (judge_latency variable)
                        judge_lat = round(judge_latency, 3)

                        yield {
                            "type": "update",
                            "image_path": image_to_display,
                            "champ_latency": champ_lat,
                            "chall_latency": chall_lat,
                            "judge_latency": judge_lat
                        }


                        # Check for trigger strings in judge reasoning if retry is configured
                        if batch_retry_attempts > 0 and batch_retry_trigger_strings and not has_trigger_string_in_attempt:
                            for trigger in batch_retry_trigger_strings:
                                if trigger in evaluation_result.reasoning:
                                    logger.warning(f"Trigger string '{trigger}' found in judge reasoning for case {case_id}. Batch will be retried.")
                                    has_trigger_string_in_attempt = True
                                    # Overwrite the winner to UNDETERMINED if retry triggered by judge
                                    evaluation_result.winner = "UNDETERMINED"
                                    evaluation_result.reasoning += "\n[Retry triggered by judge reasoning]"
                                    break

                        # Update judge success/error counts based on final verdict (after potential trigger overwrite)
                        if evaluation_result.winner != "UNDETERMINED" and evaluation_result.winner != "JUDGE_ERROR": judge_metrics["success_count"] += 1
                        else: judge_metrics["error_count"] += 1 # Count undetermined/judge_error as errors for judge metrics

                    except Exception as e:
                        logger.error(f"Error during judge evaluation for case {case_id}: {e}", exc_info=True)
                        # Create a placeholder eval result indicating judge failure
                        current_attempt_eval_results.append(EvaluationResult(
                            test_id=case_id,
                            champion_output=champ_response.output,
                            challenger_output=chall_response.output,
                            winner="JUDGE_ERROR",
                            confidence=0.0,
                            reasoning=f"Error: Judge evaluation failed critically - {e}"
                        ))
                        judge_metrics["error_count"] += 1
                        # If judge fails critically, maybe trigger retry? For now, just mark as error.
                        # has_trigger_string_in_attempt = True # Option: Trigger retry on judge exception

                # --- Batch Retry Logic ---
                if has_trigger_string_in_attempt and retry_count < batch_retry_attempts:
                    logger.warning(f"Batch {batch_num} attempt {retry_count+1} failed due to trigger strings. Retrying...")
                    retry_count += 1
                    # Clear temporary results for this failed attempt, metrics were already counted above
                    current_attempt_eval_results = []
                    continue # Go to the next iteration of the while loop (retry)
                else:
                    # Conditions to accept the batch results:
                    # 1. No trigger strings were found in this attempt.
                    # 2. Trigger strings were found, but we've exhausted retry attempts.
                    batch_success = True
                    batch_eval_results = current_attempt_eval_results # Store the results of the successful (or final) attempt

                    if has_trigger_string_in_attempt and retry_count >= batch_retry_attempts:
                        logger.warning(f"Accepting batch {batch_num} results despite trigger strings after exhausting {batch_retry_attempts} retry attempts. Some results may be marked UNDETERMINED.")

                    # Log summary for the completed batch attempt
                    batch_summary = self.aggregator.aggregate(batch_eval_results) # Aggregate results of this specific batch
                    log_prefix = f"Batch {batch_num} completed"
                    if retry_count > 0: log_prefix += f" after {retry_count} retries"
                    logger.info(f"{log_prefix}. Verdict Counts: {batch_summary['verdict_counts']}")


            # --- End of Batch Processing ---
            # Add the results of the successful batch attempt to the overall list
            all_evaluation_results.extend(batch_eval_results)
            processed_case_count += len(current_batch) # Update processed count *after* successful batch completion


        # 3. Aggregate final results across all successful batches
        aggregated_summary = self.aggregator.aggregate(all_evaluation_results)

        # 4. Calculate final metrics (using totals accumulated across all attempts)
        # Note: Parameter renamed from total_cases to processed_cases for clarity
        def calculate_avg_metrics(metrics, processed_cases):
             # Base counts on total cases attempted, errors include generation/image load issues
             total_attempts = metrics["success_count"] + metrics["error_count"]
             # Avg latency based on total attempts where latency was recorded (excludes critical failures before generation)
             valid_latency_runs = metrics["success_count"] + (metrics["error_count"] - metrics.get("image_load_errors", 0)) # Approx.
             avg_latency = round(metrics["total_latency"] / valid_latency_runs, 2) if valid_latency_runs > 0 else 0
             # Avg chars based only on successful generations
             avg_chars = int(metrics["total_output_chars"] / metrics["success_count"]) if metrics["success_count"] > 0 else 0
             # Success rate based on total test cases *processed* before stopping
             success_rate = round((metrics["success_count"] / processed_cases) * 100, 1) if processed_cases > 0 else 0

             return {
                 "avg_latency_s": avg_latency,
                 "avg_output_chars": avg_chars,
                 "success_rate_pct": success_rate, # Now calculated based on processed cases
                 "errors": metrics["error_count"],
                 "image_load_errors": metrics.get("image_load_errors", 0)
             }

        # Use processed_case_count for denominators as it reflects actual attempts before potential early stopping
        # Pass processed_case_count to the updated function parameter
        champion_avg_metrics = calculate_avg_metrics(champion_metrics, processed_case_count)
        challenger_avg_metrics = calculate_avg_metrics(challenger_metrics, processed_case_count)
        # Judge metrics are based on cases where evaluation was attempted
        judge_attempts = judge_metrics["success_count"] + judge_metrics["error_count"]
        judge_avg_metrics = calculate_avg_metrics(judge_metrics, judge_attempts)


        # 5. Determine overall decision based on aggregated results
        decision = "MAINTAIN_CHAMPION" # Default
        reason = "Insufficient data or challenger did not significantly outperform."
        win_margin_threshold = 5 # Challenger needs to win by at least 5% points
        min_determined_verdicts = max(3, int(0.1 * processed_case_count)) # Need at least 3 or 10% determined verdicts

        percentages = aggregated_summary["verdict_percentages"]
        determined_verdicts = processed_case_count - aggregated_summary["verdict_counts"].get("UNDETERMINED", 0) - aggregated_summary["verdict_counts"].get("JUDGE_ERROR", 0)

        if determined_verdicts >= min_determined_verdicts:
            champ_wins_pct = percentages.get("MODEL_A_WINS", 0)
            chall_wins_pct = percentages.get("MODEL_B_WINS", 0)
            ties_pct = percentages.get("TIE", 0)

            # Calculate confidence-weighted percentages if we have confidence scores
            avg_confidence = aggregated_summary["average_confidence"]
            confidence_factor = f" with {avg_confidence:.2f} average confidence" if avg_confidence > 0 else ""

            if chall_wins_pct > champ_wins_pct + win_margin_threshold:
                 decision = "REPLACE_WITH_CHALLENGER"
                 reason = f"Challenger won {chall_wins_pct:.1f}% vs Champion's {champ_wins_pct:.1f}%{confidence_factor} (>{win_margin_threshold}% margin based on {determined_verdicts} determined verdicts)."
            elif champ_wins_pct > chall_wins_pct + win_margin_threshold:
                 decision = "MAINTAIN_CHAMPION"
                 reason = f"Champion won {champ_wins_pct:.1f}% vs Challenger's {chall_wins_pct:.1f}%{confidence_factor} (based on {determined_verdicts} determined verdicts)."
            else:
                 # Closer results, consider ties or maintain status quo
                 decision = "MAINTAIN_CHAMPION"
                 reason = f"Results close ({champ_wins_pct:.1f}% vs {chall_wins_pct:.1f}%, {ties_pct:.1f}% ties){confidence_factor}. Challenger did not show clear superiority (based on {determined_verdicts} determined verdicts)."
        else:
            # Not enough determined verdicts for a reliable decision
            decision = "MAINTAIN_CHAMPION"
            reason = f"Insufficient determined verdicts ({determined_verdicts}/{processed_case_count}, need >= {min_determined_verdicts}) to make a reliable decision. Defaulting to maintaining champion."

        # Log final summary
        logger.info(f"--- Final Aggregated Results ({processed_case_count} cases processed) ---")
        logger.info(f"Verdict Counts: {aggregated_summary['verdict_counts']}")
        logger.info(f"Verdict Percentages (Determined Only): {aggregated_summary['verdict_percentages']}")
        logger.info(f"Average Confidence (Determined Only): {aggregated_summary['average_confidence']:.3f}")
        logger.info(f"Champion Metrics: {champion_avg_metrics}")
        logger.info(f"Challenger Metrics: {challenger_avg_metrics}")
        logger.info(f"Judge Metrics: {judge_avg_metrics}")
        logger.info(f"Decision: {decision} - {reason}")

        if progress is not None:
            final_status = "Testing completed" if not STOP_REQUESTED else "Testing stopped early"
            progress(1.0, final_status)

        final_summary = {
                "total_test_cases_processed": processed_case_count,
                "total_test_cases_loaded": num_cases,
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

        # Yield the final results dictionary
        yield {
            "type": "final",
            "evaluations": aggregated_summary["raw_evaluations"],
            "summary": final_summary
        }

# --- Gradio UI Components & Logic ---

def parse_test_data(
    file_obj,
    text_data,
    key_field_name: str = "key",
    value_field_name: str = "value",
    image_field_name: str = "image_url" # Added image field name parameter
) -> List[TestCase]:
    """
    Parses test data from Gradio file upload or text input.
    Uses specified field names for key, value, and image path/URL.
    """
    test_cases = []
    raw_data = None

    if file_obj is not None:
        # Use the temporary file path provided by Gradio
        file_path = file_obj.name
        logger.info(f"Loading test data from uploaded file: {file_path}")
        try:
            # Determine file type from extension, not relying on original name if temp name is different
            _, file_ext = os.path.splitext(file_path)
            file_ext = file_ext.lower()

            if file_ext == ".json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
            elif file_ext == ".csv":
                # Read CSV into pandas DataFrame first for easier handling
                try:
                     # Try detecting delimiter, handle potential bad lines
                     # Use sensible defaults, allow overriding later if needed
                     df = pd.read_csv(
                          file_path,
                          sep=None, # Auto-detect
                          engine='python',
                          on_bad_lines='warn',
                          quoting=csv.QUOTE_MINIMAL, # Default quoting
                          escapechar='\\' # Common escape character
                          )
                     logger.info(f"CSV loaded successfully. Columns: {df.columns.tolist()}")
                     # Convert NaN/NaT to None for cleaner processing -> convert to empty string later
                     df = df.fillna('')
                     # Convert DataFrame rows to list of dictionaries
                     raw_data = df.to_dict(orient='records')
                except Exception as e:
                     logger.error(f"Error reading CSV file '{file_path}': {e}")
                     raise ValueError(f"Error reading CSV: {e}")
            elif file_ext in (".jsonl", ".ndjson"):
                # Handle JSONL (newline-delimited JSON)
                raw_data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line: continue # Skip empty lines
                        try:
                            item = json.loads(line)
                            raw_data.append(item)
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid JSON line #{line_num + 1} in file '{file_path}': {line[:100]}...")
                if not raw_data:
                    raise ValueError("No valid JSON objects found in JSONL file.")
            else:
                allowed_extensions = ['.csv', '.json', '.jsonl', '.ndjson']
                raise ValueError(f"Invalid file type ({file_ext}). Please upload a file that is one of these formats: {allowed_extensions}")

        except Exception as e:
            logger.error(f"Error processing uploaded file {file_path}: {e}", exc_info=True)
            raise ValueError(f"Failed to process file: {e}")

    elif text_data and text_data.strip():
        logger.info("Loading test data from text input.")
        try:
            # Try parsing as JSON list first
            raw_data = json.loads(text_data)
            if not isinstance(raw_data, list):
                raise ValueError("Pasted text is valid JSON, but not a list of objects.")
        except json.JSONDecodeError as json_err:
            # If JSON fails, try treating it as line-delimited JSON (JSONL)
            logger.warning(f"Could not parse text as JSON list ({json_err}), trying as JSONL...")
            try:
                 raw_data = [json.loads(line) for line in text_data.strip().splitlines() if line.strip()]
                 if not raw_data:
                      raise ValueError("No valid JSON objects found in text input lines.")
            except json.JSONDecodeError as line_err:
                logger.error(f"Invalid JSON format in text input (checked as list and line-by-line): {line_err}")
                raise ValueError(f"Invalid JSON format in text input. Ensure it's a list of objects `[ {{\"key\": ...}}, ... ]` or one JSON object per line.")
        except Exception as e:
            logger.error(f"Error processing text input data: {e}", exc_info=True)
            raise ValueError(f"Failed to process text data: {e}")

    else:
        raise ValueError("No test data provided. Please upload a file or paste JSON/JSONL.")

    # Convert raw_data (list of dicts) to TestCase objects
    if isinstance(raw_data, list):
        for i, item in enumerate(raw_data):
            if isinstance(item, dict):
                try:
                    # Ensure the specified key field exists, value field is optional
                    # 'id' is optional (defaults to None, ModelTester assigns later if needed)
                    # 'image' field is optional
                    key = item.get(key_field_name)
                    if key is None:
                         logger.warning(f"Skipping item {i+1} due to missing '{key_field_name}' field. Data: {item}")
                         continue

                    # Get image path/url if field exists and is not empty/None
                    image_val = item.get(image_field_name) if image_field_name else None
                    image_path_or_url = str(image_val).strip() if image_val and str(image_val).strip() else None

                    test_cases.append(TestCase(
                        id=str(item.get('id', f"item-{i+1}")), # Ensure ID is string, use item index
                        key=str(key), # Ensure key is string
                        value=str(item.get(value_field_name, '')), # Ensure value is string, default empty
                        image_path_or_url=image_path_or_url,
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
    output += f"Test Cases Loaded: {summary_data.get('total_test_cases_loaded', 'N/A')}\n"
    output += f"Test Cases Processed: {summary_data.get('total_test_cases_processed', 'N/A')}\n"

    output += "\nVerdicts (Based on Processed Cases):\n"
    for verdict, count in summary_data.get('verdicts', {}).items():
        output += f"  {verdict}: {count}\n"

    output += "\nVerdict Percentages (Based on Determined Verdicts):\n"
    determined = summary_data.get('total_test_cases_processed', 0) - \
                 summary_data.get('verdicts', {}).get('UNDETERMINED', 0) - \
                 summary_data.get('verdicts', {}).get('JUDGE_ERROR', 0)
    output += f"  (Calculated from {determined} determined verdicts)\n"
    for verdict, pct in summary_data.get('verdict_percentages', {}).items():
        output += f"  {verdict}: {pct:.1f}%\n"

    avg_conf = summary_data.get('average_confidence', 0)
    output += f"\nAverage Confidence (Determined Only): {avg_conf:.3f}\n"

    output += "\nMetrics (Avg Latency / Avg Output Chars / Success Rate / Errors / Image Load Errors):\n"
    champ_metrics = summary_data.get('champion_metrics', {})
    chall_metrics = summary_data.get('challenger_metrics', {})
    judge_metrics = summary_data.get('judge_metrics', {}) # Judge metrics are calculated differently
    output += (f"  Champion:   {champ_metrics.get('avg_latency_s', 0):.2f}s / "
               f"{champ_metrics.get('avg_output_chars', 0)} / "
               f"{champ_metrics.get('success_rate_pct', 0):.1f}% / "
               f"{champ_metrics.get('errors', 0)} / "
               f"{champ_metrics.get('image_load_errors', 0)}\n")
    output += (f"  Challenger: {chall_metrics.get('avg_latency_s', 0):.2f}s / "
               f"{chall_metrics.get('avg_output_chars', 0)} / "
               f"{chall_metrics.get('success_rate_pct', 0):.1f}% / "
               f"{chall_metrics.get('errors', 0)} / "
               f"{chall_metrics.get('image_load_errors', 0)}\n")
    # Judge metrics are slightly different (no image errors, success based on valid eval)
    output += (f"  Judge:      {judge_metrics.get('avg_latency_s', 0):.2f}s / "
               f"{judge_metrics.get('avg_output_chars', 0)} / "
               f"{judge_metrics.get('success_rate_pct', 0):.1f}% / "
               f"{judge_metrics.get('errors', 0)} (Errors + Undetermined)\n")


    output += f"\nDecision: {summary_data.get('decision', 'N/A')}\n"
    output += f"Reason: {summary_data.get('reason', 'N/A')}\n"

    return output

def run_test_from_ui(
    # Model Configs (15 inputs)
    champ_name, champ_api_url, champ_model_id, champ_temp, champ_max_tokens,
    chall_name, chall_api_url, chall_model_id, chall_temp, chall_max_tokens,
    judge_name, judge_api_url, judge_model_id, judge_temp, judge_max_tokens,
    # API Key (1 input) - Potentially optional now
    api_key_input,
    # Prompts (2 inputs)
    model_prompt_template_input,
    judge_prompt_template_input,
    # Test Data (2 inputs)
    test_data_file,
    test_data_text,
    # Parameters (5 inputs)
    batch_size_input,
    batch_retry_attempts_input,
    batch_backoff_factor_input,
    batch_max_wait_input,
    batch_retry_trigger_strings_input,
    # Data Field Names (3 inputs) - Added image field name
    key_field_name_input,
    value_field_name_input,
    image_field_name_input, # Added image field name input
    # Gradio progress object
    progress=gr.Progress(track_tqdm=True)
):
    """
    Handles the logic for running the A/B test triggered by the Gradio UI button.
    """
    global STOP_REQUESTED
    STOP_REQUESTED = False # Reset stop flag at the beginning of each UI run
    logger.info("Starting test run from Gradio UI...")
    progress(0, desc="Initializing...")

    try:
        # 1. Get API Key from UI input (Treat as optional, let endpoint logic handle needs)
        # Also check environment variable as a fallback/override
        api_key_env = os.getenv("OPENROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GOOGLE_API_KEY") # Add other common key names if needed
        api_key_ui = str(api_key_input).strip() if api_key_input else None

        # Prioritize UI input if provided, otherwise use environment variable
        api_key = api_key_ui if api_key_ui else api_key_env

        if api_key_ui and api_key_env and api_key_ui != api_key_env:
             logger.warning("API Key provided via UI input overrides the environment variable.")
        elif api_key:
             logger.info(f"API Key found ({'UI input' if api_key_ui else 'environment variable'}).")
        else:
             logger.info("API Key not provided via UI input or environment variables. Only local/keyless endpoints will work.")


        progress(0.1, desc="Loading test data...")
        # 2. Load Test Cases (Pass field names from UI)
        try:
            key_field = str(key_field_name_input).strip() or "key"
            value_field = str(value_field_name_input).strip() or "value"
            image_field = str(image_field_name_input).strip() or "image_url" # Default if empty
            logger.info(f"Using data fields - Key: '{key_field}', Value: '{value_field}', Image: '{image_field}'")
            test_cases = parse_test_data(test_data_file, test_data_text, key_field, value_field, image_field)
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
        # 3. Create Model Endpoints (Pass the potentially found api_key)
        try:
            # Helper to create endpoint, ensuring types
            def create_ep(name, url, model_id, temp, max_tok, key):
                 # Strip whitespace from URL and model ID
                 url = str(url).strip() if url else ""
                 model_id = str(model_id).strip() if model_id else ""
                 # Basic validation
                 if not name: raise ValueError("Model Display Name cannot be empty.")
                 if not url: raise ValueError(f"API URL cannot be empty for model '{name}'.")
                 if not model_id: raise ValueError(f"Model ID cannot be empty for model '{name}'.")
                 # Use the potentially loaded key
                 return ModelEndpoint(
                     name=str(name), api_url=url, api_key=key, model_id=model_id,
                     temperature=float(temp), max_tokens=int(max_tok)
                 )

            champion_endpoint = create_ep(champ_name, champ_api_url, champ_model_id, champ_temp, champ_max_tokens, api_key)
            challenger_endpoint = create_ep(chall_name, chall_api_url, chall_model_id, chall_temp, chall_max_tokens, api_key)
            judge_endpoint = create_ep(judge_name, judge_api_url, judge_model_id, judge_temp, judge_max_tokens, api_key)

            # Log endpoints being used (mask key if present)
            logger.info(f"Champion Endpoint: {champion_endpoint.name}, URL: {champion_endpoint.api_url}, Model: {champion_endpoint.model_id}, Key Provided: {'Yes' if champion_endpoint.api_key else 'No'}")
            logger.info(f"Challenger Endpoint: {challenger_endpoint.name}, URL: {challenger_endpoint.api_url}, Model: {challenger_endpoint.model_id}, Key Provided: {'Yes' if challenger_endpoint.api_key else 'No'}")
            logger.info(f"Judge Endpoint: {judge_endpoint.name}, URL: {judge_endpoint.api_url}, Model: {judge_endpoint.model_id}, Key Provided: {'Yes' if judge_endpoint.api_key else 'No'}")

        except ValueError as ve:
             logger.error(f"Model Configuration Error: {ve}")
             raise gr.Error(f"Model Configuration Error: {ve}")
        except Exception as e:
             logger.error(f"Error creating ModelEndpoint objects: {e}", exc_info=True)
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
             logger.error(f"Error instantiating ModelTester: {e}", exc_info=True)
             raise gr.Error(f"Tester Initialization Error: {e}")

        # 5. Run the Test
        batch_size = int(batch_size_input) if batch_size_input is not None and batch_size_input > 0 else 1
        logger.info(f"Running test with {len(test_cases)} cases, batch size {batch_size}...")
        progress(0.3, desc="Running A/B test...")
        try:
            # Process batch retry parameters
            batch_retry_attempts = int(batch_retry_attempts_input) if batch_retry_attempts_input is not None else 0
            batch_backoff_factor = float(batch_backoff_factor_input) if batch_backoff_factor_input is not None else 2.0
            batch_max_wait = int(batch_max_wait_input) if batch_max_wait_input is not None else 60

            # Process trigger strings (convert from comma-separated string to list)
            batch_retry_trigger_strings = None
            if batch_retry_trigger_strings_input and batch_retry_trigger_strings_input.strip():
                batch_retry_trigger_strings = [s.strip().lower() for s in batch_retry_trigger_strings_input.split(',') if s.strip()] # Lowercase for case-insensitive match later
                logger.info(f"Using batch retry trigger strings: {batch_retry_trigger_strings}")

            # Make trigger strings case-insensitive in the run_test method check
            # (Already done in list comprehension above)

            # Iterate through the generator yielded by run_test
            final_results = None
            last_image_path = None # Variable to store the last image path
            # Initialize variables for individual latencies
            last_champ_latency = ""
            last_chall_latency = ""
            last_judge_latency = ""
            for result_update in tester.run_test(
                test_cases,
                batch_size=batch_size,
                progress=progress,
                batch_retry_attempts=batch_retry_attempts,
                batch_backoff_factor=batch_backoff_factor,
                batch_max_wait=batch_max_wait,
                batch_retry_trigger_strings=batch_retry_trigger_strings
            ):
                if STOP_REQUESTED: # Check stop flag during iteration
                    logger.info("Stop requested, halting UI updates.")
                    break

                if result_update.get("type") == "update":
                    # Yield intermediate update for image and runtime
                    # Order: summary, details, image, runtime
                    # Extract individual latencies from the new dictionary structure
                    last_image_path = result_update.get("image_path")
                    last_champ_latency = str(result_update.get("champ_latency", ""))
                    last_chall_latency = str(result_update.get("chall_latency", ""))
                    last_judge_latency = str(result_update.get("judge_latency", ""))
                    # Yield update for the 6 output components in the correct order
                    # Order: summary, details, image, champ_lat, chall_lat, judge_lat
                    yield None, None, last_image_path, last_champ_latency, last_chall_latency, last_judge_latency
                elif result_update.get("type") == "final":
                    # Store final results and break loop (should be the last yield)
                    final_results = result_update
                    break
                else:
                    logger.warning(f"Received unexpected update type from run_test: {result_update.get('type')}")

            # Check if the loop finished because of stop request or completion
            if STOP_REQUESTED:
                 logger.info("Test run stopped by user.")
                 # Yield a message indicating stop? Or just let the last state remain?
                 # Let's yield a final status update to the summary.
                 stopped_summary = "Test run stopped by user."
                 # Yield the stopped summary, but retain the last known image/runtime
                 # Yield stopped summary with the last known individual latencies
                 yield stopped_summary, None, last_image_path, last_champ_latency, last_chall_latency, last_judge_latency
                 return # Exit the function

            if final_results is None:
                 logger.error("Test run finished but no final results were yielded.")
                 raise gr.Error("Test Execution Error: Did not receive final results.")

            progress(0.9, desc="Formatting final results...")
            # 6. Format Final Results
            summary_data = final_results.get("summary", {})
            raw_evals = final_results.get("evaluations", [])
            summary_output = format_summary_output(summary_data)
            display_columns = ['test_id', 'winner', 'confidence', 'champion_output', 'challenger_output', 'reasoning']

            try:
                if raw_evals:
                    details_df = pd.DataFrame(raw_evals)
                    if not details_df.empty:
                        for col in display_columns:
                            if col not in details_df.columns:
                                details_df[col] = None
                        details_df = details_df[display_columns]
                    else:
                        details_df = pd.DataFrame(columns=display_columns)
                else:
                    details_df = pd.DataFrame(columns=display_columns)
                    summary_output += "\n\nNote: No evaluation results were generated."

            except Exception as df_err:
                logger.error(f"Error creating or processing DataFrame from final results: {df_err}")
                summary_output += f"\n\nError displaying detailed results: {df_err}"
                details_df = pd.DataFrame(columns=display_columns)

            logger.info("Test run completed successfully.")
            # Yield the final formatted results
            # Order: summary, details, image, runtime (image/runtime are None here)
            # Yield final summary/details along with the *last* image/runtime displayed
            # Yield final results with the last known individual latencies
            yield summary_output, details_df, last_image_path, last_champ_latency, last_chall_latency, last_judge_latency

        except Exception as e:
            logger.exception("An error occurred during the test execution loop in run_test_from_ui.")
            # Yield error message to UI components
            error_message = f"Test Execution Error: {e}"
            error_df = pd.DataFrame([{"Error": error_message}])
            # Ensure error yields also provide 6 values (None for latencies)
            yield error_message, error_df, None, None, None, None

    except gr.Error as e: # Catch Gradio-specific errors to display them directly
        logger.error(f"Gradio Error: {e}")
        error_message = str(e)
        error_df = pd.DataFrame([{"Error": error_message}])
        yield error_message, error_df, None, None, None, None
    except Exception as e:
        logger.exception("An unexpected error occurred in run_test_from_ui setup.")
        error_message = f"An unexpected setup error occurred: {e}"
        error_df = pd.DataFrame([{"Error": error_message}])
        yield error_message, error_df, None, None, None, None

    except gr.Error as e: # Catch Gradio-specific errors to display them directly
        logger.error(f"Gradio Error: {e}")
        error_message = str(e)
        error_df = pd.DataFrame([{"Error": error_message}])
        yield error_message, error_df, None, None, None, None
    except Exception as e:
        logger.exception("An unexpected setup error occurred in run_test_from_ui setup.")
        error_message = f"An unexpected setup error occurred: {e}"
        error_df = pd.DataFrame([{"Error": error_message}])
        yield error_message, error_df, None, None, None, None
    finally:
        # Ensure the stop requested flag is reset regardless of how the function exits
        # Although it's reset at the start, this is an extra safety measure.
        STOP_REQUESTED = False
# Removed the duplicated except blocks that were incorrectly indented

# Function to be called by the Stop button
def request_stop():
    global STOP_REQUESTED
    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        logger.warning("Stop requested via UI button.")
        # Use gr.Info for user feedback if Gradio version supports it well
        try:
            gr.Info("Stop request received. Finishing current batch...")
        except AttributeError: # Fallback for older Gradio versions
            print("UI: Stop request received. Finishing current batch...")

    else:
        logger.warning("Stop already requested.")
        try:
            gr.Info("Stop already requested. Please wait...")
        except AttributeError:
             print("UI: Stop already requested. Please wait...")


def create_ui():
    """Creates the Gradio web interface for the A/B testing tool."""
    logger.info("Creating Gradio UI...")

    # Default values for UI components
    default_api_url_openrouter = "https://openrouter.ai/api/v1/chat/completions"
    default_api_url_ollama = "http://localhost:11434/api/generate" # Default Ollama URL
    default_model_prompt = "User: {key}\nAssistant:" # Example prompt
    # Use the default judge prompt from the LMJudge class
    default_judge_prompt = LMJudge.DEFAULT_EVALUATION_PROMPT

    css = """
    .model-config-group .gr-form { background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    .model-config-group .gr-form > :first-child { font-weight: bold; margin-bottom: 5px; } /* Style the label */
    .results-box { border: 1px solid #ccc; padding: 15px; border-radius: 5px; margin-top: 15px; }
    .api-key-warning { color: #cc5500; font-weight: bold; margin-bottom: 15px; }
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft()) as iface:
        gr.Markdown("# Model A/B Testing & Evaluation Tool")
        gr.Markdown(
            "Configure champion, challenger, and judge models, provide test data (including optional images), "
            "and run evaluations to compare model performance."
        )
        gr.Markdown(
            """**API Key**: Optional. Enter if needed for cloud endpoints (OpenRouter, Anthropic, Gemini, etc.).
            If blank, the tool will try common environment variables (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, etc.).
            Leave blank/unset if using only local endpoints (like Ollama).""",
            elem_classes="api-key-warning"
        )
        gr.Markdown(
            """**Multimodal Input**: To use image inputs:
            1. Ensure your test data (CSV/JSON/JSONL) includes a column/field containing the **local path** or **public URL** to the image.
            2. Specify this column/field name in the 'Image Field Name' box below.
            3. Ensure your models and endpoints support multimodal input (e.g., GPT-4o, Claude 3, LLaVA via Ollama).
            4. The model prompt should instruct the model on what to do with the image (e.g., 'Describe this image.', 'What text is in the image provided?').""",
            elem_classes="api-key-warning"
        )


        with gr.Tabs():
            with gr.TabItem("Configuration"):
                # Add API Key input field
                with gr.Row():
                     api_key_input = gr.Textbox(
                          label="API Key (Optional - Overrides Environment Variable)",
                          type="password",
                          placeholder="Enter key if required and not set via ENV",
                          info="Overrides OPENROUTER_API_KEY etc. if set. Leave blank to use ENV or for local models."
                     )
                with gr.Row():
                    # Champion Model Configuration
                    with gr.Column(scale=1):
                         with gr.Group(elem_classes="model-config-group"):
                              gr.Label("Champion Model (Model A)")
                              # Updated example for Ollama Mistral 3.1 (as requested default)
                              champ_name = gr.Textbox(label="Display Name", value="Champion (LM Studio Gemma 3 12B)")
                              champ_api_url = gr.Textbox(label="API URL", value="http://localhost:1234/v1/chat/completions") # LM Studio OpenAI endpoint
                              champ_model_id = gr.Textbox(label="Model ID", value="gemma-3-12b-it") # User specified
                              champ_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.1, value=0.1)
                              champ_max_tokens = gr.Number(label="Max Tokens", value=8192, precision=0)
                    # Challenger Model Configuration
                    with gr.Column(scale=1):
                         with gr.Group(elem_classes="model-config-group"):
                              gr.Label("Challenger Model (Model B)")
                              # Updated examples for Ollama Gemma 3 27B (as requested default)
                              chall_name = gr.Textbox(label="Display Name", value="Challenger (LM Studio Gemma 3 4B)")
                              chall_api_url = gr.Textbox(label="API URL", value="http://localhost:1234/v1/chat/completions") # LM Studio OpenAI endpoint
                              chall_model_id = gr.Textbox(label="Model ID", value="gemma-3-4b-it") # User specified
                              chall_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.1, value=0.1)
                              chall_max_tokens = gr.Number(label="Max Tokens", value=8192, precision=0)
                    # Judge Model Configuration
                    with gr.Column(scale=1):
                         with gr.Group(elem_classes="model-config-group"):
                              gr.Label("Judge Model")
                              judge_name = gr.Textbox(label="Display Name", value="Judge (LM Studio Gemma 3 27B)")
                              judge_api_url = gr.Textbox(label="API URL", value="http://localhost:1234/v1/chat/completions") # LM Studio OpenAI endpoint
                              judge_model_id = gr.Textbox(label="Model ID", value="gemma-3-27b-it") # User specified
                              judge_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.0) # Judge usually deterministic
                              judge_max_tokens = gr.Number(label="Max Tokens", value=8192, precision=0) # Judge might need more tokens

                with gr.Row():
                    # Model Prompt Template
                    with gr.Column(scale=1):
                        gr.Markdown("### Model Prompt Template")
                        model_prompt_template_input = gr.Textbox(
                            label="Template for Champion/Challenger (use {key} for input)",
                            value="{key}\nUser: Provide a detailed description\nAssistant:",
                            lines=5,
                            show_copy_button=True
                        )
                    # Judge Prompt Template
                    with gr.Column(scale=1):
                        gr.Markdown("### Judge Prompt Template")
                        judge_prompt_template_input = gr.Textbox(
                            label="Template for Judge (see code/docs for available placeholders)",
                            value=default_judge_prompt,
                            lines=15,
                            show_copy_button=True
                        )

                with gr.Row():
                    # Test Data Input
                    with gr.Column(scale=1):
                        gr.Markdown("### Test Data")
                        gr.Markdown("Upload a CSV/JSON/JSONL file or paste data below. Specify the field names containing the model input (key), optional reference answer (value), and optional image path/URL. Add an `id` field for stable identification (recommended).")
                        test_data_file = gr.File(label="Upload Test Data (CSV, JSON, JSONL/NDJSON)", file_types=[".csv", ".json", ".jsonl", ".ndjson"])
                        test_data_text = gr.Textbox(label="Or Paste Test Data (JSON list or JSONL format)", lines=8, placeholder='[{"id": "t1", "prompt": "Describe image", "image_url": "/path/to/img.jpg", "reference": "..."}]\n{"id": "t2", "prompt": "Question text", "image_url": null, "reference": "..."}')
                        with gr.Row():
                             key_field_name_input = gr.Textbox(label="Key Field Name", value="name", info="Field containing the text input/prompt.") # Swapped default
                             value_field_name_input = gr.Textbox(label="Value Field Name", value="caption", info="Field containing the reference/ground truth (optional).") # Swapped default
                             image_field_name_input = gr.Textbox(label="Image Field Name", value="image_url", info="Field containing the image path or URL (optional).") # Added image field input

                    # Test Run Parameters
                    with gr.Column(scale=1):
                        gr.Markdown("### Test Parameters")
                        batch_size_input = gr.Number(label="Batch Size", value=5, minimum=1, precision=0)

                        # Add batch retry parameters
                        gr.Markdown("#### Batch Retry Settings")
                        batch_retry_attempts_input = gr.Number(
                            label="Batch Retry Attempts",
                            value=1, # Default to 1 retry
                            precision=0,
                            minimum=0,
                            info="Number of times to retry a batch if trigger strings are found (0 = no retries)"
                        )
                        batch_backoff_factor_input = gr.Slider(
                            label="Backoff Factor",
                            value=2.0,
                            minimum=1.0,
                            maximum=5.0,
                            step=0.1,
                            info="Factor for exponential backoff between retries (e.g., 2.0 waits 1s, 2s, 4s...)"
                        )
                        batch_max_wait_input = gr.Number(
                            label="Maximum Wait Time (seconds)",
                            value=60,
                            precision=0,
                            minimum=1,
                            info="Maximum wait time between retries in seconds"
                        )
                        batch_retry_trigger_strings_input = gr.Textbox(
                            label="Retry Trigger Strings (Comma-separated)",
                            placeholder="e.g., rate limit,error,timeout,empty response",
                            info="Retry batch if these strings appear in model/judge output (case-insensitive check)"
                        )
                        # Add preprocessing options later if needed

            with gr.TabItem("Monitoring"):
                gr.Markdown("### Test Execution & Results")
                with gr.Row():
                    run_button = gr.Button("Run A/B Test", variant="primary", scale=4)
                    stop_button = gr.Button("Stop Test", variant="stop", scale=1) # Add stop button
                with gr.Row(): # New row for results display
                    with gr.Column(scale=1): # Column for the new status window
                        with gr.Group(elem_classes="results-box"):
                             gr.Markdown("#### Last Processed")
                             # Placeholder for the most recent image
                             last_image_display = gr.Image(label="Last Image", type="filepath", interactive=False, height=200, value=None, show_label=True) # Explicitly show label, ensure preview area renders
                             # Placeholder for the runtime of the last image
                             # Replace single runtime display with three separate ones
                             last_champ_latency_display = gr.Textbox(label="Champion Latency (s)", value="", interactive=False)
                             last_chall_latency_display = gr.Textbox(label="Challenger Latency (s)", value="", interactive=False)
                             last_judge_latency_display = gr.Textbox(label="Judge Latency (s)", value="", interactive=False)

                    with gr.Column(scale=2): # Column for existing summary and details
                        with gr.Group(elem_classes="results-box"):
                             gr.Markdown("#### Summary")
                             summary_output = gr.Textbox(label="Overall Results", lines=15, show_copy_button=True, interactive=False)
                        with gr.Group(elem_classes="results-box"):
                             gr.Markdown("#### Detailed Evaluations")
                             details_output = gr.DataFrame(label="Individual Case Results", wrap=True, interactive=False) # Use DataFrame for better display
        # Define interactions
        run_button.click(
            fn=run_test_from_ui,
            inputs=[
                # Model Configs
                champ_name, champ_api_url, champ_model_id, champ_temp, champ_max_tokens,
                chall_name, chall_api_url, chall_model_id, chall_temp, chall_max_tokens,
                judge_name, judge_api_url, judge_model_id, judge_temp, judge_max_tokens,
                # API Key
                api_key_input,
                # Prompts
                model_prompt_template_input,
                judge_prompt_template_input,
                # Test Data
                test_data_file,
                test_data_text,
                # Parameters
                batch_size_input,
                batch_retry_attempts_input,
                batch_backoff_factor_input,
                batch_max_wait_input,
                batch_retry_trigger_strings_input,
                # Data Field Names
                key_field_name_input,
                value_field_name_input,
                image_field_name_input # Pass image field name
            ],
            # Update outputs to include the three new latency fields instead of the old one
            outputs=[
                summary_output, details_output, last_image_display,
                last_champ_latency_display, last_chall_latency_display, last_judge_latency_display
            ]
        )

        # Wire the stop button
        stop_button.click(fn=request_stop, inputs=None, outputs=None)

    return iface

def run_cli_test():
    """Runs the A/B test from the command line using hardcoded examples."""
    logger.info("Starting CLI execution of ModelTester...")

    # --- Configuration (API Key optional for local models) ---
    # Load API keys from .env file if it exists
    try:
        from dotenv import load_dotenv
        if load_dotenv():
             logger.info("Loaded environment variables from .env file.")
        else:
             logger.info(".env file not found or empty, relying on system environment variables or UI input.")
    except ImportError:
        logger.warning("python-dotenv not installed, cannot load .env file. Run 'pip install python-dotenv' or ensure packages are installed.")

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") # Still useful for judge or cloud models
    OLLAMA_API_URL = "http://localhost:11434/api/generate"

    # Define Model Endpoints
    # Using Ollama Mistral as Champion
    champion_model = ModelEndpoint(
        name="Champion (Ollama Mistral)",
        api_url=OLLAMA_API_URL,
        api_key=None, # No key needed for local Ollama
        model_id="mistral:latest", # Adjust if your model name is different
        temperature=0.1
    )

    # Using Ollama Gemma 2 9B as Challenger
    challenger_model = ModelEndpoint(
        name="Challenger (Ollama Gemma2 9B)",
        api_url=OLLAMA_API_URL,
        api_key=None, # No key needed
        model_id="hf.co/stduhpf/google-gemma-3-27b-it-qat-q4_0-gguf-small:latest", # Updated to match ollama list
        temperature=0.1,
        max_tokens=2048
    )

    # Using OpenRouter GPT-4o Mini as Judge (Requires API Key)
    if not OPENROUTER_API_KEY:
         logger.warning("OPENROUTER_API_KEY not set (checked ENV and .env). Using Champion model as Judge (less ideal).")
         judge_model = champion_model # Fallback judge
         judge_model.name = "Judge (Fallback - Ollama Mistral)"
    else:
         logger.info("Using OpenRouter API Key for Judge.")
         judge_model = ModelEndpoint(
             name="Judge (GPT-4o Mini - OR)",
             api_url="https://openrouter.ai/api/v1/chat/completions",
             api_key=OPENROUTER_API_KEY,
             model_id="openai/gpt-4o-mini",
             temperature=0.0,
             max_tokens=2048
         )

    # Define Model Prompt Template
    model_prompt = "User: {key}\nAssistant:"

    # Define Sample Test Cases (Including Multimodal Example)
    # Create a dummy image file for testing if it doesn't exist
    dummy_image_path = "dummy_test_image.png"
    if not os.path.exists(dummy_image_path):
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (100, 50), color = (73, 109, 137)) # Blueish background
            d = ImageDraw.Draw(img)
            d.text((10,10), "Test Img", fill=(255,255,0)) # Yellow text
            img.save(dummy_image_path)
            logger.info(f"Created dummy image file: {dummy_image_path}")
        except ImportError:
            logger.warning("Pillow (PIL) not installed, cannot create dummy image for CLI test. Ensure packages are installed via venv setup.")
            dummy_image_path = None # Cannot use image test case
        except Exception as e:
             logger.error(f"Failed to create dummy image: {e}")
             dummy_image_path = None


    test_cases = [
        TestCase(id="q1", key="What is the capital of France?", value="Paris", image_path_or_url=None),
        TestCase(id="q2", key="Summarize the plot of the movie 'Inception'.", value="A thief steals information by entering people's dreams.", image_path_or_url=None),
    ]
    if dummy_image_path:
         test_cases.append(TestCase(id="img1", key=f"Describe this image.", value="Blue rectangle with yellow text 'Test Img'", image_path_or_url=dummy_image_path))
    else:
         logger.warning("Skipping multimodal test case in CLI as dummy image could not be created.")


    # --- Execution ---
    try:
        # Instantiate the tester
        tester = ModelTester(
            champion_endpoint=champion_model,
            challenger_endpoint=challenger_model,
            judge_endpoint=judge_model,
            model_prompt_template=model_prompt,
            judge_prompt_template=LMJudge.DEFAULT_EVALUATION_PROMPT # Use default judge prompt
        )

        logger.info(f"Running CLI test with {len(test_cases)} test cases...")
        # Run the test
        results = tester.run_test(
            test_cases,
            batch_size=2,
            batch_retry_attempts=1,
            batch_backoff_factor=2.0,
            batch_max_wait=30,
            batch_retry_trigger_strings=["rate limit", "error", "timeout"]
        )

        # --- Output Results ---
        logger.info("Test completed. Final Results:")

        # Use the formatter function
        summary_output = format_summary_output(results.get("summary", {}))
        print("\n" + summary_output)


        # Optionally save full results to JSON
        results_filename = f"cli_results_{time.strftime('%Y%m%d-%H%M%S')}.json"
        try:
            # Need to ensure results are serializable (dataclasses might need conversion)
            # The aggregator already converts raw evals to dicts. Summary should be fine.
            with open(results_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Full results saved to {results_filename}")
            print(f"\nFull results saved to: {results_filename}")
        except TypeError as e:
             logger.error(f"Failed to save results to JSON due to serialization issue: {e}")
             print(f"\nWarning: Could not save full results to JSON: {e}")
        except Exception as e:
            logger.error(f"Failed to save results to JSON: {e}")
            print(f"\nWarning: Could not save full results to JSON: {e}")


    except Exception as e:
        logger.exception("An error occurred during the CLI execution.")
        print(f"\nAn error occurred during CLI execution: {e}")
    finally:
        # Clean up dummy image if created and path exists
        if dummy_image_path and os.path.exists(dummy_image_path):
             try:
                 os.remove(dummy_image_path)
                 logger.info(f"Removed dummy image file: {dummy_image_path}")
             except Exception as e:
                 logger.warning(f"Could not remove dummy image file {dummy_image_path}: {e}")

# ==============================================================================
# Main Execution Logic
# ==============================================================================

def main():
    """Main function to parse arguments and run either CLI or UI."""
    # Basic argument parsing: run CLI test by default, or launch UI with --ui flag
    import argparse
    parser = argparse.ArgumentParser(description="Model A/B Testing Tool")
    parser.add_argument("--ui", action="store_true", help="Launch the Gradio web UI instead of running the CLI test.")
    args = parser.parse_args()

    if args.ui:
        logger.info("Launching Gradio UI...")
        iface = create_ui()
        if iface:
             # Add share=True for public link if needed, auth=("user", "pass") for security
             # Add server_name="0.0.0.0" to listen on all interfaces if running in Docker/remote
             iface.launch(share=True)
        else:
             logger.error("Failed to create Gradio UI.")
             print("Error: Could not create the Gradio UI.")
    else:
        # Set up signal handler for CLI stop (Ctrl+C)
        def signal_handler(sig, frame):
            global STOP_REQUESTED
            if not STOP_REQUESTED:
                print("\nCtrl+C detected. Requesting stop after current batch...")
                logger.warning("Stop requested via Ctrl+C.")
                STOP_REQUESTED = True
            else:
                # Allow force exit on second Ctrl+C
                print("\nCtrl+C detected again. Forcing exit.")
                logger.error("Forced exit via second Ctrl+C.")
                sys.exit(1)

        signal.signal(signal.SIGINT, signal_handler)

        # Run the command-line test
        run_cli_test()


if __name__ == "__main__":
    # Ensure we are running in the correct virtual environment
    # ensure_venv() will handle creation, installation, and re-execution if necessary.
    # If ensure_venv() returns True, it means we are now in the correct venv.
    if ensure_venv():
        # Now that we are confirmed to be in the venv, execute the main logic
        main()
    # If ensure_venv() returned False (or exited), the script either failed or restarted itself.

