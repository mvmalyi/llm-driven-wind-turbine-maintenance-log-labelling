"""
Schema-agnostic OpenAI API client module.

This module provides the AsyncOpenAIClient class, a robust wrapper for sending
formatted prompts asynchronously. It enforces JSON schema outputs, manages
concurrency, tracks token usage, and raises custom exceptions upon failure.
"""

import asyncio
import json
import logging
import os
import random
from typing import Any, Dict, Optional, Tuple

from jsonschema import Draft202012Validator, SchemaError, ValidationError
from openai import AsyncOpenAI, OpenAIError

from src.exceptions import ConfigurationError, LLMAPIError
from src.config_manager import LLMConfig


# Configure basic logging for the module
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
logger: logging.Logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs from the underlying library
logging.getLogger("httpx").setLevel(logging.WARNING)


class AsyncOpenAIClient:
    """
    An asynchronous client wrapper for the OpenAI API configured for structured outputs,
    including retry, backoff, and timeout handling for transient failures.
    
    Attributes:
        config (LLMConfig): The configuration object containing model parameters.
        api_key (str): The OpenAI API key.
        client (AsyncOpenAI): The instantiated asynchronous OpenAI client.
        semaphore (asyncio.Semaphore): Controls the maximum number of concurrent requests.
        total_prompt_tokens (int): The cumulative count of input tokens used.
        total_completion_tokens (int): The cumulative count of output tokens generated.
    """

    def __init__(self, config: LLMConfig, api_key: Optional[str] = None) -> None:
        """
        Initialise the asynchronous OpenAI client.

        Args:
            config (LLMConfig): Configuration object detailing model and concurrency settings.
            api_key (Optional[str]): The OpenAI API key. Defaults to the environment variable.
            
        Raises:
            ConfigurationError: If no API key is provided or found in the environment.
        """
        self.config: LLMConfig = config
        self.api_key: str = api_key or os.getenv("OPENAI_API_KEY", "")
        
        if not self.api_key:
            raise ConfigurationError(
                "OpenAI API key must be provided or set as an environment variable."
            )
        
        self.client: AsyncOpenAI = AsyncOpenAI(api_key=self.api_key)
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Token tracking with an asyncio lock for thread safety
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    def _get_status_code(self, error: Exception) -> Optional[int]:
        """Extract the HTTP status code from OpenAI errors when available."""
        return getattr(error, "status_code", None)

    def _is_retryable_error(self, error: Exception, status_code: Optional[int]) -> bool:
        """Determine whether an error should be retried based on type or status code."""
        if status_code in self.config.retryable_status_codes:
            return True

        retryable_types = {
            "RateLimitError",
            "APITimeoutError",
            "APIConnectionError",
            "InternalServerError",
            "ServiceUnavailableError",
            "TimeoutError"
        }
        return type(error).__name__ in retryable_types

    def _compute_backoff(self, attempt: int) -> float:
        """Compute exponential backoff with jitter for retries."""
        base_delay = self.config.initial_backoff_seconds * (self.config.backoff_multiplier ** attempt)
        capped_delay = min(base_delay, self.config.max_backoff_seconds)
        jitter = random.uniform(0, self.config.jitter_seconds)
        return capped_delay + jitter

    def _build_response_validator(self, schema: Dict[str, Any]) -> Draft202012Validator:
        """Validate the outgoing JSON schema and build a local response validator."""
        payload_schema = schema.get("schema", schema)
        try:
            Draft202012Validator.check_schema(payload_schema)
        except SchemaError as exc:
            raise ConfigurationError(
                f"Invalid structured-output JSON schema '{schema.get('name', 'structured_output')}': {exc.message}"
            ) from exc
        return Draft202012Validator(payload_schema)

    def _format_validation_error(self, error: ValidationError) -> str:
        """Create a concise validation error message with the failing JSON path."""
        path = ".".join(str(part) for part in error.absolute_path)
        location = path or "<root>"
        return f"{location}: {error.message}"

    async def generate_structured_output(
        self, 
        system_prompt: str, 
        user_message: str, 
        schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Asynchronously send a request to the OpenAI API and enforce a JSON schema output.
        Retries transient failures using exponential backoff based on LLMConfig settings.

        Args:
            system_prompt (str): The system prompt defining the persona and instructions.
            user_message (str): The formatted data or query to be processed.
            schema (Dict[str, Any]): The JSON schema defining the required output structure.

        Returns:
            Tuple[Dict[str, Any], Dict[str, int]]: A tuple containing the parsed JSON 
            dictionary and a dictionary of the token usage for this specific call.
            
        Raises:
            LLMAPIError: If the API call fails, times out, or returns invalid JSON.
        """
        validator = self._build_response_validator(schema)

        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "developer", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema.get("name", "structured_output"),
                                "strict": True,
                                "schema": schema.get("schema", schema)
                            }
                        },
                        reasoning_effort=self.config.reasoning_effort,
                        timeout=self.config.timeout_seconds,
                        # NOTE: gpt-5.4 does not use temperature/seed controls.
                        # Uncomment the following lines for models that support these settings.
                        # temperature=0,
                        # seed=12345,
                    )

                # Safely extract usage statistics
                usage_stats = {"input": 0, "output": 0}
                if response.usage:
                    usage_stats["input"] = response.usage.prompt_tokens
                    usage_stats["output"] = response.usage.completion_tokens

                    async with self._lock:
                        self.total_prompt_tokens += usage_stats["input"]
                        self.total_completion_tokens += usage_stats["output"]

                raw_content = response.choices[0].message.content
                if not raw_content:
                    raise LLMAPIError("Received an empty response payload from the API.")

                # Clean potential markdown formatting just in case
                clean_content = raw_content.strip()
                if clean_content.startswith("```"):
                    clean_content = clean_content.strip("`").strip()
                    if clean_content.startswith("json"):
                        clean_content = clean_content[4:].strip()

                parsed_json = json.loads(clean_content)
                try:
                    validator.validate(parsed_json)
                except ValidationError as e:
                    if attempt < self.config.max_retries:
                        delay = self._compute_backoff(attempt)
                        logger.warning(
                            "Local JSON schema validation failed (attempt %s/%s). "
                            "Retrying in %.2fs: %s",
                            attempt + 1,
                            self.config.max_retries + 1,
                            delay,
                            self._format_validation_error(e)
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise LLMAPIError(
                        "Structured response failed local JSON schema validation: "
                        f"{self._format_validation_error(e)}"
                    ) from e
                return parsed_json, usage_stats

            except json.JSONDecodeError as e:
                if attempt < self.config.max_retries:
                    delay = self._compute_backoff(attempt)
                    logger.warning(
                        "JSON parsing error encountered (attempt %s/%s). Retrying in %.2fs: %s",
                        attempt + 1,
                        self.config.max_retries + 1,
                        delay,
                        e
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error("JSON Parsing Error encountered, %s", e)
                raise LLMAPIError(f"Failed to parse the structured JSON response: {str(e)}") from e
            except (OpenAIError, asyncio.TimeoutError) as e:
                status_code = self._get_status_code(e)
                retryable = self._is_retryable_error(e, status_code)
                if retryable and attempt < self.config.max_retries:
                    delay = self._compute_backoff(attempt)
                    logger.warning(
                        "Transient OpenAI error encountered (attempt %s/%s). Retrying in %.2fs: %s",
                        attempt + 1,
                        self.config.max_retries + 1,
                        delay,
                        e
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.error("OpenAI API Error encountered, %s", e)
                raise LLMAPIError(f"OpenAI API Error: {str(e)}", status_code=status_code) from e
            except LLMAPIError:
                raise
            except Exception as e:
                logger.error("Unexpected error encountered in API client, %s", e)
                raise LLMAPIError(f"An unexpected error occurred during the API call: {str(e)}") from e

    def get_total_cost(self) -> float:
        """
        Calculate the total financial cost of all API calls made by this client instance.

        Returns:
            float: The estimated total cost in USD.
        """
        return self.config.calculate_cost(
            input_tokens=self.total_prompt_tokens,
            output_tokens=self.total_completion_tokens
        )
