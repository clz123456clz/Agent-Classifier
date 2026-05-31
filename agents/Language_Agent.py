import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import APIStatusError, BadRequestError, OpenAI, RateLimitError

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class LanguageModel:
    """
    OpenAI-compatible LLM client with streaming, rate-limit throttling, and retries.
    """

    _shared_last_call_at: dict[str, float] = {}

    def __init__(
        self,
        model: str = None,
        apiKey: str = None,
        baseUrl: str = None,
        timeout: int = None,
        *,
        min_interval_sec: float | None = None,
        max_retries: int | None = None,
    ):
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        self.min_interval_sec = (
            float(min_interval_sec)
            if min_interval_sec is not None
            else float(os.getenv("LLM_MIN_INTERVAL_SEC", "0"))
        )
        self.max_retries = (
            int(max_retries)
            if max_retries is not None
            else int(os.getenv("LLM_MAX_RETRIES", "6"))
        )

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("Model ID, API key, and service address must be provided or defined in the .env file.")

        self._rate_key = f"{baseUrl}|{apiKey}"
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)
        env_temp = os.getenv("LLM_TEMPERATURE")
        self.default_temperature: Optional[float] = (
            float(env_temp) if env_temp not in (None, "") else None
        )

    def think(self, messages: List[Dict[str, str]], temperature: float | None = None) -> str:
        """Call the LLM with throttling and automatic retry on rate limits."""
        last_error: Optional[Exception] = None
        effective_temp = self.default_temperature if temperature is None else temperature
        omit_temperature = False

        for attempt in range(self.max_retries):
            self._throttle()
            print(f"🧠 Calling {self.model} model...")
            try:
                kwargs: dict = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                }
                if not omit_temperature and effective_temp is not None:
                    kwargs["temperature"] = effective_temp

                response = self.client.chat.completions.create(**kwargs)

                print("✅ Large language model response successful:")
                collected_content = []
                for chunk in response:
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
                    collected_content.append(content)
                print()
                return "".join(collected_content)

            except (RateLimitError, APIStatusError) as exc:
                last_error = exc
                if isinstance(exc, (BadRequestError, APIStatusError)) and exc.status_code == 400:
                    if not omit_temperature and "temperature" in str(exc).lower():
                        omit_temperature = True
                        print(
                            "ℹ️ Model rejected temperature parameter; "
                            "retrying with model default...",
                            flush=True,
                        )
                        continue
                if not self._should_retry(exc):
                    break
                wait = self._retry_wait_seconds(exc, attempt)
                print(
                    f"⏳ Rate limited (attempt {attempt + 1}/{self.max_retries}), "
                    f"waiting {wait:.1f}s before retry...",
                    flush=True,
                )
                time.sleep(wait)
            except Exception as exc:
                last_error = exc
                if not omit_temperature and "temperature" in str(exc).lower():
                    omit_temperature = True
                    print(
                        "ℹ️ Model rejected temperature parameter; "
                        "retrying with model default...",
                        flush=True,
                    )
                    continue
                print(f"❌ Error occurred when calling LLM API: {exc}")
                break

        print(f"❌ LLM call failed after retries: {last_error}")
        raise RuntimeError(f"LLM API call failed: {last_error}") from last_error

    def _throttle(self) -> None:
        if self.min_interval_sec <= 0:
            return
        last_call_at = self._shared_last_call_at.get(self._rate_key, 0.0)
        elapsed = time.monotonic() - last_call_at
        if elapsed < self.min_interval_sec:
            wait = self.min_interval_sec - elapsed
            print(f"⏳ Throttling: waiting {wait:.1f}s to respect RPM limits...", flush=True)
            time.sleep(wait)
        self._shared_last_call_at[self._rate_key] = time.monotonic()

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, RateLimitError):
            return True
        if isinstance(exc, APIStatusError) and exc.status_code == 429:
            return True
        message = str(exc).lower()
        return "rate_limit" in message or "rate limit" in message

    def _retry_wait_seconds(self, exc: Exception, attempt: int) -> float:
        message = str(exc)
        match = re.search(r"try again in (\d+(?:\.\d+)?)\s*s", message, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 1.0
        # Fallback: at least one full RPM window for low-tier limits.
        return max(self.min_interval_sec, min(2**attempt * 5, 60))


if __name__ == "__main__":
    try:
        llmClient = LanguageModel()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "Write a quicksort algorithm"},
        ]

        print("--- Calling LLM ---")
        responseText = llmClient.think(exampleMessages)
        print("\n\n--- Complete Model Response ---")
        print(responseText)

    except ValueError as e:
        print(e)
