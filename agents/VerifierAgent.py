from __future__ import annotations

import ast
import json
import re
from typing import Any, Literal

from agents.Language_Agent import LanguageModel
from agents.prompts.classification import VERIFIER_SYSTEM_PROMPT, build_verifier_question
from agents.prompts.verifier import VERIFIER_EXECUTOR_PROMPT, VERIFIER_PLANNER_PROMPT
from agents.schemas import LabelPrediction, WeiboRecord, extract_json_object

VerifierMode = Literal["single", "plan"]


def parse_plan_from_response(response_text: str) -> list[str]:
    """Parse a plan list from LLM output with or without markdown fences."""
    text = (response_text or "").strip()
    if not text:
        return []

    def normalize_plan(value: Any) -> list[str]:
        if not isinstance(value, list) or not value:
            return []
        steps = [str(item).strip() for item in value if str(item).strip()]
        return steps

    def try_parse(candidate: str) -> list[str]:
        candidate = candidate.strip()
        if not candidate:
            return []

        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(candidate)
                plan = normalize_plan(parsed)
                if plan:
                    return plan
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
        return []

    # 1) Fenced blocks: ```python / ```json / ```
    for match in re.finditer(r"```(?:python|json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE):
        plan = try_parse(match.group(1))
        if plan:
            return plan

    # 2) Whole response
    plan = try_parse(text)
    if plan:
        return plan

    # 3) First [...] span in the response (common for bare JSON lists)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        plan = try_parse(text[start : end + 1])
        if plan:
            return plan

    return []


class VerifierAgent:
    """Step 2: independent verification (single-call or Plan-and-Solve)."""

    def __init__(
        self,
        llm: LanguageModel,
        *,
        max_text_chars: int = 900,
        mode: VerifierMode = "single",
    ):
        self.llm = llm
        self.max_text_chars = max_text_chars
        self.mode = mode

    def verify(self, record: WeiboRecord) -> LabelPrediction:
        if self.mode == "single":
            return self._verify_single(record)
        return self._verify_plan_and_solve(record)

    def _verify_single(self, record: WeiboRecord) -> LabelPrediction:
        """One LLM call — suitable for low daily quota APIs."""
        user_prompt = build_verifier_question(record.weibo_text, max_text_chars=self.max_text_chars)
        response_text = self._call_llm(user_prompt) or ""
        prediction = self._parse_final_label(response_text)
        prediction.model = self.llm.model
        prediction.source = "verifier"
        setattr(prediction, "trace", {"mode": "single", "response": response_text})
        return prediction

    def _verify_plan_and_solve(self, record: WeiboRecord) -> LabelPrediction:
        question = build_verifier_question(record.weibo_text, max_text_chars=self.max_text_chars)
        plan = self._plan(question)
        trace: dict[str, Any] = {"mode": "plan", "plan": plan, "steps": []}

        if not plan:
            return LabelPrediction(
                label=6,
                rationale="Verifier failed to generate a valid plan.",
                model=self.llm.model,
                source="verifier",
            )

        history = ""
        final_response = ""
        for i, step in enumerate(plan):
            prompt = VERIFIER_EXECUTOR_PROMPT.format(
                question=question,
                plan=plan,
                history=history if history else "None",
                current_step=step,
            )
            response_text = self._call_llm(prompt) or ""
            history += f"Step {i + 1}: {step}\nResult: {response_text}\n\n"
            trace["steps"].append({"step": step, "result": response_text})
            final_response = response_text

        prediction = self._parse_final_label(final_response)
        prediction.model = self.llm.model
        prediction.source = "verifier"
        prediction.rationale = prediction.rationale or "Plan-and-Solve verification completed."
        setattr(prediction, "trace", trace)
        return prediction

    def _plan(self, question: str) -> list[str]:
        prompt = VERIFIER_PLANNER_PROMPT.format(question=question)
        response_text = self._call_llm(prompt) or ""
        plan = parse_plan_from_response(response_text)
        if not plan:
            print(
                "⚠️ Failed to parse verifier plan. Raw response preview:\n"
                f"{response_text[:500]}",
                flush=True,
            )
        return plan

    def _parse_final_label(self, response_text: str) -> LabelPrediction:
        try:
            payload = extract_json_object(response_text)
            return LabelPrediction.from_llm_json(payload, model=self.llm.model, source="verifier")
        except (ValueError, TypeError):
            return LabelPrediction(
                label=6,
                rationale=f"Failed to parse verifier JSON: {response_text[:200]}",
                model=self.llm.model,
                source="verifier",
            )

    def _call_llm(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self.llm.think(messages=messages) or ""
