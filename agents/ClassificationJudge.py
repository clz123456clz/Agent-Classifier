from __future__ import annotations

from agents.Language_Agent import LanguageModel
from agents.Memory import Memory
from agents.prompts.classification import JUDGE_SYSTEM_PROMPT, format_classifier_context
from agents.prompts.judge import JUDGE_DECISION_PROMPT, JUDGE_REFLECT_PROMPT
from agents.schemas import LabelPrediction, WeiboRecord, extract_json_object


class ClassificationJudge:
    """Step 4: reflection-style adjudication for disagreement or low-confidence cases."""

    def __init__(self, llm: LanguageModel, *, max_iterations: int = 1):
        self.llm = llm
        self.max_iterations = max_iterations

    def adjudicate(
        self,
        record: WeiboRecord,
        qwen: LabelPrediction,
        verifier: LabelPrediction,
    ) -> LabelPrediction:
        memory = Memory()
        critique = ""
        classifier_a_block = format_classifier_context(
            name="Classifier A (Qwen initial label)",
            label=qwen.label,
            rationale=qwen.rationale,
            cached_without_rationale=qwen.cached_without_rationale,
        )
        classifier_b_block = format_classifier_context(
            name="Classifier B (Independent verifier)",
            label=verifier.label,
            rationale=verifier.rationale,
            cached_without_rationale=False,
        )

        for _ in range(self.max_iterations):
            reflect_prompt = JUDGE_REFLECT_PROMPT.format(
                weibo_text=record.weibo_text,
                classifier_a_block=classifier_a_block,
                classifier_b_block=classifier_b_block,
            )
            critique = self._call_llm(reflect_prompt)
            memory.add_record("reflection", critique)

        decision_prompt = JUDGE_DECISION_PROMPT.format(
            weibo_text=record.weibo_text,
            classifier_a_block=classifier_a_block,
            classifier_b_block=classifier_b_block,
            critique=critique or "No critique available.",
        )
        decision_text = self._call_llm(decision_prompt)
        memory.add_record("decision", decision_text)

        try:
            payload = extract_json_object(decision_text)
            prediction = LabelPrediction.from_llm_json(
                payload,
                model=self.llm.model,
                source="judge",
            )
        except (ValueError, TypeError):
            prediction = LabelPrediction(
                label=verifier.label if verifier.label != qwen.label else qwen.label,
                rationale=f"Judge parse failed, fallback used. Raw: {decision_text[:200]}",
                model=self.llm.model,
                source="judge",
            )

        setattr(prediction, "trace", {"critique": critique, "decision_raw": decision_text})
        return prediction

    def _call_llm(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self.llm.think(messages=messages) or ""
