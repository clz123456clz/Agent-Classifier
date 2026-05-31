from __future__ import annotations

from agents.Language_Agent import LanguageModel
from agents.prompts.classification import CLASSIFICATION_SYSTEM_PROMPT, build_single_classify_user_prompt
from agents.schemas import LabelPrediction, WeiboRecord, extract_json_object, normalize_label


class InitialLabeler:
    """Optional Step 1: run Qwen to produce an initial label."""

    def __init__(self, llm: LanguageModel, *, max_text_chars: int = 900):
        self.llm = llm
        self.max_text_chars = max_text_chars

    def label(self, record: WeiboRecord) -> LabelPrediction:
        messages = [
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_single_classify_user_prompt(
                    record.weibo_text,
                    max_text_chars=self.max_text_chars,
                ),
            },
        ]
        response = self.llm.think(messages=messages) or ""
        try:
            payload = extract_json_object(response)
            return LabelPrediction.from_llm_json(
                payload,
                model=self.llm.model,
                source="initial",
            )
        except (ValueError, TypeError):
            return LabelPrediction(
                label=6,
                rationale=f"Failed to parse Qwen response: {response[:200]}",
                model=self.llm.model,
                source="initial",
            )

    def label_from_record_or_run(self, record: WeiboRecord) -> LabelPrediction:
        if record.qwen_type is not None:
            return LabelPrediction.from_qwen_type(record.qwen_type)
        return self.label(record)
