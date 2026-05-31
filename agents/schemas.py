from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


VALID_LABELS = frozenset(range(7))


def extract_json_object(content: str) -> dict[str, Any]:
    """Parse JSON from LLM output, tolerating markdown fences."""
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(content[start : end + 1])


def normalize_label(value: Any, default: int = 6) -> int:
    try:
        label = int(value)
    except (TypeError, ValueError):
        return default
    return label if label in VALID_LABELS else default


def record_id_from_raw(obj: dict[str, Any], fallback: str) -> str:
    raw_id = obj.get("_id")
    if isinstance(raw_id, dict) and "$oid" in raw_id:
        return str(raw_id["$oid"])
    if raw_id is not None:
        return str(raw_id)
    return fallback


@dataclass
class WeiboRecord:
    record_id: str
    weibo_text: str
    qwen_type: Optional[int] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, obj: dict[str, Any], *, line_no: int) -> WeiboRecord:
        return cls(
            record_id=record_id_from_raw(obj, fallback=f"line-{line_no}"),
            weibo_text=str(obj.get("weibo_text") or ""),
            qwen_type=obj.get("type"),
            raw=obj,
        )


@dataclass
class LabelPrediction:
    label: int
    rationale: str = ""
    model: str = ""
    source: str = ""
    cached_without_rationale: bool = False

    @classmethod
    def from_qwen_type(cls, label: int, model: str = "qwen-cached") -> LabelPrediction:
        return cls(
            label=normalize_label(label),
            rationale="",
            model=model,
            source="initial",
            cached_without_rationale=True,
        )

    @classmethod
    def from_llm_json(
        cls,
        payload: dict[str, Any],
        *,
        model: str,
        source: str,
        default: int = 6,
    ) -> LabelPrediction:
        label = normalize_label(payload.get("type", payload.get("label", default)), default=default)
        rationale = str(payload.get("rationale") or payload.get("reason") or "").strip()
        return cls(label=label, rationale=rationale, model=model, source=source)


@dataclass
class AgreementResult:
    agree: bool
    needs_judge: bool
    reason: str


@dataclass
class PipelineResult:
    record_id: str
    weibo_text: str
    qwen_label: LabelPrediction
    verifier_label: Optional[LabelPrediction]
    final_label: int
    confidence: float
    needs_manual_review: bool
    agreement: str
    judge_label: Optional[LabelPrediction] = None
    trace: dict[str, Any] = field(default_factory=dict)

    def to_output_fields(self) -> dict[str, Any]:
        fields = {
            "type_qwen": self.qwen_label.label,
            "type_verifier": self.verifier_label.label if self.verifier_label else None,
            "type_final": self.final_label,
            "confidence": round(self.confidence, 4),
            "needs_manual_review": self.needs_manual_review,
            "pipeline_agreement": self.agreement,
        }
        if self.judge_label:
            fields["type_judge"] = self.judge_label.label
        if self.qwen_label.rationale:
            fields["qwen_rationale"] = self.qwen_label.rationale
        if self.verifier_label and self.verifier_label.rationale:
            fields["verifier_rationale"] = self.verifier_label.rationale
        if self.judge_label and self.judge_label.rationale:
            fields["judge_rationale"] = self.judge_label.rationale
        return fields

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
