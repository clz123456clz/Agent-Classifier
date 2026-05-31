from __future__ import annotations

from typing import Literal

from agents.schemas import AgreementResult, LabelPrediction

JudgePolicy = Literal["disagree_only", "full"]

# Pairs that are often confused and should trigger judge even if labels match weakly.
AMBIGUOUS_PAIRS = frozenset(
    {
        (0, 1),
        (1, 6),
        (3, 5),
        (3, 4),
        (0, 3),
        (1, 3),
        (2, 6),
        (5, 6),
    }
)


class AgreementChecker:
    """Step 3: deterministic agreement and low-confidence rules."""

    def __init__(
        self,
        *,
        judge_policy: JudgePolicy = "disagree_only",
        low_confidence_labels: set[int] | None = None,
        force_judge_on_ambiguous_pair: bool = True,
    ):
        self.judge_policy = judge_policy
        self.low_confidence_labels = low_confidence_labels or {2, 5}
        self.force_judge_on_ambiguous_pair = force_judge_on_ambiguous_pair

    def check(self, qwen: LabelPrediction, verifier: LabelPrediction) -> AgreementResult:
        agree = qwen.label == verifier.label
        reasons: list[str] = []

        if not agree:
            reasons.append(f"label mismatch: qwen={qwen.label}, verifier={verifier.label}")

        if self.judge_policy == "full":
            if qwen.label in self.low_confidence_labels:
                reasons.append(f"qwen label {qwen.label} is boundary/low-confidence")
            if verifier.label in self.low_confidence_labels:
                reasons.append(f"verifier label {verifier.label} is boundary/low-confidence")

            pair = tuple(sorted((qwen.label, verifier.label)))
            if self.force_judge_on_ambiguous_pair and pair in AMBIGUOUS_PAIRS:
                reasons.append(f"ambiguous label pair {pair}")

            for pred, name in ((qwen, "qwen"), (verifier, "verifier")):
                text = (pred.rationale or "").lower()
                if any(token in text for token in ("uncertain", "maybe", "可能", "不确定", "ambiguous")):
                    reasons.append(f"{name} rationale indicates uncertainty")

        needs_judge = bool(reasons)
        reason = "; ".join(reasons) if reasons else "labels agree and no judge rule triggered"
        return AgreementResult(agree=agree, needs_judge=needs_judge, reason=reason)
