from __future__ import annotations

from agents.schemas import AgreementResult, LabelPrediction, PipelineResult


class ConfidenceScorer:
    """Step 5: compute confidence and manual-review flag."""

    def __init__(
        self,
        *,
        manual_review_threshold: float = 0.7,
        boundary_labels: set[int] | None = None,
    ):
        self.manual_review_threshold = manual_review_threshold
        self.boundary_labels = boundary_labels or {2, 5}

    def build_result(
        self,
        *,
        record_id: str,
        weibo_text: str,
        qwen: LabelPrediction,
        verifier: LabelPrediction | None,
        agreement: AgreementResult,
        judge: LabelPrediction | None,
        final_label: int,
        agreement_status: str,
        trace: dict | None = None,
    ) -> PipelineResult:
        confidence = self.score(
            agreement=agreement,
            qwen=qwen,
            verifier=verifier,
            judge=judge,
            final_label=final_label,
            agreement_status=agreement_status,
        )
        needs_manual_review = self.needs_manual_review(
            confidence=confidence,
            agreement_status=agreement_status,
            final_label=final_label,
            qwen=qwen,
            verifier=verifier,
            judge=judge,
        )
        return PipelineResult(
            record_id=record_id,
            weibo_text=weibo_text,
            qwen_label=qwen,
            verifier_label=verifier,
            final_label=final_label,
            confidence=confidence,
            needs_manual_review=needs_manual_review,
            agreement=agreement_status,
            judge_label=judge,
            trace=trace or {},
        )

    def score(
        self,
        *,
        agreement: AgreementResult,
        qwen: LabelPrediction,
        verifier: LabelPrediction | None,
        judge: LabelPrediction | None,
        final_label: int,
        agreement_status: str,
    ) -> float:
        if agreement_status == "agree":
            base = 0.95
            if final_label in self.boundary_labels:
                base = 0.82
            return base

        if agreement_status == "judge_resolved" and judge is not None:
            if judge.label == qwen.label or (verifier and judge.label == verifier.label):
                return 0.78
            return 0.68

        if verifier is None:
            return 0.5

        if not agreement.agree:
            return 0.4

        return 0.75

    def needs_manual_review(
        self,
        *,
        confidence: float,
        agreement_status: str,
        final_label: int,
        qwen: LabelPrediction,
        verifier: LabelPrediction | None,
        judge: LabelPrediction | None,
    ) -> bool:
        if confidence < self.manual_review_threshold:
            return True
        if agreement_status == "disagree_unresolved":
            return True
        if final_label in self.boundary_labels:
            return True
        if verifier and qwen.label != verifier.label and judge is None:
            return True
        return False
