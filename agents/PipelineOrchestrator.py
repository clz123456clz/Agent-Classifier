from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

from agents.AgreementChecker import AgreementChecker
from agents.ClassificationJudge import ClassificationJudge
from agents.ConfidenceScorer import ConfidenceScorer
from agents.InitialLabeler import InitialLabeler
from agents.VerifierAgent import VerifierAgent
from agents.pipeline_stats import PipelineStats
from agents.schemas import PipelineResult, WeiboRecord


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def iter_jsonl(path: Path, *, skip: int = 0) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < skip:
                continue
            raw = line.strip()
            if not raw:
                continue
            yield line_no, json.loads(raw)


def is_pipeline_processed(obj: dict[str, Any]) -> bool:
    return "type_final" in obj and "pipeline_agreement" in obj


class PipelineOrchestrator:
    """End-to-end orchestration with resume support."""

    def __init__(
        self,
        *,
        initial_labeler: InitialLabeler,
        verifier: VerifierAgent,
        agreement_checker: AgreementChecker,
        judge: ClassificationJudge,
        scorer: ConfidenceScorer,
        reuse_qwen_type: bool = True,
        sleep_seconds: float = 0.0,
    ):
        self.initial_labeler = initial_labeler
        self.verifier = verifier
        self.agreement_checker = agreement_checker
        self.judge = judge
        self.scorer = scorer
        self.reuse_qwen_type = reuse_qwen_type
        self.sleep_seconds = sleep_seconds

    def process_one(self, record: WeiboRecord) -> PipelineResult:
        if self.reuse_qwen_type and record.qwen_type is not None:
            qwen = self.initial_labeler.label_from_record_or_run(record)
        else:
            qwen = self.initial_labeler.label(record)

        verifier = self.verifier.verify(record)
        verifier_trace = getattr(verifier, "trace", {})

        agreement = self.agreement_checker.check(qwen, verifier)
        judge = None
        trace: dict[str, Any] = {"verifier": verifier_trace}

        if agreement.needs_judge:
            judge = self.judge.adjudicate(record, qwen, verifier)
            final_label = judge.label
            agreement_status = "judge_resolved"
            trace["judge"] = getattr(judge, "trace", {})
        elif agreement.agree:
            final_label = qwen.label
            agreement_status = "agree"
        else:
            final_label = verifier.label
            agreement_status = "disagree_unresolved"

        return self.scorer.build_result(
            record_id=record.record_id,
            weibo_text=record.weibo_text,
            qwen=qwen,
            verifier=verifier,
            agreement=agreement,
            judge=judge,
            final_label=final_label,
            agreement_status=agreement_status,
            trace=trace,
        )

    def run(
        self,
        *,
        input_path: Path,
        output_path: Path,
        resume: bool = False,
        limit: int | None = None,
        sampled_records: list[tuple[int, dict[str, Any]]] | None = None,
        stats: PipelineStats | None = None,
    ) -> int:
        if sampled_records is not None:
            skip = 0
            mode = "w"
            record_source: Iterator[tuple[int, dict[str, Any]]] = iter(sampled_records)
            remaining = len(sampled_records)
        else:
            skip = count_jsonl_rows(output_path) if resume else 0
            mode = "a" if resume and skip else "w"
            record_source = iter_jsonl(input_path, skip=skip)
            remaining = limit

        processed = 0 if sampled_records is not None else skip

        if skip and sampled_records is None:
            print(f"Resume enabled: skipping {skip} already written rows.")

        with output_path.open(mode, encoding="utf-8") as outf:
            for line_no, obj in record_source:
                if remaining is not None and remaining <= 0:
                    break

                if sampled_records is None and resume and is_pipeline_processed(obj):
                    continue

                record = WeiboRecord.from_raw(obj, line_no=line_no)
                if self.reuse_qwen_type and record.qwen_type is None:
                    print(
                        f"Warning: record {record.record_id} has no existing `type`; "
                        "will call Qwen unless --no-reuse-qwen-type is set.",
                    )

                result = self.process_one(record)
                if stats is not None:
                    stats.record(result.qwen_label.label, result.final_label)

                output_obj = dict(obj)
                output_obj.update(result.to_output_fields())
                outf.write(json.dumps(output_obj, ensure_ascii=False, separators=(",", ":")) + "\n")
                outf.flush()

                processed += 1
                if remaining is not None:
                    remaining -= 1

                print(
                    f"[{processed}] {record.record_id} "
                    f"qwen={result.qwen_label.label} "
                    f"verifier={result.verifier_label.label if result.verifier_label else 'NA'} "
                    f"final={result.final_label} "
                    f"conf={result.confidence:.2f} "
                    f"review={result.needs_manual_review}",
                )

                if self.sleep_seconds > 0:
                    time.sleep(self.sleep_seconds)

        return processed
