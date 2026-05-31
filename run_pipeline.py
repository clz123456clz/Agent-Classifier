#!/usr/bin/env python3
"""
Run the multi-stage Weibo classification pipeline on Qwen-prelabeled NDJSON.

Default behavior reuses the existing `type` field as the Qwen initial label and
only runs verifier + agreement check + optional judge + confidence scoring.

Examples:
  # Low-quota default: 1 verifier call, judge only on mismatch
  python run_pipeline.py --input ... --output ... --limit 10

  # Full Plan-and-Solve + judge on boundary labels (many API calls)
  python run_pipeline.py --input ... --output ... \\
    --verifier-mode plan --judge-policy full --limit 10

  # Random sample + stats
  python run_pipeline.py --input ... --output ... \\
    --random-sample 50 --seed 42 --stats-output pipeline_stats.json

Environment variables:
  VERIFIER_MODEL_ID / VERIFIER_API_KEY / VERIFIER_BASE_URL  (required for verifier)
  JUDGE_MODEL_ID / JUDGE_API_KEY / JUDGE_BASE_URL          (optional, defaults to verifier)
  QWEN_MODEL_ID / QWEN_API_KEY / QWEN_BASE_URL              (only if --no-reuse-qwen-type)
  LLM_*                                                     (fallback if VERIFIER_* unset)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from agents import (
    AgreementChecker,
    ClassificationJudge,
    ConfidenceScorer,
    InitialLabeler,
    LanguageModel,
    PipelineOrchestrator,
    PipelineStats,
    VerifierAgent,
    reservoir_sample_jsonl,
    write_stats_json,
)


def load_env() -> None:
    here = Path(__file__).resolve().parent
    candidates = [
        here / ".env",
        here.parent / "Basic_structures" / ".env",
    ]
    for path in candidates:
        if path.exists():
            load_dotenv(path)
            print(f"Loaded env from {path}")
            return
    load_dotenv()


def build_llm(
    prefix: str,
    fallback_prefix: str | None = "LLM",
    *,
    min_interval_sec: float | None = None,
    max_retries: int | None = None,
) -> LanguageModel:
    def pick(name: str) -> str | None:
        value = os.getenv(f"{prefix}_{name}")
        if value:
            return value
        if fallback_prefix:
            return os.getenv(f"{fallback_prefix}_{name}")
        return None

    model = pick("MODEL_ID")
    api_key = pick("API_KEY")
    base_url = pick("BASE_URL")
    timeout = int(os.getenv(f"{prefix}_TIMEOUT") or os.getenv("LLM_TIMEOUT") or "120")

    if api_key and not api_key.startswith("sk-"):
        raise ValueError(
            f"{prefix}_API_KEY looks invalid (got {api_key!r}). "
            "Set the real OpenAI key, e.g. OPENAI_API_KEY=sk-... and "
            "VERIFIER_API_KEY=${OPENAI_API_KEY} in .env — do not use a placeholder name."
        )

    return LanguageModel(
        model=model,
        apiKey=api_key,
        baseUrl=base_url,
        timeout=timeout,
        min_interval_sec=min_interval_sec,
        max_retries=max_retries,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run verifier/judge pipeline on Qwen-prelabeled Weibo NDJSON.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("weibo_light_pollution_high_recall_prefecture_plus_typed.json"),
        help="Input NDJSON with existing Qwen `type` labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NDJSON. Default: <input stem>_verified.json",
    )
    parser.add_argument(
        "--reuse-qwen-type",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing `type` as Qwen initial label (default: true).",
    )
    parser.add_argument("--resume", action="store_true", help="Append and skip written rows.")
    parser.add_argument("--limit", type=int, default=None, help="Process only N records.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between records.")
    parser.add_argument(
        "--llm-interval",
        type=float,
        default=None,
        help="Minimum seconds between LLM calls (default: LLM_MIN_INTERVAL_SEC or 0). "
        "Use 21+ if you hit OpenAI free-tier RPM limits.",
    )
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=None,
        help="Max retries on 429 rate limit (default: LLM_MAX_RETRIES or 6).",
    )
    parser.add_argument("--max-text-chars", type=int, default=900, help="Truncate weibo_text.")
    parser.add_argument(
        "--manual-review-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold below which manual review is flagged.",
    )
    parser.add_argument(
        "--judge-iterations",
        type=int,
        default=1,
        help="Reflection iterations inside ClassificationJudge.",
    )
    parser.add_argument(
        "--verifier-mode",
        choices=("single", "plan"),
        default="single",
        help="Verifier strategy: single=1 API call (default), plan=Plan-and-Solve (~5+ calls).",
    )
    parser.add_argument(
        "--judge-policy",
        choices=("disagree_only", "full"),
        default="disagree_only",
        help="Judge trigger: disagree_only=only when qwen!=verifier (default), "
        "full=also boundary/ambiguous labels.",
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N records from input (uniform). Overrides --limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --random-sample (reproducible draws).",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Write type_qwen vs type_final stats JSON to this path.",
    )
    return parser.parse_args()


def main() -> None:
    load_env()
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    if args.random_sample is not None and args.resume:
        raise SystemExit("--random-sample cannot be used together with --resume.")

    output = args.output or args.input.with_name(args.input.stem + "_verified" + args.input.suffix)
    if args.random_sample is not None and args.output is None:
        output = output.with_name(f"{output.stem}_sample{args.random_sample}{output.suffix}")

    sampled_records = None
    if args.random_sample is not None:
        if args.random_sample <= 0:
            raise SystemExit("--random-sample must be positive.")
        sampled_records = reservoir_sample_jsonl(
            args.input,
            args.random_sample,
            seed=args.seed,
        )
        print(
            f"Random sample: {len(sampled_records)} records from {args.input} "
            f"(seed={args.seed})",
        )

    stats = PipelineStats()

    try:
        verifier_llm = build_llm(
            "VERIFIER",
            min_interval_sec=args.llm_interval,
            max_retries=args.llm_retries,
        )
        judge_llm = build_llm(
            "JUDGE",
            fallback_prefix="VERIFIER",
            min_interval_sec=args.llm_interval,
            max_retries=args.llm_retries,
        )
        # Share one client when verifier/judge use the same API account.
        if (
            judge_llm.model == verifier_llm.model
            and judge_llm._rate_key == verifier_llm._rate_key
        ):
            judge_llm = verifier_llm
        qwen_llm = build_llm("QWEN", fallback_prefix="LLM") if not args.reuse_qwen_type else None
    except ValueError as exc:
        raise SystemExit(f"LLM configuration error: {exc}") from exc

    initial_labeler = InitialLabeler(
        qwen_llm or verifier_llm,
        max_text_chars=args.max_text_chars,
    )
    verifier = VerifierAgent(
        verifier_llm,
        max_text_chars=args.max_text_chars,
        mode=args.verifier_mode,
    )
    judge = ClassificationJudge(judge_llm, max_iterations=args.judge_iterations)
    orchestrator = PipelineOrchestrator(
        initial_labeler=initial_labeler,
        verifier=verifier,
        agreement_checker=AgreementChecker(judge_policy=args.judge_policy),
        judge=judge,
        scorer=ConfidenceScorer(manual_review_threshold=args.manual_review_threshold),
        reuse_qwen_type=args.reuse_qwen_type,
        sleep_seconds=args.sleep,
    )

    print(f"Input:  {args.input}")
    print(f"Output: {output}")
    print(f"Reuse Qwen type: {args.reuse_qwen_type}")
    print(f"Verifier mode: {args.verifier_mode}")
    print(f"Judge policy: {args.judge_policy}")
    if args.verifier_mode == "single" and args.judge_policy == "disagree_only":
        print("Estimated LLM calls/record: ~1 (agree) or ~3 (disagree + judge)")
    elif args.verifier_mode == "plan" and args.judge_policy == "full":
        print("Estimated LLM calls/record: ~6-8 (plan+execute+judge)")

    processed = orchestrator.run(
        input_path=args.input,
        output_path=output,
        resume=args.resume,
        limit=args.limit if args.random_sample is None else None,
        sampled_records=sampled_records,
        stats=stats,
    )
    print(f"Done. Processed {processed} rows -> {output}", file=sys.stderr)

    stats.print_summary()
    if args.stats_output:
        write_stats_json(
            args.stats_output,
            stats,
            meta={
                "input": str(args.input),
                "output": str(output),
                "random_sample": args.random_sample,
                "seed": args.seed,
                "verifier_mode": args.verifier_mode,
                "judge_policy": args.judge_policy,
                "processed": processed,
            },
        )
        print(f"Stats written to {args.stats_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
