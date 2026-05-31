from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineStats:
    """Accumulates agreement stats between Qwen initial type and pipeline final type."""

    total: int = 0
    same_count: int = 0
    diff_count: int = 0
    diff_pairs: Counter[tuple[int, int]] = field(default_factory=Counter)

    def record(self, qwen_type: int, final_type: int) -> None:
        self.total += 1
        if qwen_type == final_type:
            self.same_count += 1
        else:
            self.diff_count += 1
            self.diff_pairs[(qwen_type, final_type)] += 1

    @property
    def same_pct(self) -> float:
        return (self.same_count / self.total * 100.0) if self.total else 0.0

    @property
    def diff_pct(self) -> float:
        return (self.diff_count / self.total * 100.0) if self.total else 0.0

    def top_diff_pairs(self, n: int = 5) -> list[dict[str, Any]]:
        rows = []
        for (qwen_type, final_type), count in self.diff_pairs.most_common(n):
            rows.append(
                {
                    "type_qwen": qwen_type,
                    "type_final": final_type,
                    "count": count,
                    "pct_of_total": round(count / self.total * 100.0, 2) if self.total else 0.0,
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "same_count": self.same_count,
            "same_pct": round(self.same_pct, 2),
            "diff_count": self.diff_count,
            "diff_pct": round(self.diff_pct, 2),
            "top_diff_pairs": self.top_diff_pairs(5),
        }

    def print_summary(self) -> None:
        print("\n=== Pipeline Stats (type_qwen vs type_final) ===")
        print(f"Total processed: {self.total}")
        print(f"Same:  {self.same_count} ({self.same_pct:.2f}%)")
        print(f"Diff:  {self.diff_count} ({self.diff_pct:.2f}%)")
        if self.diff_pairs:
            print("Top differing (type_qwen -> type_final) pairs:")
            for row in self.top_diff_pairs(5):
                print(
                    f"  {row['type_qwen']} -> {row['type_final']}: "
                    f"{row['count']} ({row['pct_of_total']:.2f}%)"
                )
        else:
            print("No differing pairs.")
        print("================================================\n")


def reservoir_sample_jsonl(
    path: Path,
    sample_size: int,
    *,
    seed: int | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    """Uniformly random sample of `sample_size` NDJSON records (reservoir sampling)."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    rng = random.Random(seed)
    reservoir: list[tuple[int, dict[str, Any]]] = []
    seen = 0

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append((line_no, obj))
                continue
            replace_idx = rng.randrange(seen)
            if replace_idx < sample_size:
                reservoir[replace_idx] = (line_no, obj)

    if seen < sample_size:
        raise ValueError(
            f"Requested random sample of {sample_size}, but input only has {seen} records."
        )
    return reservoir


def write_stats_json(path: Path, stats: PipelineStats, *, meta: dict[str, Any] | None = None) -> None:
    payload = {"meta": meta or {}, "stats": stats.to_dict()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
