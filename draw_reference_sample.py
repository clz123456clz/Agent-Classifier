#!/usr/bin/env python3
"""Draw the stratified reference sample used for the strong-model label check.

A simple random sample of 1,000 posts holds very few posts of the rare categories 0 and
2. This script draws a second sample from the released dataset, stratified by the
released Qwen label (`type`) with (near-)equal allocation, so that every category has
about 143 posts. The draw is deterministic: within each stratum the posts are taken in
file order and `random.Random(SEED).sample` picks the indices, so anyone holding the
released file obtains the same sample (the `_id` list of reference_sample_labels.csv).

Output (JSON Lines, one line per sampled post):
  sample_index, _id, stratum (= released Qwen label), weibo_text (released, masked text),
  city, create_time, weight (N_k / n_k, the number of released posts each sampled post
  stands for).

Usage:
  python3 draw_reference_sample.py                      # 1,000 posts, seed 43
  python3 draw_reference_sample.py --n 700 --seed 43
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

RELEASE = Path("weibo_light_pollution_high_recall_prefecture_plus_typed.json")
OUT = Path("reference_sample_stratified.jsonl")
CATEGORIES = list(range(7))


def allocation(n_total: int, sizes: dict[int, int]) -> dict[int, int]:
    """Equal allocation; the remainder is taken from the largest strata first.

    A stratum can never receive more posts than it holds.
    """
    base, rem = divmod(n_total, len(CATEGORIES))
    alloc = {k: base for k in CATEGORIES}
    for k in sorted(CATEGORIES, key=lambda c: -sizes[c])[:rem]:
        alloc[k] += 1
    for k in CATEGORIES:
        alloc[k] = min(alloc[k], sizes[k])
    return alloc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, default=RELEASE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()

    by_cat: dict[int, list[dict]] = defaultdict(list)
    with args.release.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_cat[int(r["type"])].append(r)
    sizes = {k: len(by_cat[k]) for k in CATEGORIES}
    n_release = sum(sizes.values())
    alloc = allocation(args.n, sizes)

    rng = random.Random(args.seed)
    rows = []
    for k in CATEGORIES:
        idx = sorted(rng.sample(range(sizes[k]), alloc[k]))
        for i in idx:
            r = by_cat[k][i]
            rows.append({
                "_id": r["_id"],
                "stratum": k,
                "weibo_text": r["weibo_text"],
                "city": r.get("city"),
                "create_time": r.get("create_time"),
                "weight": sizes[k] / alloc[k],
            })
    for j, row in enumerate(rows, 1):
        row = {"sample_index": j, **row}
        rows[j - 1] = row

    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"released posts: {n_release:,}   seed: {args.seed}   sample: {len(rows)}")
    print("cat  released   share   sampled   weight(N_k/n_k)   sampling fraction")
    for k in CATEGORIES:
        print(f"{k:>3}  {sizes[k]:>8,}  {sizes[k]/n_release*100:5.2f}%  {alloc[k]:>7}  "
              f"{sizes[k]/alloc[k]:>15.2f}   {alloc[k]/sizes[k]*100:6.2f}%")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
