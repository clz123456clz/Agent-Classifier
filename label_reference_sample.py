#!/usr/bin/env python3
"""Label a sample of released posts with a strong reference model (OpenAI API).

The model sees the SAME seven-category instructions that produced the released Qwen
labels (SYSTEM_PROMPT of classify_weibo_qwen.py, Methods of the manuscript), one post
per request, with ONLY the released `weibo_text` (masked text, truncated to 900
characters as in the original run).  It never sees the Qwen label, so the reference
labels are blind.  Decoding is deterministic where the model allows it (temperature 0;
models that reject the parameter are called with their default).

Input : JSON Lines with `_id` and `weibo_text` (draw_reference_sample.py output, or
        any subset of the released file).
Output: JSON Lines, one line per post: _id, model, type_ref, rationale, finish_reason,
        usage.  Resumable: posts already present in the output are skipped.

Credentials: OPENAI_API_KEY in the environment (never written to disk by this script).

Usage:
  python3 label_reference_sample.py --input reference_sample_stratified.jsonl \
      --output reference_labels_gpt.jsonl --model gpt-5.5
  python3 label_reference_sample.py --list-models        # which gpt-5* models the key can use
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

MAX_TEXT_CHARS = 900

SYSTEM_PROMPT = """You are a careful Chinese social-media text classifier.

Task:
Classify the Weibo post using ONLY the value of `weibo_text`.
Ignore metadata such as city, location, user name, source, longitude, and latitude.
Placeholders such as [PLACE], [USER], @[USER], [PHONE], [EMAIL], [CONTACT], [NUMBER] or [SELF]
replace masked place names, mentions and contact details; treat them as neutral tokens.

Output labels:
0 = Light pollution - direct complaint or harm
    The post explicitly describes urban/artificial night light as excessive, disturbing,
    glaring, intrusive, harmful, sleep-disrupting, environmentally problematic, or as
    "光污染". Examples: excessive neon/signboard/street/building lights; lights shining
    into rooms; glare at night; complaints about lighting being too bright.

1 = Light pollution - urban night-light scene or strong artificial illumination
    The post mainly records or describes nighttime city lighting, "灯火通明", bright
    streets/buildings/skylines/LED screens/nightscape, or unusually strong artificial
    illumination. It may be neutral or aesthetic, but the core topic is city artificial
    lighting at night.

2 = Light pollution - governance, technology, ecology, measurement, or astronomy
    The post discusses management, policy, design, reduction, shielding, monitoring,
    research, ecological impact, astronomical observation, or technical aspects of
    artificial night light/light pollution.

3 = Insomnia - direct self-reported insomnia or inability to sleep
    The post directly says the author or another person cannot sleep, is sleepless,
    has insomnia, wakes too early, has difficulty falling asleep, or slept poorly.
    Examples: "失眠了", "睡不着", "一夜没睡", "难以入睡", "凌晨还醒着".

4 = Insomnia - chronic/clinical sleep disorder, treatment, medication, or help-seeking
    The post describes long-term/severe insomnia, sleep disorder, repeated episodes,
    medication/sleeping pills, hospital/doctor consultation, diagnosis, therapy, or
    explicit treatment attempts for insomnia/sleep problems.

5 = Insomnia - causes, consequences, or emotional/functional impact
    The post focuses on reasons or effects of insomnia/sleep disturbance, such as
    anxiety, stress, illness, caffeine, work shifts, exams, emotional distress, fatigue,
    daytime impairment, fear of health damage, or disrupted life due to poor sleep.

6 = Unrelated
    The post is not substantively about urban light pollution or insomnia/sleep disorder.
    This includes ordinary travel, food, scenery, celebrations, generic "耀眼" metaphors,
    sunshine/daytime brightness, general beauty of lights without urban night-light focus,
    and posts where keywords appear only in a user name, hashtag noise, quoted text, or
    irrelevant context.

Priority rules:
- If both light-pollution and insomnia signals appear, choose the most central topic.
- If a post clearly reports a person's insomnia experience, prefer labels 3-5 over
  light labels unless artificial light is the main cause/topic.
- Do not classify as light pollution merely because of "耀眼", "阳光", "刺眼", "灯光"
  if it is about daytime sunlight, photography, emotion, cosmetics, food, travel, or
  metaphor rather than urban/artificial night lighting.
- Do not classify as insomnia merely because "失眠" appears in a user name or unrelated
  account name; classify by the actual content meaning.

Return valid JSON only. No markdown, no comments.
The JSON schema must be:
{"type": <integer 0-6>, "rationale": "<one short sentence in English>"}
"""


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def parse_label(text: str) -> tuple[int | None, str]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        return None, ""
    try:
        label = int(obj.get("type"))
    except Exception:
        return None, str(obj.get("rationale", ""))
    return (label if 0 <= label <= 6 else None), str(obj.get("rationale", ""))


def call_model(client: OpenAI, model: str, text: str, reasoning_effort: str | None,
               retries: int) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "weibo_text:\n" + text},
    ]
    kwargs: dict = {"model": model, "messages": messages,
                    "response_format": {"type": "json_object"}}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    use_temperature = True
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            if use_temperature:
                resp = client.chat.completions.create(temperature=0, **kwargs)
            else:
                resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            usage = resp.usage.model_dump() if resp.usage else {}
            return {"content": choice.message.content, "finish_reason": choice.finish_reason,
                    "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                    "temperature": 0 if use_temperature else None}
        except Exception as e:  # noqa: BLE001 - we want to classify the failure
            msg = str(e)
            last_err = e
            if use_temperature and "temperature" in msg:
                use_temperature = False      # reasoning models accept only the default
                continue
            if "reasoning_effort" in msg and "reasoning_effort" in kwargs:
                kwargs.pop("reasoning_effort")
                continue
            if "response_format" in msg and "response_format" in kwargs:
                kwargs.pop("response_format")
                continue
            wait = min(60, 2 ** attempt)
            print(f"   retry {attempt + 1}/{retries} after error: {msg[:160]} (sleep {wait}s)", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"giving up: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("reference_sample_stratified.jsonl"))
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--reasoning-effort", default=None,
                    help="low / medium / high for reasoning models; omitted = model default")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8, help="parallel requests")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")
    client = OpenAI()

    if args.list_models:
        ids = sorted(m.id for m in client.models.list())
        print("\n".join(i for i in ids if i.startswith(("gpt-5", "o")) or "gpt" in i))
        return

    out = args.output or args.input.with_name(args.input.stem + f"_{args.model}.jsonl")
    done: set[str] = set()
    if out.exists():
        for r in iter_jsonl(out):
            if r.get("type_ref") is not None:
                done.add(r["_id"])
    rows, seen = [], set()
    for r in iter_jsonl(args.input):
        if r["_id"] in done or r["_id"] in seen:
            continue                      # resume, and label each distinct post once
        seen.add(r["_id"])
        rows.append(r)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(done)} already labelled, {len(rows)} to do -> {out}  (workers={args.workers})")

    def work(r: dict) -> dict:
        text = str(r.get("weibo_text") or "")
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS] + "..."
        res = call_model(client, args.model, text, args.reasoning_effort, args.retries)
        label, rationale = parse_label(res["content"])
        if label is None:
            # one blind retry for malformed output
            res = call_model(client, args.model, text, args.reasoning_effort, args.retries)
            label, rationale = parse_label(res["content"])
        return {"_id": r["_id"], "model": args.model, "type_ref": label, "rationale": rationale,
                "finish_reason": res["finish_reason"], "usage": res["usage"],
                "temperature": res["temperature"], "reasoning_effort": args.reasoning_effort,
                "raw": None if label is not None else res["content"]}

    # concurrency: the API is the bottleneck, so a handful of threads is enough
    n_ok = n_bad = 0
    tok_in = tok_out = 0
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, r) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_ok += rec["type_ref"] is not None
            n_bad += rec["type_ref"] is None
            tok_in += rec["usage"].get("prompt_tokens") or 0
            tok_out += rec["usage"].get("completion_tokens") or 0
            if i % 50 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}  ok={n_ok} unparsed={n_bad}  tokens in/out={tok_in:,}/{tok_out:,}"
                      f"  {time.time() - t0:.0f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
