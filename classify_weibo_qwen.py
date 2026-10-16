#!/usr/bin/env python3
"""
Classify Weibo NDJSON records with Qwen based on each record's `weibo_text`.

The script appends a numeric `type` key to every original JSON object:
  0-2: urban light-pollution related posts
  3-5: insomnia/sleep-disorder related posts
  6: unrelated posts

Set your API key with:
  export QWEN_API_KEY="your_api_key_here"

Example:
  python classify_weibo_qwen.py \
    --input weibo_light_pollution_high_recall_prefecture_plus.json \
    --output weibo_light_pollution_high_recall_prefecture_plus_typed.json \
    --model qwen-plus \
    --batch-size 200 \
    --resume

## validation test:
  python3 classify_weibo_qwen.py \
    --input weibo_light_pollution_high_recall_prefecture_plus.json \
    --output test_typed.json \
    --limit 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


#DEFAULT_API_URL = "https://api.zetatechs.com/v1/chat/completions"
#DEFAULT_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
#DEFAULT_API_URL = "https://api.zetatechs.online/v1/chat/completions"
#DEFAULT_API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_API_URL = "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions"
#DEFAULT_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
#DEFAULT_API_URL = "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1"

SYSTEM_PROMPT = """You are a careful Chinese social-media text classifier.

Task:
Classify each Weibo post using ONLY the value of `weibo_text`.
Ignore metadata such as city, location, user name, source, longitude, and latitude.

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
{
  "results": [
    {"id": 0, "type": 6},
    {"id": 1, "type": 3}
  ]
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify Weibo NDJSON records with Qwen and append a numeric type key."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("weibo_light_pollution_high_recall_prefecture_plus.json"),
        help="Input NDJSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NDJSON file. Default: <input stem>_typed.json",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("QWEN_API_KEY"),
        help="Qwen/DashScope API key (default: the QWEN_API_KEY environment variable).",
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="OpenAI-compatible API URL.")
    parser.add_argument("--model", default="qwen-plus", help="Qwen model name.")
    parser.add_argument("--batch-size", type=int, default=20, help="Records per API call.")
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=900,
        help="Truncate each weibo_text to this many characters before sending.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per failed API call.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between calls.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output file and skip already written records.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Classify only the first N remaining records, useful for testing.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path, skip: int = 0):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < skip:
                continue
            raw = line.strip()
            if not raw:
                continue
            yield json.loads(raw)


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    rows = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows += 1
    return rows


def build_user_prompt(batch: list[dict[str, Any]], max_text_chars: int) -> str:
    items = []
    for idx, obj in enumerate(batch):
        text = str(obj.get("weibo_text") or "")
        if len(text) > max_text_chars:
            text = text[:max_text_chars] + "..."
        items.append({"id": idx, "weibo_text": text})
    return (
        "Classify the following Weibo posts. Return JSON only, following the schema "
        "specified in the system message.\n\n"
        f"{json.dumps({'items': items}, ensure_ascii=False)}"
    )


def extract_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
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


def call_qwen(
    *,
    api_key: str,
    api_url: str,
    model: str,
    batch: list[dict[str, Any]],
    max_text_chars: int,
    temperature: float,
    max_retries: int,
) -> list[int]:
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(batch, max_text_chars)},
        ],
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        request = urllib.request.Request(api_url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            parsed = extract_json_object(content)
            results = parsed.get("results", [])
            type_by_id = {}
            for item in results:
                try:
                    item_id = int(item["id"])
                    item_type = int(item["type"])
                except (KeyError, TypeError, ValueError):
                    continue
                if item_type not in range(7):
                    item_type = 6
                type_by_id[item_id] = item_type
            return [type_by_id.get(i, 6) for i in range(len(batch))]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            wait = min(2**attempt, 30)
            print(
                f"API call failed (attempt {attempt + 1}/{max_retries}): {exc}; retry in {wait}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)

    raise RuntimeError(f"Qwen API failed after {max_retries} retries: {last_error}")


def write_batch(outf, batch: list[dict[str, Any]], labels: list[int]) -> None:
    for obj, label in zip(batch, labels):
        obj["type"] = int(label)
        outf.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    outf.flush()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Set QWEN_API_KEY or pass --api-key.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")

    output = args.output or args.input.with_name(args.input.stem + "_typed" + args.input.suffix)
    skip = count_jsonl_rows(output) if args.resume else 0
    mode = "a" if args.resume and skip else "w"
    processed = skip
    remaining_limit = args.limit

    if skip:
        print(f"Resume enabled: skipping {skip} already written rows.", flush=True)

    batch: list[dict[str, Any]] = []
    with output.open(mode, encoding="utf-8") as outf:
        for obj in iter_jsonl(args.input, skip=skip):
            if remaining_limit is not None and remaining_limit <= 0:
                break
            batch.append(obj)
            if remaining_limit is not None:
                remaining_limit -= 1
            if len(batch) < args.batch_size:
                continue

            labels = call_qwen(
                api_key=args.api_key,
                api_url=args.api_url,
                model=args.model,
                batch=batch,
                max_text_chars=args.max_text_chars,
                temperature=args.temperature,
                max_retries=args.max_retries,
            )
            write_batch(outf, batch, labels)
            processed += len(batch)
            print(f"Processed {processed} rows", flush=True)
            batch = []
            if args.sleep > 0:
                time.sleep(args.sleep)

        if batch:
            labels = call_qwen(
                api_key=args.api_key,
                api_url=args.api_url,
                model=args.model,
                batch=batch,
                max_text_chars=args.max_text_chars,
                temperature=args.temperature,
                max_retries=args.max_retries,
            )
            write_batch(outf, batch, labels)
            processed += len(batch)
            print(f"Processed {processed} rows", flush=True)

    print(f"Done: {output}", flush=True)


if __name__ == "__main__":
    main()
