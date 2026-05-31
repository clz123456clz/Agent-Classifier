from __future__ import annotations

import json
from typing import Any

# Shared by InitialLabeler, VerifierAgent, and ClassificationJudge.
LABEL_DEFINITIONS_AND_RULES = """
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
""".strip()

CLASSIFICATION_JSON_OUTPUT = """
Return valid JSON only. No markdown, no comments.
The JSON schema must be:
{
  "type": 6,
  "rationale": "brief reason in Chinese or English"
}
""".strip()

CLASSIFICATION_SYSTEM_PROMPT = (
    "You are a careful Chinese social-media text classifier.\n\n"
    f"{LABEL_DEFINITIONS_AND_RULES}\n\n"
    f"{CLASSIFICATION_JSON_OUTPUT}"
)

VERIFIER_SYSTEM_PROMPT = (
    "You are a careful Chinese social-media text classifier acting as an independent verifier.\n"
    "You must classify using ONLY the label definitions and priority rules below.\n"
    "Do not rely on any prior label from another model.\n\n"
    f"{LABEL_DEFINITIONS_AND_RULES}\n\n"
    "When a step asks for the final label, return valid JSON only:\n"
    '{"type": <0-6>, "rationale": "<brief reason>"}'
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict classification adjudication expert for Chinese Weibo posts.\n"
    "You must decide disputes using ONLY the label definitions and priority rules below.\n"
    "Always re-read the Weibo text and apply these rules directly.\n\n"
    f"{LABEL_DEFINITIONS_AND_RULES}\n\n"
    "When asked for a final decision, return valid JSON only:\n"
    '{"type": <0-6>, "rationale": "<brief reason>"}'
)


def truncate_text(text: str, max_chars: int = 900) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def build_single_classify_user_prompt(weibo_text: str, *, max_text_chars: int = 900) -> str:
    payload = {"weibo_text": truncate_text(weibo_text, max_text_chars)}
    return (
        "Classify the following Weibo post. Return JSON only, following the schema "
        "specified in the system message.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def build_verifier_question(weibo_text: str, *, max_text_chars: int = 900) -> str:
    text = truncate_text(weibo_text, max_text_chars)
    return (
        "Independently classify the following Weibo post using ONLY `weibo_text`. "
        "Do not use any prior label from another classifier.\n\n"
        f"weibo_text:\n{text}\n\n"
        "Your final step must output valid JSON only:\n"
        '{"type": <0-6>, "rationale": "<brief reason>"}'
    )


def format_classifier_context(
    *,
    name: str,
    label: int,
    rationale: str,
    cached_without_rationale: bool = False,
) -> str:
    lines = [f"# {name}", f"Label: {label}"]
    if cached_without_rationale or not rationale.strip():
        lines.append(
            "Rationale: NOT AVAILABLE. This is a cached prior label without stored reasoning. "
            "Re-evaluate whether this label fits the Weibo text under the label definitions "
            "in the system message."
        )
    else:
        lines.append(f"Rationale: {rationale}")
    return "\n".join(lines)
