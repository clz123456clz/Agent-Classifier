JUDGE_REFLECT_PROMPT = """
Review the disagreement or uncertainty between two classifiers.
Use ONLY the label definitions and priority rules from the system message.

# Weibo Text:
{weibo_text}

{classifier_a_block}

{classifier_b_block}

Analyze:
1. Which signals in the text support each label under the official definitions?
2. Which label better matches the central topic and priority rules?
3. Is either label clearly wrong because of username noise, daytime sunlight, metaphor, or weak keyword matching?
4. If Classifier A has no rationale, independently judge whether its numeric label is correct from the text.

If one label is clearly correct, say which one and why.
If both are partially reasonable, explain the ambiguity and which label is safer.

Output your critique directly. Do not output JSON in this step.
"""

JUDGE_DECISION_PROMPT = """
Decide the final label for this disputed Weibo post.
Use ONLY the label definitions and priority rules from the system message.

# Weibo Text:
{weibo_text}

{classifier_a_block}

{classifier_b_block}

# Adjudication Critique:
{critique}

Choose the best final label from 0 to 6.
Return valid JSON only:
{{"type": <0-6>, "rationale": "<brief reason>"}}
"""
