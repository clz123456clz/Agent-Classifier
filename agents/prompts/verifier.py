VERIFIER_PLANNER_PROMPT = """
You are a top AI planning expert for Chinese social-media text classification.

Your task is to decompose a Weibo classification problem into a short, logical action plan.
Each step must be an independent reasoning subtask leading to a final label from 0 to 6.

Classification question:
{question}

Please output a Python list of 3 to 5 concise steps.
The last step must explicitly say to output the final JSON label.

Strict output format:
```python
["Step 1", "Step 2", "Step 3"]
```
"""

VERIFIER_EXECUTOR_PROMPT = """
You are a top AI execution expert for Chinese social-media text classification.
Follow the plan strictly and solve only the current step.

# Original Question:
{question}

# Complete Plan:
{plan}

# Historical Steps and Results:
{history}

# Current Step:
{current_step}

Output only the result for the current step.
If this is the final step, output valid JSON only:
{{"type": <0-6>, "rationale": "<brief reason>"}}
"""
