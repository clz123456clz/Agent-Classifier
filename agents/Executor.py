EXECUTOR_PROMPT_TEMPLATE = """
You are a top AI execution expert. Your task is to strictly follow the given plan and solve the problem step by step.
You will receive the original question, the complete plan, and the steps and results completed so far.
Please focus on solving the "current step" and only output the final answer for that step, without any additional explanations or dialogue.

# Original Question:
{question}

# Complete Plan:
{plan}

# Historical Steps and Results:
{history}

# Current Step:
{current_step}

Please only output the answer for the "current step":
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Executor:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        """
        Execute step by step according to the plan and solve the problem.
        """
        history = "" # String to store historical steps and results

        print("\n--- Executing Plan ---")

        for i, step in enumerate(plan):
            print(f"\n-> Executing step {i+1}/{len(plan)}: {step}")

            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history if history else "None", # If it's the first step, history is empty
                current_step=step
            )
            if history:
                logger.info(f"History: {history}")

            messages = [{"role": "user", "content": prompt}]

            response_text = self.llm_client.think(messages=messages) or ""

            # Update history for the next step
            history += f"Step {i+1}: {step}\nResult: {response_text}\n\n"

            print(f"✅ Step {i+1} completed, result: {response_text}")

        # After the loop ends, the last step's response is the final answer
        final_answer = response_text
        return final_answer
