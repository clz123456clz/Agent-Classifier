from typing import List, Dict, Any, Optional

class Memory:
    """
    A simple short-term memory module for storing the agent's action and reflection trajectory.
    """

    def __init__(self):
        """
        Initialize an empty list to store all records.
        """
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        Add a new record to memory.

        Parameters:
        - record_type (str): Type of record ('execution' or 'reflection').
        - content (str): Specific content of the record (e.g., generated code or reflection feedback).
        """
        record = {"type": record_type, "content": content}
        self.records.append(record)
        print(f"📝 Memory updated, added a '{record_type}' record.")

    def get_trajectory(self) -> str:
        """
        Format all memory records into a coherent string text for building prompts.
        """
        trajectory_parts = []
        for record in self.records:
            if record['type'] == 'execution':
                trajectory_parts.append(f"--- Previous Attempt (Code) ---\n{record['content']}")
            elif record['type'] == 'reflection':
                trajectory_parts.append(f"--- Reviewer Feedback ---\n{record['content']}")

        return "\n\n".join(trajectory_parts)

    def get_last_execution(self) -> Optional[str]:
        """
        Get the most recent execution result (e.g., the latest generated code).
        Returns None if it doesn't exist.
        """
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None

