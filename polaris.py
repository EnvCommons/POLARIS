"""
POLARIS Environment - Math reasoning problem evaluation

Dataset: POLARIS-Project/Polaris-Dataset-53K (53,291 problems)
Task: Single-turn answer verification for math reasoning
"""

from __future__ import annotations

import pandas as pd
from math_verify import parse, verify
from pydantic import BaseModel, Field
import os

from openreward.environments import Environment, JSONObject, Server, TextBlock, ToolOutput, tool

# Load dataset at module import time (AIME pattern)

if os.path.exists("/orwd_data"):
    polaris_tasks_df = pd.read_parquet("/orwd_data/data/polaris_tasks.parquet")
else:
    polaris_tasks_df = pd.read_parquet("data/polaris_tasks.parquet")

polaris_tasks = polaris_tasks_df.to_dict(orient="records")

# Add task IDs for tracking
for i, task in enumerate(polaris_tasks):
    task["id"] = str(i)


class PolarisTaskSpec(BaseModel):
    """Task specification for a single POLARIS problem"""
    id: str
    problem: str
    answer: str
    difficulty: str


class AnswerParams(BaseModel):
    """Input parameters for the answer tool"""
    answer: str = Field(..., description="Your final answer to the problem")


class Polaris(Environment):
    """
    POLARIS environment for evaluating mathematical reasoning.

    Provides 53,291 math problems spanning various difficulty levels.
    Agent submits a single answer which is verified using symbolic math comparison.
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = PolarisTaskSpec.model_validate(task_spec)

    @classmethod
    def list_splits(cls) -> list[str]:
        """Return available data splits"""
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """Return task specifications for the given split"""
        if split == "train":
            return polaris_tasks
        raise ValueError(f"Unknown split: {split}. Available splits: {cls.list_splits()}")

    def get_prompt(self) -> list[TextBlock]:
        """Generate the task prompt for the agent"""
        prompt_text = (
            f"{self.config.problem}\n\n"
            "Submit your final answer using the answer tool. "
            "Only include the final number or expression, no other text."
        )
        return [TextBlock(type="text", text=prompt_text)]

    @tool
    async def answer(self, params: AnswerParams) -> ToolOutput:
        """
        Submit your final answer for evaluation.

        The answer will be verified using symbolic math comparison,
        so equivalent mathematical expressions will be accepted.
        """
        # Parse both answers using math-verify
        try:
            gold_parsed = parse(self.config.answer)
            submitted_parsed = parse(params.answer)

            # Verify equivalence
            is_correct = verify(gold_parsed, submitted_parsed)
        except Exception:
            # If parsing fails, do string comparison as fallback
            is_correct = params.answer.strip() == self.config.answer.strip()

        # Determine reward and feedback
        reward = 1.0 if is_correct else 0.0
        feedback = "Correct!" if is_correct else f"Incorrect. Expected: {self.config.answer}"

        return ToolOutput(
            blocks=[TextBlock(type="text", text=feedback)],
            metadata={
                "task_id": self.config.id,
                "submitted_answer": params.answer,
                "expected_answer": self.config.answer,
                "correct": is_correct,
                "difficulty": self.config.difficulty,
            },
            reward=reward,
            finished=True,
        )


if __name__ == "__main__":
    Server([Polaris]).run()
