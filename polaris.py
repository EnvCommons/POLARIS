"""
POLARIS Environment - Math reasoning problem evaluation

Dataset: POLARIS-Project/Polaris-Dataset-53K (53,291 problems)
Task: Single-turn answer verification for math reasoning
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
from math_verify import parse, verify
from pydantic import BaseModel, Field

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

# "train_old" is the subset of "train" whose problem text passes a
# pre-2000-knowledge cutoff filter (regex prefilter + deepseek-v4.1-flash
# verdict, with a standalone re-verification step before any learned term
# is folded into the shared regex -- see cutoff1999.py/filter_data.py).
# old_ids.json holds the (string) ids of surviving rows, computed offline.
_OLD_IDS_PATH = Path(__file__).parent / "old_ids.json"
_old_ids_cache: set[str] | None = None


def _load_old_ids() -> set[str]:
    global _old_ids_cache
    if _old_ids_cache is None:
        _old_ids_cache = set(json.loads(_OLD_IDS_PATH.read_text()))
    return _old_ids_cache


CURRENCY_MARKER = re.compile(r"(?<!\\)\$(?=[\d.\-])")


def parse_math_answer(text: str) -> list:
    text = text.strip().replace("\\%", "%")
    text = text.replace("\\$", "").replace("{,}", "")
    delimited = len(text) > 1 and text.startswith("$") and text.endswith("$")
    if not delimited:
        text = CURRENCY_MARKER.sub("", text)
    if delimited or "\\boxed" in text or not text:
        return parse(text)
    return parse(f"${text}$") or parse(text)


# Reward for a submission made after the task has already been graded. Negative
# so repeat submissions are actively discouraged, not merely left unscored.
REPEAT_SUBMISSION_PENALTY = -0.1


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

        # Graded submissions this session. Only the first is rewarded: an
        # incorrect answer reports "Expected: <answer>", so an uncapped tool
        # would let the agent read the answer and resubmit it.
        self.submitted = 0

    @classmethod
    def list_splits(cls) -> list[str]:
        """Return available data splits"""
        return ["train", "train_old"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """Return task specifications for the given split"""
        if split == "train":
            return polaris_tasks
        if split == "train_old":
            old_ids = _load_old_ids()
            return [t for t in polaris_tasks if t["id"] in old_ids]
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
        if self.submitted > 0:
            return ToolOutput(
                blocks=[TextBlock(type="text", text="An answer has already been submitted for "
                                  "this task. This episode is over: it is not re-graded, and "
                                  "repeat submissions are penalised (reward -0.1).")],
                metadata={"task_id": self.config.id, "already_submitted": True,
                          "submission_count": self.submitted},
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        # Parse both answers using math-verify
        try:
            gold_parsed = parse_math_answer(self.config.answer)
            submitted_parsed = parse_math_answer(params.answer)

            # Verify equivalence
            is_correct = verify(gold_parsed, submitted_parsed)
        except Exception:
            # If parsing fails, do string comparison as fallback
            is_correct = params.answer.strip() == self.config.answer.strip()

        # Determine reward and feedback
        reward = 1.0 if is_correct else 0.0
        feedback = "Correct!" if is_correct else f"Incorrect. Expected: {self.config.answer}"

        self.submitted += 1

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
