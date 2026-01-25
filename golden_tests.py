"""
Golden tests for POLARIS environment.
Tests correct answer acceptance and wrong answer rejection.
"""
import pytest
import random

from polaris import Polaris, AnswerParams
from openreward.environments import JSONObject

# Random seed for reproducible sampling
_rng = random.Random(42)
SAMPLED_TASKS = None


@pytest.fixture(scope="module")
def sampled_tasks():
    """Sample 200 tasks for testing"""
    global SAMPLED_TASKS
    if SAMPLED_TASKS is None:
        import asyncio
        tasks = asyncio.run(Polaris.list_tasks("train"))
        SAMPLED_TASKS = _rng.sample(tasks, min(200, len(tasks)))
    return SAMPLED_TASKS


@pytest.mark.asyncio
@pytest.mark.parametrize("task_idx", range(200))
async def test_golden_answer(sampled_tasks: list[JSONObject], task_idx: int):
    """Test that correct answers are accepted"""
    task = sampled_tasks[task_idx]
    env = Polaris(task_spec=task)

    # Submit the correct answer
    output = await env.answer(AnswerParams(answer=env.config.answer))

    assert output.reward == 1.0, f"Expected reward 1.0 for correct answer, got {output.reward}"
    assert output.finished is True, "Expected finished=True"
    assert output.metadata["correct"] is True, "Expected correct=True in metadata"
    assert len(output.blocks) > 0, "Expected feedback blocks"


@pytest.mark.asyncio
@pytest.mark.parametrize("task_idx", range(200))
async def test_wrong_answer(sampled_tasks: list[JSONObject], task_idx: int):
    """Test that incorrect answers are rejected"""
    task = sampled_tasks[task_idx]
    env = Polaris(task_spec=task)

    # Submit an obviously wrong answer
    output = await env.answer(AnswerParams(answer="definitely_wrong_answer_12345"))

    assert output.reward == 0.0, f"Expected reward 0.0 for wrong answer, got {output.reward}"
    assert output.finished is True, "Expected finished=True"
    assert output.metadata["correct"] is False, "Expected correct=False in metadata"
    assert len(output.blocks) > 0, "Expected feedback blocks"
