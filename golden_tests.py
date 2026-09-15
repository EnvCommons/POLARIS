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
        tasks = Polaris.list_tasks("train")
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


def test_train_old_split_size():
    assert len(Polaris.list_tasks("train_old")) == 45840


def test_train_old_is_subset_of_train():
    old_problems = {t["problem"] for t in Polaris.list_tasks("train_old")}
    full_problems = {t["problem"] for t in Polaris.list_tasks("train")}
    assert old_problems <= full_problems


def test_no_base_post_2000_markers_in_train_old():
    from cutoff1999 import mentions_post_2000_regex_marker
    for task in Polaris.list_tasks("train_old"):
        assert not mentions_post_2000_regex_marker(task["problem"])


def test_unknown_split_still_raises():
    with pytest.raises(ValueError):
        Polaris.list_tasks("test")
