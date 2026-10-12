"""
Tests for the repaired POLARIS dataset and the LLM-judge grading path (local only).

    python -m pytest test_fixed_grading.py -q                 # offline tests (judge mocked)
    POLARIS_LIVE=1 python -m pytest test_fixed_grading.py -q  # + live judge calls (OPENAI_API_KEY;
                                                              #   OPENAI_BASE_URL=https://infer.gr.inc/v1 for the gateway)

The env loads /orwd_data/data/polaris_tasks.parquet when that exists, so these tests point
POLARIS_DATA_PATH at the local rebuilt data/polaris_tasks.parquet before importing it.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
os.environ.setdefault("POLARIS_DATA_PATH", str(HERE / "data" / "polaris_tasks.parquet"))

import polaris  # noqa: E402
from polaris import AnswerParams, Polaris, key_parses, parse_judge_verdict, parse_proof_score  # noqa: E402

LIVE = os.environ.get("POLARIS_LIVE") == "1" and bool(os.environ.get("OPENAI_API_KEY"))
TASKS = Polaris.list_tasks("train")
DF = polaris.polaris_tasks_df


# ----------------------------------------------------------------------------- helpers

class MockJudge:
    """Stands in for openai.AsyncClient: serves canned replies as a raw SSE body through
    chat.completions.with_streaming_response.create (or raises), and records requests."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []
        self.drained: list[bool] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(
            with_streaming_response=SimpleNamespace(create=self._create)))

    def _create(self, **kwargs):
        mock = self

        class Ctx:
            async def __aenter__(self_inner):
                mock.prompts.append(kwargs["messages"][0]["content"])
                mock.kwargs.append(kwargs)
                reply = mock.replies.pop(0) if len(mock.replies) > 1 else mock.replies[0]
                if isinstance(reply, BaseException):
                    raise reply
                assert kwargs.get("stream") is True
                ev = lambda d: ("data: " + json.dumps(d, ensure_ascii=False) + "\n\n").encode()
                body = ev({"choices": [{"delta": {"content": None, "reasoning_content": "thinking"}}]})
                for k in range(0, len(reply), 7):
                    body += ev({"choices": [{"delta": {"content": reply[k:k + 7]}, "finish_reason": None}]})
                body += ev({"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]})
                body += ev({"choices": [], "usage": {"completion_tokens": 3}}) + b"data: [DONE]\n\n"
                body += b": trailing bytes after DONE\n\n"
                mock.drained.append(False)

                async def iter_bytes():
                    for k in range(0, len(body), 13):  # arbitrary chunking, splits UTF-8 and lines
                        yield body[k:k + 13]
                    mock.drained[-1] = True
                return SimpleNamespace(iter_bytes=iter_bytes)

            async def __aexit__(self_inner, *exc):
                return False
        return Ctx()


def make_env(spec: dict, judge=None, api_key: str | None = "test-key") -> Polaris:
    env = Polaris(task_spec=spec, secrets={"openai_api_key": api_key} if api_key else {})
    if judge is not None:
        env._judge_client = judge
    return env


def offline_env(spec: dict) -> Polaris:
    """An env whose judge is unreachable (no API key), so judge rows take the fallback path."""
    env = Polaris(task_spec=spec, secrets={})
    env._judge_secrets = {}
    return env


def assert_no_reference(out, spec: dict):
    """The agent sees the whole tool result, so it must not carry the key, the proof or the rubric."""
    seen = out.blocks[0].text + json.dumps(out.metadata, ensure_ascii=False)
    assert "Expected:" not in out.blocks[0].text
    assert not {"expected_answer", "grading_response", "grading_notes"} & set(out.metadata)
    for field in ("answer", "grading_notes"):
        ref = (spec.get(field) or "").strip()
        if len(ref) >= 8:  # short keys like "2" occur in any text by chance
            assert ref not in seen, field


@pytest.fixture(autouse=True)
def no_ambient_judge_credentials(monkeypatch):
    """Offline tests must not pick up a judge key or endpoint from the developer's shell
    (resolve_backend's legacy mode falls back to OPENAI_API_KEY / OPENAI_BASE_URL)."""
    if not LIVE:
        for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CHAT_API_KEY", "CHAT_BASE_URL", "CHAT_PROVIDER"):
            monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    async def no_sleep(_):
        return None
    monkeypatch.setattr(polaris.asyncio, "sleep", no_sleep)


def untouched_mask():
    return DF["edit_source"].isna() if "edit_source" in DF.columns else DF["answer"].notna()


# ----------------------------------------------------------------------------- dataset shape

def test_dataset_shape_and_order():
    assert len(TASKS) == 53291
    assert [t["id"] for t in TASKS[:3]] == ["0", "1", "2"]
    if "original_problem" in DF.columns:
        # untouched rows keep the corpus text; every row keeps the original in its own column
        u = untouched_mask()
        assert (DF.loc[u, "problem"] == DF.loc[u, "original_problem"]).all()
        assert (DF.loc[u, "answer"] == DF.loc[u, "original_answer"]).all()
    json.dumps(TASKS[:500])  # task specs are plain JSON


def test_untouched_spec_defaults():
    spec = {"id": "0", "problem": "1+1?", "answer": "2", "difficulty": "1/8"}
    c = Polaris(task_spec=spec).config
    assert (c.grading, c.answer_format, c.is_proof, c.grading_notes, c.case) == (None, "math", False, None, None)
    assert not Polaris(task_spec=spec).uses_judge()


def test_is_proof_and_answer_format_stay_in_sync():
    base = {"id": "0", "problem": "p", "answer": "a", "difficulty": "1/8"}
    assert Polaris(task_spec={**base, "is_proof": True}).config.answer_format == "proof"
    assert Polaris(task_spec={**base, "answer_format": "proof"}).config.is_proof is True


# ----------------------------------------------------------------------------- golden (untouched rows)
# golden_tests.py runs the same two checks on a 200-row sample of every row, skipping
# judge-graded rows (they need a live judge). These run them on rows the repair did not
# touch, with a mocked judge for the untouched rows whose key math_verify cannot parse.

_rng = random.Random(42)
_UNTOUCHED_IDX = [i for i, keep in enumerate(untouched_mask().tolist()) if keep]
GOLDEN_SAMPLE = _rng.sample(_UNTOUCHED_IDX, 200)


@pytest.mark.asyncio
@pytest.mark.parametrize("row", GOLDEN_SAMPLE)
async def test_golden_untouched(row):
    env = offline_env(TASKS[row])
    ok = await env.answer(AnswerParams(answer=env.config.answer))
    assert ok.reward == 1.0 and ok.finished and ok.metadata["correct"] is True
    # a fresh env for the wrong answer: only the first answer of a session is graded
    if key_parses(env.config.answer):
        assert ok.metadata["grading_mode"] == "math_verify"
        bad = await offline_env(TASKS[row]).answer(AnswerParams(answer="definitely_wrong_answer_12345"))
    else:  # judge row: a mocked INCORRECT verdict stands in for the live judge
        bad = await make_env(TASKS[row], MockJudge(["<verdict>INCORRECT</verdict>"])).answer(
            AnswerParams(answer="definitely_wrong_answer_12345"))
    assert bad.reward == 0.0 and bad.finished and bad.metadata["correct"] is False
    assert_no_reference(bad, TASKS[row])


@pytest.mark.asyncio
async def test_math_verify_decisions_unchanged_on_parseable_keys():
    """For parseable keys, the new grader agrees with the original decision on a spread of submissions."""
    from math_verify import verify
    rows = [i for i in GOLDEN_SAMPLE if key_parses(TASKS[i]["answer"])][:100]
    for i in rows:
        key = TASKS[i]["answer"]
        for sub in (key, f"${key}$", "0", "-1", "x+1"):
            try:
                old = verify(polaris.parse_math_answer(key), polaris.parse_math_answer(sub))
            except Exception:
                old = sub.strip() == key.strip()
            out = await offline_env(TASKS[i]).answer(AnswerParams(answer=sub))
            assert out.metadata["correct"] == bool(old), (i, key, sub)


# ----------------------------------------------------------------------------- routing

UNPARSEABLE_KEY = r"^2=\frac{1}{3}(a_1^2+b_1^2+c_1^2)-\frac{1}{9}(^2+b^2+^2)"  # corpus row 1


def test_unparseable_key_detection():
    assert not key_parses(UNPARSEABLE_KEY)
    assert not key_parses("")
    assert key_parses("9375") and key_parses(r"\frac{\sqrt{3}}{4}")
    # every row the dataset marks llm-by-default really has an unparseable key
    if "key_parses" in DF.columns:
        auto_llm = DF[(DF.grading == "llm") & (DF.answer_format == "math") & untouched_mask()]
        assert not auto_llm["key_parses"].any()
        assert len(auto_llm) > 0  # shrinks as specialists repair keys


@pytest.mark.asyncio
async def test_unparseable_key_routes_to_judge():
    judge = MockJudge(["The submission matches.\n<verdict>CORRECT</verdict>"])
    env = make_env({"id": "1", "problem": "Find the median length.", "answer": UNPARSEABLE_KEY, "difficulty": "5/8"}, judge)
    assert env.uses_judge()
    out = await env.answer(AnswerParams(answer="m_a^2 = (2b^2+2c^2-a^2)/4"))
    assert out.reward == 1.0 and out.metadata["grading_mode"] == "llm"
    assert env.last_judge_response.endswith("<verdict>CORRECT</verdict>")
    assert "grading_response" not in out.metadata
    assert len(judge.prompts) == 1 and UNPARSEABLE_KEY in judge.prompts[0]
    kw = judge.kwargs[0]
    # only parameters both the gateway and api.openai.com accept (no temperature, no glm flags)
    assert set(kw) == {"model", "messages", "stream"} and kw["model"] == "gpt-5-mini" and kw["stream"] is True
    assert out.metadata["judge_call"] == {"finish_reason": "stop", "reasoning_chars": 8, "attempts": 1,
                                          "connect_failures": 0}
    assert judge.drained == [True]  # body read past [DONE] so the pooled connection is reusable


@pytest.mark.asyncio
async def test_unparseable_dataset_rows_route_to_judge():
    rows = [i for i in GOLDEN_SAMPLE if not key_parses(TASKS[i]["answer"])]
    rows += [i for i, t in enumerate(TASKS[:3000]) if not key_parses(t["answer"])][:5]
    assert rows
    for i in rows:
        judge = MockJudge(["<verdict>INCORRECT</verdict>"])
        out = await make_env(TASKS[i], judge).answer(AnswerParams(answer="something else entirely"))
        assert out.metadata["grading_mode"] == "llm" and out.reward == 0.0 and len(judge.prompts) == 1


@pytest.mark.asyncio
async def test_parseable_key_does_not_call_judge():
    judge = MockJudge([RuntimeError("must not be called")])
    out = await make_env({"id": "0", "problem": "p", "answer": "9375", "difficulty": "7/8"}, judge).answer(
        AnswerParams(answer="9375.0"))
    assert out.reward == 1.0 and out.metadata["grading_mode"] == "math_verify" and not judge.prompts


@pytest.mark.asyncio
async def test_llm_marked_and_text_rows_route_to_judge():
    for extra in ({"grading": "llm"}, {"grading": "llm", "answer_format": "text",
                                       "grading_notes": "MODE=ANY: accept any one of: 13; 85."}):
        judge = MockJudge(["<verdict>CORRECT</verdict>"])
        spec = {"id": "13763", "problem": "p", "answer": "13 or 85", "difficulty": "3/8", **extra}
        out = await make_env(spec, judge).answer(AnswerParams(answer="85"))
        assert out.metadata["grading_mode"] == "llm" and out.reward == 1.0
        if "grading_notes" in extra:
            assert "MODE=ANY" in judge.prompts[0] and "more than one valid answer" in judge.prompts[0]


@pytest.mark.asyncio
async def test_exact_key_match_skips_judge():
    judge = MockJudge([RuntimeError("must not be called")])
    spec = {"id": "1", "problem": "p", "answer": UNPARSEABLE_KEY, "difficulty": "5/8"}
    out = await make_env(spec, judge).answer(AnswerParams(answer=UNPARSEABLE_KEY + "  "))
    assert out.reward == 1.0 and out.metadata["grading_mode"] == "exact_match" and not judge.prompts


PROOF_SPEC = {
    "id": "p1", "difficulty": "4/8", "is_proof": True, "case": "4",
    "problem": "Prove that for every positive integer $n$, $n^3 - n$ is divisible by $6$.",
    "answer": "STATEMENT: 6 | n^3 - n for all positive integers n.\n"
              "REFERENCE SKETCH: n^3 - n = (n-1)n(n+1) is a product of three consecutive integers, "
              "so it has a factor divisible by 2 and one divisible by 3; since gcd(2,3)=1, 6 divides it.",
    "grading_notes": "RUBRIC (7 pts): factor as (n-1)n(n+1) [2]; divisibility by 2 justified [2]; "
                     "divisibility by 3 justified [2]; combine via coprimality [1]. "
                     "FATAL: checking small n only; asserting divisibility without reason.",
}
GOOD_PROOF = (
    "We have $n^3-n = n(n^2-1) = (n-1)n(n+1)$, a product of three consecutive integers. "
    "Among any two consecutive integers one is even, so $2 \\mid (n-1)n(n+1)$. Among any three consecutive "
    "integers exactly one is a multiple of 3 (they cover all residues mod 3), so $3 \\mid (n-1)n(n+1)$. "
    "Since $\\gcd(2,3)=1$, both divide the product implies $6 \\mid n^3-n$. $\\blacksquare$"
)


@pytest.mark.asyncio
async def test_proof_row_routes_to_rubric_judge():
    judge = MockJudge(["Analysis: complete except a terse coprimality step.\n\nScore: 5"])
    env = make_env(PROOF_SPEC, judge)
    out = await env.answer(AnswerParams(answer=GOOD_PROOF))
    md = out.metadata
    assert md["grading_mode"] == "llm_rubric" and md["is_proof"] is True
    assert md["score"] == 5 and out.reward == pytest.approx(5 / 7) and md["correct"] is False
    assert env.last_judge_response.startswith("Analysis:")
    assert out.blocks[0].text == "Score: 5/7"  # the analysis can quote the reference proof
    assert_no_reference(out, PROOF_SPEC)
    assert "RUBRIC (7 pts)" in judge.prompts[0] and "Score: <a single integer from 0 to 7>" in judge.prompts[0]
    out7 = await make_env(PROOF_SPEC, MockJudge(["Analysis: ok\nScore: 7"])).answer(AnswerParams(answer=GOOD_PROOF))
    assert out7.reward == 1.0 and out7.metadata["correct"] is True


@pytest.mark.asyncio
async def test_proof_copy_of_reference_still_goes_to_judge():
    judge = MockJudge(["Analysis: restates the reference sketch\nScore: 6"])
    out = await make_env(PROOF_SPEC, judge).answer(AnswerParams(answer=PROOF_SPEC["answer"]))
    assert out.metadata["grading_mode"] == "llm_rubric" and len(judge.prompts) == 1


# ----------------------------------------------------------------------------- failure handling

def assert_ungraded(out, needle=""):
    md = out.metadata
    assert out.reward is None and out.finished is True
    assert md["ungraded"] is True and md["grading_mode"] == "ungraded" and md["correct"] is None
    assert needle in md["judge_error"]
    json.dumps(md)


@pytest.mark.asyncio
async def test_judge_failure_is_ungraded_not_zero():
    spec = {"id": "x", "problem": "p", "answer": "13 or 85", "difficulty": "3/8", "grading": "llm", "answer_format": "text"}
    for sub in ("something", "85 or 13"):  # no math_verify fallback, even for a symbolically equal answer
        judge = MockJudge([TimeoutError("read timeout")])
        out = await make_env(spec, judge).answer(AnswerParams(answer=sub))
        assert_ungraded(out, "read timeout")
        assert len(judge.prompts) == polaris.JUDGE_ATTEMPTS == 4


@pytest.mark.asyncio
async def test_unparseable_key_judge_failure_is_ungraded():
    spec = {"id": "1", "problem": "p", "answer": UNPARSEABLE_KEY, "difficulty": "5/8"}
    assert_ungraded(await make_env(spec, MockJudge([ConnectionError("conn reset")])).answer(AnswerParams(answer="foo")),
                    "conn reset")


@pytest.mark.asyncio
async def test_unparseable_verdict_is_retried_then_used():
    judge = MockJudge(["I think it is right.", "<verdict>maybe</verdict>", "ok <verdict>**CORRECT**</verdict>"])
    spec = {"id": "1", "problem": "p", "answer": UNPARSEABLE_KEY, "difficulty": "5/8"}
    out = await make_env(spec, judge).answer(AnswerParams(answer="foo"))
    assert out.metadata["grading_mode"] == "llm" and out.reward == 1.0 and len(judge.prompts) == 3
    assert out.metadata["ungraded"] is False


@pytest.mark.asyncio
async def test_no_api_key_is_ungraded_without_crashing():
    spec = {"id": "1", "problem": "p", "answer": UNPARSEABLE_KEY, "difficulty": "5/8"}
    assert_ungraded(await offline_env(spec).answer(AnswerParams(answer="foo")), "No judge API key")


@pytest.mark.asyncio
async def test_proof_judge_failure_is_ungraded():
    judge = MockJudge(["Analysis: no score line here"])
    out = await make_env(PROOF_SPEC, judge).answer(AnswerParams(answer=GOOD_PROOF))
    assert_ungraded(out, "Score")
    assert "score" not in out.metadata and len(judge.prompts) == polaris.JUDGE_ATTEMPTS


def _connect_error():
    import httpx
    import openai
    e = openai.APIConnectionError(request=httpx.Request("POST", "https://infer.gr.inc/v1/chat/completions"))
    e.__cause__ = httpx.ConnectError("All connection attempts failed")
    return e


@pytest.mark.asyncio
async def test_connect_failures_retried_without_spending_attempts():
    replies = [_connect_error() for _ in range(10)] + ["<verdict>CORRECT</verdict>"]
    judge = MockJudge(replies)
    spec = {"id": "1", "problem": "p", "answer": UNPARSEABLE_KEY, "difficulty": "5/8"}
    out = await make_env(spec, judge).answer(AnswerParams(answer="foo"))
    assert out.reward == 1.0 and out.metadata["judge_call"]["connect_failures"] == 10
    assert out.metadata["judge_call"]["attempts"] == 1


@pytest.mark.asyncio
async def test_persistent_connect_failure_is_ungraded():
    judge = MockJudge([_connect_error()])
    spec = {"id": "1", "problem": "p", "answer": UNPARSEABLE_KEY, "difficulty": "5/8"}
    out = await make_env(spec, judge).answer(AnswerParams(answer="foo"))
    assert_ungraded(out, "connect failures")
    assert len(judge.prompts) == polaris.JUDGE_CONNECT_RETRIES + polaris.JUDGE_ATTEMPTS


@pytest.mark.asyncio
async def test_stream_parsing_survives_unicode_line_separators():
    reply = "analysis with \u2028 and \u0085 and é inside\n<verdict>CORRECT</verdict>"
    judge = MockJudge([reply])
    spec = {"id": "1", "problem": "p", "answer": UNPARSEABLE_KEY, "difficulty": "5/8"}
    env = make_env(spec, judge)
    out = await env.answer(AnswerParams(answer="foo"))
    assert out.reward == 1.0 and env.last_judge_response == reply and judge.drained == [True]


def test_verdict_and_score_parsers():
    assert parse_judge_verdict("incorrectly stated... <verdict>CORRECT</verdict>") is True
    assert parse_judge_verdict("<verdict>CORRECT</verdict> wait <verdict>INCORRECT</verdict>") is False
    with pytest.raises(RuntimeError):
        parse_judge_verdict("CORRECT")
    assert parse_proof_score("Analysis: Score: 3 was tempting\n\nScore: 6") == 6
    assert parse_proof_score("Score: **9**") == 7
    with pytest.raises(RuntimeError):
        parse_proof_score("Analysis: great proof, 7 out of 7")


def test_judge_settings_from_secrets_and_env(monkeypatch):
    spec = {"id": "0", "problem": "p", "answer": "1", "difficulty": "1/8"}
    env = Polaris(task_spec=spec, secrets={"openai_api_key": "k", "judge_model": "m"})
    assert env.judge_model == "m"
    monkeypatch.setenv("POLARIS_JUDGE_MODEL", "m2")
    env2 = Polaris(task_spec=spec, secrets={})
    assert env2.judge_model == "m2" and env2.judge_timeout_s >= 7200
    assert Polaris(task_spec=spec).judge_model == "m2"
    monkeypatch.delenv("POLARIS_JUDGE_MODEL")
    assert Polaris(task_spec=spec).judge_model == "gpt-5-mini"


@pytest.mark.asyncio
async def test_judge_follows_openai_base_url_and_ignores_chat_api_key(monkeypatch):
    """Training: enigma sets OPENAI_BASE_URL to grader_base_url (infer.gr.inc) and also forwards
    a chat_api_key for the graders pinned to OpenAI. POLARIS must stay on the gateway."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://infer.gr.inc/v1")
    env = Polaris(task_spec=UNPARSEABLE_SPEC, secrets={"openai_api_key": "gw-key", "chat_api_key": "oa-key"})
    client = env._get_judge_client()
    assert str(client.base_url).rstrip("/") == "https://infer.gr.inc/v1"
    assert client.api_key == "gw-key" and client.max_retries == 0


@pytest.mark.asyncio
async def test_judge_goes_to_openai_without_a_gateway():
    env = Polaris(task_spec=UNPARSEABLE_SPEC, secrets={"openai_api_key": "k"})
    client = env._get_judge_client()
    assert str(client.base_url).rstrip("/") == "https://api.openai.com/v1" and client.api_key == "k"


# ----------------------------------------------------------------------------- attempt rules (from main) on judge rows

TEXT_SPEC = {"id": "t1", "problem": "p", "answer": "13 or 85", "difficulty": "3/8", "grading": "llm",
             "answer_format": "text", "grading_notes": "Accept any ONE of: 13; 85"}
UNPARSEABLE_SPEC = {"id": "1", "problem": "p", "answer": UNPARSEABLE_KEY, "difficulty": "5/8"}


@pytest.mark.asyncio
async def test_only_the_first_graded_answer_counts_on_judge_rows():
    judge = MockJudge(["<verdict>INCORRECT</verdict>", "<verdict>CORRECT</verdict>"])
    env = make_env(TEXT_SPEC, judge)
    first = await env.answer(AnswerParams(answer="49"))
    assert first.reward == 0.0 and first.finished and env.submitted == 1
    assert first.blocks[0].text == "Incorrect."
    assert_no_reference(first, TEXT_SPEC)
    again = await env.answer(AnswerParams(answer="85"))
    assert again.reward == polaris.REPEAT_SUBMISSION_PENALTY and again.finished
    assert again.metadata["already_submitted"] is True and len(judge.prompts) == 1


@pytest.mark.asyncio
async def test_ungraded_does_not_use_up_the_attempt():
    env = make_env(TEXT_SPEC, MockJudge([TimeoutError("read timeout")]))
    assert_ungraded(await env.answer(AnswerParams(answer="85")), "read timeout")
    assert env.submitted == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["", "   ", "\\boxed{}", "\\boxed{ }", "$"])
@pytest.mark.parametrize("spec", [TEXT_SPEC, PROOF_SPEC, UNPARSEABLE_SPEC], ids=["text", "proof", "unparseable_key"])
async def test_empty_answer_on_judge_rows_is_not_graded(spec, answer):
    judge = MockJudge(["Analysis: x\nScore: 7"] if spec.get("is_proof") else ["<verdict>CORRECT</verdict>"])
    env = make_env(spec, judge)
    out = await env.answer(AnswerParams(answer=answer))
    assert out.reward == 0.0 and out.finished is False and env.submitted == 0 and not judge.prompts
    out = await env.answer(AnswerParams(answer=GOOD_PROOF if spec.get("is_proof") else "85"))
    assert out.finished is True and out.reward == 1.0 and env.submitted == 1


# ----------------------------------------------------------------------------- prompts

def _prompt(spec):
    return Polaris(task_spec=spec).get_prompt()[0].text


def test_prompt_depends_on_answer_format():
    base = {"id": "0", "problem": "PROBLEM", "answer": "1", "difficulty": "1/8"}
    math_p = _prompt(base)
    assert "exactly one correct answer" in math_p and "Only include the final number" in math_p
    text_p = _prompt({**base, "answer_format": "text"})
    assert "exactly one" not in text_p and "separated by ' or '" in text_p and "Only include the final number" not in text_p
    proof_p = _prompt({**base, "is_proof": True})
    assert "exactly one" not in proof_p and "complete, rigorous proof" in proof_p and "final number" not in proof_p.split("not just")[0]
    assert all(p.startswith("PROBLEM\n\n") for p in (math_p, text_p, proof_p))
    mc_p = _prompt({**base, "case": "mc"})
    assert "Give the value of the correct option, not its letter." in mc_p and "exactly one correct answer" in mc_p
    assert "not its letter" not in math_p
    assert "not its letter" not in _prompt({**base, "case": "mc", "answer_format": "text"})
    instr_p = _prompt({**base, "answer_format": "text", "answer_instruction": "List all possible values of n."})
    assert "List all possible values of n.\nIf the problem has more than one" in instr_p


# ----------------------------------------------------------------------------- smoke: fixed rows of each case

def _rows_by_case(n=2):
    if "case" not in DF.columns:
        return []
    out = []
    for case, grp in DF.groupby(DF["case"].fillna("unclassified")):
        edited = grp[grp["edit_source"].notna()] if "edit_source" in grp else grp.iloc[:0]
        pick = edited if len(edited) else grp
        out += [(case, int(i)) for i in pick.index[:n]]
    return out


SMOKE_ROWS = _rows_by_case()


@pytest.mark.asyncio
@pytest.mark.parametrize("case,row", SMOKE_ROWS or [pytest.param(None, None, marks=pytest.mark.skip("no case column"))])
async def test_smoke_end_to_end(case, row):
    """list_tasks -> env -> get_prompt -> answer, with a mocked judge, for fixed rows of every case."""
    spec = TASKS[row]
    json.dumps(spec)
    judge = MockJudge(["Analysis: fine\n\nScore: 7"] if spec.get("is_proof") else ["<verdict>CORRECT</verdict>"])
    env = make_env(spec, judge)
    prompt = env.get_prompt()[0].text
    assert prompt.startswith(spec["problem"])
    assert ("exactly one correct answer" in prompt) == (env.config.answer_format == "math")
    out = await env.answer(AnswerParams(answer=spec["answer"]))
    assert out.finished and out.reward == 1.0, out.metadata
    assert out.metadata["task_id"] == str(row) and out.metadata["case"] == spec.get("case")
    expected = "llm_rubric" if env.config.is_proof else ("math_verify" if not env.uses_judge() else None)
    if expected:
        assert out.metadata["grading_mode"] == expected


# ----------------------------------------------------------------------------- live judge (infer.gr.inc)

CASE2_SPEC = {
    "id": "13763", "difficulty": "2/8", "case": "2", "grading": "llm", "answer_format": "text",
    "problem": TASKS[13763]["problem"] if len(TASKS) > 13763 else "",
    "answer": "13 or 85",
    "grading_notes": "MODE=ANY: accept any one of: 13; 85. Also accept a list whose members are all valid. "
                     "Reject a list containing an invalid value.",
}


@pytest.mark.skipif(not LIVE, reason="set POLARIS_LIVE=1 (and OPENAI_API_KEY) for live judge calls")
@pytest.mark.asyncio
async def test_live_judge_three_examples():
    import asyncio as aio
    assert "ab" in CASE2_SPEC["problem"].replace(" ", "") or CASE2_SPEC["problem"]
    good, bad, proof = await aio.gather(
        Polaris(task_spec=CASE2_SPEC).answer(AnswerParams(answer="85")),
        Polaris(task_spec=CASE2_SPEC).answer(AnswerParams(answer="49")),
        Polaris(task_spec=PROOF_SPEC).answer(AnswerParams(answer=GOOD_PROOF)),
    )
    for o in (good, bad, proof):
        print(json.dumps({k: (v[-400:] if isinstance(v, str) else v) for k, v in o.metadata.items()}, indent=1))
        assert "grading_response" not in o.metadata and "Expected:" not in o.blocks[0].text
    assert good.metadata["grading_mode"] == "llm" and good.reward == 1.0
    assert bad.metadata["grading_mode"] == "llm" and bad.reward == 0.0
    assert proof.metadata["grading_mode"] == "llm_rubric" and proof.metadata["score"] >= 6


def _live_smoke_rows():
    if "edit_source" not in DF.columns:
        return []
    ed = DF[DF["edit_source"].notna()]
    out = []
    for case, grp in ed.groupby("case"):
        judged = [i for i in grp.index if Polaris(task_spec=TASKS[i]).uses_judge() or TASKS[i].get("is_proof")]
        out += [(case, int(i)) for i in (judged or list(grp.index))[:1]]
    return out


@pytest.mark.skipif(not LIVE, reason="set POLARIS_LIVE=1 (and OPENAI_API_KEY) for live judge calls")
@pytest.mark.asyncio
async def test_live_smoke_edited_rows():
    """One repaired row per case, live judge, submitting a wrong answer (proofs: a bare restatement)."""
    import asyncio as aio
    rows = _live_smoke_rows()
    assert rows
    envs = [Polaris(task_spec=TASKS[r]) for _, r in rows]
    outs = await aio.gather(*[e.answer(AnswerParams(answer="I don't know.")) for e in envs])
    for (case, r), e, o in zip(rows, envs, outs):
        md = o.metadata
        print(case, r, md["grading_mode"], o.reward, md.get("score"), (e.last_judge_response or md.get("judge_error") or "")[-150:].replace("\n", " "))
    for o in outs:
        assert o.finished and o.metadata["grading_mode"] != "ungraded", o.metadata
        assert o.reward == 0.0
