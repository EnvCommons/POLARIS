"""Offline pre-2000-knowledge-cutoff filter for POLARIS-Dataset-53K, following
the doctrine in cutoff1999.py (shared with tulu3_compare/GSM8K/MATH's
equivalent pipelines): a row is DROPPED if its `problem` text references
information from after December 31, 1999, or is post-1999-*motivated*.

Two stages, stage 2 feeding stage 1:
  1. Regex hard-drop (cutoff1999.YEAR_RE / a TERMS_RE rebuilt from
     cutoff1999.POST_2000_TERMS plus every term learned so far this run).
  2. deepseek-v4.1-flash verdict (via the internal infer.gr.inc gateway) on
     every regex survivor, using cutoff1999.SYSTEM_PROMPT. A newly-named
     disqualifying term is checked against TERM_DENYLIST, then independently
     re-verified with a second, standalone, zero-context LLM call
     (verify_term_is_universal) before being folded into the regex.

POLARIS has no stable id field -- this script uses the row's positional
index into the dataset's own iteration order as the stable key (the
dataset is loaded once, deterministically, by `datasets.load_dataset`, so
this index is stable across runs as long as the upstream data doesn't
change).

Resumable: verdicts append to filter_verdicts_1999.jsonl; learned terms
accumulate in learned_terms.json. Requires KIMI_API_KEY.
"""
import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import cast

import datasets
import httpx
from datasets import DatasetDict

from cutoff1999 import (POST_2000_TERMS, SYSTEM_PROMPT, TERM_DENYLIST,
                        TERM_VERIFY_SYSTEM_PROMPT, YEAR_RE, build_terms_regex)

HERE = Path(__file__).parent
VERDICTS = HERE / "filter_verdicts_1999.jsonl"
LEARNED_TERMS_FILE = HERE / "learned_terms.json"
OLD_IDS_FILE = HERE / "old_ids.json"
REPORT = HERE / "filter_report.json"

URL = "https://infer.gr.inc/hi/v1/chat/completions"
MODEL = "deepseek-v4.1-flash"
CONCURRENCY = 20
MAX_ATTEMPTS = 4
MAX_CHARS = 16000

MIN_TERM_LEN = 3
MAX_TERM_WORDS = 5
_VALID_TERM_RE = re.compile(r"^[a-z0-9][a-z0-9 ./#'\-]*[a-z0-9]$")

SWEEP_EVERY = 3000


def record_text(row: dict) -> str:
    text = row["problem"]
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    return text


async def verify_term_is_universal(client: httpx.AsyncClient, api_key: str, term: str) -> bool:
    prompt = TERM_VERIFY_SYSTEM_PROMPT.replace("%%TERM%%", repr(term))
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = await client.post(
                URL, headers={"Authorization": f"Bearer {api_key}"},
                json={"model": MODEL,
                      "messages": [{"role": "system", "content": prompt},
                                   {"role": "user", "content": f"Candidate term: {term!r}"}],
                      "max_tokens": 20, "temperature": 0.0},
                timeout=20.0,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
            start, end = content.find("{"), content.rfind("}")
            parsed = json.loads(content[start:end + 1])
            return bool(parsed.get("safe_as_global_marker"))
        except Exception:
            await asyncio.sleep(min(2 ** attempt, 10))
    return False


class LearnedTerms:
    def __init__(self, initial: list[str], client: httpx.AsyncClient, api_key: str) -> None:
        self._terms: set[str] = set(initial)
        self._base_lower = {t.lower() for t in POST_2000_TERMS}
        self.current_re = build_terms_regex(sorted(self._terms))
        self.lock = asyncio.Lock()
        self._client = client
        self._api_key = api_key
        self.rejected: set[str] = set()

    def _sanitize(self, raw: str) -> str | None:
        term = raw.strip().lower()
        if not term or len(term) < MIN_TERM_LEN or len(term.split()) > MAX_TERM_WORDS:
            return None
        if not _VALID_TERM_RE.match(term):
            return None
        if term in TERM_DENYLIST:
            return None
        return term

    async def add(self, raw_terms: list[str]) -> list[str]:
        candidates = {c for c in (self._sanitize(t) for t in raw_terms) if c}
        async with self.lock:
            candidates -= self._terms
            candidates -= self.rejected
            candidates = {t for t in candidates if not self.current_re.search(t)}
        if not candidates:
            return []
        verdicts = await asyncio.gather(*(
            verify_term_is_universal(self._client, self._api_key, t) for t in candidates
        ))
        async with self.lock:
            new = set()
            for term, safe in zip(candidates, verdicts):
                if term in self._terms:
                    continue
                if safe:
                    new.add(term)
                else:
                    self.rejected.add(term)
            if not new:
                return []
            self._terms |= new
            self.current_re = build_terms_regex(sorted(self._terms))
            LEARNED_TERMS_FILE.write_text(json.dumps(sorted(self._terms - self._base_lower), indent=2))
            return sorted(new)

    async def matches(self, text: str) -> re.Match | None:
        async with self.lock:
            regex = self.current_re
        return regex.search(text)


async def classify(client: httpx.AsyncClient, api_key: str, row: dict) -> dict:
    text = record_text(row)
    verdict: dict = {"stage": "llm"}
    last_error = "unknown"
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = await client.post(
                URL, headers={"Authorization": f"Bearer {api_key}"},
                json={"model": MODEL,
                      "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user", "content": text}],
                      "max_tokens": 120, "temperature": 0.0},
                timeout=30.0,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
            start, end = content.find("{"), content.rfind("}")
            parsed = json.loads(content[start:end + 1])
            verdict["keep"] = bool(parsed.get("keep"))
            verdict["reason"] = str(parsed.get("reason", ""))[:200]
            verdict["disqualifying_terms"] = [
                str(t)[:60] for t in parsed.get("disqualifying_terms", []) if isinstance(t, str)
            ][:10]
            return verdict
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            await asyncio.sleep(min(2 ** attempt, 20))
    verdict["keep"] = False
    verdict["reason"] = f"classification_failed: {last_error}"[:200]
    verdict["disqualifying_terms"] = []
    return verdict


def _load_done() -> dict[str, dict]:
    done = {}
    if VERDICTS.exists():
        for line in open(VERDICTS):
            line = line.strip()
            if line:
                v = json.loads(line)
                done[str(v["id"])] = v
    return done


async def sweep_kept(rows_by_id: dict, kept_ids: set, learned: LearnedTerms, out_f, counts: Counter) -> None:
    async with learned.lock:
        regex = learned.current_re
    newly_dropped = []
    for rid in list(kept_ids):
        text = record_text(rows_by_id[rid])
        match = regex.search(text)
        if match:
            newly_dropped.append((rid, match))
    for rid, match in newly_dropped:
        out_f.write(json.dumps({"id": rid, "stage": "regex_retroactive", "keep": False,
                                 "reason": f"regex (retroactive sweep): {match.group(0)[:40]}"}) + "\n")
        kept_ids.discard(rid)
        counts["retroactive_drop"] += 1
    out_f.flush()
    print(f"  sweep at {counts['done']}: retroactively dropped "
          f"{len(newly_dropped)}/{len(kept_ids) + len(newly_dropped)} previously-kept records", flush=True)


async def main() -> None:
    api_key = os.environ["KIMI_API_KEY"]
    ds = cast(DatasetDict, datasets.load_dataset("POLARIS-Project/Polaris-Dataset-53K", streaming=False))
    rows = list(ds["train"])
    rows_by_id = {str(i): rows[i] for i in range(len(rows))}
    print(f"{len(rows)} parsed rows", flush=True)

    judged = _load_done()
    print(f"resuming: {len(judged)} verdicts already recorded", flush=True)
    kept_ids = {rid for rid, v in judged.items() if v.get("keep")}

    initial_learned = []
    if LEARNED_TERMS_FILE.exists():
        initial_learned = json.loads(LEARNED_TERMS_FILE.read_text())
        print(f"resuming with {len(initial_learned)} previously learned terms", flush=True)

    to_process = [(rid, row) for rid, row in rows_by_id.items() if rid not in judged]
    print(f"{len(to_process)} rows left to process", flush=True)

    counts: Counter = Counter()
    counts["done"] = len(judged)
    lock = asyncio.Lock()

    async def worker(queue: "asyncio.Queue", client, api_key, learned, out_f):
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            rid, row = item
            text = record_text(row)
            match = YEAR_RE.search(text) or await learned.matches(text)
            if match:
                verdict = {"id": rid, "stage": "regex", "keep": False,
                           "reason": f"regex: {match.group(0)[:40]}"}
            else:
                verdict = await classify(client, api_key, row)
                verdict["id"] = rid
                if not verdict["keep"] and verdict.get("disqualifying_terms"):
                    added = await learned.add(verdict["disqualifying_terms"])
                    if added:
                        async with lock:
                            counts["terms_learned"] += len(added)
                            print(f"  learned term(s) from row {rid}: {added}", flush=True)
            do_sweep = False
            async with lock:
                out_f.write(json.dumps(verdict) + "\n")
                out_f.flush()
                counts["done"] += 1
                counts[f"{verdict['stage']}_drop" if not verdict["keep"] else f"{verdict['stage']}_keep"] += 1
                if verdict["keep"]:
                    kept_ids.add(rid)
                if counts["done"] % 500 == 0:
                    print(f"  processed {counts['done']} | learned_terms={counts['terms_learned']}", flush=True)
                if counts["done"] % SWEEP_EVERY == 0:
                    do_sweep = True
            if do_sweep:
                await sweep_kept(rows_by_id, kept_ids, learned, out_f, counts)
            queue.task_done()

    with open(VERDICTS, "a") as out_f:
        async with httpx.AsyncClient() as client:
            learned = LearnedTerms(initial_learned, client, api_key)
            if kept_ids:
                print(f"resuming: sweeping {len(kept_ids)} previously-kept rows...", flush=True)
                await sweep_kept(rows_by_id, kept_ids, learned, out_f, counts)
            if to_process:
                queue: asyncio.Queue = asyncio.Queue()
                for item in to_process:
                    queue.put_nowait(item)
                for _ in range(CONCURRENCY):
                    queue.put_nowait(None)
                workers = [asyncio.create_task(worker(queue, client, api_key, learned, out_f))
                           for _ in range(CONCURRENCY)]
                await asyncio.gather(*workers)
            if kept_ids:
                await sweep_kept(rows_by_id, kept_ids, learned, out_f, counts)

    print(f"stage summary: {dict(counts)}", flush=True)

    judged = _load_done()
    old_ids = sorted((rid for rid, v in judged.items() if v.get("keep")), key=int)
    OLD_IDS_FILE.write_text(json.dumps(old_ids))
    report = {
        "total": len(rows), "kept": len(old_ids),
        "survival_rate_pct": round(100 * len(old_ids) / max(len(rows), 1), 2),
        "terms_learned_total": len(json.loads(LEARNED_TERMS_FILE.read_text())) if LEARNED_TERMS_FILE.exists() else 0,
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
