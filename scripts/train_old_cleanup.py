"""Remove web links from the train_old problem statements.

Some train_old statements were scraped with their web links: wiki links around a
word ("[square](https://artofproblemsolving.com/wiki/...)"), forum notes, source
links, and figures hosted on image sites. This script writes two files that
polaris.py applies to the train_old split (the train split is served unchanged):

  train_old_problem_fixes.json  {id: cleaned problem} for statements that stand
                                without the link (the link is removed, the word kept)
  train_old_drops.json          [{id, reason}] for tasks removed from train_old
                                (the figure the problem depends on is only a link,
                                the statement holds Asymptote [asy] figure code,
                                or the problem names a later technology)

Ids are unchanged. Run from the repo root against the full dataset:
    POLARIS_DATA_PATH=data/polaris_tasks.parquet python scripts/train_old_cleanup.py
The run fails if any served train_old statement still contains a link.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import polaris  # noqa: E402

# A markdown link around a word, pointing at the AoPS wiki: keep the word.
WIKI_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(https?://(?:www\.)?artofproblemsolving\.com/wiki/[^)\s]*\)")
# An image the statement shows.
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)|\[img\]", re.IGNORECASE)
# Figure code for Asymptote, a later drawing language: the whole task is dropped.
ASY = re.compile(r"\[asy\]", re.IGNORECASE)
LINK = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|org|net|kz|ro|gy|be)/", re.IGNORECASE)

# Notes appended to a statement (forum remarks, source and video links).
# Each old text must occur exactly once in its problem.
EDITS: dict[str, list[tuple[str, str]]] = {
    "9349": [(
        "\n\n\n=Video Solution for Problems 21-25\n"
        "[https://youtu.be/-mi3qziCuec](https://artofproblemsolving.comhttps://youtu.be/-mi3qziCuec)",
        "",
    )],
    "15193": [(
        "\n\nP.S.: This is basically the same problem as "
        "http://www.artofproblemsolving.com/Forum/viewtopic.php?f=57&t=548550.",
        "",
    )],
    "21424": [(
        "\n\nProblem was post earlier [here](http://www.artofproblemsolving.com/Forum/viewtopic.php?f=67&t=29456&hilit=silk+road)"
        " , but solution not gives and olympiad doesn't indicate, so I post it again  :blush:\n\n"
        "Official solution [here](http://www.artofproblemsolving.com/Forum/viewtopic.php?f=125&t=365714&p=2011659#p2011659)",
        "",
    )],
    "27668": [(" [source](http://matol.kz/comments/3373/show)", "")],
    "30932": [(
        "\n\nrelated:\nhttps://artofproblemsolving.com/community/c6h515011\n"
        "https://artofproblemsolving.com/community/c6h108341 (Putnam 1976)\n"
        "https://artofproblemsolving.com/community/c6h368488\n"
        "https://artofproblemsolving.com/community/c6h445330\n"
        "https://artofproblemsolving.com/community/c6h378928",
        "",
    )],
}

MANUAL_DROPS = {
    "36759": "the figure the problem depends on is only a web link",
    "27031": "the figure the problem depends on is only an external image",
    "46409": "names pandas (a later library)",
    "48036": "browser tabs and Google searches as everyday habits (later setting)",
}


def main() -> None:
    old_ids = polaris._load_old_ids()
    tasks = [t for t in polaris.polaris_tasks if t["id"] in old_ids]
    fixes: dict[str, str] = {}
    drops: list[dict[str, str]] = []
    for task in tasks:
        tid, problem = task["id"], task["problem"]
        if tid in MANUAL_DROPS:
            drops.append({"id": tid, "reason": MANUAL_DROPS[tid]})
            continue
        if ASY.search(problem):
            drops.append({"id": tid, "reason": "Asymptote figure code (2004)"})
            continue
        new = problem
        for old, repl in EDITS.get(tid, []):
            assert new.count(old) == 1, f"task {tid}: edit text not found exactly once"
            new = new.replace(old, repl)
        new = WIKI_LINK.sub(lambda m: m.group(1).strip(), new)
        if LINK.search(new) and IMAGE.search(new):
            drops.append({"id": tid, "reason": "the figure the problem depends on is only an external image"})
            continue
        if LINK.search(new):
            raise SystemExit(f"task {tid}: a link is left that no rule handles")
        if new != problem:
            fixes[tid] = new
    missing = (set(EDITS) | set(MANUAL_DROPS)) - {t["id"] for t in tasks}
    assert not missing, f"ids not in train_old: {sorted(missing)}"
    drops.sort(key=lambda d: int(d["id"]))
    (ROOT / "train_old_problem_fixes.json").write_text(
        json.dumps(dict(sorted(fixes.items(), key=lambda kv: int(kv[0]))), indent=4, ensure_ascii=False) + "\n"
    )
    (ROOT / "train_old_drops.json").write_text(json.dumps(drops, indent=4, ensure_ascii=False) + "\n")
    print(f"train_old: {len(tasks)} tasks, {len(fixes)} statements cleaned, {len(drops)} dropped")


if __name__ == "__main__":
    main()
