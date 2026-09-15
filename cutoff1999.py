"""Shared December-31-1999 cutoff machinery (regexes + filter prompt).

Same doctrine as HHH-2000/HelpSteer-2000/RedTeam-2000/SysPrompt-2000/
WildChat-2000: a regex hard-drop on years 2000-2099 and unambiguous
post-1999 markers, then an LLM verdict with the motivation rule. Copied
from those environments' cutoff1999.py, except:
  - the stage-2 judge call in filter_data.py uses deepseek-v4.1-flash via
    the internal infer.gr.inc gateway instead of gpt-5-mini via OpenAI
    directly (per explicit instruction for this dataset);
  - SYSTEM_PROMPT additionally asks the judge to name the specific
    disqualifying term(s) on a drop, and `build_terms_regex` lets
    filter_data.py fold newly-named terms the regex missed back into the
    regex mid-run -- stage 2 teaches stage 1, so later records (in the same
    run, and future reruns) that use the same term get caught for free
    instead of costing another LLM call. The base POST_2000_TERMS list and
    the rest of the doctrine are otherwise unchanged.
"""

import re

YEAR_RE = re.compile(r"\b20\d{2}\b")  # 2000-2099: even the year 2000 is out

POST_2000_TERMS = [
    # devices & tech products
    "iphone", "ipad", "ipod", "android phone", "smartphone", "smart phone",
    "app store", "macbook", "chromebook", "kindle", "airpods", "apple watch",
    "smartwatch", "smart tv", "fitbit", "gopro", "oculus", "vr headset",
    "usb-c", "tablet computer", "e-reader", "ebook reader", "self-driving",
    "electric vehicle charging", "tesla model", "spacex", "starlink",
    # software & the modern web
    "facebook", "instagram", "tiktok", "snapchat", "whatsapp", "youtube",
    "reddit", "linkedin", "wikipedia", "gmail", "google maps", "googled",
    "googling", "myspace", "hulu", "spotify", "uber", "lyft", "airbnb",
    "doordash", "instacart", "venmo", "zelle", "skype", "zoom call",
    "zoom meeting", "twitch.tv", "onlyfans", "podcast", "selfie", "hashtag",
    "emoji", "social media", "streaming service", "dating app", "mobile app",
    "smart speaker", "alexa", "siri", "cortana", "chatgpt", "gpt-3", "gpt-4",
    "openai", "large language model", "machine learning model", "deepfake",
    "cryptocurrency", "bitcoin", "ethereum", "blockchain", "nft",
    "ransomware", "phishing app",
    # events & people in post-2000 roles
    "9/11", "twin towers attack", "war on terror", "iraq war",
    "hurricane katrina", "fukushima", "brexit", "covid", "omicron",
    "social distancing", "barack obama", "obama", "obamacare", "biden",
    "affordable care act", "donald trump", "president trump", "joe biden",
    "president biden", "kamala harris", "elon musk", "mark zuckerberg",
    "jeff bezos amazon", "black lives matter", "#metoo", "me too movement",
    "january 6", "capitol riot",
    # culture
    "kanye", "taylor swift", "justin bieber", "lady gaga", "rihanna",
    "billie eilish", "beyoncé solo", "game of thrones", "breaking bad",
    "stranger things", "the office us", "squid game", "marvel cinematic",
    "mcu", "avengers movie", "fortnite", "minecraft", "roblox", "pokemon go",
    "candy crush", "angry birds", "league of legends", "world of warcraft",
    "call of duty", "gta v", "gta 5", "xbox", "playstation 3", "ps3", "ps4",
    "ps5", "nintendo switch", "wii", "vape", "vaping", "e-cigarette",
]

def build_terms_regex(extra_terms: "list[str] | None" = None) -> re.Pattern:
    """POST_2000_TERMS plus any additional learned terms, compiled fresh.
    Cheap to call repeatedly (a few hundred terms at most) -- filter_data.py
    recompiles this every time a new term is learned mid-run."""
    terms = list(POST_2000_TERMS) + list(extra_terms or [])
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)


TERMS_RE = build_terms_regex()

SYSTEM_PROMPT = """You are curating a conversation dataset for a language model whose knowledge ends on 31 December 1999. Judge whether a conversation is safe for that cutoff.

Answer KEEP only if BOTH hold:
1. Every fact, event, product, technology, person-in-role, work of culture, and cultural reference mentioned anywhere in the conversation existed or had happened by 31 December 1999.
2. The requests themselves would plausibly have been made before the year 2000. Drop conversations that are post-1999-MOTIVATED even when their surface entities existed earlier -- e.g. a question about Joe Biden driven by his later prominence as vice president or president, or any request whose natural occasion is a later event, technology, or cultural moment.

Answer DROP if anything violates either condition, or if the conversation otherwise leaks a post-1999 setting (dates, prices that only make sense later, later norms or slang).

Anticipating the year 2000 as a future event (Y2K preparations, millennium plans) is acceptable. Ignore the framing that this is a conversation with an AI assistant -- that framing is out of scope.

If you answer false, also name the specific post-1999 term(s) or short phrase(s) (as they literally appear in the text) that are the disqualifying evidence -- but ONLY terms that are inherently and unambiguously post-1999 IN ISOLATION, out of context, such that the bare term/phrase could not plausibly appear in an authentic pre-2000 document at all. This means: a specific brand/product/company name, a named event, or a real person's name used specifically in a role or context that only exists after 1999.

Do NOT name: generic English words or phrases (however evocative), general scientific/technical/academic terms or concepts (even ones popularly associated with modern AI -- e.g. "machine learning" and "artificial intelligence" are 1950s terms), programming languages/libraries/keywords that predate 2000 (e.g. "python", "var"), a bare first name or common surname, or anything that is disqualifying only because of the motivation rule or surrounding context rather than the term itself. When in doubt, name fewer terms rather than more -- omit it entirely if any doubt exists. It is fine to answer {"keep": false} with an empty disqualifying_terms list.

Reply with a single JSON object: {"keep": true} or {"keep": false, "reason": "<short reason>", "disqualifying_terms": ["<term>", ...]}."""

# A second, standalone LLM check re-verifies each NEWLY harvested term before
# it's folded into the shared regex (see filter_data.py's LearnedTerms) --
# defense in depth, since the classify() call above already demonstrated it
# can mis-name generic/pre-2000 terms (python, numpy, "machine learning",
# "house", "minimize", ...) as disqualifying when judging a specific record
# in context. This prompt asks the SAME question with zero record context,
# which is exactly the property a safe global regex term needs.
TERM_VERIFY_SYSTEM_PROMPT = """You are vetting a candidate entry for a regex denylist used to flag post-1999 content in ANY document, out of context. The candidate term/phrase is: %%TERM%%

Answer YES only if this exact term/phrase is so specifically and unambiguously tied to something that came into existence after 31 December 1999 that its mere appearance -- in absolutely any document, regardless of subject -- is a reliable signal of post-1999 content. This means a specific brand/product/company name, named event, or coined technology term.

Answer NO if the term is a generic English word or phrase, a general scientific/technical/academic concept (even one popularly associated with modern technology), a programming language/library/keyword, a bare first name or common surname, an ordinary place name, or anything whose post-1999-ness depends on context rather than the term itself.

Reply with a single JSON object: {"safe_as_global_marker": true} or {"safe_as_global_marker": false}."""

# Hard denylist: terms already demonstrated (in this project's own
# development) to be exactly the kind of generic/pre-2000 mistake described
# above. Checked before even asking the verify prompt -- free, and a backstop
# in case the verify call itself is ever wrong.
TERM_DENYLIST = frozenset({
    "python", "numpy", "scipy", "scipy.optimize", "sympy", "bfgs", "nelder-mead",
    "minimize", "machine learning", "artificial intelligence", "house", "ghost",
    "eloquent", "var", "taylor", "hamilton", "concussions", "esports",
    "hybrid strategies", "predictive analytics", "analytics and data tracking",
    "composite materials", "high-definition broadcasts", "cyber warfare",
    "digital platforms", "mobile apps", "real-time data", "smart parking",
    "meta shifts", "game patches", "professional matches", "differential loading",
    "good people", "better together", "upside down", "banana pancakes",
    "middleware", "aleppo", "gender identity issues", "transgender children",
    "keep america first", "transgender law center",
})


def mentions_post_2000_regex_marker(text: str) -> bool:
    """Cheap, base-list-only check (does not include terms learned by
    filter_data.py's LLM pass) -- useful for a quick regression test that
    the base doctrine's obvious markers never appear in a kept row."""
    return bool(YEAR_RE.search(text)) or bool(TERMS_RE.search(text))
