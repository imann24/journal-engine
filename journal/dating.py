"""Date inference — the part most likely to be silently wrong, so it is the part
with the most tests (tests/test_dating.py).

Resolution order for a file-backed entry:
    1. a date in the filename       -> source "filename"
    2. a date in the first few lines -> source "header"
    3. the file's modification time  -> source "mtime"

For pasted / typed entries there is no filename or mtime, so the caller passes an
explicit date (source "manual") or we fall back to the header, then to today
(source "today").
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

from dateutil import parser as dateparser

from . import config

# How many leading lines of the body to scan for a header date.
HEADER_LINES = 4

# Plausible range for a journal entry date. Anything outside is treated as a
# false positive (e.g. "1024" in prose, a year like 2400).
MIN_YEAR = 1990
MAX_YEAR = 2100

# Filename date patterns, tried in this order:
#   compact:    exactly 8 digits, YYYYMMDD               (20130504)
#   year-first: 4-digit year, then 1-2 digit M and D     (2013-05-04, 2013_5_4)
#   year-last:  1-2 digit X and Y, then 2-4 digit year   (1.03.25, 01/03/2025)
# Separators are any of . - / _ .
_FILENAME_COMPACT_RE = re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)")
_FILENAME_YMD_RE = re.compile(r"(?<!\d)(\d{4})[-_./](\d{1,2})[-_./](\d{1,2})(?!\d)")
_FILENAME_MDY_RE = re.compile(r"(?<!\d)(\d{1,2})[-_./](\d{1,2})[-_./](\d{2,4})(?!\d)")


def _normalize_year(y: int) -> int:
    """Expand a 2-digit year with a 1970–2069 pivot (25 -> 2025, 99 -> 1999)."""
    if y < 100:
        return 2000 + y if y <= 69 else 1900 + y
    return y

# Textual dates allowed in a header line.
_TEXTUAL_DATE_RE = re.compile(
    r"\b("
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"                         # 2013-05-04
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"
    r")\b",
    re.IGNORECASE,
)


def _valid_ymd(y: int, mo: int, d: int) -> date | None:
    """Build a date, returning None for out-of-range or impossible dates
    (date() rejects e.g. Feb 30, month 13, day 32)."""
    if not (MIN_YEAR <= y <= MAX_YEAR):
        return None
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def date_from_filename(name: str) -> date | None:
    """Extract a date from a filename stem, or None.

    Handles YYYYMMDD, year-first (2013-05-04), and year-last short forms with
    2- or 4-digit years (1.03.25, 01/03/2025). Ambiguous month/day order
    defaults to US month-first; flip with JOURNAL_DATE_DAYFIRST.
    """
    m = _FILENAME_COMPACT_RE.search(name)
    if m:
        got = _valid_ymd(int(m[1]), int(m[2]), int(m[3]))
        if got:
            return got

    m = _FILENAME_YMD_RE.search(name)              # YYYY sep M sep D
    if m:
        got = _valid_ymd(int(m[1]), int(m[2]), int(m[3]))
        if got:
            return got

    m = _FILENAME_MDY_RE.search(name)              # X sep Y sep YY[YY]
    if m:
        a, b, year = int(m[1]), int(m[2]), _normalize_year(int(m[3]))
        orders = [(b, a), (a, b)] if config.DATE_DAYFIRST else [(a, b), (b, a)]
        for mo, d in orders:
            got = _valid_ymd(year, mo, d)
            if got:
                return got
    return None


def date_from_header(body: str) -> date | None:
    """Extract a date from the first few lines of the body, or None."""
    head = "\n".join(body.splitlines()[:HEADER_LINES])
    m = _TEXTUAL_DATE_RE.search(head)
    if not m:
        return None
    try:
        parsed = dateparser.parse(m.group(1), fuzzy=True).date()
    except (ValueError, OverflowError, TypeError):
        return None
    if MIN_YEAR <= parsed.year <= MAX_YEAR:
        return parsed
    return None


def infer_date_for_file(path: Path, body: str) -> tuple[date, str]:
    """(date, source) for a file-backed entry: filename -> header -> mtime."""
    got = date_from_filename(Path(path).stem)
    if got:
        return got, "filename"

    got = date_from_header(body)
    if got:
        return got, "header"

    return datetime.fromtimestamp(Path(path).stat().st_mtime).date(), "mtime"


def infer_date_for_text(
    body: str,
    explicit: str | date | None = None,
    today: date | None = None,
) -> tuple[date, str]:
    """(date, source) for a pasted/typed entry with no filename or mtime.

    explicit (an ISO string or a date) wins as source "manual"; otherwise we try
    the header, then fall back to today's date (source "today").
    """
    if explicit:
        parsed = explicit if isinstance(explicit, date) else _parse_iso(explicit)
        if parsed:
            return parsed, "manual"

    got = date_from_header(body)
    if got:
        return got, "header"

    return (today or date.today()), "today"


def _parse_iso(s: str) -> date | None:
    s = s.strip()
    if not s:
        return None
    try:
        parsed = dateparser.parse(s, fuzzy=False).date()
    except (ValueError, OverflowError, TypeError):
        return None
    if MIN_YEAR <= parsed.year <= MAX_YEAR:
        return parsed
    return None


def to_date_int(d: date) -> int:
    """YYYYMMDD as an int, for cheap range filters in LanceDB."""
    return d.year * 10000 + d.month * 100 + d.day
