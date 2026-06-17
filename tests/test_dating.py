"""Tests for date inference — the part most likely to be silently wrong.

Covers the full resolution order (filename -> header -> mtime / today), the
plausibility bounds, rejection of impossible dates, and the explicit/manual path.
"""

from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path

import pytest

from journal.dating import (
    date_from_filename,
    date_from_header,
    infer_date_for_file,
    infer_date_for_text,
    to_date_int,
)


# --- filename ------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,expected",
    [
        ("2013-05-04", date(2013, 5, 4)),
        ("2013_5_4", date(2013, 5, 4)),
        ("2013.05.04", date(2013, 5, 4)),
        ("2013/05/04", date(2013, 5, 4)),
        ("20130504", date(2013, 5, 4)),
        ("journal-2019-12-31-evening", date(2019, 12, 31)),
        ("entry_20200229", date(2020, 2, 29)),  # leap day, valid
        ("diary 2024-01-01 draft", date(2024, 1, 1)),
    ],
)
def test_date_from_filename_hits(name, expected):
    assert date_from_filename(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "no-date-here",
        "notes",
        "2013-13-01",       # month 13
        "2013-02-30",       # impossible day
        "1899-01-01",       # before MIN_YEAR
        "12345",            # not an 8-digit date
        "201305",           # too short
        "v2-final",
    ],
)
def test_date_from_filename_misses(name):
    assert date_from_filename(name) is None


def test_compact_8digit_not_matched_inside_longer_run():
    # 12 digits should not be misread as a date by the compact pattern.
    assert date_from_filename("201305041230") is None


# --- short / year-last filename dates (e.g. 1.03.25) ---------------------- #
@pytest.mark.parametrize(
    "name,expected",
    [
        ("1.03.25", date(2025, 1, 3)),          # M.D.YY, US default
        ("01.03.2025", date(2025, 1, 3)),       # MM.DD.YYYY
        ("12.31.24", date(2024, 12, 31)),       # M.D.YY end of year
        ("1-3-25", date(2025, 1, 3)),           # dash separators
        ("1/3/25", date(2025, 1, 3)),           # slash separators
        ("journal 3.14.2013 notes", date(2013, 3, 14)),
        ("13.05.25", date(2025, 5, 13)),        # first>12 -> auto day-first
        ("25.12.2024", date(2024, 12, 25)),     # day-first, unambiguous
        ("1.03.99", date(1999, 1, 3)),          # 2-digit year pivot -> 1999
    ],
)
def test_year_last_filename_dates(name, expected):
    assert date_from_filename(name) == expected


def test_dayfirst_toggle(monkeypatch):
    from journal import config
    # Ambiguous 03.04.25: US default = Mar 4; day-first = Apr 3.
    monkeypatch.setattr(config, "DATE_DAYFIRST", False)
    assert date_from_filename("03.04.25") == date(2025, 3, 4)
    monkeypatch.setattr(config, "DATE_DAYFIRST", True)
    assert date_from_filename("03.04.25") == date(2025, 4, 3)


def test_short_date_needs_two_digit_year():
    # 1.3.5 is too ambiguous (1-digit year) and must not match.
    assert date_from_filename("1.3.5") is None


# --- header --------------------------------------------------------------- #
@pytest.mark.parametrize(
    "body,expected",
    [
        ("2013-05-04\nDear diary...", date(2013, 5, 4)),
        ("May 4, 2013\nWhat a day.", date(2013, 5, 4)),
        ("4 May 2013\nWhat a day.", date(2013, 5, 4)),
        ("Sept. 9, 2021\nrain.", date(2021, 9, 9)),
        ("Title line\n2018/07/22\nbody", date(2018, 7, 22)),
    ],
)
def test_date_from_header_hits(body, expected):
    assert date_from_header(body) == expected


def test_header_only_scans_first_lines():
    # A date buried on line 10 must NOT be picked up.
    body = "\n".join(["intro"] * 9 + ["2013-05-04"])
    assert date_from_header(body) is None


def test_header_no_date():
    assert date_from_header("just some thoughts\nno dates at all") is None


# --- file resolution order ------------------------------------------------ #
def test_filename_beats_header(tmp_path):
    f = tmp_path / "2013-05-04.txt"
    f.write_text("June 1, 2020\nbody")
    d, src = infer_date_for_file(f, f.read_text())
    assert (d, src) == (date(2013, 5, 4), "filename")


def test_header_used_when_filename_blank(tmp_path):
    f = tmp_path / "untitled.txt"
    f.write_text("June 1, 2020\nbody")
    d, src = infer_date_for_file(f, f.read_text())
    assert (d, src) == (date(2020, 6, 1), "header")


def test_mtime_fallback(tmp_path):
    f = tmp_path / "untitled.txt"
    f.write_text("no date anywhere\njust text")
    target = time.mktime(date(2016, 3, 15).timetuple())
    os.utime(f, (target, target))
    d, src = infer_date_for_file(f, f.read_text())
    assert src == "mtime"
    assert d == date(2016, 3, 15)


# --- pasted / typed text -------------------------------------------------- #
def test_explicit_manual_date_wins():
    d, src = infer_date_for_text("Some text with June 1, 2020 in it",
                                 explicit="2013-05-04")
    assert (d, src) == (date(2013, 5, 4), "manual")


def test_text_falls_back_to_header():
    d, src = infer_date_for_text("May 4, 2013\nbody", explicit=None)
    assert (d, src) == (date(2013, 5, 4), "header")


def test_text_falls_back_to_today():
    today = date(2026, 6, 17)
    d, src = infer_date_for_text("no date here", explicit=None, today=today)
    assert (d, src) == (today, "today")


def test_blank_explicit_is_ignored():
    d, src = infer_date_for_text("May 4, 2013\nbody", explicit="   ")
    assert src == "header"


def test_garbage_explicit_falls_through():
    d, src = infer_date_for_text("no date", explicit="not a date",
                                 today=date(2026, 6, 17))
    assert src == "today"


# --- helpers -------------------------------------------------------------- #
def test_to_date_int():
    assert to_date_int(date(2013, 5, 4)) == 20130504
    assert to_date_int(date(2024, 12, 31)) == 20241231
