"""Split a free-text eligibility section into individual, typed criteria.

Handles enumeration hierarchies, inclusion/exclusion headers, parenthetical
protection, and semicolon/sentence splitting in one pass. The sentence step is
PySBD-inspired (mask -> split -> restore): decimals, single-letter initials/genus
(``E. coli``), and a curated abbreviation list are protected so boundaries never
cut mid-abbreviation.

Public API: ``split_eligibility_criteria(text) -> list[{"type", "criterion"}]``
where ``type`` is "inclusion", "exclusion", or "unknown".
"""

from __future__ import annotations

import re

# --- Header detection -------------------------------------------------------
_HEADER_CORE = re.compile(
    r"^(?:key|main|general|principal|primary|additional|specific|major)?\s*"
    r"(inclusion|exclusion)(?:\s+criteria)?(?:\s+for\s+.+)?$"
)
_HEADER_FOLLOWING = re.compile(r"following\s+(inclusion|exclusion)\s+criteria")

_LEADING_HEADER = re.compile(
    r"^\s*(?:key|main|general|principal|primary|additional|specific|major)?\s*"
    r"(inclusion|exclusion)(?:\s+criteria)?\s*:\s*(.*)$",
    re.IGNORECASE,
)

# --- Enumeration markers ----------------------------------------------------
# Leading list marker at line start. Numeric markers need a trailing "."/")"; an
# uppercase single letter is a marker only before a capital/digit, not a genus ("E. coli").
_LEADING_MARKER = re.compile(
    r"^\s*(?:"
    r"[-–—*•·]"  # - – — * • ·
    r"|\(?[0-9]{1,2}(?:\.[0-9]{1,2}){1,3}[.)]?"  # multi-level: 1.2  1.2.3  1.2.
    r"|\(?[0-9]{1,2}[.)]"  # single: 1. 1)
    r"|\([0-9]{1,2}\)"  # (1)
    r"|\([a-zA-Z]\)"  # (a)
    r"|\(?(?:i{1,3}|iv|v|vi{0,3}|ix|x)[.)]"  # roman i. ii) (iv)
    r"|[a-z][.)]"  # lowercase single-letter list marker: a. b)
    r"|[A-Z][.)](?=\s+[A-Z0-9(])"  # uppercase letter marker ONLY before capital/digit
    r")\s+"
)

# Mid-line marker (preceded and followed by whitespace).
_MIDLINE_MARKER = re.compile(
    r"\s("
    r"[0-9]{1,2}(?:\.[0-9]{1,2}){0,3}[.)]"
    r"|\([0-9a-zA-Z]{1,3}\)"
    r"|[a-z][.)]"
    r"|[-–—•*]"
    r")\s"
)

# A bare "N." mid-line is ambiguous (decimal/ordinal vs list item), so it only
# counts as a split point inside a line that is already a list.
_BARE_NUMERIC_MARKER = re.compile(r"[0-9]{1,2}\.")


def detect_header(line: str) -> str | None:
    """Return 'inclusion'/'exclusion' if the line is a section header, else None."""
    stripped = line.strip().rstrip(":").strip()
    core = re.sub(r"[^a-z ]+", " ", stripped.casefold())
    core = re.sub(r"\s+", " ", core).strip()
    if not core:
        return None
    match = _HEADER_CORE.match(core)
    if match:
        return match.group(1)
    match = _HEADER_FOLLOWING.search(core)
    if match:
        return match.group(1)
    return None


def _mask_parens(text: str) -> str:
    """Blank out balanced parenthesized spans so markers inside them aren't split points.

    A stray unbalanced opener is left intact so it doesn't swallow the rest of the line.
    """
    out = list(text)
    stack: list[int] = []
    for i, char in enumerate(text):
        if char in "([{":
            stack.append(i)
        elif char in ")]}" and stack:
            for j in range(stack.pop(), i + 1):
                out[j] = " "
    return "".join(out)


def _split_inline(line: str) -> list[str]:
    """Split a line into segments at mid-line enumeration markers (paren-safe)."""
    masked = _mask_parens(line)
    # Trust a bare "N." mid-line marker only when the line already starts as a list.
    in_list = _starts_new_criterion(line)
    cut_points = [0]
    for match in _MIDLINE_MARKER.finditer(masked):
        if not in_list and _BARE_NUMERIC_MARKER.fullmatch(match.group(1)):
            continue
        start = match.start(1)
        if start > 0:
            cut_points.append(start)
    cut_points = sorted(set(cut_points))
    segments = [
        line[cut_points[i] : cut_points[i + 1]] for i in range(len(cut_points) - 1)
    ]
    segments.append(line[cut_points[-1] :])
    return segments


def _starts_new_criterion(line: str) -> bool:
    return bool(_LEADING_MARKER.match(line))


def _strip_marker(line: str) -> str:
    return _LEADING_MARKER.sub("", line, count=1)


def _clean(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" :-\t")
    if detect_header(cleaned):
        return ""
    return cleaned


def _is_useful(text: str) -> bool:
    return len(text) >= 3 and detect_header(text) is None


# --- Secondary splits -------------------------------------------------------
def _split_semicolons(text: str) -> list[str]:
    """Split on semicolons not enclosed in parentheses/brackets."""
    masked = _mask_parens(text)
    parts: list[str] = []
    last = 0
    for i, char in enumerate(masked):
        if char == ";":
            parts.append(text[last:i])
            last = i + 1
    parts.append(text[last:])
    return [p.strip() for p in parts if p.strip()]


# Abbreviations whose trailing period is not a sentence boundary. Stored lower-cased
# without the trailing dot, matched case-insensitively.
_ABBREVIATIONS = frozenset(
    {
        # titles
        "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "messrs",
        # latin / academic
        "e.g", "i.e", "etc", "cf", "vs", "viz", "et", "al", "ca", "approx",
        "incl", "excl", "esp", "ibid",
        # reference / numbering
        "no", "nos", "fig", "figs", "ref", "refs", "vol", "ch", "sec", "eq",
        # measurement / clinical units & qualifiers
        "mg", "kg", "mcg", "ug", "ng", "ml", "dl", "cm", "mm", "nm",
        "hr", "hrs", "wk", "wks", "mo", "mos", "yr", "yrs", "min", "mins",
        "sec", "iu", "meq", "mmol", "mol", "max", "avg", "approx",
        # organizations / general
        "inc", "ltd", "co", "corp", "dept", "univ", "assn",
    }
)

_PRD = "\x00"  # placeholder for a protected (non-boundary) period

# longest-first so multi-dot abbreviations ("e.g") win over their prefixes
_ABBREV_RE = re.compile(
    r"(?<![A-Za-z0-9.])(" + "|".join(
        re.escape(a) for a in sorted(_ABBREVIATIONS, key=len, reverse=True)
    ) + r")\.",
    re.IGNORECASE,
)
_DECIMAL = re.compile(r"(\d)\.(?=\d)")
# single-letter initial / genus: "E. coli", "J. Smith", "A. B. Patient"
_INITIAL = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])\.(?=\s)")
# a boundary is sentence-ending punctuation + whitespace + a sentence-like start
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def _protect_periods(text: str) -> str:
    text = _DECIMAL.sub(lambda m: m.group(1) + _PRD, text)
    text = _ABBREV_RE.sub(lambda m: m.group(1).replace(".", _PRD) + _PRD, text)
    text = _INITIAL.sub(lambda m: m.group(1) + _PRD, text)
    return text


def _split_sentences(text: str) -> list[str]:
    """Split into sentences; decimals/initials/abbreviations are protected from becoming boundaries."""
    protected = _protect_periods(text)
    pieces = _SENTENCE_BOUNDARY.split(protected)
    out = [piece.replace(_PRD, ".").strip() for piece in pieces]
    return [s for s in out if s]


def _emit(text: str, criterion_type: str, entries: list[dict[str, str]]) -> None:
    """Clean, semicolon-split, sentence-split, filter, and append criteria."""
    cleaned = _clean(text)
    if not cleaned:
        return
    for part in _split_semicolons(cleaned):
        for sentence in _split_sentences(part):
            sentence = sentence.strip(" :-\t")
            if _is_useful(sentence):
                entries.append({"type": criterion_type, "criterion": sentence})


def split_eligibility_criteria(text: str) -> list[dict[str, str]]:
    if not text or not text.strip():
        return []

    current_type = "unknown"
    entries: list[dict[str, str]] = []
    buffered: list[str] = []

    def flush() -> None:
        if not buffered:
            return
        joined = " ".join(buffered)
        buffered.clear()
        _emit(joined, current_type, entries)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue

        leading = _LEADING_HEADER.match(line)
        if leading:
            flush()
            current_type = leading.group(1).lower()
            remainder = leading.group(2).strip()
            if not remainder:
                continue
            line = remainder
        else:
            header = detect_header(line)
            if header:
                flush()
                current_type = header
                continue

        for segment in _split_inline(line):
            segment = segment.strip()
            if not segment:
                continue
            if _starts_new_criterion(segment):
                flush()
                buffered.append(_strip_marker(segment))
            else:
                buffered.append(segment)

    flush()
    if entries:
        return entries

    _emit(text, "unknown", entries)
    return entries
