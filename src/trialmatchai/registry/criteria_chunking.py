"""Eligibility-criteria chunking.

Splits a free-text eligibility section into individual, typed criteria. This
folds the domain knowledge from the legacy regex preprocessor — multi-level
enumeration hierarchies, varied inclusion/exclusion headers, parenthetical
protection, semicolon splitting, long-criterion full-stop re-splitting, and
decimal/abbreviation split-exceptions — into one streamlined pass over the text,
so chunk boundaries match the legacy pipeline.

Public API: ``split_eligibility_criteria(text) -> list[{"type", "criterion"}]``
where ``type`` is "inclusion", "exclusion", or "unknown".
"""

from __future__ import annotations

import re

# --- Header detection -------------------------------------------------------
# A line is a section header when, reduced to alpha words, it is essentially the
# inclusion/exclusion label (with optional qualifiers / "for <cohort>"), or it
# is a "...the following inclusion/exclusion criteria" lead-in. This is precise
# enough not to misfire on a real criterion that merely mentions "inclusion".
_HEADER_CORE = re.compile(
    r"^(?:key|main|general|principal|primary|additional|specific|major)?\s*"
    r"(inclusion|exclusion)(?:\s+criteria)?(?:\s+for\s+.+)?$"
)
_HEADER_FOLLOWING = re.compile(r"following\s+(inclusion|exclusion)\s+criteria")

# A header that begins a line and may be followed (after a colon) by inline
# criteria, e.g. "Exclusion Criteria: 1. Pregnancy 2. Active infection".
_LEADING_HEADER = re.compile(
    r"^\s*(?:key|main|general|principal|primary|additional|specific|major)?\s*"
    r"(inclusion|exclusion)(?:\s+criteria)?\s*:\s*(.*)$",
    re.IGNORECASE,
)

# --- Enumeration markers ----------------------------------------------------
# A leading list marker at the start of a line. Numeric markers REQUIRE a
# trailing "." or ")" so a bare "18 years" is not mistaken for item 18, and the
# mandatory trailing whitespace means "e.g." / "i.e." are not treated as markers.
_LEADING_MARKER = re.compile(
    r"^\s*(?:"
    r"[-–—*•·]"  # - – — * • ·
    r"|\(?[0-9]{1,2}(?:\.[0-9]{1,2}){1,3}[.)]?"  # multi-level: 1.2  1.2.3  1.2.
    r"|\(?[0-9]{1,2}[.)]"  # single: 1. 1)
    r"|\([0-9]{1,2}\)"  # (1)
    r"|\([a-zA-Z]\)"  # (a)
    r"|\(?(?:i{1,3}|iv|v|vi{0,3}|ix|x)[.)]"  # roman i. ii) (iv)
    r"|[a-zA-Z][.)]"  # a. b)
    r")\s+"
)

# Mid-line marker (preceded and followed by whitespace) — used to split several
# criteria packed onto one line.
_MIDLINE_MARKER = re.compile(
    r"\s("
    r"[0-9]{1,2}(?:\.[0-9]{1,2}){0,3}[.)]"
    r"|\([0-9a-zA-Z]{1,3}\)"
    r"|[a-z][.)]"
    r"|[-–—•*]"
    r")\s"
)


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
    """Blank out parenthesized spans so markers inside them are not split points."""
    out = list(text)
    depth = 0
    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
            out[i] = " "
        elif char in ")]}":
            depth = max(0, depth - 1)
            out[i] = " "
        elif depth > 0:
            out[i] = " "
    return "".join(out)


def _split_inline(line: str) -> list[str]:
    """Split a line into segments at mid-line enumeration markers (paren-safe)."""
    masked = _mask_parens(line)
    cut_points = [0]
    for match in _MIDLINE_MARKER.finditer(masked):
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


# --- Secondary splits (faithful to the legacy preprocessor) -----------------
# Each assembled criterion is further split on semicolons that fall outside
# parentheses, and any resulting criterion longer than this many characters is
# split into sentences on real full stops (decimal- and abbreviation-aware) —
# matching the legacy split_lines_on_semicolon and split_large_sentences /
# split_on_full_stops, so chunk boundaries stay identical to main.
_LONG_CRITERION_CHARS = 200
_FULLSTOP_SPLIT = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")


def _split_semicolons(text: str) -> list[str]:
    """Split on semicolons not enclosed in parentheses/brackets."""
    masked = _mask_parens(text)  # in-paren chars (incl. ';') become spaces
    parts: list[str] = []
    last = 0
    for i, char in enumerate(masked):
        if char == ";":
            parts.append(text[last:i])
            last = i + 1
    parts.append(text[last:])
    return [p.strip() for p in parts if p.strip()]


def _split_long_criterion(text: str) -> list[str]:
    """Split a long (>200 char) criterion into sentences; short text passes through."""
    if len(text) <= _LONG_CRITERION_CHARS:
        return [text]
    sentences = [s.strip() for s in _FULLSTOP_SPLIT.split(text) if s.strip()]
    return sentences or [text]


def _emit(text: str, criterion_type: str, entries: list[dict[str, str]]) -> None:
    """Clean, semicolon-split, long-criterion-split, filter, and append criteria."""
    cleaned = _clean(text)
    if not cleaned:
        return
    for part in _split_semicolons(cleaned):
        for sentence in _split_long_criterion(part):
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
        text = " ".join(buffered)
        buffered.clear()
        _emit(text, current_type, entries)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue

        # A header at the start of the line sets the section type. If the colon
        # form carries inline criteria after it, process the remainder below.
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

    # Unstructured text (no markers/headers): still semicolon/long-split it.
    _emit(text, "unknown", entries)
    return entries
