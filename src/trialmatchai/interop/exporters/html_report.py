"""Self-contained HTML results report for a matched patient.

Joins ``ranked_trials.json`` + the per-trial CoT eligibility evaluations + trial
metadata + the patient matching summary into one offline ``report.html`` (no
server, no build step). All dynamic content is embedded as a JSON island and
rendered client-side via safe DOM APIs, so there is no server-side HTML
templating to escape and no templating dependency.
"""

from __future__ import annotations

import html as _html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

_TEMPLATE = Path(__file__).parent / "templates" / "report.html"
_LOGO = Path(__file__).parent / "templates" / "logo.png"
_DATA_PLACEHOLDER = "__REPORT_DATA__"
_LOGO_PLACEHOLDER = "__LOGO_SRC__"
# the sentinel eligibility_base writes for an unparseable model response
_ERROR_SENTINEL = "invalid_json_response"
_META_FIELDS = (
    "brief_title",
    "brief_summary",
    "overall_status",
    "phase",
    "study_type",
    "condition",
    "gender",
    "minimum_age",
    "maximum_age",
    "source_url",
)


def _read_json(path: Path) -> Any | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _read_cot(path: Path) -> str | None:
    """Extract the model's <think>…</think> reasoning from a per-trial .txt.

    Returns None when the file is absent or has no thinking block (e.g. no-think
    runs), so the reasoning panel only appears when there is something to show.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return None
    low = raw.lower()
    i = low.find("<think>")
    if i == -1:
        return None
    start = i + len("<think>")
    end = low.find("</think>", start)
    text = (raw[start:end] if end != -1 else raw[start:]).strip()
    return text or None


def _ranked_records(ranked_json: Any) -> list[dict]:
    """``ranked_trials.json`` is ``{"RankedTrials": [...]}`` or a bare list."""
    if isinstance(ranked_json, Mapping):
        return list(ranked_json.get("RankedTrials", []))
    if isinstance(ranked_json, list):
        return ranked_json
    return []


def _narrative(val: Any) -> str:
    if isinstance(val, (list, tuple)):
        return " ".join(str(x) for x in val if str(x).strip())
    return str(val) if val else ""


def _trial_meta(raw: Any) -> dict:
    """Display metadata only; the ``*_vector`` embedding fields are dropped."""
    if not isinstance(raw, Mapping):
        return {}
    meta = {k: raw[k] for k in _META_FIELDS if raw.get(k) not in (None, "")}
    names = [
        i.get("name")
        for i in (raw.get("intervention") or [])
        if isinstance(i, Mapping) and i.get("name")
    ]
    if names:
        meta["interventions"] = names
    return meta


def _criteria(items: Any) -> list[dict]:
    out: list[dict] = []
    for it in items or []:
        if isinstance(it, Mapping):
            out.append(
                {
                    "criterion": str(it.get("Criterion", "")).strip(),
                    "classification": str(it.get("Classification", "")).strip(),
                    "justification": str(it.get("Justification", "")).strip(),
                }
            )
    return out


def build_report_model(
    *,
    patient_summary: Mapping[str, Any],
    ranked: Any,
    eligibility_by_id: Mapping[str, Any],
    meta_by_id: Mapping[str, Any],
    cot_by_id: Mapping[str, Any] | None = None,
    generated_at: str,
    run_info: Mapping[str, Any] | None = None,
) -> dict:
    """Pure join of a patient's result artifacts into a render-ready model.

    No I/O — the caller supplies already-loaded data, so this is unit-testable.
    Trials keep ``ranked_trials.json`` order; ``rank`` is the 1-based position.
    """
    trials: list[dict] = []
    for rank, rec in enumerate(_ranked_records(ranked), start=1):
        tid = str(rec.get("TrialID", "")).strip()
        if not tid:
            continue
        elig = eligibility_by_id.get(tid) or {}
        # an unparseable model response left an error sentinel, not an evaluation
        reasoning_ok = bool(elig) and elig.get("error") != _ERROR_SENTINEL
        meta = _trial_meta(meta_by_id.get(tid))
        trials.append(
            {
                "rank": rank,
                "trial_id": tid,
                "score": rec.get("Score"),
                "reranker_score": rec.get("RerankerScore"),
                "first_level_score": rec.get("FirstLevelScore"),
                "meta": meta,
                "metadata_available": bool(meta),
                "final_decision": elig.get("Final Decision") if reasoning_ok else None,
                "inclusion": _criteria(elig.get("Inclusion_Criteria_Evaluation")) if reasoning_ok else [],
                "exclusion": _criteria(elig.get("Exclusion_Criteria_Evaluation")) if reasoning_ok else [],
                "reasoning_available": reasoning_ok,
                "cot": (cot_by_id or {}).get(tid),
            }
        )
    return {
        "patient": {
            "id": patient_summary.get("patient_id"),
            "age": patient_summary.get("age"),
            "sex": patient_summary.get("gender"),
            "main_conditions": list(patient_summary.get("main_conditions", [])),
            "other_conditions": list(patient_summary.get("other_conditions", [])),
            "narrative": _narrative(patient_summary.get("patient_narrative")),
        },
        "trials": trials,
        "generated_at": generated_at,
        "run": dict(run_info or {}),
    }


def _logo_data_uri() -> str:
    """Inline the logo thumbnail as a base64 data URI (empty string if missing)."""
    try:
        import base64

        return "data:image/png;base64," + base64.b64encode(_LOGO.read_bytes()).decode("ascii")
    except Exception:
        return ""


def render_html_report(model: Mapping[str, Any]) -> str:
    """Embed the model as a tag-safe JSON island in the static template."""
    data = json.dumps(model, ensure_ascii=False, default=str)
    # neutralize </script> injection — LLM free text may contain "</script>";
    # JSON.parse reads the escaped sequences back unchanged.
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = _TEMPLATE.read_text(encoding="utf-8").replace(_DATA_PLACEHOLDER, data)
    return html.replace(_LOGO_PLACEHOLDER, _logo_data_uri())


def _load_meta(trial_id: str, folders: list[Path]) -> dict | None:
    for folder in folders:
        raw = _read_json(folder / f"{trial_id}.json")
        if isinstance(raw, Mapping):
            return raw
    return None


def profile_to_html_report(
    patient_dir: str | Path,
    *,
    summary_dir: str | Path | None = None,
    trial_meta_folders: list[str | Path] | None = None,
    generated_at: str | None = None,
    run_info: Mapping[str, Any] | None = None,
) -> str:
    """Read a patient's result dir and return a self-contained HTML report.

    ``patient_dir`` is ``<output_dir>/<patient_id>/``. Metadata folders are tried
    in order (processed_trials, then trials_jsons); ids with no metadata (e.g.
    Dutch ``NL…`` registry trials) degrade to id + score + verdict only.
    """
    patient_dir = Path(patient_dir)
    ranked = _read_json(patient_dir / "ranked_trials.json") or {}
    records = _ranked_records(ranked)
    folders = [Path(f) for f in (trial_meta_folders or ["data/processed_trials", "data/trials_jsons"])]

    eligibility_by_id: dict[str, Any] = {}
    meta_by_id: dict[str, Any] = {}
    cot_by_id: dict[str, Any] = {}
    for rec in records:
        tid = str(rec.get("TrialID", "")).strip()
        if not tid:
            continue
        elig = _read_json(patient_dir / f"{tid}.json")
        if elig is not None:
            eligibility_by_id[tid] = elig
        if tid not in meta_by_id:
            meta = _load_meta(tid, folders)
            if meta:
                meta_by_id[tid] = meta
        cot = _read_cot(patient_dir / f"{tid}.txt")
        if cot:
            cot_by_id[tid] = cot

    patient_id = patient_dir.name
    summary = _read_json(Path(summary_dir) / f"{patient_id}.json") if summary_dir else None
    summary = summary or _read_json(patient_dir / "keywords.json") or {"patient_id": patient_id}

    model = build_report_model(
        patient_summary=summary,
        ranked=ranked,
        eligibility_by_id=eligibility_by_id,
        meta_by_id=meta_by_id,
        cot_by_id=cot_by_id,
        generated_at=generated_at or datetime.now().strftime("%Y-%m-%d %H:%M"),
        run_info=run_info,
    )
    return render_html_report(model)


_INDEX_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TrialMatchAI — Match Reports</title>
<style>
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         color:#1a1a1a; line-height:1.5; }
  .wrap { max-width:720px; margin:0 auto; padding:32px 20px 64px; }
  .eyebrow { font:600 12px/1 ui-monospace,Menlo,Consolas,monospace; letter-spacing:.12em;
             text-transform:uppercase; color:#0072b2; }
  h1 { font-size:22px; margin:8px 0 20px; }
  ul { list-style:none; margin:0; padding:0; }
  li { border-bottom:1px solid #e2e2e2; padding:14px 4px; display:flex; align-items:baseline; gap:12px; }
  li a { color:#0072b2; text-decoration:none; font-weight:600; }
  li a:hover { text-decoration:underline; }
  .n { color:#5a5a5a; font-size:13px; margin-left:auto; font-family:ui-monospace,Menlo,Consolas,monospace; }
  footer { color:#8a8a8a; font:12px/1.5 ui-monospace,Menlo,Consolas,monospace; margin-top:18px; }
</style></head><body><div class="wrap">
  <div class="eyebrow">TrialMatchAI</div>
  <h1>Match Reports (__N__ patients)</h1>
  <ul>
__ITEMS__
  </ul>
  <footer>Generated __WHEN__</footer>
</div></body></html>"""


def render_index_html(entries: Sequence[Mapping[str, Any]], generated_at: str) -> str:
    """A minimal index page linking each patient's ``report.html``.

    ``entries`` items: ``{"patient_id", "n_trials", "href"}``.
    """
    items = "\n".join(
        '    <li><a href="{href}">Patient {pid}</a><span class="n">{n} trials</span></li>'.format(
            href=_html.escape(str(e["href"]), quote=True),
            pid=_html.escape(str(e["patient_id"])),
            n=int(e.get("n_trials", 0)),
        )
        for e in entries
    )
    return (
        _INDEX_TEMPLATE.replace("__ITEMS__", items)
        .replace("__WHEN__", _html.escape(str(generated_at)))
        .replace("__N__", str(len(entries)))
    )
