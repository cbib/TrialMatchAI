# Analysis Results

## Test A: Backward-Compatibility (Single Patient Report)

### Scenario 1: Single patient loads with route()
- route() checks: if PATIENTS.length === 1
- Calls: showPatient(PATIENTS[0], true)
- In showPatient(p, solo=true):
  - document.getElementById("overview").hidden = true
  - document.getElementById("topbar").hidden = true (because !!solo = !!true = true)
  - document.getElementById("patient-view").hidden = false
  - mountPatient(p) called
  - Expected behavior: NO front page, NO back link, straight into patient view

**Result**: PASS - backward-compat works as intended

### Scenario 2: render_html_report wraps single model
- model = {"patient": {...}, "trials": [...]}
- render_html_report checks: if "patients" not in model
- Wraps: model = {"patients": [dict(model)], "generated_at": model.get("generated_at", "")}
- Result: {"patients": [{"patient": {...}, "trials": [...]}]}
- Template reads: PATIENTS = DATA.patients || [] → length = 1
- route() logic applies

**Result**: PASS - single model correctly wrapped into one-element island

### Issue Found? Topbar State
Line 198 in template: `<div class="topbar no-print" id="topbar" hidden>`
Topbar starts with hidden attribute (hardcoded).
In showPatient(p, solo=true): topbar.hidden = true (no change)
In showPatient(p, solo=false): topbar.hidden = false (unhides it)

**Status**: SAFE - topbar stays hidden for solo=true path

---

## Test B: Overview Rendering Edge Cases

### Scenario 1: Patient with zero trials
- overviewRow(p) where p.trials = []
- trials.length = 0
- bits.push(0 + " trials") → "0 trials" in pmeta
- top = trials.slice().sort(...)[0] → undefined
- if (top) check → skipped, no "Top match" shown
- tally loop: for each tier, n = 0 → no tally items rendered
- href: "#/p/" + encodeURIComponent(String(pt.id))

**Issue**: If pt.id is null, String(null) = "null" string literal
href becomes "#/p/null" - this will render in the overview but clicking won't find a match in route()

**Severity**: LOW - zero-trial patients are edge case, "null" ID is handled with fallback "—" display

### Scenario 2: Missing patient metadata
- overviewRow(p) where p.patient is undefined or missing
- pt = p.patient || {} → pt = {}
- pt.id → undefined → "Patient —" displayed ✓
- pt.age → undefined → skipped ✓
- pt.sex → undefined/null → skipped ✓
- verdictOf uses t.reasoning_available check → consistent ✓

**Result**: SAFE - fallbacks work

### Scenario 3: Verdict tally counts
- tally uses: verdictOf(t) which returns t.reasoning_available ? verdictLabel(...) : "Not assessed"
- verdictLabel(null) or verdictLabel(undefined) → returns "Not assessed"
- TALLY array includes: { cls: "neutral", glyph: "—", title: "Not assessed", match: ["Not assessed"] }
- Filter counts: trials.filter(t => tier.match.includes(verdictOf(t)))

**Check**: Is verdictOf used consistently?
- Line 327 (overview tally): verdictOf(t) ✓
- Line 465 (per-patient verdict filter): verdictOf used in fillVerdict() ✓
- Line 481 (search filter): verdictOf(t) in apply() ✓

**Result**: CONSISTENT - verdictOf is used everywhere

---

## Test C: CLI report.py Regressions

### Scenario 1: --all with missing ranked_trials.json
- Code (line 52-66):
  ```python
  valid_dirs = []
  for pdir in patient_dirs:
    if (pdir / "ranked_trials.json").exists():
      valid_dirs.append(pdir)
    else:
      logger.warning("Skipping %s: no ranked_trials.json", pdir.name)
  if not valid_dirs:
    logger.error("No patient result dirs...")
    return 1
  ```
- Directories without ranked_trials.json are warned and skipped
- Others are still processed

**Result**: PASS - skips gracefully with warning

### Scenario 2: --all with all dirs invalid
- valid_dirs = []
- Code line 78: if not models: logger.error(...); return 1
- Exit code 1 returned

**Result**: PASS - error handling correct

### Scenario 3: Per-patient model build raises exception
- Code (line 73-77):
  ```python
  try:
    models.append(profile_to_model(...))
  except Exception:
    logger.exception("Skipping %s: failed to build report model", pdir.name)
  ```
- Exception caught, logged, directory skipped
- Other directories continue

**Result**: PASS - resilient to individual failures

### Scenario 4: Dead imports check
- Line 10: import json → used at line 46: json.loads(...)
- Line 11: from datetime import datetime → used at line 57: datetime.now().strftime(...)

**Result**: PASS - both imports are used, no dead code

### Scenario 5: Multi-file behavior regression
- Old behavior (inferred): multiple report.html files per patient
- Current --patient behavior (lines 87-93):
  ```python
  pdir = valid_dirs[0]
  html = profile_to_html_report(...)
  out_path = Path(args.out) if args.out else pdir / "report.html"
  out_path.write_text(html, encoding="utf-8")
  ```
- Still generates report.html per patient ✓

- Current --all behavior (lines 68-84):
  ```python
  render_unified_html(models, generated_at)
  out_path = output_dir / "index.html"
  out_path.write_text(...)
  ```
- Generates ONE index.html with unified report

**Change**: --all now creates index.html instead of individual report.html files
**Potential regression**: Users expecting <output_dir>/<patient_id>/report.html for each patient won't get them with --all

**Severity**: MEDIUM - breaking change in --all behavior, but --patient path is unchanged

