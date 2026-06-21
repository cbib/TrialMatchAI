# Registry Updater

`trialmatchai-update-registry` updates the local LanceDB-backed trial registry from ClinicalTrials.gov v2. It is command-based and safe to run from cron, systemd timers, or GitHub Actions.

## Basic Usage

```bash
uv run trialmatchai-update-registry --max-studies 500
```

When no keyword is provided, TrialMatchAI uses broad default keyword queries covering oncology, cardiology, neurology, rare disease, immunology, infectious disease, metabolic disease, hematology, pediatrics, and precision medicine.

Use explicit keywords:

```bash
uv run trialmatchai-update-registry \
  --keyword "lung cancer" \
  --keyword "EGFR" \
  --since 2026-06-01 \
  --max-studies 250
```

Use a keyword file:

```bash
uv run trialmatchai-update-registry --keywords-file data/registry/keywords.txt
```

Dry-run:

```bash
uv run trialmatchai-update-registry --dry-run --max-studies 25
```

Dry-runs do not write raw JSON, normalized trial JSON, manifests, reports, or LanceDB tables unless an explicit `--report-path` is provided.

## Idempotency

Each fetched source record is hashed after canonical JSON serialization. If the latest manifest entry for an NCT ID has the same source hash, the updater skips preparation and indexing.

Changed studies are written and then upserted:

- Trial rows are replaced by `nct_id`.
- Criteria rows are replaced by `nct_id`.
- Failures are recorded per study and do not stop the run unless the failure rate exceeds `TRIALMATCHAI_REGISTRY_FAILURE_THRESHOLD`.

## Cron

Run daily at 02:30:

```cron
30 2 * * * cd /opt/TrialMatchAI && /usr/local/bin/uv run trialmatchai-update-registry --max-studies 1000 >> logs/registry-update.log 2>&1
```

## systemd

Service:

```ini
[Unit]
Description=TrialMatchAI registry update

[Service]
Type=oneshot
WorkingDirectory=/opt/TrialMatchAI
ExecStart=/usr/local/bin/uv run trialmatchai-update-registry --max-studies 1000
```

Timer:

```ini
[Unit]
Description=Daily TrialMatchAI registry update

[Timer]
OnCalendar=*-*-* 02:30:00
Persistent=true

[Install]
WantedBy=timers.target
```

## Healthcheck

```bash
uv run trialmatchai-healthcheck --registry --require-tables
```

Use `--require-tables` after the first successful update or indexing run.
