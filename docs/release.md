# Development and release runbook

The [production roadmap](production-roadmap.md) tracks the remaining clinical,
ranking, agent, and deployment work. These delivery gates establish the first
foundation; they do not qualify the GPU models or a clinical deployment.

## Develop and exercise the CLI

Use Python 3.11 and uv 0.11.24, matching CI. Work on a short-lived branch and keep
`uv.lock` committed. `uv sync --frozen` installs the declared development environment.

```bash
uv sync --frozen
make lint
make test
make demo
make e2e
```

`demo` creates a temporary workspace with synthetic trials and a FHIR patient,
then runs preparation, a real LanceDB index, import, matching, and HTML reporting.
It uses deterministic CPU embeddings and disables model stages. The output states
that eligibility reasoning is unavailable, identifies the report, and prints a
resume command. Use `trialmatchai demo --workdir /path/to/empty-directory` to choose
the location. Initialization publishes its fixtures atomically; if it is interrupted
before completion, retry the same command without `--resume`. Once initialized,
`--resume` requires the original, unmodified fixture configuration;
create a new workspace to try a different configuration. Resume also regenerates a
missing or truncated patient report from the saved ranking/evidence without rerunning
matching. Runtime state is retained
for inspection. No real patient data is used by this example or by CI.

`trialmatchai --version` prints the package version. The new artifact commands
support `--json`: one JSON object on stdout, exit 0 for success or 1 for verification
failure. Argument errors return 2. Demo interruption returns 130. Existing commands
are being standardized separately in roadmap P10.

## Verify transferred artifacts

```bash
trialmatchai artifacts manifest /path/to/artifacts --json
trialmatchai artifacts verify /path/to/artifacts/SHA256SUMS --require-exact --json
```

The manifest uses GNU `sha256sum` text syntax. Paths are relative to its directory,
with nested directories allowed. Verification rejects symlinked manifests, missing or corrupt files,
unsafe paths, duplicate entries, and symlink artifacts. `--require-exact` also
rejects extra files. `--directory` selects the artifact root when the manifest is
stored elsewhere. A manifest hashes every regular file, including hidden files,
except itself; keep the directory limited to the intended deliverables.

A checksum verifies content against an expected value. Obtain that value from a
trusted release or controlled artifact source; hashing an untrusted download does
not establish its publisher. Preserve build provenance alongside release digests.

For bootstrap, obtain a trusted manifest listing `processed_trials.tar.gz` and
`criteria_part_0.zip` through `criteria_part_5.zip` (adjust to `--criteria-chunks`).
Include `models.tar.gz` with `--with-models` and `finetuning_datasets.zip` with
`--finetune-data`.

```bash
trialmatchai bootstrap-data --root /path/to/workspace \
  --checksum-manifest /path/to/trusted/SHA256SUMS
```

The manifest implies `--require-checksums`. Alternatively, strict mode accepts
`TRIALMATCHAI_PROCESSED_TRIALS_SHA256`, `TRIALMATCHAI_CRITERIA_PART_<n>_SHA256`,
`TRIALMATCHAI_MODELS_SHA256`, and `TRIALMATCHAI_FINETUNE_DATA_SHA256`. Every requested
digest must exist and contain exactly 64 hexadecimal characters before work starts.
Cached archives are hashed before extraction. A corrupt cache is retained under a
hidden `.corrupt-<id>` filename and replaced with one fresh download attempt;
a repeatedly corrupt source fails without accepting a different digest. Downloads
use `.part` files; completion markers must be regular files, never symlinks or FIFOs.

Each changed stage extracts into a new sibling directory before replacing its
managed tree. Dedicated criteria/trial/training trees contain only the new archive
contents. The shared `models/` tree replaces incoming and previously recorded owned
roots, preserving unrelated top-level models. Schema-2 completion markers record
archive hashes and owned roots. Strict mode re-extracts legacy markers to migrate
them; legacy model entries without ownership metadata are conservatively preserved
unless an incoming archive replaces the same top-level name.

One bootstrap writer holds a workspace lock. Publication renames the old tree to
`.NAME.bootstrap-previous` and then publishes the staged tree. Failure restores the
old tree; if killed between renames, the next bootstrap recovers it. Successful
replacement retains the old tree in `.NAME-backup-*`. Stop readers during updates:
the two renames leave a short gap, and stages are published individually, not as
one atomic trial/criteria/index snapshot. Keep sufficient disk space for staging,
backups, and quarantined downloads. Review these retained files before removal;
unrelated preserved model files may share hardlinks with their backup copies.

Strict resume checks matching archive provenance and the presence of managed roots.
It does not rehash every extracted file. End-to-end snapshot identity, index
publication, and per-file corpus verification remain roadmap P05 work. Default
bootstrap remains compatible with legacy archives whose digests have not been supplied.

## Local release rehearsal

From a clean checkout with no distributions from older versions:

```bash
make release-check
```

This validates workflows with actionlint and the lockfile, lints, runs tests, builds distributions, creates and
checks their manifest, installs the wheel in an isolated environment, exercises the
public CLI, scans for secrets, and audits installed dependencies. The installed
smoke verifies packaged embedder catalogs, synthetic patient import, a real index,
adult/pediatric filtering, ranked output, HTML content, resume, missing/truncated
report repair without ranking changes, and checksum CLI.
Model downloads are disabled. The smoke is also callable via
`scripts/installed_smoke.py --workspace /empty/path --expect-installed` using an
installed environment's Python.

The dependency advisory exceptions currently live in `Makefile`; CI calls `make
audit` so the lists cannot drift. The default audit covers installed base and dev
dependencies. Optional GPU dependency auditing, dated exception reviews, SBOMs,
and actual model load/infer checks still need P09/P14 qualification. CPU import
compatibility for the entity extra is required in CI.

## CI and publishing

`ci.yml` calls `verify.yml` on pull requests, supported branch pushes, merge groups,
and manual runs. Required verification includes frozen dependencies, tests, lint,
secret/dependency checks, model imports, a single package build, and the installed
wheel e2e. Jobs have timeouts; test reports and synthetic failure artifacts have
limited retention. The `verification / required` aggregate succeeds only when all
its dependencies succeed. Use the exact check name displayed by the first run when
configuring required status checks.

Publishing a GitHub release calls the same reusable verification at that commit.
The tag must equal `v<version>` and `pyproject.toml` must agree with `__version__`.
The build records the source commit, Python/uv versions, and lockfile hash. The
publisher downloads the previously tested artifacts, verifies the manifest, creates
[GitHub provenance](https://docs.github.com/en/actions/concepts/security/artifact-attestations),
and publishes the wheel/sdist using [PyPI OIDC](https://docs.pypi.org/trusted-publishers/).
The release package build omits uv's housekeeping `.gitignore`, so the manifest
covers the wheel, sdist, and build metadata. These files remain available in the
workflow artifact for its declared retention period. It does not rebuild. Only the publish job receives write-capable identity permissions.
A failed verification blocks publication. Diagnose a partial publish before retrying;
do not overwrite an existing PyPI version.

Repository administrators must configure these settings; workflow files do not
activate them automatically:

- Protect `main` with reviewed pull requests and the aggregate verification check;
  include merge-group checks if using a merge queue. Restrict bypass and force pushes.
- Protect the `pypi` environment, select permitted release refs, and configure its
  trusted publisher for this repository and `release.yml` on PyPI.
- Review pinned Action updates from Dependabot. Use trusted isolated runners for
  GPU jobs and keep them inaccessible to untrusted pull-request code.
- Define the staging/production target, immutable model/data versions, health
  probes, access and retention policy, backup/restore, and rollback criteria before
  enabling deployment. Package publication is not application deployment.

Staging and production automation will be implemented for the chosen deployment
profile in P14. No deployment credentials, infrastructure, or clinical users are
assumed by the initial foundation PR.
