"""Exercise public CLI commands from an installed wheel, outside the checkout.

Run with the installed environment's Python and --expect-installed in CI.
This checks the software path with synthetic CPU retrieval, not model accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from importlib import metadata, resources
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--expect-installed", action="store_true")
    args = parser.parse_args()
    root = args.workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)

    import trialmatchai
    from trialmatchai.config.config_loader import load_config

    if args.expect_installed:
        source = Path(__file__).resolve().parents[1] / "src"
        if Path(trialmatchai.__file__).resolve().is_relative_to(source):
            raise RuntimeError("Smoke test imported the checkout, not the installed wheel")

    command = Path(sys.executable).parent / "trialmatchai"
    env = {key: value for key, value in os.environ.items() if not key.startswith("TRIALMATCHAI_")}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    def run(*arguments):
        result = subprocess.run([str(command), *arguments], cwd=root, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        print(result.stdout, end="", flush=True)
        if result.returncode:
            raise RuntimeError(f"CLI {arguments} returned {result.returncode}")
        return result.stdout

    assert run("--version").strip() == f"trialmatchai {metadata.version('trialmatchai')}"
    package = resources.files("trialmatchai")
    for name, expected_model in (("medcpt", "ncbi/MedCPT-Article-Encoder"), ("bge-m3", "BAAI/bge-m3")):
        config = json.loads(package.joinpath("config/config.json").read_text())
        config["embedder"] = {"model": name}
        path = root / f"{name}.json"
        path.write_text(json.dumps(config))
        resolved = load_config(path)
        assert resolved["embedder"]["model_name"] == expected_model
        assert Path(resolved["entity_extraction"]["schema_path"]).is_file()

    demo = root / "demo"
    run("demo", "--workdir", str(demo))
    result = json.loads((demo / "demo-result.json").read_text())
    assert result["mode"] == "synthetic_retrieval_only"
    assert result["ranked_trial_ids"][0] == "NCT00000001"
    assert "NCT00000002" not in result["ranked_trial_ids"]  # Pediatric age filter.
    report = Path(result["report"]).read_text()
    assert "demo-patient" in report and "NCT00000001" in report
    assert '"reasoning_available": false' in report
    ranked = demo / "results/demo-patient/ranked_trials.json"
    payload = json.loads(ranked.read_text())
    assert payload["Run"]["mode"] == "retrieval_only"
    assert payload["Run"]["assessment_status"] == "disabled"
    assert '"mode": "retrieval_only"' in report and "Retrieval-only" in report
    original = (ranked.read_bytes(), ranked.stat().st_mtime_ns)
    run("demo", "--workdir", str(demo), "--resume")
    assert (ranked.read_bytes(), ranked.stat().st_mtime_ns) == original
    assert json.loads((demo / "demo-result.json").read_text())["resumed"] is True
    report_path = Path(result["report"])
    for state in ("missing", "truncated"):
        if state == "missing":
            report_path.unlink()
        else:
            report_path.write_text("<!DOCTYPE html><html>")
        run("demo", "--workdir", str(demo), "--resume")
        repaired = report_path.read_text()
        assert repaired.rstrip().endswith("</html>")
        assert "demo-patient" in repaired and "NCT00000001" in repaired
        assert (ranked.read_bytes(), ranked.stat().st_mtime_ns) == original
    run("artifacts", "verify", str(demo / "SHA256SUMS"), "--json")
    print("Installed CLI smoke passed: catalog, import, real index, retrieval, report repair, resume, checksums.")


if __name__ == "__main__":
    main()
