"""Create and verify portable SHA-256 manifests for artifact promotion."""

from __future__ import annotations

import argparse
import json
import sys

from trialmatchai.utils.integrity import verify_manifest, write_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="trialmatchai artifacts")
    commands = parser.add_subparsers(dest="action", required=True)
    create = commands.add_parser("manifest", help="Write SHA256SUMS for an artifact directory")
    create.add_argument("directory")
    create.add_argument("--json", action="store_true", help="Emit machine-readable output")
    verify = commands.add_parser("verify", help="Verify every artifact in a SHA-256 manifest")
    verify.add_argument("manifest")
    verify.add_argument("--directory", default=None, help="Artifact root (default: manifest directory)")
    verify.add_argument("--require-exact", action="store_true", help="Reject unlisted files too")
    verify.add_argument("--json", action="store_true", help="Emit machine-readable output")
    args = parser.parse_args(argv)
    try:
        if args.action == "manifest":
            path = write_manifest(args.directory)
            result = {"ok": True, "manifest": str(path)}
            message = f"Wrote {path}"
        else:
            names = verify_manifest(args.manifest, directory=args.directory, require_exact=args.require_exact)
            result = {"ok": True, "verified": names, "count": len(names)}
            message = f"Verified {len(names)} artifact(s)."
    except (OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"Artifact verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result) if args.json else message)
    return 0
