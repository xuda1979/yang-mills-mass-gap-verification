"""certificate_runner.py

Deterministic certificate-grade entrypoint.

This script is designed to be the *single* reproducible runner for the CAP suite.
It:
  1) Runs the verification exporter (which itself runs the main checks).
  2) Emits an artifact bundle with hashes + environment metadata.

Artifacts are written under: verification/artifacts/<UTC timestamp>/

Why a wrapper when export_results_to_latex.py already exists?
- It makes the "certificate contract" explicit.
- It collects reproducibility metadata in one place.
- It pins what files are considered the certificate outputs.

Usage:
    python certificate_runner.py

Exit codes:
    0  PASS
    1  FAIL (verification failed)
    2  ERROR (exception / cannot produce artifacts)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from typing import Dict, List


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _copy_into_artifacts(src: str, dst_dir: str) -> str:
    _ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    return dst


def _collect_environment() -> Dict[str, object]:
    return {
        "utc_generated": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
    }


def main() -> int:
    # Ensure imports resolve relative to verification/ directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)

    # Import late so sys.path is set.
    import export_results_to_latex  # noqa: WPS433
    import verify_uv_handoff
    import verify_ir_limit
    import verify_axioms

    # Run exporter (returns 0/1)
    status_code = 0
    try:
        # 1. Main Verification Phase (Tube Contraction)
        print("\n[CERT] Running Phase 1: Main Verification (Tube Contraction)...")
        if export_results_to_latex.main() != 0:
            status_code = 1
        
        # 2. UV Handoff Check
        print("\n[CERT] Running Phase 2: UV Handoff Verification...")
        if verify_uv_handoff.verify_uv_condition() != 0:
            status_code = 1
            
        # 3. IR Limit Check
        print("\n[CERT] Running Phase 3: IR Limit Verification...")
        if verify_ir_limit.verify_ir_condition() != 0:
            status_code = 1

        # 4. Axiom Compliance Audit
        print("\n[CERT] Running Phase 4: Axiom Compliance Audit...")
        if verify_axioms.verify_axiom_compliance() != 0:
            status_code = 1

    except SystemExit as e:
        if e.code != 0:
            status_code = 1
    except Exception as e:
        print(f"[CERT] ERROR: Verification suite crashed: {e}")
        return 2

    if status_code != 0:
        print("[CERT] FAILURE: One or more verification phases failed.")
        return 1

    # Locate outputs (as defined by exporter)
    paper_dir = os.path.join(script_dir, "..", "single")
    latex_out = os.path.abspath(os.path.join(paper_dir, "verification_results.tex"))
    json_out = os.path.abspath(os.path.join(script_dir, "verification_results.json"))

    expected: List[str] = [latex_out, json_out]
    missing = [p for p in expected if not os.path.exists(p)]
    if missing:
        print("[CERT] ERROR: expected outputs missing:")
        for p in missing:
            print(f"  - {p}")
        return 2

    # Create artifact bundle
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifacts_root = os.path.join(script_dir, "artifacts", ts)
    _ensure_dir(artifacts_root)

    copied = []
    for p in expected:
        copied.append(_copy_into_artifacts(p, artifacts_root))

    manifest = {
        "environment": _collect_environment(),
        "status": "PASS" if status_code == 0 else "FAIL",
        "outputs": {
            os.path.basename(p): {
                "source": os.path.abspath(p),
                "sha256": _sha256_file(os.path.join(artifacts_root, os.path.basename(p))),
            }
            for p in expected
        },
    }

    manifest_path = os.path.join(artifacts_root, "certificate_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"[CERT] Bundle written to: {artifacts_root}")
    print(f"[CERT] Status: {manifest['status']}")

    return 0 if status_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
