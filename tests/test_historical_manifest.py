"""Règle des deux checks (REGLES.md §7.1) pour reports/historical/mu_j/manifest.sha256."""
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "reports/historical/mu_j/manifest.sha256"
TRACES = REPO_ROOT / "reports/historical/mu_j/traces"


def test_manifest_verifies_and_covers_every_traced_file():
    """Check positif : shasum -c sort en 0, et le manifeste couvre tous les fichiers réels."""
    result = subprocess.run(
        ["shasum", "-a", "256", "-c", "reports/historical/mu_j/manifest.sha256"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    manifest_lines = MANIFEST.read_text().splitlines()
    files_on_disk = [p for p in TRACES.rglob("*") if p.is_file()]
    assert len(manifest_lines) == len(files_on_disk)


def test_single_byte_alteration_is_caught_as_exactly_one_mismatch(tmp_path):
    """Check négatif : un octet altéré dans une copie doit produire exactement 1 FAILED."""
    copy_traces = tmp_path / "traces"
    shutil.copytree(TRACES, copy_traces)

    # Réécrit le manifeste avec des chemins pointant vers la copie altérée.
    altered_manifest = tmp_path / "manifest.sha256"
    lines = []
    for line in MANIFEST.read_text().splitlines():
        digest, _, rel_path = line.partition("  ")
        rel_to_traces = Path(rel_path).relative_to("reports/historical/mu_j/traces")
        lines.append(f"{digest}  traces/{rel_to_traces}\n")
    altered_manifest.write_text("".join(lines))

    target = next(p for p in copy_traces.rglob("*") if p.is_file())
    data = bytearray(target.read_bytes())
    data[0] ^= 0xFF
    target.write_bytes(bytes(data))

    result = subprocess.run(
        ["shasum", "-a", "256", "-c", "manifest.sha256"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert result.stdout.count("FAILED") == 1
