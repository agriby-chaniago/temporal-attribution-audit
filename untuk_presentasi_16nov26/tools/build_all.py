"""Build slides from the bundled evidence snapshot. Never run research code."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", action="store_true", help="Also export PDF with LibreOffice")
    args = parser.parse_args()
    subprocess.run(["node", "build_presentation.cjs"], cwd=HERE, check=True)
    subprocess.run([sys.executable, "tools/finalize_deck.py"], cwd=HERE, check=True)
    if args.pdf:
        if not shutil.which("soffice"):
            raise RuntimeError("Install LibreOffice to export PDF; the PPTX has been built.")
        with tempfile.TemporaryDirectory(prefix="attention-presentation-lo-") as profile:
            subprocess.run([
                "soffice", "-env:UserInstallation=" + Path(profile).as_uri(),
                "--headless", "--convert-to", "pdf", "--outdir", str(HERE),
                str(HERE / "presentasi_attention_16nov2026.pptx")], check=True)
    print("Built presentation from stored evidence only; no research computation.")


if __name__ == "__main__":
    main()
