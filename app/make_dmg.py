#!/usr/bin/env python
"""
Package dist/Nemotron Dictate.app into dist/Nemotron Dictate.dmg.

    cd ~/apps/nemotron-dictate
    source .venv/bin/activate
    python app/build_app.py     # builds the .app first
    python app/make_dmg.py      # -> dist/Nemotron Dictate.dmg

The DMG contains the app + an /Applications symlink so users drag-to-install.
Weights are NOT in the DMG (downloaded on first run) -> small download.
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "dist", "Nemotron Dictate.app")
DMG = os.path.join(ROOT, "dist", "Nemotron Dictate.dmg")
VOLNAME = "Nemotron Dictate"


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    if not os.path.isdir(APP):
        print("ERROR: build the app first (python app/build_app.py). Missing:", APP)
        sys.exit(1)
    if os.path.exists(DMG):
        os.remove(DMG)

    staging = tempfile.mkdtemp(prefix="nemodmg_")
    try:
        # copy app into staging, add /Applications symlink
        dst = os.path.join(staging, "Nemotron Dictate.app")
        print("staging app…", flush=True)
        shutil.copytree(APP, dst, symlinks=True)
        os.symlink("/Applications", os.path.join(staging, "Applications"))

        run([
            "hdiutil", "create",
            "-volname", VOLNAME,
            "-srcfolder", staging,
            "-ov", "-format", "UDZO",   # compressed
            DMG,
        ])
        size = subprocess.run(["du", "-sh", DMG], capture_output=True, text=True).stdout.strip()
        print("\nDONE:", DMG)
        print("size:", size)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
