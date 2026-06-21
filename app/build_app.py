#!/usr/bin/env python
"""
Repeatable build for Nemotron Dictate.app.

    cd ~/apps/nemotron-dictate
    source .venv/bin/activate
    python app/build_app.py

Steps:
  1. regenerate the icon
  2. clean build/ and dist/
  3. py2app build  -> dist/Nemotron Dictate.app
  4. ad-hoc codesign the whole bundle (so macOS will load the bundled dylibs)
  5. (run `python app/make_dmg.py` afterwards to produce the .dmg)

We ad-hoc sign (sign identity '-') because we don't have an Apple Developer ID.
This makes the app launchable locally; users still need the Gatekeeper
right-click -> Open the first time. See DISTRIBUTION.md.
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
APP_PATH = os.path.join(ROOT, "dist", "Nemotron Dictate.app")


def run(cmd, **kw):
    print("+", " ".join(cmd) if isinstance(cmd, list) else cmd, flush=True)
    return subprocess.run(cmd, check=True, **kw)


def main():
    # 1. icon
    run([sys.executable, os.path.join(HERE, "make_icon.py")])

    # 2. clean
    for d in ("build", "dist"):
        p = os.path.join(ROOT, d)
        if os.path.isdir(p):
            shutil.rmtree(p)
            print("cleaned", p)

    # 3. build
    run([sys.executable, os.path.join(HERE, "setup.py"), "py2app"], cwd=ROOT)

    if not os.path.isdir(APP_PATH):
        print("ERROR: app not produced at", APP_PATH)
        sys.exit(1)

    # 4. ad-hoc codesign the bundle (deep). macOS 26 refuses unsigned bundled dylibs.
    print("ad-hoc codesigning bundle…")
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", APP_PATH])
    # verify the main exe at least
    subprocess.run(["codesign", "-v", "--verbose=2", APP_PATH])

    size = subprocess.run(["du", "-sh", APP_PATH], capture_output=True, text=True).stdout.strip()
    print("\nDONE:", APP_PATH)
    print("size:", size)
    print("next: python app/make_dmg.py")


if __name__ == "__main__":
    main()
