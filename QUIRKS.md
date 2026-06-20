# QUIRKS — nemotron-asr dictation

One row per anomaly. Categories: ORIGINAL-BUG · DECOMPILER · DEAD-CODE · COSMETIC · DATA · ENV.

| Where | Category | What | Handling | Status |
|---|---|---|---|---|
| `menubar_dictate.py` run loop | ENV | Ctrl-C / SIGINT in a terminal doesn't stop the menu-bar app — rumps/PyObjC `runEventLoop` installs its own SIGINT handler on `run()`, overriding ours; an idle app's run loop also sleeps, so the signal isn't processed promptly. | Re-assert our SIGINT/SIGTERM handlers on the first `rumps.Timer` tick (runs Python on the main thread). Now Ctrl-C fires the clean `_quit` within ~0.15s. Menu **Quit** and `launchctl unload` (SIGTERM) also tear down cleanly. | RESOLVED |
| NeMo streaming on MPS | ENV | NeMo logs `CUDA is not available → Cuda graphs with while loops disabled, decoding slower`. | Expected on Apple Silicon — not an error. MPS path still runs ~48x real-time (batch). Streaming RTF measured separately. | NOTED |
| Model output | DATA | Transcripts carry language tags like `<en-US>` / `<fr-FR>` at the start. | Stripped via regex `\s*<[a-z]{2}-[A-Z]{2}>\s*` before injecting. | RESOLVED |
