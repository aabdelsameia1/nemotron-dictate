# Nemotron Dictate

Local, real-time dictation for macOS (Apple Silicon) that replaces Apple's built-in
dictation with NVIDIA's **`nemotron-3.5-asr-streaming-0.6b`**. Fully offline — audio
never leaves the machine. English + French (auto-detected).

Tap your mic key → speak → words stream into whatever app is focused, **live**.

## Features
- **Real-time streaming** transcription (cache-aware NeMo, ~0.11 RTF on MPS — 4–9× real-time)
- **Pure-append** output → live typing with zero backspacing/glitches
- **Floating indicator** — a glass pill under the notch, on whichever screen you're using
- **Audio ducking** — lowers other apps' volume while you talk, fades it back gently
- **Pause / Resume** — unloads the model (GPU → 0 MB) to free the GPU for other work
- **Remembers settings** (language, ducking) across restarts; **error popups** on mic/model failure
- **Menu-bar app**, runs from a LaunchAgent so it loads once at login (no terminal) + auto-restarts on crash

## Requirements
- Apple Silicon Mac, macOS 13+
- ~2.6 GB free memory while running; ~2.4 GB disk for the model
- [Karabiner-Elements](https://karabiner-elements.pqrs.org/) to map the mic key (F5 🎤) → F18

## Quick start
```bash
# 1. create the env (uv) and install deps
uv venv && uv pip install -r requirements.txt    # NeMo is installed from git (see requirements.txt)

# 2. run the live menu-bar app (production trigger = F18 from Karabiner)
./.venv/bin/python live_dictate.py

# testing without Karabiner (uses Right Option as the trigger):
./.venv/bin/python live_dictate.py --trigger alt_r --lang auto

# transcribe a file instead:
./.venv/bin/python run.py --audio some.wav --lang fr-FR --device mps --out out.txt
```

See **[DICTATION_SETUP.md](DICTATION_SETUP.md)** for the full setup (Karabiner remap,
turning off Apple dictation, permissions, always-on LaunchAgent).

## Files
| File | What |
|---|---|
| `live_dictate.py` | the main app — real-time streaming + pill + ducking + pause |
| `stream_engine.py` | cache-aware streaming core (`StreamingTranscriber`) |
| `live_inject.py` | delta-typer (diffs growing text → minimal keystrokes) |
| `menubar_dictate.py` | older batch app (tap-record-then-paste) |
| `dictate.py` | push-to-talk daemon + shared recorder/model/inject |
| `run.py` | file → text transcriber |
| `QUIRKS.md` | logged anomalies + fixes |

## Notes
- Streaming requires **float32** (NeMo's streaming path rejects fp16). It's still
  comfortably real-time on MPS.
- License of the model: **OpenMDW-1.1** (open).

🤖 Built with [Claude Code](https://claude.com/claude-code)
