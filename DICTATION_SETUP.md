# Nemotron Dictation — setup & status

Local, real-time speech-to-text that replaces Apple dictation with
`nvidia/nemotron-3.5-asr-streaming-0.6b` on Apple Silicon. Fully offline.

## How it works
Tap **F5 (🎤 mic key)** → 🔴 listening → words stream into the focused app **as you speak**
→ tap again → done.
- F5 is remapped by **Karabiner** → **F18** (a phantom key).
- The menu-bar app `live_dictate.py` listens for F18, streams audio to Nemotron
  (cache-aware streaming, MPS), and types the growing text live (pure-append → no glitches).

## The app: `live_dictate.py` (this is the main one)
Menu-bar 🎤. Features:
- **Real-time streaming** dictation (EN + FR, ~0.11 RTF on MPS)
- **Floating indicator** 🔴 on screen while recording
- **Audio ducking** — lowers other apps' volume to 20% while you talk, restores after
- **Pause / Resume** — unloads the model (GPU → 0 MB) so you can run HyperRead; reload on demand
- Clean teardown (Ctrl-C / menu Quit / launchctl all safe)

Older files: `menubar_dictate.py` (batch, tap-to-record-then-paste), `run.py` (file → text),
`dictate.py` (push-to-talk + shared recorder/inject), `stream_engine.py` (streaming core),
`live_inject.py` (delta-typer).

## Run it
```bash
cd ~/apps/nemotron-dictate
# production (after restart: F5 → F18 via Karabiner):
./.venv/bin/python live_dictate.py
# testing before the restart (uses Right Option as the trigger):
./.venv/bin/python live_dictate.py --trigger alt_r --lang auto
```

## Resource facts
- Idle (not talking): ~0% GPU, ~2.6 GB held in memory. Other apps unaffected.
- Pause: unloads to **0 MB** — use before heavy HyperRead/TTS sessions.
- Active: brief ~35ms GPU bursts per 320ms of speech.
- ⚠️ One caution: dictating *while* HyperRead is actively generating = two models on the
  GPU. Idle-alongside is fine; for simultaneous heavy use, hit **Pause**.

## Always-on (no terminal, loads once at login)
LaunchAgent staged at `~/Library/LaunchAgents/com.abdallah.nemotron-dictate.plist`
(points at `live_dictate.py`). Enable after the restart + granting the venv `python`
its permissions:
```bash
launchctl load ~/Library/LaunchAgents/com.abdallah.nemotron-dictate.plist   # start + auto-start at login
launchctl unload ~/Library/LaunchAgents/com.abdallah.nemotron-dictate.plist # stop
```

## TO FINISH (needs your restart) ⏳
1. Karabiner: enable **Driver Extension** → **restart Mac**.
2. Keyboard settings → **Dictation OFF** (frees F5).
3. Verify F5 → F18: Karabiner → EventViewer → tap F5 (should show `f18`).
4. Grant the venv `python`: **Input Monitoring** + **Accessibility** + **Microphone**.
5. `launchctl load` the plist → 🎤 lives in the menu bar at every login.

## Permissions needed
Microphone · Input Monitoring (listen for the key) · Accessibility (type into apps).
During terminal testing these attach to **Terminal**; under the LaunchAgent they attach to
the venv **python** binary.
