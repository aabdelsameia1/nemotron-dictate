# Nemotron Dictate — Build & Distribution

Local, private dictation menu-bar app for macOS. Replaces cloud dictation with
NVIDIA Nemotron 3.5 ASR running fully on-device via ONNX Runtime (CPU). No audio
ever leaves the Mac. ONNX-only — does NOT bundle torch / NeMo.

**The packaged app is the FULL live experience** (it wraps `live_dictate.LiveDictateApp`):
- **Live streaming** — words appear in the focused field AS YOU SPEAK (not record-then-paste).
- **Glass pill under the notch** — a Dynamic-Island-style "Listening" indicator while recording.
- **Audio ducking** — other apps' volume gently fades down while you talk, fades back after.
- **Pause/Resume** + language and ducking settings persisted across restarts.
The bundle's entry class is `PackagedLiveDictateApp(LiveDictateApp)` (adds first-run download
+ main-thread model load); all the live UX is inherited from `LiveDictateApp`.

## What gets built
- `dist/Nemotron Dictate.app` — the app (~433 MB; Python runtime + onnxruntime + numpy + native libs)
- `dist/Nemotron Dictate.dmg` — drag-to-install disk image (~220 MB compressed)
- Model weights (~2.4 GB) are **NOT** bundled — downloaded on first run into
  `~/Library/Application Support/Nemotron Dictate/onnx_weights/`.

## Build (developer)
```bash
cd ~/apps/nemotron-dictate
source .venv/bin/activate
python app/build_app.py     # icon -> clean -> py2app -> ad-hoc codesign  => dist/Nemotron Dictate.app
python app/make_dmg.py      # => dist/Nemotron Dictate.dmg
```
`build_app.py` ad-hoc codesigns the bundle (`codesign --sign -`) because we have no
Apple Developer ID. That makes it launchable locally; see Gatekeeper note below.

Headless smoke test (no mic, transcribes a WAV) to verify the bundle works:
```bash
"dist/Nemotron Dictate.app/Contents/MacOS/Nemotron Dictate" \
  --selftest /Users/abdallah.abdelsameia/apps/nemotron-dictate/assets/test_en.wav
# -> [selftest] The quick brown fox jumps over the lazy dog speech recognition on Apple Silicon is working.
```

## Install (end user)
1. Open `Nemotron Dictate.dmg`, drag **Nemotron Dictate** to **Applications**.
2. **First launch (unsigned-app Gatekeeper caveat):** double-clicking shows
   "Apple could not verify… malware". This is expected for an app without a paid
   Apple Developer signature. To open it the first time:
   - **Right-click** (or Control-click) the app → **Open** → **Open** in the dialog.
   - (or: System Settings → Privacy & Security → scroll to the blocked-app notice → **Open Anyway**.)
   After the first open, it launches normally.
3. **First run downloads the model (~2.4 GB)** — the menu-bar icon shows ⬇️ and a
   "Downloading…" status. One time only; cached in Application Support afterwards.
4. **Grant permissions** when prompted (see below). The app opens the right Settings
   pane for you via menu → **Open permissions…**.

## Permissions the user must grant (the app cannot grant these itself)
| Permission | Why | Where |
|---|---|---|
| **Microphone** | record your speech | System Settings → Privacy & Security → Microphone |
| **Accessibility** | type the transcript into the focused app (Cmd-V) | … → Accessibility |
| **Input Monitoring** | detect the global double-tap Right-⌘ hotkey | … → Input Monitoring |
The app degrades gracefully if denied (logs it, no crash) and the menu has
**Open permissions…** to jump straight to the panes.

## Using it
- **Default hotkey (no Karabiner needed): double-tap Right-Command (⌘).**
  Double-tap → 🔴 glass pill appears under the notch + other audio ducks → speak and
  the words **stream live** into the focused app → double-tap again → ✍️ finalize → 🎤 idle.
- Menu also has **Start / Stop (manual)** if you prefer clicking.
- **Pause (free GPU)** in the menu unloads the model; **Resume** reloads it.
- **Duck audio while recording** toggle in the menu (on by default).
- **Language:** menu → Set language → `auto` / `en-US` / `fr-FR` (auto-detects by default).
- **Optional advanced hotkey (F18 via Karabiner):** the dev script also supports
  `--trigger f18`; the packaged app ships with double-tap Right-⌘ so Karabiner is not needed.

## Run the full app from source (dev)
```bash
python live_dictate.py --engine onnx --trigger doubletap-rcmd      # same UX as the packaged app
```

## Logs / troubleshooting
- Log file: `~/Library/Application Support/Nemotron Dictate/dictate.log`
  (menu → **Show log**). Look for `recorder ready`, `model ready`.
- Quit cleanly via the menu **Quit** (or Ctrl-C if launched from a terminal). The
  app always releases the mic on quit — never needs a force-kill.

## Code-signing & notarization (TODO — needs a paid Apple Developer account)
We currently ship **ad-hoc signed** (identity `-`), which is why users hit the
Gatekeeper right-click→Open step. To remove that and allow normal double-click /
distribution outside your own Mac, you need an **Apple Developer ID** (paid, $99/yr
— Abdallah does not have one yet). With it:
```bash
# 1. sign with a Developer ID Application cert
codesign --force --deep --options runtime \
  --sign "Developer ID Application: <Your Name> (TEAMID)" "dist/Nemotron Dictate.app"
# 2. notarize the dmg
xcrun notarytool submit "dist/Nemotron Dictate.dmg" \
  --apple-id <id> --team-id <TEAMID> --password <app-specific-pw> --wait
# 3. staple the ticket
xcrun stapler staple "dist/Nemotron Dictate.dmg"
```
Until then, the right-click→Open instruction above is the supported path.

## Technical notes (build gotchas already handled)
- **liblzma / librosa removed.** The mel frontend is pure-numpy (no librosa →
  no pooch → no lzma), which also shrank the app and fixed a dylib-signature crash.
- **PortAudio (sounddevice) un-zipped.** py2app zips package data, but dylibs can't
  dlopen from a zip; `libportaudio.dylib` is copied to `Contents/Resources/` and the
  app points sounddevice at it on startup.
- **Native libs imported on the main thread.** Importing onnxruntime/sentencepiece
  off the main thread deadlocks dyld during app init; the model is loaded on the
  main thread (via a one-shot timer; menu bar is already visible).
- **Real model files, not symlinks.** macOS file-access (TCC) blocks an unsigned
  app from `open()`-ing symlinks that point outside Application Support — first run
  downloads real files into Application Support.
- **Ad-hoc codesign required** on macOS 26: unsigned bundled dylibs are refused by dyld.
```
```
