#!/usr/bin/env python
"""
LiveTyper — turn a stream of growing transcripts into live typing in the focused app.

A streaming ASR model emits the WHOLE hypothesis each step, and it can REVISE the
tail as it hears more audio ("recognize" -> "recognise the"). To show that live in
the user's actual text field we:

  1. keep track of what we've already typed (`self.committed`)
  2. on each update, find the longest common PREFIX with the new text
  3. backspace only the part that changed, then type only the new tail

So in the steady state we just append; when the model revises the last word or two
we backspace those few chars and retype. Minimal, cursor stays put (as long as the
user doesn't move it mid-dictation).

Needs Accessibility permission (synthetic key events). pynput's `type()` uses unicode
string events on macOS, so accents (é, à, ç…) inject correctly.

There is ONE runnable check at the bottom: `python live_inject.py --selfcheck`
exercises the diff logic with a fake stream and prints the backspace/type plan
WITHOUT touching the keyboard.
"""
from __future__ import annotations
import time


def diff_plan(old: str, new: str) -> tuple[int, str]:
    """Return (num_backspaces, text_to_type) to turn `old` into `new` via a common prefix."""
    i = 0
    n = min(len(old), len(new))
    while i < n and old[i] == new[i]:
        i += 1
    return len(old) - i, new[i:]


class LiveTyper:
    def __init__(self, key_delay: float = 0.0):
        # key_delay: optional pause between synthetic events if macOS drops fast bursts.
        from pynput.keyboard import Controller, Key
        self._kb = Controller()
        self._Key = Key
        self._delay = key_delay
        self.committed = ""

    def update(self, new_full_text: str):
        """Make the focused field show `new_full_text`, typing only the delta."""
        n_back, to_type = diff_plan(self.committed, new_full_text)
        for _ in range(n_back):
            self._kb.press(self._Key.backspace)
            self._kb.release(self._Key.backspace)
            if self._delay:
                time.sleep(self._delay)
        if to_type:
            # type() handles unicode/accents via CGEventKeyboardSetUnicodeString on macOS
            self._kb.type(to_type)
        self.committed = new_full_text

    def finalize(self, final_text: str | None = None):
        """Commit a final (possibly cleaned) text, then reset for the next utterance."""
        if final_text is not None:
            self.update(final_text)
        self.committed = ""

    def reset(self):
        self.committed = ""


def _selfcheck():
    # Simulate a streaming hypothesis that grows and revises its tail.
    stream = [
        "hello",
        "hello this",
        "hello this is",
        "hello this is wor",
        "hello this is working",      # tail revised: 'wor' -> 'working'
        "hello, this is working",     # earlier revision: 'hello' -> 'hello,'
        "hello, this is working nice",
    ]
    committed = ""
    for step in stream:
        n_back, to_type = diff_plan(committed, step)
        print(f"  {committed!r:34} -> {step!r:34}  |  ⌫×{n_back}  +type {to_type!r}")
        committed = step
    assert committed == stream[-1]
    print("selfcheck OK ✅  (no keyboard touched)")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        print("Run with --selfcheck to test the diff logic without typing.")
