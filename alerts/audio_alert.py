"""Audio alert channel using gTTS (text-to-speech) + pygame playback.

Example:
    >>> from alerts.audio_alert import AudioAlert
    >>> _ = AudioAlert()
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.logger import logger


@dataclass
class AudioResult:
    ok: bool
    path: Optional[str]
    error: Optional[str]


class AudioAlert:
    """Generate and optionally play an audio warning."""

    def __init__(self) -> None:
        self._pygame_ready = False

    def _ensure_pygame(self) -> None:
        if self._pygame_ready:
            return
        try:
            import pygame

            pygame.mixer.init()
            self._pygame_ready = True
        except Exception as e:
            logger.warning(f"pygame audio unavailable (will only save mp3): {e}")
            self._pygame_ready = False

    def trigger(self, text: str) -> AudioResult:
        """Generate TTS mp3 and play if possible."""

        try:
            from gtts import gTTS
        except Exception as e:
            return AudioResult(ok=False, path=None, error=f"gTTS unavailable: {e}")

        try:
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "pyrosense_alert.mp3"
                gTTS(text=text, lang="en").save(str(out))
                self._ensure_pygame()
                if self._pygame_ready:
                    try:
                        import pygame

                        pygame.mixer.music.load(str(out))
                        pygame.mixer.music.play()
                    except Exception as e:
                        logger.warning(f"Audio playback failed: {e}")
                # persist to a stable location too
                stable = Path("data") / "processed" / "audio"
                stable.mkdir(parents=True, exist_ok=True)
                stable_out = stable / "last_alert.mp3"
                stable_out.write_bytes(out.read_bytes())
                return AudioResult(ok=True, path=str(stable_out), error=None)
        except Exception as e:
            return AudioResult(ok=False, path=None, error=str(e))

