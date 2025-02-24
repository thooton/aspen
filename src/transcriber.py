import io
import time
import json
import wave
from anthropic import NoneType
import numpy as np
from threading import Event
from queue import Empty, SimpleQueue
from typing import Optional
from numpy.typing import NDArray
from groq import Groq


class Transcriber:
    """Transcribes audio into text using Groq API."""

    # Class constants
    NUM_CHANNELS: int = 1  # Mono audio
    SAMPLE_WIDTH: int = 2  # 16-bit audio
    MAX_RETRIES: int = 3  # Maximum API retry attempts

    def __init__(
        self,
        input_queue: SimpleQueue[NDArray[np.float32]],
        output_queue: SimpleQueue[str],
        exit_event: Event,
        sample_rate: int,
        api_key: str,
        model: str,
        lang: str,
    ) -> None:
        """Initialize the Transcriber with required parameters."""
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.exit_event = exit_event
        self.sample_rate = sample_rate
        self.client = Groq(api_key=api_key)
        self.model = model
        self.lang = lang

    def audio_to_text(self, audio: NDArray[np.float32]) -> Optional[str]:
        """Convert audio array to text using Groq API transcription."""
        # Process audio
        assert len(audio.shape) == 1, "Audio must be mono"
        audio = np.clip(audio, -1, 1)  # Ensure audio range
        processed_audio = (audio * 32767).astype(np.int16)  # Convert to 16-bit

        # Create WAV buffer
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(self.NUM_CHANNELS)
            wf.setsampwidth(self.SAMPLE_WIDTH)
            wf.setframerate(self.sample_rate)
            wf.writeframes(processed_audio.tobytes())
        buffer.seek(0)

        # Transcribe with retry logic
        result = self.client.audio.transcriptions.create(
            file=("audio.wav", buffer.read()),
            model=self.model,
            language=self.lang,
        )
        text = result.text.strip()
        return text if text else None

    def loop(self) -> None:
        """Main transcription loop handling audio processing and transcription."""
        # Warm-up
        try:
            self.audio_to_text(np.zeros(1600, dtype=np.float32))
        except Exception:
            pass  # Silent fail for warm-up

        while not self.exit_event.is_set():
            # Get audio from queue with timeout
            try:
                audio = self.input_queue.get(True, 0.25)
            except Empty:
                continue

            # Transcribe audio and handle output
            for attempt in range(self.MAX_RETRIES):
                try:
                    text = self.audio_to_text(audio)
                    if text:
                        self.output_queue.put(text)
                    break
                except Exception as e:
                    if attempt < self.MAX_RETRIES - 1:
                        continue
                    if self.exit_event.is_set():
                        return
                    error_msg = f"Transcription failed after {self.MAX_RETRIES} attempts: {str(e)}"
                    self.output_queue.put(error_msg)
