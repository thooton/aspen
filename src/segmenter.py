from dataclasses import dataclass
from queue import Empty, SimpleQueue
from threading import Event
from typing import Any
import numpy as np
from numpy.typing import NDArray
from collections import deque
import torch
torch.set_num_threads(1)

class ModelInterface:
    def __call__(self, chunk_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return torch.Tensor()
    def eval(self) -> None:
        pass


class Segmenter:
    """Segments continuous audio stream into discrete utterances using Silero VAD."""

    SPEECH_THRESHOLD = 0.4  # Silero VAD probability threshold
    PRE_SPEECH_BUFFER = 25  # Samples to keep before speech
    SILENCE_LIMIT = 24  # Samples of silence to end segment (approx 0.8s at 32ms/sample)
    MIN_SPEECH_CHUNKS = (
        3  # Required speech chunks to start (approx 0.1s at 32ms/sample)
    )

    current_speech: list[NDArray[np.float32]]
    pre_buffer: deque[NDArray[np.float32]]
    speech_samples: int
    silence_samples: int
    recording: bool
    input: SimpleQueue[NDArray[np.float32]]
    output: SimpleQueue[NDArray[np.float32]]
    speaking_event: Event
    exit_event: Event
    model: ModelInterface  # Silero VAD model
    count: int
    sample_rate: int
    window_size: int
    buffer: NDArray[np.float32] | None

    def __init__(
        self,
        input_chan: SimpleQueue[NDArray[np.float32]],
        output_chan: SimpleQueue[NDArray[np.float32]],
        speaking_event: Event,
        exit_event: Event,
        sample_rate: int,
        model_path: str,
    ):
        self.input = input_chan
        self.output = output_chan
        self.speaking_event = speaking_event
        self.exit_event = exit_event
        self.model = torch.jit.load(model_path, map_location=torch.device("cpu"))
        self.model.eval()
        self.pre_buffer = deque(maxlen=self.PRE_SPEECH_BUFFER)
        self.current_speech = []
        self.speech_samples = 0
        self.silence_samples = 0
        self.recording = False
        self.count = 0
        if sample_rate != 8000 and sample_rate != 16000:
            raise ValueError("unsupported sample rate", sample_rate)
        self.sample_rate = sample_rate
        self.window_size = 512 if sample_rate == 16000 else 256
        self.buffer = None

    def detect_speech(self, audio_chunk: NDArray[np.float32]) -> float:
        """Run Silero VAD on the audio chunk and return speech probability."""
        # Convert numpy array to torch tensor
        chunk_tensor = torch.tensor(audio_chunk, dtype=torch.float32)

        # Run Silero VAD model
        with torch.no_grad():
            speech_prob = self.model(chunk_tensor, self.sample_rate).item()

        return speech_prob

    def loop(self):
        """Process audio stream continuously"""
        while not self.exit_event.is_set():
            # Try to get audio chunk from input queue with timeout
            try:
                audio_chunk = self.input.get(True, 0.25)
            except Empty:
                continue

            # Verify audio chunk is 1-dimensional
            assert len(audio_chunk.shape) == 1

            # If there's leftover buffer from previous chunk, prepend it
            if self.buffer is not None:
                audio_chunk = np.concatenate([self.buffer, audio_chunk], axis=-1)
                self.buffer = None

            sample_count = audio_chunk.shape[0]

            # If chunk is too small, store it in buffer for next iteration
            if sample_count < self.window_size:
                self.buffer = audio_chunk
                continue

            # If chunk is too large, keep excess in buffer and truncate current chunk
            if sample_count > self.window_size:
                self.buffer = audio_chunk[self.window_size :]
                audio_chunk = audio_chunk[: self.window_size]

            # Run Silero VAD on audio sample
            speech_prob = self.detect_speech(audio_chunk)
            is_speech = speech_prob > self.SPEECH_THRESHOLD

            # Push sample to pre-speech buffer
            self.pre_buffer.append(audio_chunk)

            if not self.recording:
                self.speech_samples = self.speech_samples + 1 if is_speech else 0
                if self.speech_samples >= self.MIN_SPEECH_CHUNKS:
                    # Start recording, speech detected
                    self.current_speech = list(self.pre_buffer)
                    self.recording = True
                    self.silence_samples = 0
                    self.speaking_event.set()
            else:
                self.current_speech.append(audio_chunk)
                self.silence_samples = 0 if is_speech else self.silence_samples + 1
                if self.silence_samples >= self.SILENCE_LIMIT and self.current_speech:
                    # Stop recording, silence detected
                    self.output.put(np.concatenate(self.current_speech))
                    self.current_speech = []
                    self.recording = False
                    self.speech_samples = 0
                    self.count += 1
                    self.speaking_event.clear()
