import numpy as np
from numpy.typing import NDArray
import sounddevice as sd
from queue import Queue, SimpleQueue, Empty
import time
from threading import Event
from conversation import Conversation


class Speaker:
    """Plays audio data through the system speakers"""

    def __init__(
        self,
        input_queue: SimpleQueue[tuple[str, NDArray[np.float32]]],
        speaking_event: Event,
        exit_event: Event,
        sample_rate: int,
        conversation: Conversation,
    ):
        """Initialize Speaker with channels for fragment and associated audio data"""
        self.input_queue = input_queue
        self.speaking_event = speaking_event
        self.exit_event = exit_event
        self.sample_rate = sample_rate
        self.conversation = conversation

    def loop(self) -> None:
        """Continuously play audio data from the channel"""
        while not self.exit_event.is_set():
            # Get audio data from channel
            try:
                text, audio_data = self.input_queue.get(True, 0.25)
            except Empty:
                continue
            if self.speaking_event.is_set():
                # User is speaking - don't talk, clear queue
                continue

            # Ensure data is 1D and normalized
            audio_data = audio_data.squeeze()
            audio_data = audio_data.astype(np.float32)
            max_amplitude = np.max(np.abs(audio_data))
            if max_amplitude > 1:
                audio_data = audio_data / max_amplitude

            # Play audio
            sd.play(audio_data, self.sample_rate)

            # Estimate audio boundaries for words
            words = text.split()
            audio_duration = len(audio_data) / self.sample_rate
            word_duration = audio_duration / len(words)

            # Append text word-by-word to conversation
            for word in words:
                if self.speaking_event.wait(word_duration):
                    # user is speaking, stop
                    sd.stop()
                    break
                if self.exit_event.is_set():
                    return
                self.conversation.append("assistant", word)

            # Wait for audio to finish playing
            sd.wait()
