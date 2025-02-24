from threading import Event
import numpy as np
from numpy.typing import NDArray
import sounddevice as sd
from sounddevice import CallbackFlags
from queue import SimpleQueue
from typing import Optional


class Microphone:
    """Streams audio data from the microphone"""

    SAMPLE_LENGTH = 32  # ms per sample

    sample_queue: SimpleQueue[NDArray[np.float32]]
    exit_event: Event

    def __init__(
        self,
        sample_rate: int,
        sample_queue: SimpleQueue[NDArray[np.float32]],
        exit_event: Event,
    ):
        """
        Initialize the Microphone with a sample queue.

        Parameters:
            sample_queue (Queue): Queue to store audio samples and VAD results
        """
        self.sample_rate = sample_rate
        self.sample_queue = sample_queue
        self.exit_event = exit_event

    def loop(self):
        """Start the audio input stream loop"""

        def callback(
            indata: np.dtype[np.float32],
            frames: int,
            time: sd.CallbackStop,
            status: CallbackFlags,
        ) -> None:
            # Reduce to single channel if necessary
            self.sample_queue.put(np.array(indata).copy().squeeze())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=callback,
            blocksize=int(self.sample_rate * self.SAMPLE_LENGTH / 1000),
        ):
            self.exit_event.wait()
