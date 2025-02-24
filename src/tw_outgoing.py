import time
import json
import base64
import numpy as np
from queue import SimpleQueue, Empty
from threading import Event
from websockets.sync.server import ServerConnection
from collections import deque
import g711
from conversation import Conversation


class WordQueue:
    """Manages the queue of spoken words."""

    def __init__(self, conversation: Conversation):
        self.conversation = conversation
        self.word_queue = deque()  # Queue of (duration, word) pairs
        self.last_update_time = 0.0  # Initialized in `add_words`

    def clear(self) -> None:
        self.word_queue.clear()

    def is_empty(self) -> bool:
        return len(self.word_queue) == 0

    def update(self) -> None:
        """Adds spoken words to the conversation based on time passed."""
        current_time = time.monotonic()
        time_passed = current_time - self.last_update_time
        self.last_update_time = current_time

        # Process words in the queue based on elapsed time
        while self.word_queue and time_passed > 0:
            duration, word = self.word_queue.popleft()  # Get the next word
            time_passed -= duration
            if time_passed < 0:
                # Not enough time passed, put the word back
                self.word_queue.appendleft((-time_passed, word))
            else:
                # Word was "spoken," add it to the conversation
                self.conversation.append("assistant", word)

    def add_words(self, text: str, audio_data: np.ndarray, sample_rate: int) -> None:
        """Splits text into words and estimates duration per word."""
        words = text.split()
        duration_per_word = len(audio_data) / sample_rate / max(len(words), 1)
        if len(self.word_queue) == 0:
            self.last_update_time = time.monotonic()
        for word in words:
            self.word_queue.append((duration_per_word, word))


class TwOutgoing:
    """Sends audio to Twilio over a WebSocket and tracks spoken words."""

    def __init__(
        self,
        stream_sid_queue: SimpleQueue[str],  # Queue to get the stream ID
        exit_event: Event,  # Signal to stop the class
        ws: ServerConnection,  # WebSocket connection to Twilio
        input_queue: SimpleQueue[
            tuple[str, np.ndarray]
        ],  # Queue with text and audio data
        speaking_event: Event,  # Signal when someone is speaking
        sample_rate: int,  # Audio sample rate (e.g., 8000 Hz)
        conversation: Conversation,  # Object to store conversation history
    ):
        self.stream_sid = ""  # Unique ID for the Twilio stream
        self.stream_sid_queue = stream_sid_queue
        self.ws = ws
        self.input_queue = input_queue
        self.exit_event = exit_event
        self.speaking_event = speaking_event
        self.sample_rate = sample_rate
        self.word_queue = WordQueue(conversation)

    def encode_audio(self, audio_data: np.ndarray) -> str:
        """Turns audio into a format Twilio understands."""
        mulaw_bytes = g711.encode_ulaw(audio_data)  # Compress audio to μ-law
        return base64.b64encode(mulaw_bytes).decode("utf-8")  # Encode as base64 string

    def send_audio(self, audio_data: np.ndarray) -> None:
        """Sends audio to Twilio over the WebSocket."""
        payload = self.encode_audio(audio_data)
        message = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": payload},
        }
        try:
            self.ws.send(json.dumps(message))  # Send JSON message
        except Exception:
            self.exit_event.set()  # Stop if there's an error

    def interrupt(self) -> None:
        """Tells Twilio to stop its current output."""
        try:
            self.ws.send(json.dumps({"event": "clear", "streamSid": self.stream_sid}))
        except Exception:
            self.exit_event.set()  # Stop if there's an error

    def run(self) -> None:
        """Main loop: waits for audio/text and sends it to Twilio."""
        # Step 1: Wait for the stream ID from Twilio
        while not self.exit_event.is_set():
            try:
                self.stream_sid = self.stream_sid_queue.get(timeout=0.25)
                break
            except Empty:
                continue  # Keep waiting if queue is empty

        # Step 2: Process audio and text forever (until exit)
        while not self.exit_event.is_set():
            # If speaking, pause and clear word queue
            if self.speaking_event.is_set():
                if not self.word_queue.is_empty():
                    self.interrupt()  # Stop Twilio's output
                    self.word_queue.clear()
                time.sleep(0.25)  # Wait til we're not speaking anymore
                continue

            # Update word queue
            self.word_queue.update()

            # Get audio and text from the input queue
            try:
                text, audio_data = self.input_queue.get(timeout=0.25)
            except Empty:
                continue  # Keep waiting if queue is empty

            # Clean up audio data
            audio_data = np.squeeze(audio_data).astype(np.float32)
            if np.max(np.abs(audio_data)) > 1:
                audio_data /= np.max(np.abs(audio_data))  # Normalize loud audio

            # Send audio to Twilio
            self.send_audio(audio_data)

            # Add words to queue
            self.word_queue.add_words(text, audio_data, self.sample_rate)
