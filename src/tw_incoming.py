from threading import Event
from numpy.typing import NDArray
from websockets.sync.server import ServerConnection
from queue import SimpleQueue
import numpy as np
import json
import base64
import g711


class TwIncoming:
    """Streams incoming data from Twilio Media Streams via WebSocket"""

    def __init__(
        self,
        stream_sid_queue: SimpleQueue[str],
        exit_event: Event,
        ws: ServerConnection,
        output_queue: SimpleQueue[NDArray[np.float32]],
    ):
        """Initialize with WebSocket connection and output queue for processed audio"""
        self.stream_sid_queue = stream_sid_queue
        self.ws = ws
        self.output_queue = output_queue
        self.exit_event = exit_event

    def decode_mulaw(self, payload: str) -> NDArray[np.float32]:
        """Convert base64 μ-law audio to float32 samples using g711 library"""
        audio_bytes = base64.b64decode(payload)
        return g711.decode_ulaw(audio_bytes)

    def process_message(self, message: str | bytes) -> None:
        """Process incoming WebSocket message"""
        data = json.loads(message)
        event_type = data.get("event")

        if event_type == "start":
            self.stream_sid_queue.put(str(data.get("streamSid")))

        elif event_type == "media":
            payload = data["media"].get("payload")
            if payload:
                audio_data = self.decode_mulaw(payload)
                self.output_queue.put(audio_data)

        elif event_type == "stop":
            self.exit_event.set()

    def loop(self) -> None:
        """Start processing incoming WebSocket messages"""
        while not self.exit_event.is_set():
            try:
                message = self.ws.recv()
            except Exception:
                self.exit_event.set()
                continue
            self.process_message(message)
