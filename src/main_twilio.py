#!/usr/bin/env python3

"""Twilio server for speech-to-speech conversation using WebSocket streaming."""

from queue import SimpleQueue
from threading import Lock, Thread, Event
import time
from websockets import Headers
from websockets.http11 import Request, Response
from websockets.sync.server import ServerConnection, serve
from twilio.twiml.voice_response import VoiceResponse, Connect
import numpy as np
from segmenter import Segmenter
from transcriber import Transcriber
from responder import Responder
from synthesizer import Synthesizer
from tw_incoming import TwIncoming
from tw_outgoing import TwOutgoing
from conversation import Conversation
from config import load_config, Config

# Constants
CONFIG_PATH = "./priv/config.json"
TW_SAMPLE_RATE = 8000

# Static variables
START_TIME = time.monotonic()
def elapsed():
    return time.monotonic() - START_TIME

CONFIG: Config = load_config(CONFIG_PATH)

CLOSED_IDS: set[str] = set()
CLOSED_IDS_LOCK = Lock()

def handler(ws: ServerConnection) -> None:
    """Handle WebSocket connection for Twilio media streaming."""
    id = ws.id.hex

    # if connection is closed, save resources by returning early
    with CLOSED_IDS_LOCK:
        if id in CLOSED_IDS:
            CLOSED_IDS.discard(id)
            return

    id = id[:8]

    start = time.monotonic()
    print(f"aspen @ {elapsed():.2f}s: call {id} connected")

    # Create queues for communication between components
    stream_sid_queue = SimpleQueue()  # For Twilio stream SID
    incoming_audio_queue = SimpleQueue()  # From Twilio to segmenter
    segment_queue = SimpleQueue()  # Segmented audio
    transcript_queue = SimpleQueue()  # Transcribed text
    response_queue = SimpleQueue()  # AI responses
    audio_queue = SimpleQueue()  # Synthesized audio

    # Create events
    speaking_event = Event()
    exit_event = Event()

    # Create conversation object
    conversation = Conversation()

    # Initialize components
    tw_incoming = TwIncoming(
        stream_sid_queue=stream_sid_queue,
        exit_event=exit_event,
        ws=ws,
        output_queue=incoming_audio_queue,
    )

    segmenter = Segmenter(
        input_chan=incoming_audio_queue,
        output_chan=segment_queue,
        speaking_event=speaking_event,
        exit_event=exit_event,
        sample_rate=TW_SAMPLE_RATE,
        model_path=CONFIG.segmenter.model_path,
    )

    transcriber = Transcriber(
        input_queue=segment_queue,
        output_queue=transcript_queue,
        exit_event=exit_event,
        sample_rate=TW_SAMPLE_RATE,
        api_key=CONFIG.transcriber.api_key,
        model=CONFIG.transcriber.model,
        lang=CONFIG.transcriber.language,
    )

    responder = Responder(
        input_queue=transcript_queue,
        output_queue=response_queue,
        speaking_event=speaking_event,
        exit_event=exit_event,
        conversation=conversation,
        system_message=CONFIG.responder.system_message,
        model=CONFIG.responder.model,
        max_tokens=CONFIG.responder.max_tokens,
        api_key=CONFIG.responder.api_key,
    )

    synthesizer = Synthesizer(
        input_queue=response_queue,
        output_queue=audio_queue,
        speaking_event=speaking_event,
        exit_event=exit_event,
        credentials_path=CONFIG.synthesizer.credentials_path,
        voice_language_code=CONFIG.synthesizer.voice_language_code,
        voice_name=CONFIG.synthesizer.voice_name,
        voice_gender=CONFIG.synthesizer.voice_gender,
        sample_rate=TW_SAMPLE_RATE,
    )

    tw_outgoing = TwOutgoing(
        stream_sid_queue=stream_sid_queue,
        exit_event=exit_event,
        ws=ws,
        input_queue=audio_queue,
        speaking_event=speaking_event,
        sample_rate=TW_SAMPLE_RATE,
        conversation=conversation,
    )

    # Initial greeting
    response_queue.put(CONFIG.general.initial_greeting)

    # Start processing threads
    threads = [
        Thread(target=tw_incoming.loop, daemon=True),
        Thread(target=segmenter.loop, daemon=True),
        Thread(target=transcriber.loop, daemon=True),
        Thread(target=responder.loop, daemon=True),
        Thread(target=synthesizer.loop, daemon=True),
        Thread(target=tw_outgoing.run, daemon=True),
    ]

    for thread in threads:
        thread.start()

    # Wait for exit event
    exit_event.wait()

    # Cleanup
    for thread in threads:
        thread.join()

    # Stats
    duration = time.monotonic() - start
    print(f"aspen @ {elapsed():.2f}s: call {id} disconnected after {duration:.2f}s")


def on_request(ws: ServerConnection, request: Request) -> Response | None:
    """Handle HTTP requests and return TwiML for incoming calls."""
    if request.path.startswith("/incoming-call"):
        host = request.headers.get("Host") or ""

        response = VoiceResponse()
        connect = Connect()
        connect.stream(url=f"wss://{host}/media-stream")
        response.append(connect)

        body = str(response).encode()

        headers = Headers()
        headers["Content-Type"] = "application/xml"
        headers["Content-Length"] = str(len(body))
        headers["Connection"] = "close"

        with CLOSED_IDS_LOCK:
            CLOSED_IDS.add(ws.id.hex)
        return Response(200, "OK", headers, body)

    elif request.path.startswith("/media-stream"):
        return None

    else:
        body = b"Page not found"
        headers = Headers()
        headers["Content-Length"] = str(len(body))
        headers["Connection"] = "close"

        with CLOSED_IDS_LOCK:
            CLOSED_IDS.add(ws.id.hex)
        return Response(404, "Not Found", headers, body)


def main():
    """Start the WebSocket server for Twilio integration."""
    host = CONFIG.general.tw_host
    port = CONFIG.general.tw_port
    with serve(handler, host, port, process_request=on_request) as server:
        print(f"aspen @ {elapsed():.2f}s: twilio server started on {host}:{port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print(f"\naspen @ {elapsed():.2f}s: done")

if __name__ == "__main__":
    main()
