#!/usr/bin/env python3

import datetime
import time
from queue import SimpleQueue
import numpy as np
from threading import Thread, Event
from microphone import Microphone
from segmenter import Segmenter
from conversation import Conversation
from transcriber import Transcriber
from responder import Responder
from synthesizer import Synthesizer
from speaker import Speaker
from config import load_config, Config

CONFIG_PATH = "./priv/config.json"


def main():
    """Local demo that combines microphone input with audio segmentation, transcription, response generation, and speech synthesis"""
    # Load and validate configuration
    try:
        config: Config = load_config(CONFIG_PATH)
    except ValueError as e:
        print(f"Error loading configuration: {e}")
        return

    # Create queues for communication between components
    microphone_queue = SimpleQueue()
    segment_queue = SimpleQueue()
    transcript_queue = SimpleQueue()
    response_queue = SimpleQueue()
    audio_queue = SimpleQueue()

    # Create event for mic interruption
    speaking_event = Event()

    # Create exit event for graceful shutdown
    exit_event = Event()

    # Create conversation
    conversation = Conversation()

    # Initialize components with config values
    microphone = Microphone(
        sample_rate=config.microphone.sample_rate,
        sample_queue=microphone_queue,
        exit_event=exit_event,
    )
    segmenter = Segmenter(
        input_chan=microphone_queue,
        output_chan=segment_queue,
        speaking_event=speaking_event,
        exit_event=exit_event,
        sample_rate=config.microphone.sample_rate,
        model_path=config.segmenter.model_path,
    )
    transcriber = Transcriber(
        input_queue=segment_queue,
        output_queue=transcript_queue,
        exit_event=exit_event,
        sample_rate=config.microphone.sample_rate,
        api_key=config.transcriber.api_key,
        model=config.transcriber.model,
        lang=config.transcriber.language,
    )
    responder = Responder(
        input_queue=transcript_queue,
        output_queue=response_queue,
        speaking_event=speaking_event,
        exit_event=exit_event,
        conversation=conversation,
        system_message=config.responder.system_message,
        model=config.responder.model,
        max_tokens=config.responder.max_tokens,
        api_key=config.responder.api_key,
    )
    synthesizer = Synthesizer(
        input_queue=response_queue,
        output_queue=audio_queue,
        speaking_event=speaking_event,
        exit_event=exit_event,
        credentials_path=config.synthesizer.credentials_path,
        voice_language_code=config.synthesizer.voice_language_code,
        voice_name=config.synthesizer.voice_name,
        voice_gender=config.synthesizer.voice_gender,
        sample_rate=config.synthesizer.sample_rate,
    )
    speaker = Speaker(
        input_queue=audio_queue,
        speaking_event=speaking_event,
        exit_event=exit_event,
        sample_rate=config.synthesizer.sample_rate,
        conversation=conversation,
    )

    # Initial greeting
    response_queue.put(config.general.initial_greeting)

    # Start processing threads
    threads = [
        Thread(target=microphone.loop, daemon=True),
        Thread(target=segmenter.loop, daemon=True),
        Thread(target=transcriber.loop, daemon=True),
        Thread(target=responder.loop, daemon=True),
        Thread(target=synthesizer.loop, daemon=True),
        Thread(target=speaker.loop, daemon=True),
    ]

    for thread in threads:
        thread.start()

    print("aspen @ 0.00s: starting, press CTRL+C to exit")
    start = time.monotonic()
    try:
        while True:
            if exit_event.wait(0.1):
                break
    except KeyboardInterrupt:
        pass
    elapsed = time.monotonic() - start
    print(f"\naspen @ {elapsed:.2f}s: exiting")
    exit_event.set()

    for thread in threads:
        thread.join()

    elapsed = time.monotonic() - start
    print(f"aspen @ {elapsed:.2f}s: done")


if __name__ == "__main__":
    main()
