from queue import Empty, SimpleQueue
from threading import Event
from typing_extensions import Literal
from numpy.typing import NDArray
from google.oauth2.service_account import Credentials
import numpy as np
import io
import wave
import time
from google.cloud import texttospeech


class Synthesizer:
    input_queue: SimpleQueue[str]
    output_queue: SimpleQueue[tuple[str, NDArray[np.float32]]]
    speaking_event: Event
    exit_event: Event
    client: texttospeech.TextToSpeechClient
    voice: texttospeech.VoiceSelectionParams
    audio_config: texttospeech.AudioConfig

    def __init__(
        self,
        input_queue: SimpleQueue[str],
        output_queue: SimpleQueue[tuple[str, NDArray[np.float32]]],
        speaking_event: Event,
        exit_event: Event,
        credentials_path: str,
        voice_language_code: str,
        voice_name: str,
        voice_gender: Literal["male", "female", "neutral"],
        sample_rate: int,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.speaking_event = speaking_event
        self.exit_event = exit_event

        # Initialize the TTS client
        credentials = Credentials.from_service_account_file(credentials_path)
        self.client = texttospeech.TextToSpeechClient(credentials=credentials)

        # Configure voice settings
        self.voice = texttospeech.VoiceSelectionParams(
            language_code=voice_language_code,
            name=voice_name,
            ssml_gender={
                "male": texttospeech.SsmlVoiceGender.MALE,
                "female": texttospeech.SsmlVoiceGender.FEMALE,
                "neutral": texttospeech.SsmlVoiceGender.NEUTRAL,
            }[voice_gender],
        )

        # Configure audio settings
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
        )

    def text_to_audio(self, text: str) -> NDArray[np.float32]:
        # Create synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Get audio response
        response = self.client.synthesize_speech(
            input=synthesis_input, voice=self.voice, audio_config=self.audio_config
        )

        # Read WAV data from memory
        wav_io = io.BytesIO(response.audio_content)
        with wave.open(wav_io, "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()

            # Read raw audio data
            raw_data = wav_file.readframes(n_frames)

            # Convert raw bytes to numpy array based on sample width
            if sample_width == 1:
                samples = np.frombuffer(raw_data, dtype=np.uint8)
                samples = (samples - 128) / 128.0
            elif sample_width == 2:
                samples = np.frombuffer(raw_data, dtype=np.int16)
                samples = samples / 32768.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            # Convert to mono if stereo
            if n_channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)

            samples = samples.astype(np.float32)

        return samples

    def loop(self):
        try:
            self.text_to_audio("warming up")
        except Exception:
            pass

        while not self.exit_event.is_set():
            # Get text from input channel
            try:
                text = self.input_queue.get(True, 0.25)
            except Empty:
                continue

            if self.speaking_event.is_set():
                continue

            # Convert text to audio array with retry logic
            audio_array = None
            for attempt in range(3):
                try:
                    audio_array = self.text_to_audio(text)
                    break
                except Exception:
                    if self.speaking_event.is_set():
                        break
                    if self.exit_event.is_set():
                        return
                    continue

            if audio_array is not None:
                self.output_queue.put((text, audio_array))
