# config.py
from pathlib import Path
import json
from pydantic import BaseModel, Field, ValidationError, PositiveInt
from typing import Literal


class MicrophoneConfig(BaseModel):
    sample_rate: PositiveInt = Field(..., description="Microphone sample rate in Hz")


class SegmenterConfig(BaseModel):
    model_path: str = Field(..., description="Segmenter torch JIT model path")


class TranscriberConfig(BaseModel):
    api_key: str = Field(..., min_length=1, description="Transcription API key")
    model: str = Field(..., min_length=1, description="Transcription model name")
    language: str = Field(..., min_length=1, description="Transcription language code")


class ResponderConfig(BaseModel):
    system_message: str = Field(
        ..., min_length=1, description="System message for AI responder"
    )
    model: str = Field(..., min_length=1, description="Responder AI model name")
    max_tokens: PositiveInt = Field(..., description="Maximum tokens for response")
    api_key: str = Field(..., min_length=1, description="Responder API key")


class SynthesizerConfig(BaseModel):
    credentials_path: str = Field(
        ..., min_length=1, description="Path to synthesizer credentials"
    )
    voice_language_code: str = Field(
        ..., min_length=1, description="Voice language code"
    )
    voice_name: str = Field(..., min_length=1, description="Voice name")
    voice_gender: Literal["male", "female", "neutral"] = Field(
        ..., description="Voice gender"
    )
    sample_rate: PositiveInt = Field(..., description="Synthesizer sample rate in Hz")


class GeneralConfig(BaseModel):
    initial_greeting: str = Field(
        ..., min_length=1, description="Initial greeting message"
    )
    tw_host: str = Field(
        ..., description="Listen host for the web server when Twilio is used"
    )
    tw_port: PositiveInt = Field(
        ..., description="Listen port for the web server when Twilio is used"
    )


class Config(BaseModel):
    microphone: MicrophoneConfig
    segmenter: SegmenterConfig
    transcriber: TranscriberConfig
    responder: ResponderConfig
    synthesizer: SynthesizerConfig
    general: GeneralConfig


def load_config(config_path: str) -> Config:
    """Load and validate configuration from file"""
    try:
        with open(config_path, "rb") as f:
            data = json.load(f)
        return Config.model_validate(data)
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file {config_path}")
    except ValidationError as e:
        raise ValueError(f"Configuration validation failed:\n{e}")
