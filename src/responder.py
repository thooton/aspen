from queue import Empty, SimpleQueue
from threading import Event
from typing import List
import time
import re
from anthropic import Anthropic
from anthropic.types.message_param import MessageParam
from conversation import Conversation

END_PUNCTUATIONS = [".", "!", "?", "。", "！", "？", "...", "。。。"]
ABBREVIATIONS = [
    "Mr.",
    "Mrs.",
    "Dr.",
    "Prof.",
    "Inc.",
    "Ltd.",
    "Jr.",
    "Sr.",
    "e.g.",
    "i.e.",
    "vs.",
    "St.",
    "Rd.",
]


def segment_text_by_regex(text: str) -> tuple[list[str], str]:
    """
    Segment text into complete sentences using regex pattern matching.

    Args:
        text: Text to segment into sentences

    Returns:
        tuple: (list of complete sentences, remaining incomplete text)
    """
    if not text:
        return [], ""

    complete_sentences = []
    remaining_text = text.strip()

    # Create pattern for matching sentences ending with any end punctuation
    escaped_punctuations = [re.escape(p) for p in END_PUNCTUATIONS]
    pattern = r"(.*?(?:[" + "|".join(escaped_punctuations) + r"]))"

    while remaining_text:
        match = re.search(pattern, remaining_text)
        if not match:
            break

        end_pos = match.end(1)
        potential_sentence = remaining_text[:end_pos].strip()

        # Skip if sentence ends with abbreviation
        if any(potential_sentence.endswith(abbrev) for abbrev in ABBREVIATIONS):
            remaining_text = remaining_text[end_pos:].lstrip()
            continue

        complete_sentences.append(potential_sentence)
        remaining_text = remaining_text[end_pos:].lstrip()

    return complete_sentences, remaining_text


class Responder:
    """Handles text responses in a conversation using Anthropic API."""

    MAX_RETRIES = 5
    RETRY_DELAY = 1.0

    def __init__(
        self,
        input_queue: SimpleQueue[str],
        output_queue: SimpleQueue[str],
        speaking_event: Event,
        exit_event: Event,
        conversation: Conversation,
        system_message: str,
        model: str,
        max_tokens: int,
        api_key: str,
    ):
        """Initialize responder with communication channels and Anthropic client."""
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.speaking_event = speaking_event
        self.exit_event = exit_event
        self.conversation = conversation
        self.client = Anthropic(api_key=api_key)
        self.system_message = system_message
        self.model = model
        self.max_tokens = max_tokens

    def loop(self) -> None:
        """Continuously process incoming messages and generate responses."""
        try:
            self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}],
            )
        except Exception:
            pass

        while not self.exit_event.is_set():
            try:
                user_message = self.input_queue.get(True, 0.25)
            except Empty:
                continue

            messages = self.conversation.append("user", user_message)

            if self.speaking_event.is_set():
                continue
            if self.exit_event.is_set():
                return

            for attempt in range(self.MAX_RETRIES):
                try:
                    with self.client.messages.stream(
                        model=self.model,
                        system=self.system_message,
                        max_tokens=self.max_tokens,
                        messages=messages,
                    ) as stream:
                        buffer = ""
                        for text in stream.text_stream:
                            if self.speaking_event.is_set():
                                break
                            if self.exit_event.is_set():
                                return
                            buffer += text
                            sentences, buffer = segment_text_by_regex(buffer)
                            for sentence in sentences:
                                if sentence:
                                    self.output_queue.put(sentence)
                        if buffer.strip():
                            self.output_queue.put(buffer.strip())
                    break
                except Exception as e:
                    if attempt == self.MAX_RETRIES - 1:
                        self.output_queue.put(
                            f"Error: Failed after {self.MAX_RETRIES} attempts: {e}"
                        )
                    if self.speaking_event.wait(self.RETRY_DELAY):
                        break
