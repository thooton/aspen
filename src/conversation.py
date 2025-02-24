from threading import Lock
from typing import Literal, List
from anthropic.types.message_param import MessageParam


class Conversation:
    def __init__(self):
        self.messages: List[MessageParam] = []
        self.lock = Lock()

    def append(
        self, role: Literal["user", "assistant"], content: str
    ) -> List[MessageParam]:
        with self.lock:
            if self.messages and self.messages[-1]["role"] == role:
                last_message = self.messages[-1]
                last_content = last_message["content"]
                if isinstance(last_content, str):
                    spacer = (
                        " "
                        if (
                            last_content
                            and not content.startswith((".", "!", "?", ","))
                        )
                        else ""
                    )
                    last_message["content"] = f"{last_content}{spacer}{content}"
            else:
                self.messages.append({"role": role, "content": content})
            return self.messages.copy()

    def get(self) -> List[MessageParam]:
        with self.lock:
            return self.messages.copy()

    def reset(self) -> None:
        with self.lock:
            self.messages = []
