"""
Simple in-memory conversation memory; not persistent. Use for short-lived sessions in the chatbot demo.
"""
from collections import deque
from typing import Deque, Tuple

class Memory:
    def __init__(self, max_len: int = 20):
        self.store: Deque[Tuple[str,str]] = deque(maxlen=max_len)

    def add(self, user: str, bot: str):
        self.store.append((user, bot))

    def get(self):
        return list(self.store)


# singleton for demo
GLOBAL_MEMORY = Memory()
