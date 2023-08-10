from typing import Any, Protocol

class Model(Protocol):
    def run(self, prompt: str) -> Any: ...
