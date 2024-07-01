from abc import ABC, abstractmethod


class Tokenizer(ABC):

    def __init__(self):
        self.vocab: dict[int, bytes]
        self.substitutions: dict[tuple[int, int], int]
    
    @abstractmethod
    def decode(token_ids: list[int]) -> str:
        ...
    
    @abstractmethod
    def encode(text: str) -> list[int]:
        ...
    
    @abstractmethod
    @classmethod
    def load(...) -> ...:
        ...

    @abstractmethod
    def save(...) -> ...:
        ...

    @abstractmethod
    def train(training_data: str, num_merges: int) -> None:
        ...