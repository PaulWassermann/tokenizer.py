from abc import ABC, abstractmethod
from os import PathLike
from typing import Self


class Tokenizer(ABC):
    """
    Base class defining an interface for concrete tokenizers.
    
    Attributes
    ----------
    vocab : dict (int -> bytes)
        A mapping from the token ids to the actual bytes they encode. The bytes can be 
        converted to text using UTF-8 decoding. However, there is no guarantee that the
        bytes are actually valid UTF-8.

    merges: dict ((int, int) -> str)  
        A mapping from pairs of token ids to the 'merged' token ids.
    """
    def __init__(self):
        self.vocab: dict[int, bytes] = {bytes(chr(i), encoding="utf-8") for i in range(256)} 
        self.merges: dict[tuple[int, int], int] = {}
    
    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        ...
    
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        ...
    
    @abstractmethod
    @classmethod
    def load(self, path: str | PathLike) -> Self:
        ...

    @abstractmethod
    def save(self, path: str | PathLike) -> None:
        ...

    @abstractmethod
    def train(self, training_data: str, num_merges: int) -> None:
        ...