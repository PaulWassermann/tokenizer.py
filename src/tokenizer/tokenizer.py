from abc import ABC, abstractmethod
from os import PathLike
from typing import Self


class Tokenizer(ABC):
    """
    Base class defining an interface for concrete tokenizers.
    
    Attributes
    ----------
    vocab : dict (int -> bytes)
        A mapping from the token ids to the bytes they encode. The bytes can be converted to text using UTF-8 decoding. However, there is no guarantee that the bytes are actually valid UTF-8.

    merges: dict ((int, int) -> str)  
        A mapping from pairs of token ids to the 'merged' token ids.
    """
    def __init__(self):
        self.vocab: dict[int, bytes] = {
            bytes(chr(i), encoding="utf-8") for i in range(256)
            } 
        self.merges: dict[tuple[int, int], int] = {}
    
    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """
        Decodes a list of token ids to a human-readable text. Decoding is performed assuming UTF-8. In case of an invalid bytes sequence, said sequence is simply replaced with a special character.

        Parameters
        ----------
        token_ids: list of int
            A list of integer tokens, with values greater than zero and inferior than the vocabulary size.

        Returns
        -------
        str
            The decoded string.
        """
    
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        ...
    
    @abstractmethod
    @classmethod
    def load(self, path: str | PathLike) -> Self:
        ...

    def _merge(
            self, token_ids: list[int], id_pair: tuple[int, int], new_id: int
            ) -> list[int]:
        new_token_ids = []
        for current_pair in zip(token_ids, token_ids[1:]):
            if current_pair == id_pair:
                new_token_ids.append(new_id)
            else:
                new_token_ids.append(current_pair[0])
        
        return new_token_ids

    @abstractmethod
    def save(self, path: str | PathLike) -> None:
        ...

    @abstractmethod
    def train(self, training_data: str, num_merges: int) -> None:
        ...