from .tokenizer import Tokenizer


class SimpleTokenizer(Tokenizer):
    """
    The `SimpleTokenizer` class implements the interface defined in :class:`Tokenizer`.

    The :method:`train` method implements a simple Byte Pair Encoding (BPE) algorithm,
    with no consideration whatsoever for wasteful / useless merges. This tokenizer does
    not support special token handling.

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
        super().__init__()

    def decode(self):
        ...

    def encode(self):
        ...

    def train(self):
        ...