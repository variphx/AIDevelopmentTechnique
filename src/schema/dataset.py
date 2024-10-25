"""Subclasses or torch.utils.data.Dataset for loading required data"""

from torch.utils.data import Dataset as _Dataset
from transformers import PreTrainedTokenizerBase
import re as _re


class RawTrafficRulesDataset(_Dataset):
    """Raw text dataset derived from Main-QCVN 41_2019-BGTVT"""

    def __init__(self, raw_file: str, chunk_size: int, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self._tokenizer = tokenizer
        with open(raw_file, "r") as f:
            self._raw_file = f.read()
        self._raw_file = _re.sub(r"\s+", " ", self._raw_file)
        self._raw_file = self._raw_file.strip()

        self._chunks = [
            self._raw_file[i : i + chunk_size]
            for i in range(0, len(self._raw_file), chunk_size)
        ]

    def __len__(self):
        return len(self._chunks)

    def __getitem__(self, index):
        item = self._chunks[index]
        tokenized = self._tokenizer(item)
        return tokenized
