from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset

SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]


class Vocab:
    def __init__(self, texts: List[str], min_freq: int = 1):
        counter = Counter()
        for t in texts:
            counter.update(t.split())

        words = [(w, c) for w, c in counter.items() if c >= min_freq]
        words.sort(key=lambda x: (-x[1], x[0]))  # determinista

        self.itos = SPECIALS + [w for w, _ in words]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def from_itos(cls, itos: List[str]) -> "Vocab":
        obj = cls.__new__(cls)
        obj.itos = list(itos)
        obj.stoi = {w: i for i, w in enumerate(obj.itos)}
        return obj

    def encode(self, text: str) -> List[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(w, unk) for w in text.split()]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids]


@dataclass(frozen=True)
class Batch:
    src: torch.Tensor       # [B, Tx]
    trg: torch.Tensor       # [B, Ty] (incluye <sos> y <eos>)
    src_mask: torch.Tensor  # [B, Tx] (1 real, 0 pad)
    src_lens: torch.Tensor  # [B]
    trg_lens: torch.Tensor  # [B]


class TranslationDataset(Dataset):
    """
    Lee pares 'es ||| en' y construye (src, trg) según direction.
    direction:
      - "es-en": src=es, trg=en
      - "en-es": src=en, trg=es
    """
    def __init__(self, path: str, direction: str = "es-en", min_freq: int = 1):
        if direction not in ("es-en", "en-es"):
            raise ValueError("direction must be 'es-en' or 'en-es'")

        self.direction = direction
        self.pairs: List[Tuple[str, str]] = []

        es_texts, en_texts = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                left, right = line.split("|||")
                es = left.strip()
                en = right.strip()
                self.pairs.append((es, en))
                es_texts.append(es)
                en_texts.append(en)

        # Dos vocabularios (origen/destino) según direction
        if direction == "es-en":
            src_texts, trg_texts = es_texts, en_texts
        else:
            src_texts, trg_texts = en_texts, es_texts

        self.src_vocab = Vocab(src_texts, min_freq=min_freq)
        self.trg_vocab = Vocab(trg_texts, min_freq=min_freq)

        self.src_pad_idx = self.src_vocab.stoi["<pad>"]
        self.trg_pad_idx = self.trg_vocab.stoi["<pad>"]
        self.trg_sos_idx = self.trg_vocab.stoi["<sos>"]
        self.trg_eos_idx = self.trg_vocab.stoi["<eos>"]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        es, en = self.pairs[idx]
        if self.direction == "es-en":
            src_text, trg_text = es, en
        else:
            src_text, trg_text = en, es

        src_ids = self.src_vocab.encode(src_text)
        trg_ids = [self.trg_sos_idx] + self.trg_vocab.encode(trg_text) + [self.trg_eos_idx]

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(trg_ids, dtype=torch.long),
        )


def collate_fn(batch, src_pad_value: int, trg_pad_value: int) -> Batch:
    src_batch, trg_batch = zip(*batch)

    src_lens = torch.tensor([len(x) for x in src_batch], dtype=torch.long)
    trg_lens = torch.tensor([len(x) for x in trg_batch], dtype=torch.long)

    src_pad = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=src_pad_value
    )
    trg_pad = torch.nn.utils.rnn.pad_sequence(
        trg_batch, batch_first=True, padding_value=trg_pad_value
    )

    src_mask = (src_pad != src_pad_value).int()

    return Batch(
        src=src_pad,
        trg=trg_pad,
        src_mask=src_mask,
        src_lens=src_lens,
        trg_lens=trg_lens,
    )
