from collections import Counter
import torch
from torch.utils.data import Dataset

SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]

class Vocab:
    def __init__(self, texts, min_freq=1):
        counter = Counter()
        for t in texts:
            counter.update(t.split())

        # Orden determinista: primero por frecuencia desc, luego alfabético
        words = [(w, c) for w, c in counter.items() if c >= min_freq]
        words.sort(key=lambda x: (-x[1], x[0]))

        self.itos = SPECIALS + [w for w, _ in words]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text):
        unk = self.stoi["<unk>"]
        return [self.stoi.get(w, unk) for w in text.split()]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


class ReverseDataset(Dataset):
    def __init__(self, path, min_freq=1):
        self.pairs = []
        src_texts, trg_texts = [], []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                src, trg = line.split("|||")
                src, trg = src.strip(), trg.strip()
                self.pairs.append((src, trg))
                src_texts.append(src)
                trg_texts.append(trg)

        self.src_vocab = Vocab(src_texts, min_freq=min_freq)
        self.trg_vocab = Vocab(trg_texts, min_freq=min_freq)

        self.pad_idx = self.trg_vocab.stoi["<pad>"]
        self.sos_idx = self.trg_vocab.stoi["<sos>"]
        self.eos_idx = self.trg_vocab.stoi["<eos>"]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, trg = self.pairs[idx]
        src_ids = self.src_vocab.encode(src)
        trg_ids = [self.sos_idx] + self.trg_vocab.encode(trg) + [self.eos_idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)


def collate_fn(batch, pad_value=0):
    src_batch, trg_batch = zip(*batch)

    src_lens = torch.tensor([len(x) for x in src_batch], dtype=torch.long)
    trg_lens = torch.tensor([len(x) for x in trg_batch], dtype=torch.long)

    src_pad = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=pad_value
    )
    trg_pad = torch.nn.utils.rnn.pad_sequence(
        trg_batch, batch_first=True, padding_value=pad_value
    )

    src_mask = (src_pad != pad_value).int()

    return src_pad, trg_pad, src_mask, src_lens, trg_lens
