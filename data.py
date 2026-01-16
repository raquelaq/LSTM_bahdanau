from collections import Counter
import torch
from torch.utils.data import Dataset

SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]

class Vocab:
    def __init__(self, texts, min_freq=1):
        counter = Counter()
        for t in texts:
            counter.update(t.split())

        self.itos = SPECIALS + [w for w, c in counter.items() if c >= min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text):
        return [self.stoi.get(w, self.stoi["<unk>"]) for w in text.split()]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


class ReverseDataset(Dataset):
    def __init__(self, path):
        self.pairs = []
        src_texts, trg_texts = [], []

        with open(path, encoding="utf-8") as f:
            for line in f:
                src, trg = line.strip().split("|||")
                src, trg = src.strip(), trg.strip()
                self.pairs.append((src, trg))
                src_texts.append(src)
                trg_texts.append(trg)

        self.src_vocab = Vocab(src_texts)
        self.trg_vocab = Vocab(trg_texts)

        self.pad_idx = self.trg_vocab.stoi["<pad>"]
        self.sos_idx = self.trg_vocab.stoi["<sos>"]
        self.eos_idx = self.trg_vocab.stoi["<eos>"]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, trg = self.pairs[idx]
        src_ids = self.src_vocab.encode(src)
        trg_ids = [self.sos_idx] + self.trg_vocab.encode(trg) + [self.eos_idx]
        return torch.tensor(src_ids), torch.tensor(trg_ids)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)

    src_lens = [len(x) for x in src_batch]
    trg_lens = [len(x) for x in trg_batch]

    src_pad = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=0
    )
    trg_pad = torch.nn.utils.rnn.pad_sequence(
        trg_batch, batch_first=True, padding_value=0
    )

    src_mask = (src_pad != 0).int()

    return src_pad, trg_pad, src_mask

