import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import ReverseDataset, collate_fn
from model import Encoder, Decoder, Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "data/reverse_dataset.txt"

BATCH_SIZE = 32
EPOCHS = 25
LR = 5e-4
CLIP = 1.0

EMB_DIM = 64
ENC_HID_DIM = 128
DEC_HID_DIM = 128
ATTN_DIM = 64

BASE_TEACHER_FORCING = 0.6

dataset = ReverseDataset(DATA_PATH)

PAD_IDX = dataset.pad_idx
SOS_IDX = dataset.sos_idx
EOS_IDX = dataset.eos_idx

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, pad_value=PAD_IDX)
)

SRC_VOCAB_SIZE = len(dataset.src_vocab.itos)
TRG_VOCAB_SIZE = len(dataset.trg_vocab.itos)

print("SRC vocab size:", len(dataset.src_vocab.itos))
print("TRG vocab size:", len(dataset.trg_vocab.itos))
print("Sample vocab:", dataset.src_vocab.itos[:20])


encoder = Encoder(
    vocab_size=SRC_VOCAB_SIZE,
    emb_dim=EMB_DIM,
    hid_dim=ENC_HID_DIM,
    n_layers=1,
    dropout=0.1,
    bidir=True,
    pad_idx=PAD_IDX
)

# enc_hid_dim ahora es *2 por bidir
decoder = Decoder(
    vocab_size=TRG_VOCAB_SIZE,
    emb_dim=EMB_DIM,
    enc_hid_dim=ENC_HID_DIM * 2,
    dec_hid_dim=DEC_HID_DIM,
    attn_dim=ATTN_DIM,
    dropout=0.1,
    pad_idx=PAD_IDX
)

model = Seq2Seq(
    encoder=encoder,
    decoder=decoder,
    sos_idx=SOS_IDX,
    eos_idx=EOS_IDX,
    device=DEVICE
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5
)

def train_epoch(model, loader, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0.0

    for src, trg, src_mask, src_lens, _ in loader:
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)
        src_mask = src_mask.to(DEVICE)
        src_lens = src_lens.to(DEVICE)

        optimizer.zero_grad()

        outputs, _ = model(
            src=src,
            trg=trg,
            src_lengths=src_lens,
            mask=src_mask,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        output_dim = outputs.shape[-1]

        outputs = outputs[:, 1:, :].reshape(-1, output_dim)
        trg_gold = trg[:, 1:].reshape(-1)

        loss = criterion(outputs, trg_gold)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)

best_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    teacher_forcing = max(0.1, BASE_TEACHER_FORCING * (0.95 ** (epoch - 1)))

    loss = train_epoch(model, loader, teacher_forcing_ratio=teacher_forcing)
    print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | TF: {teacher_forcing:.3f}")

    scheduler.step(loss)

    if loss < best_loss:
        best_loss = loss
        torch.save(
            {"model_state_dict": model.state_dict()},
            "seq2seq_bahdanau.pt"
        )
        #print("  ✅ Nuevo mejor modelo guardado (seq2seq_bahdanau.pt)")

print("Entrenamiento terminado.")
