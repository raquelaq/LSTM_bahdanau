import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import ReverseDataset, collate_fn
from model import Encoder, Decoder, Seq2Seq

# -------------------------
# Configuración general
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "data/reverse_dataset.txt"

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
TEACHER_FORCING = 0.6
CLIP = 1.0

EMB_DIM = 64
ENC_HID_DIM = 128
DEC_HID_DIM = 128
ATTN_DIM = 64

# -------------------------
# Dataset y DataLoader
# -------------------------
dataset = ReverseDataset(DATA_PATH)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

SRC_VOCAB_SIZE = len(dataset.src_vocab.itos)
TRG_VOCAB_SIZE = len(dataset.trg_vocab.itos)

PAD_IDX = dataset.pad_idx
SOS_IDX = dataset.sos_idx
EOS_IDX = dataset.eos_idx

# -------------------------
# Modelo
# -------------------------
encoder = Encoder(
    vocab_size=SRC_VOCAB_SIZE,
    emb_dim=EMB_DIM,
    hid_dim=ENC_HID_DIM,
    n_layers=1,
    dropout=0.1,
    bidir=False
)

decoder = Decoder(
    vocab_size=TRG_VOCAB_SIZE,
    emb_dim=EMB_DIM,
    enc_hid_dim=ENC_HID_DIM,
    dec_hid_dim=DEC_HID_DIM,
    attn_dim=ATTN_DIM,
    dropout=0.1
)

model = Seq2Seq(
    encoder=encoder,
    decoder=decoder,
    sos_idx=SOS_IDX,
    eos_idx=EOS_IDX,
    device=DEVICE
).to(DEVICE)

# -------------------------
# Pérdida y optimizador
# -------------------------
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Training loop
# -------------------------
def train_epoch(model, loader):
    model.train()
    epoch_loss = 0.0

    for src, trg, src_mask in loader:
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)
        src_mask = src_mask.to(DEVICE)

        optimizer.zero_grad()

        outputs, _ = model(
            src,
            trg,
            mask=src_mask,
            teacher_forcing_ratio=TEACHER_FORCING
        )

        # outputs: [B, Ty, vocab]
        # trg:     [B, Ty]
        output_dim = outputs.shape[-1]

        outputs = outputs[:, 1:, :].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(outputs, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)

# -------------------------
# Entrenamiento
# -------------------------
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(model, loader)
    print(f"Epoch {epoch:02d} | Loss: {loss:.4f}")

# -------------------------
# Guardar modelo
# -------------------------
torch.save({
    "model_state_dict": model.state_dict()
}, "seq2seq_bahdanau.pt")

print("Modelo guardado en seq2seq_bahdanau.pt")

