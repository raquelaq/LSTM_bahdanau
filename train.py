import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TranslationDataset, collate_fn
from model import Encoder, Decoder, Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "data/es-en.txt"

BATCH_SIZE = 64
EPOCHS = 20
LR = 5e-4
CLIP = 1.0


EMB_DIM = 64
ENC_HID_DIM = 128
DEC_HID_DIM = 128
ATTN_DIM = 64

BASE_TEACHER_FORCING = 0.6


def build_model_and_data(direction: str):
    dataset = TranslationDataset(DATA_PATH, direction=direction, min_freq=1)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, src_pad_value=dataset.src_pad_idx, trg_pad_value=dataset.trg_pad_idx),
    )

    SRC_VOCAB_SIZE = len(dataset.src_vocab.itos)
    TRG_VOCAB_SIZE = len(dataset.trg_vocab.itos)

    encoder = Encoder(
        vocab_size=SRC_VOCAB_SIZE,
        emb_dim=EMB_DIM,
        hid_dim=ENC_HID_DIM,
        n_layers=1,
        dropout=0.1,
        bidir=True,
        pad_idx=dataset.src_pad_idx,
    )

    decoder = Decoder(
        vocab_size=TRG_VOCAB_SIZE,
        emb_dim=EMB_DIM,
        enc_hid_dim=ENC_HID_DIM * 2,
        dec_hid_dim=DEC_HID_DIM,
        attn_dim=ATTN_DIM,
        dropout=0.1,
        pad_idx=dataset.trg_pad_idx,
    )

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        sos_idx=dataset.trg_sos_idx,
        eos_idx=dataset.trg_eos_idx,
        device=DEVICE,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.trg_pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    return dataset, loader, model, criterion, optimizer, scheduler


def train_epoch(model, loader, criterion, optimizer, teacher_forcing_ratio: float):
    model.train()
    epoch_loss = 0.0

    for batch in loader:
        src = batch.src.to(DEVICE)
        trg = batch.trg.to(DEVICE)
        src_mask = batch.src_mask.to(DEVICE)
        src_lens = batch.src_lens.to(DEVICE)

        optimizer.zero_grad()

        outputs, _ = model(
            src=src,
            trg=trg,
            src_lengths=src_lens,
            mask=src_mask,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        # Ignoramos t=0 (<sos>) en la loss
        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:, :].reshape(-1, output_dim)
        trg_gold = trg[:, 1:].reshape(-1)

        loss = criterion(outputs, trg_gold)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / max(1, len(loader))


def train_direction(direction: str):
    dataset, loader, model, criterion, optimizer, scheduler = build_model_and_data(direction)

    print(f"\nEntrenando dirección: {direction}")
    print("SRC vocab size:", len(dataset.src_vocab.itos))
    print("TRG vocab size:", len(dataset.trg_vocab.itos))
    print("Ejemplo SRC vocab:", dataset.src_vocab.itos[:15])
    print("Ejemplo TRG vocab:", dataset.trg_vocab.itos[:15])

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        teacher_forcing = max(0.1, BASE_TEACHER_FORCING * (0.95 ** (epoch - 1)))

        loss = train_epoch(model, loader, criterion, optimizer, teacher_forcing_ratio=teacher_forcing)
        scheduler.step(loss)

        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | TF: {teacher_forcing:.3f}")

        if loss < best_loss:
            best_loss = loss

            ckpt_name = f"seq2seq_bahdanau_{direction}.pt"
            torch.save(
                {
                    "direction": direction,
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "emb_dim": EMB_DIM,
                        "enc_hid_dim": ENC_HID_DIM,
                        "dec_hid_dim": DEC_HID_DIM,
                        "attn_dim": ATTN_DIM,
                        "bidir": True,
                    },
                    "vocabs": {
                        "src_itos": dataset.src_vocab.itos,
                        "trg_itos": dataset.trg_vocab.itos,
                    },
                    "special": {
                        "src_pad_idx": dataset.src_pad_idx,
                        "trg_pad_idx": dataset.trg_pad_idx,
                        "trg_sos_idx": dataset.trg_sos_idx,
                        "trg_eos_idx": dataset.trg_eos_idx,
                    },
                },
                ckpt_name,
            )

    print(f"Terminado {direction}. Mejor loss: {best_loss:.4f}")







