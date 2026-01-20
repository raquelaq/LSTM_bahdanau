import numpy as np
import torch
import matplotlib.pyplot as plt

from model import Encoder, Decoder, Seq2Seq
from dataset import Vocab

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 30


def load_model(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    cfg = ckpt["config"]
    src_vocab = Vocab.from_itos(ckpt["vocabs"]["src_itos"])
    trg_vocab = Vocab.from_itos(ckpt["vocabs"]["trg_itos"])

    src_pad_idx = ckpt["special"]["src_pad_idx"]
    trg_pad_idx = ckpt["special"]["trg_pad_idx"]
    trg_sos_idx = ckpt["special"]["trg_sos_idx"]
    trg_eos_idx = ckpt["special"]["trg_eos_idx"]

    encoder = Encoder(
        vocab_size=len(src_vocab.itos),
        emb_dim=cfg["emb_dim"],
        hid_dim=cfg["enc_hid_dim"],
        n_layers=1,
        dropout=0.0,
        bidir=cfg["bidir"],
        pad_idx=src_pad_idx,
    )

    decoder = Decoder(
        vocab_size=len(trg_vocab.itos),
        emb_dim=cfg["emb_dim"],
        enc_hid_dim=cfg["enc_hid_dim"] * (2 if cfg["bidir"] else 1),
        dec_hid_dim=cfg["dec_hid_dim"],
        attn_dim=cfg["attn_dim"],
        dropout=0.0,
        pad_idx=trg_pad_idx,
    )

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        sos_idx=trg_sos_idx,
        eos_idx=trg_eos_idx,
        device=DEVICE,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, src_vocab, trg_vocab, src_pad_idx, trg_sos_idx, trg_eos_idx


@torch.no_grad()
def translate_with_attention(model, src_vocab, trg_vocab, src_pad_idx, sos_idx, eos_idx, sentence: str, max_len: int = MAX_LEN):
    tokens = sentence.strip().split()
    if not tokens:
        return [], np.zeros((0, 0)), []

    unk = src_vocab.stoi["<unk>"]
    src_ids = [src_vocab.stoi.get(t, unk) for t in tokens]

    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    src_len = torch.tensor([len(src_ids)], dtype=torch.long).to(DEVICE)
    src_mask = (src_tensor != src_pad_idx).int()  # [1, Tx]

    enc_outputs, (h, c) = model.encoder(src_tensor, src_len)
    hidden = model._init_decoder_hidden(h, c)

    y = torch.tensor([sos_idx], device=DEVICE, dtype=torch.long)
    out_tokens = []
    attn_rows = []

    for _ in range(max_len):
        logits, hidden, alpha = model.decoder(y, hidden, enc_outputs, mask=src_mask)
        next_id = int(logits.argmax(dim=-1).item())

        if next_id == eos_idx:
            break

        out_tokens.append(trg_vocab.itos[next_id])
        attn_rows.append(alpha.squeeze(0).detach().cpu().numpy())  # [Tx]
        y = torch.tensor([next_id], device=DEVICE, dtype=torch.long)

    attn = np.array(attn_rows) if len(attn_rows) > 0 else np.zeros((0, len(tokens)))
    return out_tokens, attn, tokens


def show_attention(attn, src_tokens, out_tokens, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(attn, aspect="auto")

    ax.set_xticks(np.arange(len(src_tokens)))
    ax.set_yticks(np.arange(len(out_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticklabels(out_tokens)

    ax.set_xlabel("Tokens origen")
    ax.set_ylabel("Tokens destino")
    ax.set_title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()



