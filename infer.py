import argparse
import torch

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
def translate_sentence(model, src_vocab, trg_vocab, src_pad_idx, trg_sos_idx, trg_eos_idx, sentence: str, max_len=MAX_LEN):
    tokens = sentence.strip().split()
    if not tokens:
        return []

    unk = src_vocab.stoi["<unk>"]
    src_ids = [src_vocab.stoi.get(t, unk) for t in tokens]

    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)  # [1, Tx]
    src_len = torch.tensor([len(src_ids)], dtype=torch.long).to(DEVICE)
    src_mask = (src_tensor != src_pad_idx).int()

    enc_outputs, (h, c) = model.encoder(src_tensor, src_len)
    hidden = model._init_decoder_hidden(h, c)

    y = torch.tensor([trg_sos_idx], device=DEVICE, dtype=torch.long)
    generated = []

    for _ in range(max_len):
        logits, hidden, _alpha = model.decoder(y, hidden, enc_outputs, mask=src_mask)
        next_token = logits.argmax(dim=-1).item()

        if next_token == trg_eos_idx:
            break

        generated.append(trg_vocab.itos[next_token])
        y = torch.tensor([next_token], device=DEVICE, dtype=torch.long)

    return generated


def run_inference(direction: str):
    ckpt_path = f"seq2seq_bahdanau_{direction}.pt"
    model, src_vocab, trg_vocab, src_pad_idx, trg_sos_idx, trg_eos_idx = load_model(ckpt_path)

    print(f"\n=== INFERENCIA {direction} ===")
    print("Escribe una frase tokenizada por espacios. 'q' para salir.\n")

    while True:
        s = input("> ").strip()
        if s.lower() == "q":
            break

        out = translate_sentence(
            model=model,
            src_vocab=src_vocab,
            trg_vocab=trg_vocab,
            src_pad_idx=src_pad_idx,
            trg_sos_idx=trg_sos_idx,
            trg_eos_idx=trg_eos_idx,
            sentence=s,
        )
        print(" ".join(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["es-en", "en-es"], required=True)
    args = parser.parse_args()
    run_inference(args.direction)


if __name__ == "__main__":
    main()
