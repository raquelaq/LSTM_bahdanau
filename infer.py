import torch

from model import Encoder, Decoder, Seq2Seq
from data import ReverseDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "seq2seq_bahdanau.pt"
DATA_PATH = "data/reverse_dataset.txt"

EMB_DIM = 64
ENC_HID_DIM = 128
DEC_HID_DIM = 128
ATTN_DIM = 64
MAX_LEN = 30

dataset = ReverseDataset(DATA_PATH)

SRC_VOCAB_SIZE = len(dataset.src_vocab.itos)
TRG_VOCAB_SIZE = len(dataset.trg_vocab.itos)

PAD_IDX = dataset.pad_idx
SOS_IDX = dataset.sos_idx
EOS_IDX = dataset.eos_idx

encoder = Encoder(
    vocab_size=SRC_VOCAB_SIZE,
    emb_dim=EMB_DIM,
    hid_dim=ENC_HID_DIM,
    n_layers=1,
    dropout=0.0,
    bidir=True,
    pad_idx=PAD_IDX
)

decoder = Decoder(
    vocab_size=TRG_VOCAB_SIZE,
    emb_dim=EMB_DIM,
    enc_hid_dim=ENC_HID_DIM * 2,
    dec_hid_dim=DEC_HID_DIM,
    attn_dim=ATTN_DIM,
    dropout=0.0,
    pad_idx=PAD_IDX
)

model = Seq2Seq(
    encoder=encoder,
    decoder=decoder,
    sos_idx=SOS_IDX,
    eos_idx=EOS_IDX,
    device=DEVICE
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

@torch.no_grad()
def translate_sentence(sentence, max_len=MAX_LEN):
    tokens = sentence.strip().split()
    if len(tokens) == 0:
        return [], []

    src_ids = [dataset.src_vocab.stoi.get(t, dataset.src_vocab.stoi["<unk>"]) for t in tokens]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)  # [1, Tx]
    src_len = torch.tensor([len(src_ids)], dtype=torch.long).to(DEVICE)
    src_mask = (src_tensor != PAD_IDX).int()

    enc_outputs, (h, c) = model.encoder(src_tensor, src_len)
    hidden = model._init_decoder_hidden(h, c)

    y = torch.tensor([SOS_IDX], device=DEVICE, dtype=torch.long)

    generated = []
    attentions = []

    for _ in range(max_len):
        logits, hidden, alpha = model.decoder(y, hidden, enc_outputs, mask=src_mask)
        next_token = logits.argmax(dim=-1).item()

        if next_token == EOS_IDX:
            break

        generated.append(dataset.trg_vocab.itos[next_token])
        attentions.append(alpha.squeeze(0).cpu())

        y = torch.tensor([next_token], device=DEVICE, dtype=torch.long)

    return generated, attentions

def run_inference():
    while True:
        sentence = input("\nInput sentence (or 'q' to quit): ").strip()
        if sentence.lower() == "q":
            break

        out, attn = translate_sentence(sentence)
        print("Output:", " ".join(out))
        print("Attention steps:", len(attn))
