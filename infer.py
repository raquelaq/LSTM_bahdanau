import torch
import torch.nn.functional as F

from model import Encoder, Decoder, Seq2Seq
from data import ReverseDataset

# -------------------------
# Configuración
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "seq2seq_bahdanau.pt"
DATA_PATH = "data/reverse_dataset.txt"

EMB_DIM = 64
ENC_HID_DIM = 128
DEC_HID_DIM = 128
ATTN_DIM = 64

MAX_LEN = 20

# -------------------------
# Cargar dataset y vocabularios
# -------------------------
dataset = ReverseDataset(DATA_PATH)

SRC_VOCAB_SIZE = len(dataset.src_vocab.itos)
TRG_VOCAB_SIZE = len(dataset.trg_vocab.itos)

PAD_IDX = dataset.pad_idx
SOS_IDX = dataset.sos_idx
EOS_IDX = dataset.eos_idx

# -------------------------
# Cargar modelo
# -------------------------
encoder = Encoder(
    vocab_size=SRC_VOCAB_SIZE,
    emb_dim=EMB_DIM,
    hid_dim=ENC_HID_DIM
)

decoder = Decoder(
    vocab_size=TRG_VOCAB_SIZE,
    emb_dim=EMB_DIM,
    enc_hid_dim=ENC_HID_DIM,
    dec_hid_dim=DEC_HID_DIM,
    attn_dim=ATTN_DIM
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

# -------------------------
# Greedy decoding
# -------------------------
def translate_sentence(sentence, max_len=MAX_LEN):
    tokens = sentence.strip().split()
    src_ids = [dataset.src_vocab.stoi.get(t, dataset.src_vocab.stoi["<unk>"]) for t in tokens]

    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)  # [1, Tx]
    src_mask = (src_tensor != PAD_IDX).int()

    with torch.no_grad():
        enc_outputs, (h, c) = model.encoder(src_tensor)
        hidden = (h[-1:].contiguous(), c[-1:].contiguous())

    y = torch.tensor([SOS_IDX], device=DEVICE)

    generated_tokens = []
    attentions = []

    for _ in range(max_len):
        with torch.no_grad():
            logits, hidden, alpha = model.decoder(
                y, hidden, enc_outputs, mask=src_mask
            )

        next_token = logits.argmax(dim=-1).item()

        if next_token == EOS_IDX:
            break

        generated_tokens.append(dataset.trg_vocab.itos[next_token])
        attentions.append(alpha.squeeze(0).cpu())

        y = torch.tensor([next_token], device=DEVICE)

    return generated_tokens, attentions

def run_inference():
    while True:
        sentence = input("\nInput sentence (or 'q' to quit): ")
        if sentence.lower() == "q":
            break

        translation, attn = translate_sentence(sentence)
        print("Output:", " ".join(translation))
        print("Attention steps:", len(attn))

