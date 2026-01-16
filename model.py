import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super().__init__()
        self.W = nn.Linear(enc_hid_dim + dec_hid_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        """
        dec_hidden: [B, dec_hid_dim]
        enc_outputs: [B, Tx, enc_hid_dim]
        mask: [B, Tx] con 1 en tokens reales y 0 en pad (opcional)
        """
        B, Tx, _ = enc_outputs.shape

        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, Tx, 1)   # [B, Tx, dec_hid_dim]
        energy_in = torch.cat([dec_hidden, enc_outputs], dim=-1) # [B, Tx, enc+dec]

        energy = torch.tanh(self.W(energy_in))                  # [B, Tx, attn_dim]
        scores = self.v(energy).squeeze(-1)                     # [B, Tx]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        alpha = F.softmax(scores, dim=1)                        # [B, Tx]
        context = torch.bmm(alpha.unsqueeze(1), enc_outputs)     # [B, 1, enc_hid_dim]
        context = context.squeeze(1)                            # [B, enc_hid_dim]

        return context, alpha

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers=1, dropout=0.1, bidir=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            emb_dim, hid_dim, num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0, bidirectional=bidir
        )
        self.dropout = nn.Dropout(dropout)
        self.bidir = bidir
        self.hid_dim = hid_dim

    def forward(self, src, src_lengths=None):
        # src: [B, Tx]
        emb = self.dropout(self.embedding(src))  # [B, Tx, emb_dim]
        outputs, (h, c) = self.rnn(emb)          # outputs: [B, Tx, H*(2 si bidir)]
        return outputs, (h, c)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, attn_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attn = BahdanauAttention(enc_hid_dim, dec_hid_dim, attn_dim)
        self.rnn = nn.LSTM(emb_dim + enc_hid_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y_prev, hidden, enc_outputs, mask=None):
        """
        y_prev: [B] token previo
        hidden: (h, c) donde h,c: [1, B, dec_hid_dim] (si 1 layer)
        enc_outputs: [B, Tx, enc_hid_dim]
        """
        h, c = hidden
        dec_hidden = h[-1]  # [B, dec_hid_dim]

        context, alpha = self.attn(dec_hidden, enc_outputs, mask=mask)  # context [B, enc_hid_dim]

        emb = self.dropout(self.embedding(y_prev)).unsqueeze(1)         # [B, 1, emb_dim]
        rnn_in = torch.cat([emb, context.unsqueeze(1)], dim=-1)         # [B, 1, emb+enc]

        output, (h_new, c_new) = self.rnn(rnn_in, (h, c))               # output [B,1,dec]
        logits = self.fc_out(output.squeeze(1) + context)

        return logits, (h_new, c_new), alpha

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, sos_idx, eos_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def forward(self, src, trg, mask=None, teacher_forcing_ratio=0.5):
        """
        src: [B, Tx]
        trg: [B, Ty] (incluye <sos> al inicio normalmente)
        """
        B, Ty = trg.shape
        vocab_size = self.decoder.fc_out.out_features

        enc_outputs, (h, c) = self.encoder(src)

        # Si tus dims no encajan (bidir, etc.), aquí harías proyección.
        # Para simplificar: asume que enc_hid_dim == dec_hid_dim o ajusta con Linear.

        outputs = torch.zeros(B, Ty, vocab_size, device=self.device)
        attn_weights = []

        y = trg[:, 0]  # primer token: <sos>

        hidden = (h[-1:].contiguous(), c[-1:].contiguous())

        for t in range(1, Ty):
            logits, hidden, alpha = self.decoder(y, hidden, enc_outputs, mask=mask)
            outputs[:, t, :] = logits
            attn_weights.append(alpha)

            use_tf = torch.rand(1).item() < teacher_forcing_ratio
            y = trg[:, t] if use_tf else logits.argmax(dim=-1)

        # attn_weights: lista de Ty-1 tensores [B, Tx]
        return outputs, attn_weights

