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
        dec_hidden:  [B, dec_hid_dim]
        enc_outputs: [B, Tx, enc_hid_dim]
        mask:        [B, Tx] (1 real, 0 pad)
        """
        B, Tx, _ = enc_outputs.shape

        dec_hidden = dec_hidden.unsqueeze(1).expand(B, Tx, dec_hidden.size(-1))
        energy_in = torch.cat([dec_hidden, enc_outputs], dim=-1)

        energy = torch.tanh(self.W(energy_in))          # [B, Tx, attn_dim]
        scores = self.v(energy).squeeze(-1)             # [B, Tx]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        alpha = F.softmax(scores, dim=1)                # [B, Tx]
        context = torch.bmm(alpha.unsqueeze(1), enc_outputs).squeeze(1)  # [B, enc_hid_dim]
        return context, alpha


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers=1, dropout=0.1, bidir=True, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            emb_dim, hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidir
        )
        self.dropout = nn.Dropout(dropout)
        self.bidir = bidir
        self.hid_dim = hid_dim
        self.n_layers = n_layers

    def forward(self, src, src_lengths):
        # src: [B, Tx]
        emb = self.dropout(self.embedding(src))  # [B, Tx, emb_dim]

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, (h, c) = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # outputs: [B, Tx, hid_dim * num_directions]
        return outputs, (h, c)


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, attn_dim, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attn = BahdanauAttention(enc_hid_dim, dec_hid_dim, attn_dim)

        self.rnn = nn.LSTM(emb_dim + enc_hid_dim, dec_hid_dim, batch_first=True)

        # ✅ CORRECCIÓN IMPORTANTE: combinar output y context por concatenación
        self.fc_out = nn.Linear(dec_hid_dim + enc_hid_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, y_prev, hidden, enc_outputs, mask=None):
        """
        y_prev:     [B]
        hidden:     (h, c) con h,c: [1, B, dec_hid_dim]
        enc_outputs:[B, Tx, enc_hid_dim]
        """
        h, c = hidden
        dec_hidden = h[-1]  # [B, dec_hid_dim]

        context, alpha = self.attn(dec_hidden, enc_outputs, mask=mask)  # [B, enc_hid_dim]

        emb = self.dropout(self.embedding(y_prev)).unsqueeze(1)         # [B, 1, emb_dim]
        rnn_in = torch.cat([emb, context.unsqueeze(1)], dim=-1)         # [B, 1, emb+enc]

        output, (h_new, c_new) = self.rnn(rnn_in, (h, c))               # output [B,1,dec]
        output = output.squeeze(1)                                      # [B, dec_hid_dim]

        logits = self.fc_out(torch.cat([output, context], dim=-1))      # [B, vocab]
        return logits, (h_new, c_new), alpha


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, sos_idx, eos_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

        # ✅ Proyección encoder → decoder (lo que me preguntaste)
        enc_out_dim = encoder.hid_dim * (2 if encoder.bidir else 1)
        dec_hid_dim = decoder.rnn.hidden_size
        self.fc_h = nn.Linear(enc_out_dim, dec_hid_dim)
        self.fc_c = nn.Linear(enc_out_dim, dec_hid_dim)

    def _init_decoder_hidden(self, h, c):
        """
        h,c del encoder:
          - unidireccional: [n_layers, B, hid_dim]
          - bidireccional:  [n_layers*2, B, hid_dim]
        Aquí construimos un vector [B, enc_out_dim] y lo proyectamos a dec_hid_dim.
        """
        if self.encoder.bidir:
            # Tomamos última capa: forward y backward
            # índices: -2 (forward), -1 (backward)
            h_enc = torch.cat([h[-2], h[-1]], dim=1)  # [B, hid_dim*2]
            c_enc = torch.cat([c[-2], c[-1]], dim=1)
        else:
            h_enc = h[-1]  # [B, hid_dim]
            c_enc = c[-1]

        h_dec = torch.tanh(self.fc_h(h_enc)).unsqueeze(0)  # [1, B, dec_hid_dim]
        c_dec = torch.tanh(self.fc_c(c_enc)).unsqueeze(0)
        return (h_dec, c_dec)

    def forward(self, src, trg, src_lengths, mask=None, teacher_forcing_ratio=0.5):
        """
        src:         [B, Tx]
        trg:         [B, Ty] (incluye <sos>)
        src_lengths: [B]
        """
        B, Ty = trg.shape
        vocab_size = self.decoder.fc_out.out_features

        enc_outputs, (h, c) = self.encoder(src, src_lengths)
        hidden = self._init_decoder_hidden(h, c)

        outputs = torch.zeros(B, Ty, vocab_size, device=self.device)
        attn_weights = []

        y = trg[:, 0]  # <sos>

        for t in range(1, Ty):
            logits, hidden, alpha = self.decoder(y, hidden, enc_outputs, mask=mask)
            outputs[:, t, :] = logits
            attn_weights.append(alpha)

            use_tf = torch.rand(1).item() < teacher_forcing_ratio
            y = trg[:, t] if use_tf else logits.argmax(dim=-1)

        return outputs, attn_weights
