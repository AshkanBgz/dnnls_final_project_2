import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Text Autoencoder
# ---------------------------------------------------------------------------

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_seq, hidden, cell):
        embedded = self.embedding(input_seq)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        return self.out(output), hidden, cell


class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        _, hidden, cell = self.encoder(input_seq)
        predictions, _, _ = self.decoder(target_seq[:, :-1], hidden, cell)
        return predictions


# ---------------------------------------------------------------------------
# Visual Autoencoder
# ---------------------------------------------------------------------------

class Backbone(nn.Module):
    def __init__(self, latent_dim=16, output_w=8, output_h=16):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=2, padding=3),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1),
        )
        self.flatten_dim = 64 * output_w * output_h
        self.fc1 = nn.Sequential(nn.Linear(self.flatten_dim, latent_dim), nn.ReLU())

    def forward(self, x):
        x = self.encoder_conv(x)
        x = x.view(-1, self.flatten_dim)
        return self.fc1(x)


class VisualEncoder(nn.Module):
    def __init__(self, latent_dim=16, output_w=8, output_h=16):
        super().__init__()
        self.context_backbone = Backbone(latent_dim, output_w, output_h)
        self.content_backbone = Backbone(latent_dim, output_w, output_h)
        self.projection = nn.Linear(2 * latent_dim, latent_dim)

    def forward(self, x):
        z = torch.cat((self.content_backbone(x), self.context_backbone(x)), dim=1)
        return self.projection(z)


class VisualDecoder(nn.Module):
    def __init__(self, latent_dim=16, output_w=8, output_h=16):
        super().__init__()
        self.imh = 60
        self.imw = 125
        self.output_w = output_w
        self.output_h = output_h
        self.flatten_dim = 64 * output_w * output_h

        self.fc1 = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 3, kernel_size=7, stride=2, padding=3, output_padding=(1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc1(z)
        return self.decode_image(x), self.decode_image(x)

    def decode_image(self, x):
        x = x.view(-1, 64, self.output_w, self.output_h)
        x = self.decoder_conv(x)
        return x[:, :, :self.imh, :self.imw]


class VisualAutoencoder(nn.Module):
    def __init__(self, latent_dim=16, output_w=8, output_h=16):
        super().__init__()
        self.encoder = VisualEncoder(latent_dim, output_w, output_h)
        self.decoder = VisualDecoder(latent_dim, output_w, output_h)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Attention + Sequence Predictor  (baseline)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, rnn_outputs):
        energy = self.attn(rnn_outputs).squeeze(2)
        attn_weights = self.softmax(energy)
        context = torch.bmm(attn_weights.unsqueeze(1), rnn_outputs)
        return context.squeeze(1)


class SequencePredictor(nn.Module):
    """Baseline sequence predictor: CNN encoder + simple concat fusion + GRU."""

    def __init__(self, visual_autoencoder, text_autoencoder, latent_dim, gru_hidden_dim):
        super().__init__()
        self.image_encoder = visual_autoencoder.encoder
        self.text_encoder = text_autoencoder.encoder

        fusion_dim = latent_dim * 2
        self.temporal_rnn = nn.GRU(fusion_dim, gru_hidden_dim, batch_first=True)
        self.attention = Attention(gru_hidden_dim)
        self.projection = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, latent_dim),
            nn.ReLU(),
        )

        self.image_decoder = visual_autoencoder.decoder
        self.text_decoder = text_autoencoder.decoder

        self.fused_to_h0 = nn.Linear(latent_dim, 16)
        self.fused_to_c0 = nn.Linear(latent_dim, 16)

    def forward(self, image_seq, text_seq, target_seq):
        batch_size, seq_len, C, H, W = image_seq.shape

        img_flat = image_seq.view(batch_size * seq_len, C, H, W)
        txt_flat = text_seq.view(batch_size * seq_len, -1)

        z_v_flat = self.image_encoder(img_flat)
        _, hidden, cell = self.text_encoder(txt_flat)

        z_v_seq = z_v_flat.view(batch_size, seq_len, -1)
        z_t_seq = hidden.squeeze(0).view(batch_size, seq_len, -1)

        z_fusion_flat = torch.cat((z_v_flat, hidden.squeeze(0)), dim=1)
        z_fusion_seq = z_fusion_flat.view(batch_size, seq_len, -1)

        zseq, h = self.temporal_rnn(z_fusion_seq)
        h = h.squeeze(0)
        context = self.attention(zseq)
        z = self.projection(torch.cat((h, context), dim=1))

        pred_image_content, pred_image_context = self.image_decoder(z)

        h0 = self.fused_to_h0(z).unsqueeze(0)
        c0 = self.fused_to_c0(z).unsqueeze(0)
        decoder_input = target_seq[:, :, :-1].squeeze(1)
        predicted_text_logits_k, _, _ = self.text_decoder(decoder_input, h0, c0)

        return pred_image_content, pred_image_context, predicted_text_logits_k, h0, c0, z_v_seq, z_t_seq
