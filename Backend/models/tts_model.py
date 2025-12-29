import torch
import torch.nn as nn

class SimpleTTS(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, mel_dim):
        super(SimpleTTS, self).__init__()
        self.hidden_dim = hidden_dim
        self.mel_dim = mel_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(mel_dim, hidden_dim, batch_first=True)  # 🔧 FIXED HERE
        self.linear = nn.Linear(hidden_dim, mel_dim)

    def forward(self, text, text_lens, max_audio_len):
        emb = self.embedding(text)
        enc_out, _ = self.encoder(emb)

        batch_size = text.size(0)
        decoder_input = torch.zeros(batch_size, 1, self.mel_dim).to(text.device)  # match input size
        hidden = None

        outputs = []
        for _ in range(max_audio_len):
            dec_out, hidden = self.decoder(decoder_input, hidden)
            frame = self.linear(dec_out)
            outputs.append(frame)
            decoder_input = frame  # feedback loop

        return torch.cat(outputs, dim=1)
