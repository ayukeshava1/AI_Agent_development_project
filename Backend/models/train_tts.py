import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tts_model import SimpleTTS  # Import net
import numpy as np
import os
import librosa  # For mel spectrogram target (toy)

class TTSDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.audio_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        self.num_samples = len(self.audio_files)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        file_idx = int(self.audio_files[idx].split('_')[1].split('.')[0])
        audio = torch.load(os.path.join(self.data_dir, self.audio_files[idx])).float()
        text_path = os.path.join(self.data_dir, f'text_{file_idx}.txt')
        with open(text_path, 'r') as f:
            text = f.read().strip()
        # Char encode (same as STT)
        char_to_idx = {c: i+1 for i, c in enumerate('abcdefghijklmnopqrstuvwxyz ')}
        text_idx = torch.tensor([char_to_idx.get(c.lower(), 0) for c in text], dtype=torch.long)
        # Toy mel target: STFT mag (80 bins, simple)
        mel = librosa.feature.melspectrogram(y=audio.numpy(), sr=22050, n_mels=80, fmax=8000)
        mel = torch.tensor(mel).T.float()  # (time, 80)
        audio_len = mel.size(0)
        text_len = len(text_idx)
        return text_idx, mel, text_len, audio_len

def collate_fn(batch):
    texts, mels, text_lens, audio_lens = zip(*batch)
    max_text = max(text_lens)
    max_audio = max(audio_lens)
    padded_texts = torch.zeros(len(batch), max_text, dtype=torch.long)
    padded_mels = torch.zeros(len(batch), max_audio, 80)
    for i in range(len(batch)):
        padded_texts[i, :text_lens[i]] = texts[i]
        padded_mels[i, :audio_lens[i]] = mels[i]
    return padded_texts, padded_mels, torch.tensor(text_lens), torch.tensor(audio_lens)

def train_tts(epochs=10, batch_size=4, lr=1e-3):  # Small batch for audio
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training TTS on {device}")

    dataset = TTSDataset('data/tts')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = SimpleTTS(vocab_size=29, embed_dim=64, hidden_dim=128, mel_dim=80).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for texts, mels, text_lens, audio_lens in loader:
            texts, mels = texts.to(device), mels.to(device)
            pred_mels = model(texts, text_lens, audio_lens.max().item())  # Use max len
            loss = criterion(pred_mels, mels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'models/tts.pth')
    print("TTS training done! Saved tts.pth")

if __name__ == "__main__":
    train_tts(epochs=10, batch_size=2)  # Tiny for CPU


# Quick synth test
print("Testing TTS synthesis...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # ✅ Add this line

model = SimpleTTS(vocab_size=29, embed_dim=64, hidden_dim=128, mel_dim=80).to(device)
model.load_state_dict(torch.load('models/tts.pth', map_location=device))
model.eval()

# Sample text "hello" → indices
char_to_idx = {c: i+1 for i, c in enumerate('abcdefghijklmnopqrstuvwxyz ')}
text = torch.tensor([[char_to_idx.get(c.lower(), 0) for c in "hello "]]).to(device)
text_len = 6
audio_len = 50  # Toy frames

with torch.no_grad():
    mel = model(text, [text_len], audio_len)[0]  # (audio_len, 80)

print("Mel shape:", mel.shape)  # [50, 80]

# To WAV (simple inverse—griffin lim approx)
from librosa.feature.inverse import mel_to_audio
audio = mel_to_audio(mel.cpu().numpy(), sr=22050)
import soundfile as sf
sf.write('test_synth.wav', audio, 22050)
print("✅ Saved test_synth.wav — play to hear robotic 'hello'!")
