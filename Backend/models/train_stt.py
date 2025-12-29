import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Fixed: For log_softmax
from torch.utils.data import Dataset, DataLoader
from stt_model import SimpleSTT  # Import net
from torch.nn import CTCLoss
import numpy as np

class STTDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.audio_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        self.num_samples = len(self.audio_files)
        if self.num_samples == 0:
            raise ValueError(f"No .pt files in {data_dir}! Run prep_data.py first.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        # Parse index from filename (e.g., 'audio_5.pt' → 5)
        file_idx = int(audio_file.split('_')[1].split('.')[0])
        audio = torch.load(os.path.join(self.data_dir, audio_file)).float()  # To float32
        text_path = os.path.join(self.data_dir, f'text_{file_idx}.txt')
        with open(text_path, 'r') as f:
            text = f.read().strip()
        # Simple char encode: a=1, b=2..., space=27, blank=0 (CTC end), pad=28
        char_to_idx = {c: i+1 for i, c in enumerate('abcdefghijklmnopqrstuvwxyz ')}
        text_idx = [char_to_idx.get(c.lower(), 28) for c in text] + [0]  # Blank for CTC
        return torch.unsqueeze(audio, -1), torch.tensor(text_idx, dtype=torch.long)  # (seq, 1), targets

def collate_fn(batch):
    # Pad audio to max len; keep targets as list (CTC handles var len)
    audios, targets = zip(*batch)
    max_len = max(a.size(0) for a in audios)
    padded_audios = torch.zeros(len(batch), max_len, 1)
    for i, audio in enumerate(audios):
        padded_audios[i, :audio.size(0)] = audio
    return padded_audios, list(targets)  # List of tensors, not stacked!

def train_stt(epochs=5, batch_size=4, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device} with {STTDataset('data/stt').num_samples} samples")

    dataset = STTDataset('data/stt')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = SimpleSTT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CTCLoss(blank=0, zero_infinity=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for audio, targets in loader:  # targets now list
            audio = audio.to(device)
            log_probs = model(audio)  # (batch, seq, classes)
            log_probs = F.log_softmax(log_probs, dim=2)  # Use F
            input_lengths = torch.full((audio.size(0),), audio.size(1), dtype=torch.long)
            target_lengths = torch.tensor([len(t) - 1 for t in targets], dtype=torch.long)  # -1 for blank
            loss = criterion(log_probs, torch.stack(targets), input_lengths, target_lengths)  # Stack for criterion input
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Stability
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'models/stt.pth')
    print("STT training done! Saved stt.pth")

if __name__ == "__main__":
    train_stt(epochs=5)  # Short for test; increase later