import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class SimpleSTT(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=29):  # Vocab: a-z + space + blank (CTC)
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Bi-dir

    def forward(self, x):
        # x: (batch, seq_len, 1) – raw audio
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out  # Logits for CTC

# Quick test: model = SimpleSTT(); print(model)