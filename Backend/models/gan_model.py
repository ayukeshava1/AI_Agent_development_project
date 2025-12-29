import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=64, text_dim=10, img_channels=1, img_size=28, hidden_dim=128):
        super().__init__()
        self.text_emb = nn.Embedding(10, text_dim)  # Toy: Digit labels as "text" (0-9)
        self.gen = nn.Sequential(
            nn.Linear(z_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, img_channels * img_size * img_size),
            nn.Tanh()  # [-1,1] for images
        )

    def forward(self, z, text_labels):
        text_emb = self.text_emb(text_labels)  # (B, text_dim)
        x = torch.cat([z, text_emb], dim=1)  # Fixed: Full emb (B, z_dim + text_dim)
        img = self.gen(x).view(-1, 1, 28, 28)  # (B, C, H, W)
        return img

class Discriminator(nn.Module):
    def __init__(self, text_dim=10, img_channels=1, img_size=28, hidden_dim=128):
        super().__init__()
        self.text_emb = nn.Embedding(10, text_dim)
        self.disc = nn.Sequential(
            nn.Linear(text_dim + img_channels * img_size * img_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Real/fake prob
        )

    def forward(self, img, text_labels):
        text_emb = self.text_emb(text_labels)  # (B, text_dim)
        img_flat = img.view(img.size(0), -1)
        x = torch.cat([text_emb, img_flat], dim=1)  # Fixed: Full emb (B, text_dim + flat)
        return self.disc(x)