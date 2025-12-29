import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gan_model import Generator, Discriminator  # Import nets
import numpy as np
import os

class GANDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_files = sorted([f for f in os.listdir(data_dir) if f.startswith('img_') and f.endswith('.pt')])
        self.labels = []
        for f in self.img_files:
            i = int(f.split('_')[1].split('.')[0])  # Filename index
            prompt_path = os.path.join(data_dir, f'prompt_{i}.txt')
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r') as pf:
                    prompt = pf.read().strip()
                # Parse "draw digit X" → X
                label_str = prompt.split()[-1]  # Last word = digit
                label = int(label_str)
                self.labels.append(label)
            else:
                raise ValueError(f"No prompt for {f}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = torch.load(os.path.join(self.data_dir, self.img_files[idx]))
        label = self.labels[idx]
        return img, label

def train_gan(epochs=20, batch_size=8, lr=2e-4, z_dim=64):  # Smaller batch for CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training GAN on {device}")

    data_dir = 'data/gan'
    dataset = GANDataset(data_dir)
    if len(dataset) == 0:
        print("No data! Run prep_data.py first.")
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    gen = Generator(z_dim).to(device)
    disc = Discriminator().to(device)
    g_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_z = torch.randn(16, z_dim).to(device)  # For viz samples
    fixed_labels = torch.tensor([i // 4 for i in range(16)]).to(device)  # 0-3 repeated

    for epoch in range(epochs):
        d_total, g_total = 0, 0
        num_batches = 0
        for real_imgs, labels in loader:
            batch_size_cur = real_imgs.size(0)
            real_imgs = real_imgs.to(device).float()
            labels = labels.to(device).long()  # Ensure long for Embedding

            # Train Disc
            d_opt.zero_grad()
            real_pred = disc(real_imgs, labels)
            d_real_loss = criterion(real_pred, torch.ones_like(real_pred))
            z = torch.randn(batch_size_cur, z_dim).to(device)
            fake_imgs = gen(z, labels)
            fake_pred = disc(fake_imgs.detach(), labels)
            d_fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_opt.step()
            d_total += d_loss.item()

            # Train Gen
            g_opt.zero_grad()
            fake_pred = disc(fake_imgs, labels)
            g_loss = criterion(fake_pred, torch.ones_like(fake_pred))
            g_loss.backward()
            g_opt.step()
            g_total += g_loss.item()

            num_batches += 1

        avg_d = d_total / num_batches
        avg_g = g_total / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Avg D Loss: {avg_d:.4f}, Avg G Loss: {avg_g:.4f}")

        if epoch % 5 == 0:
            # Save intermediate
            torch.save(gen.state_dict(), f'models/gan_epoch_{epoch}.pth')
            print(f"Checkpoint saved: gan_epoch_{epoch}.pth")

    torch.save(gen.state_dict(), 'models/gan.pth')
    print("GAN training done! Saved gan.pth")

if __name__ == "__main__":
    train_gan(epochs=20, batch_size=8)  # CPU-friendly


# Quick gen test (run after training)
print("Testing GAN generation...")
gen = Generator(64)
gen.load_state_dict(torch.load('models/gan.pth'))
gen.eval()
z = torch.randn(4, 64)
labels = torch.tensor([0, 1, 2, 3]).long()
fakes = gen(z, labels)
print("Generated shapes:", fakes.shape)  # [4,1,28,28]

# Viz (needs matplotlib)
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 4, figsize=(8, 2))
for i in range(4):
    axs[i].imshow(fakes[i, 0].detach().cpu().numpy(), cmap='gray')
    axs[i].set_title(f'Digit {labels[i].item()}')
    axs[i].axis('off')
plt.tight_layout()
plt.show()