import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from llm_model import SimpleTransformer  # Import your transformer model
import os

# Dataset class
class TextDataset(Dataset):
    def __init__(self, file_path, block_size=128):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# Training function
def train_llm(file_path='data/llm/input.txt', epochs=10, batch_size=32, lr=1e-3, block_size=128):
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training LLM on {device}")

    # Prepare dataset and dataloader
    dataset = TextDataset(file_path, block_size=block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = SimpleTransformer(
        vocab_size=dataset.vocab_size,
        n_embd=64,
        n_head=4,
        n_layer=4,
        block_size=block_size
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    # Ensure models folder exists
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/llm.pth')
    print("LLM training done! Saved models/llm.pth")

    return model, dataset  # Return model and dataset if needed

# Main
if __name__ == "__main__":
    model, dataset = train_llm(epochs=1, batch_size=4, block_size=32)
    # model and dataset are now available for further use
