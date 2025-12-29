import os
import torch
import numpy as np
import librosa  # For resample (kept for GAN if needed)
import requests  # New: For URL downloads
from torchvision import datasets, transforms  # GAN

# Helper: Save tensor/audio/text
def save_sample(audio, text, idx, folder):
    audio_path = os.path.join(folder, f'audio_{idx}.pt')
    text_path = os.path.join(folder, f'text_{idx}.txt')
    torch.save(torch.tensor(audio), audio_path)
    with open(text_path, 'w') as f:
        f.write(text)
    print(f"Saved {idx}: {audio.shape} audio, '{text[:50]}...' text")

# 0.1.1: STT Data (Synthetic: Sine waves + random sentences – toy for now)
def prep_stt(num_samples=50):
    print("Prepping synthetic STT data...")
    folder = 'data/stt'
    os.makedirs(folder, exist_ok=True)
    sentences = ["Hello world.", "This is a test.", "Convert video to PDF.", "AI agent rocks!", "Training from scratch."] * 10  # Repeat for variety
    for i in range(num_samples):
        # Synthetic audio: Sine wave (1s @ 16kHz, freq random 200-800Hz)
        sr = 16000
        t = np.linspace(0, 1, sr)
        freq = 200 + 600 * np.random.rand()
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
        # Add noise for realism
        audio += 0.1 * np.random.randn(sr)
        text = sentences[i % len(sentences)] + f" [sample {i}]"
        save_sample(audio, text, i, folder)
    print(f"STT prep done: {num_samples} synthetic samples in {folder}")

# 0.1.2: LLM Data (Direct URL: Tiny Shakespeare)
def prep_llm():
    print("Prepping LLM data...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    folder = 'data/llm'
    os.makedirs(folder, exist_ok=True)
    response = requests.get(url)
    if response.status_code == 200:
        text = response.text
        with open(os.path.join(folder, 'input.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"LLM prep done: Full text saved in {folder}/input.txt (len: {len(text):,})")
    else:
        print(f"LLM download error: {response.status_code}. Skipping...")

# 0.1.3: GAN Data (MNIST: Keep as-is, worked!)
def prep_gan(num_samples=100):
    print("Prepping GAN data...")
    try:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST(root='data/gan', train=True, download=True, transform=transform)
        folder = 'data/gan'
        os.makedirs(folder, exist_ok=True)
        for i in range(min(num_samples, len(dataset))):
            img, label = dataset[i]
            prompt = f"draw digit {label}"
            img_path = os.path.join(folder, f'img_{i}.pt')
            prompt_path = os.path.join(folder, f'prompt_{i}.txt')
            torch.save(img, img_path)
            with open(prompt_path, 'w') as pf:
                pf.write(prompt)
            if i % 50 == 0: print(f"GAN sample {i}: {img.shape}, prompt: {prompt}")
        print(f"GAN prep done: {min(num_samples, len(dataset))} samples in {folder}")
    except Exception as e:
        print(f"GAN prep error: {e}. Skipping...")

# 0.1.4: TTS Data (Synthetic: Same as STT, for seq2seq toy)
def prep_tts(num_samples=50):
    print("Prepping synthetic TTS data...")
    folder = 'data/tts'
    os.makedirs(folder, exist_ok=True)
    sentences = ["Read this text.", "Voice synthesis works.", "PDF to video narration.", "Custom model success!", "Let's roar!"] * 10
    for i in range(num_samples):
        # Synthetic: Chirp wave (0.5s @ 22kHz, freq ramp)
        sr = 22050
        t = np.linspace(0, 0.5, sr // 2)
        freq = 300 + 400 * t  # Ramp
        audio = np.sin(2 * np.pi * np.cumsum(freq) / sr).astype(np.float32)
        audio += 0.05 * np.random.randn(len(audio))
        text = sentences[i % len(sentences)] + f" [synth {i}]"
        save_sample(audio, text, i, folder)
    print(f"TTS prep done: {num_samples} synthetic samples in {folder}")

if __name__ == "__main__":
    prep_stt(20)
    prep_llm()
    prep_gan(100)  # Reuse your existing
    prep_tts(20)
    print("Phase 0.1 COMPLETE: Data ready for training!")