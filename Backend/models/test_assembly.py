from utils import assemble_pdf_from_video, assemble_video_from_pdf
import torch
import numpy as np
import librosa

# 🧩 Dummy model stubs (safe placeholders)
class DummyGAN:
    def __call__(self, z, labels):
        # Return a random grayscale image (64x64)
        return torch.rand(1, 64, 64)

class DummyTTS:
    def __call__(self, text_idx, text_lens, audio_len):
        # Return random mel spectrogram (audio_len x 80)
        mel = torch.rand(audio_len, 80)
        return mel.unsqueeze(0)  # Batch dim

print("Testing PDF assembly...")
pdf = assemble_pdf_from_video('dummy.wav', None, None)
print(f"PDF saved: {pdf}")

print("Testing Video assembly...")
gan = DummyGAN()
tts = DummyTTS()
video = assemble_video_from_pdf("Toy PDF text", None, gan, tts)
print(f"Video saved: {video}")
