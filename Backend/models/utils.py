import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wav_write
import imageio
import librosa

# ==========================
# PDF generation from video
# ==========================
def assemble_pdf_from_video(video_path, stt_model, llm_model):
    """
    Converts a video/audio into a text PDF.
    Uses dummy transcript and summary for demo.
    """
    # Dummy audio
    sr, audio = 22050, np.random.randn(22050 * 5)  # 5 sec
    audio_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(-1).float()

    # STT transcription (dummy)
    transcript = stt_model(audio_tensor) if callable(stt_model) else "Dummy transcript."

    # LLM summary (dummy)
    summary = llm_model(transcript) if callable(llm_model) else "Bullet 1: Hello.\nBullet 2: World."

    # Create PDF
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.text(0.1, 0.9, summary, fontsize=12, va='top', wrap=True)
    ax.axis('off')
    plt.savefig('output.pdf', format='pdf')
    plt.close(fig)
    return 'output.pdf'


# ==========================
# Video generation from PDF
# ==========================
def assemble_video_from_pdf(pdf_text, llm_model, gan_model, tts_model):
    """
    Converts PDF text to video using GAN images + TTS audio.
    """
    prompts = ["draw chart", "draw graph"]
    images = []

    for p in prompts:
        label = int(p.split()[-1]) if p.split()[-1].isdigit() else 5
        z = torch.randn(1, 64)

        # GAN image
        img = gan_model(z, torch.tensor([label]).long()).detach()

        # Handle shapes: [1,1,H,W] or [1,H,W] or [H,W]
        if img.ndim == 4:  # [B,C,H,W]
            img = img.squeeze(0).squeeze(0)
        elif img.ndim == 3 and img.shape[0] == 1:  # [1,H,W]
            img = img.squeeze(0)
        elif img.ndim != 2:
            raise ValueError(f"Unexpected img shape from GAN: {img.shape}")

        # Convert to uint8
        img = (img * 255).clip(0, 255).numpy().astype(np.uint8)

        # Convert grayscale -> RGB
        img_rgb = np.stack([img]*3, axis=-1)
        images.append(img_rgb)

    # Dummy TTS audio
    text_idx = torch.tensor([[1, 8, 12, 12, 15]])
    mel = tts_model(text_idx, [5], 50)[0].detach().numpy()
    audio = librosa.feature.inverse.mel_to_audio(mel, sr=22050)
    wav_write('temp_audio.wav', 22050, audio)

    # Write video
    video_writer = imageio.get_writer('output.mp4', fps=5)
    for img_rgb in images * 10:
        img_rgb = np.asarray(img_rgb, dtype=np.uint8)
        video_writer.append_data(img_rgb)
    video_writer.close()

    return 'output.mp4'
