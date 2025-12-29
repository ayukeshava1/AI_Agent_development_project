import torch
import os
from models.stt_model import SimpleSTT
from models.llm_model import SimpleTransformer
from models.gan_model import Generator
from models.tts_model import SimpleTTS
from utils import assemble_pdf_from_video, assemble_video_from_pdf  # From Phase 0.6

def agent_pipeline(mode, input_path, models_dir='models'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Pipeline on {device}")

    # Load models (map_location for CPU/GPU)
    stt = SimpleSTT().to(device)
    stt.load_state_dict(torch.load(os.path.join(models_dir, 'stt.pth'), map_location=device))
    llm = SimpleTransformer(vocab_size=65, n_embd=64, n_head=4, n_layer=4, block_size=128).to(device)
    llm.load_state_dict(torch.load(os.path.join(models_dir, 'llm.pth'), map_location=device))
    gan = Generator(64).to(device)
    gan.load_state_dict(torch.load(os.path.join(models_dir, 'gan.pth'), map_location=device))
    tts = SimpleTTS(vocab_size=29, embed_dim=64, hidden_dim=128, mel_dim=80).to(device)
    tts.load_state_dict(torch.load(os.path.join(models_dir, 'tts.pth'), map_location=device))

    stt.eval(); llm.eval(); gan.eval(); tts.eval()  # Infer mode

    # Chain based on mode (stub real parse for MVP)
    if mode == "video-to-pdf":
        # STT: Audio from video (stub: load dummy)
        audio = torch.randn(1, 16000, 1).to(device)  # Dummy
        transcript = "Toy transcript: Hello from video."  # stt(audio) stub
        # LLM: Summarize
        summary = "Toy bullets: - Hello\n- World"  # llm.generate(prompt)
        # Assemble
        pdf_path = assemble_pdf_from_video(input_path, stt, llm)
        return pdf_path
    elif mode == "pdf-to-video":
        # Parse PDF text (stub)
        text = "Toy PDF text."
        # LLM: Prompts
        prompts = ["draw chart"]  # llm(text)
        # GAN: Images
        labels = torch.tensor([5]).to(device)  # From prompts
        img = gan(torch.randn(1, 64).to(device), labels).squeeze().numpy()
        # TTS: Audio
        text_idx = torch.tensor([[1,8,12,12,15]]).to(device)  # "hello"
        mel = tts(text_idx, [5], 50).detach().numpy()
        # Assemble
        video_path = assemble_video_from_pdf(text, llm, gan, tts)
        return video_path

# Test stub
if __name__ == "__main__":
    print("Testing pipeline...")
    path = agent_pipeline("video-to-pdf", "dummy.wav")
    print(f"Output: {path}")