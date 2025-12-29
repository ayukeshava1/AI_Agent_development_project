import torch
from stt_model import SimpleSTT
from llm_model import SimpleTransformer
from gan_model import Generator
from tts_model import SimpleTTS
from utils import assemble_pdf_from_video, assemble_video_from_pdf
import numpy as np

def test_all():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading models...")

    # Load (toy params—match your trains)
    stt = SimpleSTT().to(device)
    stt.load_state_dict(torch.load('models/stt.pth', map_location=device))
    llm = SimpleTransformer(vocab_size=65, n_embd=64, n_head=4, n_layer=4, block_size=128).to(device)
    llm.load_state_dict(torch.load('models/llm.pth', map_location=device))
    gan = Generator(64).to(device)
    gan.load_state_dict(torch.load('models/gan.pth', map_location=device))
    tts = SimpleTTS(vocab_size=29, embed_dim=64, hidden_dim=128, mel_dim=80).to(device)
    tts.load_state_dict(torch.load('models/tts.pth', map_location=device))

    print("Models loaded OK!")

    # Test chain: Video → PDF
    pdf = assemble_pdf_from_video('dummy.wav', stt, llm)  # Real: Pass video path
    print(f"Full chain test: PDF generated at {pdf}")

    # PDF → Video
    video = assemble_video_from_pdf("Dummy PDF text", llm, gan, tts)
    print(f"Full chain test: Video generated at {video}")

    print("ALL MODELS PASS! Phase 0 COMPLETE 🚀")

if __name__ == "__main__":
    test_all()