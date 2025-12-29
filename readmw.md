ai-converter-app/                          # Root (your VS Code workspace)
├── README.md                               # Project overview, setup instructions
├── .env                                    # API keys (e.g., Supabase; gitignore this)
├── .gitignore                              # Ignore venv, node_modules, .env
├── backend/                                # Python/FastAPI + ML (Phase 0-7)
│   ├── main.py                             # FastAPI app (endpoints like /convert, /login)
│   ├── requirements.txt                    # Pip deps (fastapi, torch, etc.)
│   ├── venv/                               # Virtual env (ignored)
│   ├── models/                             # Phase 0: Custom PyTorch models & training scripts
│   │   ├── __init__.py
│   │   ├── train_stt.py                    # 0.2: STT training script
│   │   ├── stt_model.py                    # STT net definition (LSTM/Transformer)
│   │   ├── stt.pth                         # Saved trained model (post-0.2)
│   │   ├── train_llm.py                    # 0.3: LLM training
│   │   ├── llm_model.py                    # Tiny Transformer
│   │   ├── llm.pth                         # Saved model
│   │   ├── train_gan.py                    # 0.4: GAN training
│   │   ├── gan_model.py                    # Generator/Discriminator
│   │   ├── gan.pth                         # Saved model
│   │   ├── train_tts.py                    # 0.5: TTS training
│   │   ├── tts_model.py                    # Seq2Seq net
│   │   ├── tts.pth                         # Saved model
│   │   ├── test_models.py                  # 0.7: Unit tests for all models
│   │   └── utils.py                        # Shared helpers (e.g., data loaders)
│   ├── data/                               # Phase 0.1: Datasets (subfolders per model)
│   │   ├── stt/                            # LibriSpeech subsets (audio.pt, text.txt)
│   │   ├── llm/                            # Shakespeare/input.txt
│   │   ├── gan/                            # MNIST/COCO (images/, prompts.txt)
│   │   └── tts/                            # LJSpeech (audio.wav, metadata.csv)
│   ├── agent/                              # Phase 7: Pipeline orchestration
│   │   ├── __init__.py
│   │   └── pipeline.py                     # agent_pipeline(mode, input_file) chains models
│   ├── outputs/                            # Temp outputs (PDFs/videos; cleanup script later)
│   ├── temp/                               # Temp uploads (auto-clean)
│   └── tests/                              # Backend unit tests (pytest)
├── frontend/                               # React/Vite app (Phases 1-6)
│   ├── package.json                        # NPM deps (react, tailwind, etc.)
│   ├── vite.config.js                      # Vite build config
│   ├── tailwind.config.js                  # Tailwind setup
│   ├── postcss.config.js                   # PostCSS for Tailwind
│   ├── index.html                          # Entry HTML
│   ├── src/
│   │   ├── main.jsx                        # App entry (ReactDOM.render)
│   │   ├── App.jsx                         # Root (Router wrapper)
│   │   ├── index.css                       # Global styles (gradients)
│   │   ├── components/                     # Reusable UI (Phases 1-3)
│   │   │   ├── Header.jsx                   # Top bar (logo, sign-in, subscribe)
│   │   │   ├── Sidebar.jsx                  # Left nav (Home, Converter, Files, etc.)
│   │   │   ├── ChatContainer.jsx            # Phase 3: Message bubbles
│   │   │   ├── InputBox.jsx                 # Phase 3: Chat input + file attach
│   │   │   ├── SignInModal.jsx              # Phase 2: Login form
│   │   │   ├── MessageBubble.jsx            # Phase 3: User/AI bubbles (progress, preview)
│   │   │   ├── PreviewEmbed.jsx             # Phase 4: PDF/video embeds
│   │   │   └── Toast.jsx                    # Phase 6: Notifications
│   │   ├── contexts/                       # State management (Phase 2)
│   │   │   └── AuthContext.jsx              # isLoggedIn, token
│   │   ├── pages/                          # Routes (Phase 1)
│   │   │   ├── Home.jsx                    # Landing/teaser
│   │   │   ├── Converter.jsx                # Chat page
│   │   │   ├── Files.jsx                    # Tabbed history
│   │   │   ├── Gallery.jsx                  # Samples + user feed
│   │   │   └── Resources.jsx                # Guides/accordion
│   │   └── hooks/                          # Custom hooks (Phase 3+)
│       └── useChat.js                      # Message handling, commands
│   └── public/                             # Static assets (Phase 6 PWA)
│       ├── manifest.json                   # PWA config
│       └── favicon.ico
├── docs/                                   # Specs, notes (from Phase 1 setup)
│   ├── project-spec.md                     # Your vision/tweaks
│   └── wireframes/                         # Figma exports (optional)
└── deploy/                                 # Phase 8: Scripts/configs
    ├── vercel.json                         # Frontend deploy
    └── render.yaml                          # Backend deploy