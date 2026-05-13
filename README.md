# 🎬 AI Dub Studio

> Production-grade AI dubbing system with emotion-preserving voice cloning,
> per-speaker voice profiles, and a professional desktop interface.

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)
![PyQt6](https://img.shields.io/badge/UI-PyQt6-green)
![CUDA](https://img.shields.io/badge/CUDA-12.4-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- 🎙 **Emotion-preserving dubbing** — full emotional fidelity, not just translation
- 🔒 **Offline and private** — no cloud dependency except Google Translate
- 🌍 **Universal source language** — dub from ANY language into English
- 👥 **Per-speaker voice cloning** — each character sounds like themselves
- 🎵 **Non-lexical sound preservation** — hmm, uh, ah preserved naturally
- 📺 **Professional desktop UI** — DaVinci Resolve-style interface
- 🎬 **Scalable** — short-form reels to full movies and series

---

## 🏗️ Architecture

```
Input Video (ANY language)
      ↓
Stage 00: Vocals Extraction    → separates speech from BGM (Demucs)
      ↓
Stage 01: ASR                  → transcription with word timestamps (Whisper)
      ↓
Stage 01b: Speaker Diarization → per-speaker identification (TitaNet)
      ↓
Stage 02: Emotion Detection    → emotion + intensity per dialogue (wav2vec2)
      ↓
Stage 03: Translation          → target language translation (Google Translate)
      ↓
Stage 04: TTS Synthesis        → voice cloning per speaker (XTTS-v2)
      ↓
Stage 05: Assembly             → final video with dubbed audio
      ↓
Stage 06: Frontend             → desktop UI for review and control
```

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 11 + WSL2 | Windows 11 + WSL2 |
| CPU | 6 cores | 8+ cores |
| RAM | 16GB | 32GB |
| GPU | NVIDIA 4GB VRAM | NVIDIA 8GB+ VRAM |
| Storage | 50GB free | 100GB+ free |
| CUDA | 12.x | 12.4 |

---

## 🚀 Quick Start

### Option A — Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/Nagakiran777/ai-dub-studio
cd ai-dub-studio

# 2. Copy environment config
cp .env.example .env
# Edit .env to set your cache paths

# 3. Download models (first time only — ~11GB)
bash scripts/download_models.sh

# 4. Build and run
docker-compose up --build

# 5. Launch the desktop UI
bash scripts/launch_ui.sh
```

### Option B — Manual Setup (WSL2 + Conda)

```bash
# 1. Clone the repository
git clone https://github.com/Nagakiran777/ai-dub-studio
cd ai-dub-studio

# 2. Run the setup script (creates all 8 conda environments)
bash scripts/setup.sh

# 3. Download models
bash scripts/download_models.sh

# 4. Launch
bash scripts/launch.sh
# OR double-click DubStudio.bat on Windows
```

---

## 📁 Project Structure

```
ai-dub-studio/
├── config.yaml              ← pipeline configuration
├── pipeline.py              ← main pipeline orchestrator
├── DubStudio.bat            ← one-click Windows launcher
├── stages/
│   ├── 00_vocals/           ← Demucs vocal separation
│   ├── 01_asr/              ← Whisper speech recognition
│   ├── 01b_diarization/     ← TitaNet speaker diarization
│   ├── 02_emotion/          ← wav2vec2 emotion detection
│   ├── 03_translation/      ← Google Translate
│   ├── 04_tts/              ← XTTS-v2 voice cloning
│   ├── 05_assembly/         ← ffmpeg audio assembly
│   └── 06_frontend/         ← PyQt6 desktop UI
├── data/
│   ├── input/               ← place input videos here
│   ├── audio/               ← generated audio (auto)
│   └── outputs/             ← dubbed videos (auto)
├── jobs/                    ← per-job data (auto)
├── logs/                    ← pipeline logs (auto)
├── scripts/
│   ├── setup.sh             ← environment setup
│   ├── download_models.sh   ← model downloader
│   └── launch.sh            ← WSL2 launcher
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## 🐳 Docker Details

The Docker image contains:
- Ubuntu 22.04 + CUDA 12.4
- Miniconda with all 8 pipeline environments
- All project code

The following are mounted from your host machine (not baked into image):
- Model caches (~11GB) — downloaded once, reused forever
- `data/` folder — your input/output videos
- `jobs/` folder — job data

---

## 🎮 Using the Desktop UI

1. Launch DubStudio Pro (via `DubStudio.bat` or `scripts/launch.sh`)
2. Click **Upload Video** — select any video file
3. Run stages in order using the Pipeline panel
4. After Stage 01b — review speaker assignments in the timeline
5. Click **Confirm Speakers** — builds voice profiles from confirmed assignments
6. Run Stage 02 → 03 → edit translations → run TTS → assemble
7. Watch the dubbed video in the built-in player

---

## 🔧 Configuration

Edit `config.yaml` to change:
- Source / target languages
- ASR model size (tiny → large)
- TTS sample rate
- Speaker detection sensitivity
- Hardware settings (GPU/CPU)

---

## 📦 Models Used

| Model | Stage | Size | Source |
|---|---|---|---|
| Demucs mdx_extra | Vocals | ~300MB | Facebook Research |
| Whisper medium | ASR | ~1.5GB | OpenAI |
| TitaNet Large | Diarization | ~100MB | NVIDIA NeMo |
| wav2vec2-large | Emotion | ~1.2GB | Audeering |
| distilroberta | Emotion (text) | ~80MB | Hugging Face |
| XTTS-v2 | TTS | ~1.8GB | Coqui AI |

---

## 🛣️ Roadmap

- [ ] Whisper large-v3 upgrade
- [ ] Pyannote 3.0 diarization
- [ ] F5-TTS / Zonos TTS upgrade
- [ ] Telugu, Hindi, Tamil target languages
- [ ] Batch processing multiple videos
- [ ] Fine-tuned speaker voice models

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Credits

Built with:
- [Demucs](https://github.com/facebookresearch/demucs) — source separation
- [OpenAI Whisper](https://github.com/openai/whisper) — speech recognition
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — speaker diarization
- [Coqui TTS / XTTS-v2](https://github.com/coqui-ai/TTS) — voice cloning
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) — desktop UI
