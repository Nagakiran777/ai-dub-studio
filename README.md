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
| OS | Windows 10/11 + WSL2 | Windows 11 + WSL2 |
| CPU | 6 cores | 8+ cores |
| RAM | 16GB | 32GB |
| GPU | NVIDIA 4GB VRAM | NVIDIA 8GB+ VRAM |
| Storage | 50GB free | 100GB+ free |
| CUDA | 12.x | 12.4 |

---

## 🚀 Quick Start (WSL2 + Conda)

### Step 1 — Install WSL2 (Windows only)

Open PowerShell as Administrator and run:

```powershell
wsl --install
```

Restart your PC. Ubuntu will be installed automatically.

### Step 2 — Install Miniconda inside WSL2

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, say yes to everything
source ~/.bashrc
```

### Step 3 — Clone the repository

```bash
git clone https://github.com/Nagakiran777/ai-dub-studio
cd ai-dub-studio
```

### Step 4 — Run setup (creates all 8 conda environments)

```bash
bash scripts/setup.sh
```

> ⚠️ This takes 30–60 minutes the first time. It installs PyTorch, Whisper, XTTS-v2 and all dependencies.

### Step 5 — Download models

```bash
bash scripts/download_models.sh
```

Downloads ~5GB of AI models. Only needed once.

### Step 6 — Launch

```bash
bash scripts/launch.sh
```

Or on Windows — double-click `DubStudio.bat`

---

## 📁 Project Structure

```
ai-dub-studio/
├── config.yaml                  ← pipeline configuration
├── pipeline.py                  ← main pipeline orchestrator
├── DubStudio.bat                ← one-click Windows launcher
├── stages/
│   ├── 00_vocals/               ← Demucs vocal separation
│   │   ├── run.py
│   │   ├── environment.yml
│   │   └── environment_frozen.yml
│   ├── 01_asr/                  ← Whisper speech recognition
│   ├── 01b_diarization/         ← TitaNet speaker diarization
│   ├── 02_emotion/              ← wav2vec2 emotion detection
│   ├── 03_translation/          ← Google Translate
│   ├── 04_tts/                  ← XTTS-v2 voice cloning
│   ├── 05_assembly/             ← ffmpeg audio assembly
│   └── 06_frontend/             ← PyQt6 desktop UI
│       ├── main.py
│       ├── launch.sh
│       └── app/
│           ├── job_manager.py
│           ├── pipeline_api.py
│           ├── models/
│           │   ├── job_model.py
│           │   └── speaker_model.py
│           └── ui/
│               ├── main_window.py
│               ├── timeline_widget.py
│               ├── dialogue_panel.py
│               ├── speaker_manager.py
│               ├── pipeline_panel.py
│               ├── video_player.py
│               ├── widgets.py
│               └── design.py
├── data/
│   ├── input/                   ← place input videos here
│   ├── audio/                   ← generated audio (auto)
│   └── outputs/                 ← dubbed videos (auto)
├── jobs/                        ← per-job data (auto)
├── logs/                        ← pipeline logs (auto)
└── scripts/
    ├── setup.sh                 ← environment setup
    ├── download_models.sh       ← model downloader
    └── launch.sh                ← WSL2 launcher
```

---

## 🎮 Using the Desktop UI

1. Launch DubStudio Pro via `DubStudio.bat` or `bash scripts/launch.sh`
2. Click **Upload Video** — select any video file
3. Run stages in order using the Pipeline panel
4. After Stage 01b — review speaker assignments in the timeline
5. Click **Confirm Speakers** — builds voice profiles from confirmed assignments
6. Run Stage 02 → 03 → edit translations if needed → Stage 04 TTS → Stage 05 Assembly
7. Watch the dubbed video in the built-in player

---

## 🔧 Configuration

Edit `config.yaml` to change:
- Source / target languages
- ASR model size (tiny → large-v3)
- TTS sample rate
- Speaker detection sensitivity
- Hardware settings (GPU/CPU)
- Cache paths for models

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

**Total model download: ~5GB** (downloaded once by `scripts/download_models.sh`)

---

## ⚙️ Conda Environments

Each stage runs in its own isolated conda environment:

| Environment | Stage | Python |
|---|---|---|
| dub_vocals | Stage 00 — Vocals Extraction | 3.10 |
| dub_asr | Stage 01 — ASR Transcription | 3.10 |
| dub_diar | Stage 01b — Speaker Diarization | 3.10 |
| dub_emotion | Stage 02 — Emotion Detection | 3.10 |
| dub_translate | Stage 03 — Translation | 3.10 |
| dub_tts | Stage 04 — TTS Voice Cloning | 3.11 |
| dub_assembly | Stage 05 — Audio Assembly | 3.10 |
| dub_frontend | Stage 06 — Desktop UI | 3.10 |

---

## 🐛 Troubleshooting

**UI doesn't open on Windows:**
- Make sure you launched via `DubStudio.bat` not directly from WSL2
- Check that WSL2 GUI support is enabled (Windows 11 has it built-in)

**CUDA not detected:**
- Run `nvidia-smi` in WSL2 to verify GPU is visible
- Make sure CUDA 12.x drivers are installed on Windows (not inside WSL2)

**Stage fails with import error:**
- Make sure `bash scripts/setup.sh` completed fully without errors
- Check the `logs/` folder for detailed error messages

**Models not found:**
- Run `bash scripts/download_models.sh` again
- Check `config.yaml` for correct cache paths

**Setup takes too long or fails:**
- Make sure you have at least 50GB free storage
- Check your internet connection
- Try running setup stage by stage manually

---

## 🛣️ Roadmap

- [ ] Whisper large-v3 upgrade
- [ ] Pyannote 3.0 diarization
- [ ] F5-TTS / Zonos TTS upgrade
- [ ] Telugu, Hindi, Tamil target languages
- [ ] Batch processing multiple videos
- [ ] Fine-tuned speaker voice models
- [ ] Docker image for one-click setup

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
- [Audeering wav2vec2](https://github.com/audeering/w2v2-how-to) — emotion detection