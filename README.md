# 🔥 PyroSense AI

**Real-time Fire & Smoke Detection with Explainable Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![Last Commit](https://img.shields.io/github/last-commit/your-org/pyrosense-ai)](https://github.com/your-org/pyrosense-ai)

![Demo](assets/demo.gif)

PyroSense AI is a **production-grade, hackathon-ready** system that detects **fire and smoke** from **webcams**, **RTSP IP cameras**, **YouTube streams**, **uploaded videos**, and **static images** using a fine-tuned **YOLOv8** detector, a secondary **EfficientNetV2** verifier, **Grad-CAM explainability**, **LLM incident reporting**, and a polished **Streamlit dashboard** with a **FastAPI** backend.

## ✨ Features

- **⚡ Real-time detection**: Webcam, RTSP, YouTube URL, image/video upload
- **🧠 Ensemble inference**: YOLOv8 + EfficientNetV2 weighted voting to reduce false positives
- **🔎 Explainable AI (XAI)**: Grad-CAM/EigenCAM heatmaps (Original | Heatmap | Blend)
- **🧾 AI incident reporter**: LLaMA3 summaries via **Groq** (cloud) or **Ollama** (local) with rule-based fallback
- **📣 Multi-modal alerts**: Email, Telegram, webhook, and audio (gTTS)
- **🧠 Similar incident search**: CLIP embeddings + FAISS top-3 similar historical detections
- **📊 Risk score engine**: Composite risk score (0–100) with severity bands (LOW→CRITICAL)
- **🗃️ Incident logging**: SQLite (dev) + PostgreSQL-ready connection string; snapshots + heatmaps + delivery logs
- **🧪 Training & tracking**: Albumentations (incl. smoke simulation) + MLflow metrics and runs
- **📦 Edge deployment**: ONNX export + Raspberry Pi script with HTTP alert forwarding

## 🧩 Architecture (high-level)

```
   Sources
 (Webcam/RTSP/YouTube/File)
            |
            v
     Frame Ingestion
   (OpenCV / yt-dlp)
            |
            v
   YOLOv8 Detector (Ultralytics)
            |
      boxes/scores/classes
            |
            +------------------------------+
            |                              |
            v                              v
 EfficientNetV2 Verifier             Grad-CAM Explainer
   (secondary confidence)          (Original|Heatmap|Blend)
            |                              |
            +--------------+---------------+
                           v
                    Ensemble + Risk Score
                           |
                           v
        +------------------+------------------+
        |                                     |
        v                                     v
   DB + FAISS history                    Alert Manager
(snapshots, heatmaps,                   (email/telegram/
 summaries, metrics)                    webhook/audio)
        |                                     |
        +------------------+------------------+
                           v
        Streamlit Dashboard <-> FastAPI REST/WebSocket API
```

## 🚀 Quick Start (Docker)

1. Copy env file:

```bash
cp .env.example .env
```

2. Start services:

```bash
docker-compose up --build
```

- **Dashboard**: `http://localhost:8501`
- **API**: `http://localhost:8000`
- **API docs**: `http://localhost:8000/docs`
- **MLflow**: `http://localhost:5000`

## 🧰 Manual Setup (Windows/macOS/Linux)

```bash
python -m venv .venv
```

Activate:

- Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Configure:

```bash
cp .env.example .env
```

Initialize the database:

```bash
python -m database.migrations.init_db
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

Run the API:

```bash
uvicorn api.main:app --reload --port 8000
```

## 🧪 Training (YOLOv8 + MLflow)

1. Download datasets:

```bash
python data/download_datasets.py
```

2. Prepare/augment to `data/processed/` (see `training/augmentation.py`) and then train:

```bash
python training/train_yolo.py
```

Resume an interrupted run:

```bash
python training/train_yolo.py --resume
```

Launch MLflow UI:

```bash
python training/launch_mlflow.py
```

## 🔌 API Documentation

- Swagger UI is available at `http://localhost:8000/docs` when running the API.
- REST: `POST /api/v1/detect` for image upload
- WebSocket: `/api/v1/ws/stream` for real-time frame inference

## 🤖 Telegram Bot Setup

1. Create a bot with BotFather and get a token.
2. Set in `.env`:
   - `TELEGRAM_ENABLED=true`
   - `TELEGRAM_BOT_TOKEN=...`
   - `TELEGRAM_CHAT_ID=...`
3. Start the API (or dashboard; both can call the alert system).

Supported commands:
- `/status` — current system status and last detection
- `/snapshot` — returns current camera frame
- `/history` — last 5 detections
- `/threshold 0.7` — update confidence threshold remotely

## 🥧 Edge Deployment (Raspberry Pi 4)

The edge script lives at `edge_deploy/raspberry_pi.py` and supports:
- ONNX Runtime inference (CPU)
- Picamera2 frame capture (if available) with fallback to OpenCV
- Sending detections to the central FastAPI server via HTTP webhook
- Optional local display overlay

Minimal steps:

```bash
pip install onnxruntime opencv-python requests
```

Then:

```bash
python edge_deploy/raspberry_pi.py --server http://<server-ip>:8000 --source picamera
```

## 🗂️ Project Structure (condensed)

```
pyrosense-ai/
  api/        # FastAPI (REST + WebSocket)
  dashboard/  # Streamlit multi-page dashboard
  models/     # YOLO, EfficientNetV2, ONNX, ensemble
  inference/  # detector engine + explainability
  alerts/     # email/telegram/audio/webhook
  llm/        # Groq/Ollama summarizer + FAISS history
  database/   # SQLAlchemy models + CRUD + init script
  training/   # training scripts + notebooks + MLflow
  data/       # raw/processed/samples + downloader
  tests/      # pytest unit/integration tests
```

## 🧾 Tech Stack

| Layer | Libraries / Tools |
|------|--------------------|
| Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [PyTorch](https://pytorch.org/), [OpenCV](https://opencv.org/) |
| Secondary classifier | EfficientNetV2 (PyTorch) |
| Explainability | [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) (EigenCAM) |
| API | [FastAPI](https://fastapi.tiangolo.com/), WebSockets, Uvicorn |
| Dashboard | [Streamlit](https://streamlit.io/) |
| Similarity search | [Transformers](https://huggingface.co/docs/transformers) (CLIP), [FAISS](https://github.com/facebookresearch/faiss) |
| LLM summaries | [Groq](https://groq.com/) or [Ollama](https://ollama.com/) |
| Storage | SQLite (dev) + PostgreSQL-ready config |
| Tracking | [MLflow](https://mlflow.org/) |
| DevOps | Docker, GitHub Actions CI (ruff + pytest + build) |

## 📄 License

MIT License. See `LICENSE` (or add one if you publish the repo).

