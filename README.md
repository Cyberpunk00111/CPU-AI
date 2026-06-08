# CogniCore SLM

CogniCore is a CPU-native Small Language Model scaffold for Phase 1 experimentation. It implements a GPT-style decoder with BitLinear ternary quantisation, RMSNorm, RoPE, grouped-query attention, SwiGLU, CPU training utilities, FastAPI streaming, and a React UI.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Phase 1 Training

```bash
bash scripts/train_phase1.sh
```

The script downloads the open datasets, trains the BPE tokenizer, writes train/validation token shards, and starts CPU training.

On Windows, use the Python runner:

```bash
python scripts/run_phase1_pipeline.py --config configs/phase1_config.yaml
```

## Configuration

All model dimensions, training hyperparameters, data paths, CPU switches, and logging paths live in `configs/phase1_config.yaml`. Phase 2 and Phase 3 config files are included as forward-looking scale targets.

## API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /api/health`
- `GET /api/status`
- `POST /api/query`
- `GET /api/stream`

To serve a trained checkpoint:

```bash
export COGNICORE_CONFIG=configs/phase1_config.yaml
export COGNICORE_CHECKPOINT=logs/checkpoints/cognicore_step_50000.pt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Repository Map

- `cognicore/model`: transformer architecture, BitLinear, training, inference, tokenizer.
- `data`: dataset download, preprocessing, memory-mapped token shards.
- `api`: FastAPI backend, status, chat, streaming, optional DuckDuckGo RAG.
- `frontend`: Vite React client with streaming chat UI.
- `tests`: focused validation for model math, config, training utilities, and API.
