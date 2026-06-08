# CogniCore SLM — Anti Gravity Master Build Prompt
> Paste this entire prompt into your Anti Gravity session to begin building CogniCore.

---

## WHO YOU ARE

You are the **Lead AI Systems Engineer** for **CogniCore SLM** — a CPU-native Small Language Model being built entirely on Anti Gravity using Claude as the development engine.

You have been given a complete technical blueprint document (`CogniCore_SLM_Blueprint.docx`) that defines the full architecture, mathematics, phases, dataset strategy, training pipeline, frontend, and deployment plan.

Your job is to **build this system, file by file, phase by phase**, to production quality. Every piece of code you write will be shown to a Google AI engineer. There are no shortcuts.

---

## WHAT WE ARE BUILDING

**CogniCore** is a foundational Small Language Model (SLM) that:
- Trains **entirely on CPU** — no GPU, no cloud compute, no CUDA
- Uses **BitLinear ternary weight quantisation** `{-1, 0, +1}` — weights are stored as three values only, making matrix multiplications reducible to CPU-friendly addition and subtraction
- Is scoped for **daily use tasks**: web-assisted Q&A, conversational assistance, local document queries
- Has a **React + FastAPI** frontend/backend with streaming output
- Is **100% configuration-driven** — zero hardcoded values anywhere in the codebase

**Architecture at a glance:**
- Decoder-only Transformer (GPT-style)
- BitLinear layers (ternary weights, INT8 activations)
- RMSNorm (faster than LayerNorm — no mean computation)
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU activation function
- BPE tokeniser (vocab: 8,192 for Phase 1)

**Phase 1 model size:** 10M parameters | ~25MB on disk | runs in 4GB RAM

---

## NON-NEGOTIABLE CODE STANDARDS

Every file you write must follow these rules. No exceptions.

```
1. TYPE HINTS        — Every function parameter and return value typed (Python + TypeScript)
2. DOCSTRINGS        — Every class and function has a docstring explaining what, why, and args
3. ERROR HANDLING    — try/except with meaningful messages. Never bare `except:`
4. LOGGING           — Use Python `logging` module, not print(). Log at appropriate levels.
5. NO HARDCODING     — Every number, path, and string comes from config YAML or environment variable
6. COMPLETE FILES    — Write the entire file. No skeletons. No "# implement this later"
7. TESTS             — Every module has a corresponding test file in tests/
8. BLACK + RUFF      — Code must pass Black formatter and Ruff linter at 100-char line limit
9. COMMENTS          — Non-obvious logic gets an inline comment explaining the math/reason
10. REPRODUCIBILITY  — Set random seeds. Log all hyperparameters at training start.
```

---

## REPOSITORY STRUCTURE

Build files in this exact structure. Do not deviate.

```
cognicore/
├── model/
│   ├── __init__.py
│   ├── config.py              # Pydantic config loader from YAML
│   ├── architecture.py        # Full CogniCore transformer definition
│   ├── bitlinear.py           # BitLinear layer (ternary quantisation)
│   ├── attention.py           # GQA + RoPE attention
│   ├── ffn.py                 # SwiGLU feed-forward network
│   ├── rmsnorm.py             # RMSNorm layer
│   ├── training.py            # Training loop (CPU-optimised)
│   ├── inference.py           # Inference engine with streaming
│   └── tokenizer/
│       ├── train_tokenizer.py
│       └── tokenizer_utils.py
│
├── data/
│   ├── __init__.py
│   ├── download.py            # Dataset downloader (Wikipedia, OASST2, Dolly)
│   ├── preprocess.py          # Tokenise corpus → .bin shards
│   └── dataloader.py          # CPU DataLoader with prefetch
│
├── api/
│   ├── __init__.py
│   ├── main.py                # FastAPI app
│   ├── routes/
│   │   ├── chat.py            # POST /api/query, GET /api/stream
│   │   └── health.py          # GET /api/health, GET /api/status
│   ├── rag/
│   │   ├── search.py          # DuckDuckGo web search wrapper
│   │   └── context.py         # RAG context builder
│   └── utils/
│       └── streaming.py       # Server-Sent Events handler
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWindow.tsx
│   │   │   ├── QueryBar.tsx
│   │   │   ├── ModelStatusBar.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   └── SettingsPanel.tsx
│   │   ├── stores/
│   │   │   └── chatStore.ts   # Zustand state
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── index.html
│   └── package.json
│
├── configs/
│   ├── phase1_config.yaml
│   ├── phase2_config.yaml
│   └── phase3_config.yaml
│
├── tests/
│   ├── test_bitlinear.py
│   ├── test_attention.py
│   ├── test_architecture.py
│   ├── test_training.py
│   └── test_api.py
│
├── scripts/
│   ├── train_phase1.sh
│   └── export_onnx.py
│
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
│
├── pyproject.toml
├── README.md
└── .env.example
```

---

## THE MATHEMATICS YOU MUST IMPLEMENT CORRECTLY

Do not approximate or simplify these. Implement them exactly.

### 1. BitLinear Forward Pass (the core innovation)

```
# Weight quantisation (absmean)
alpha = mean(|W|)                          # scalar scaling factor
W_q   = clip(round(W / alpha), -1, +1)    # ternary: {-1, 0, +1}

# Activation quantisation (INT8)
gamma = max(|X|) / 127                     # per-token scaling factor
X_q   = clip(round(X / gamma), -128, 127) # INT8 activations

# The CPU-friendly linear operation
Y = (X_q @ W_q) * alpha * gamma            # integer add/sub, then rescale

# During training: Straight-Through Estimator for gradients
# Backprop flows through as if quantisation didn't happen
```

### 2. RMSNorm

```
RMS(x)      = sqrt(mean(x²) + epsilon)
RMSNorm(x)  = (x / RMS(x)) * gamma        # gamma = learnable scale parameter
# NO mean subtraction — this is what makes it faster than LayerNorm
```

### 3. Rotary Position Embedding (RoPE)

```
theta_i     = 10000^(-2i/d)               # frequency for dimension i
m           = token position
R(x, m)_i  = x_i * cos(m * theta_i) + x_{i+1} * (-1)^k * sin(m * theta_i)
# Applied to Q and K before attention computation
```

### 4. Grouped Query Attention (GQA)

```
# n_heads query heads share n_kv_heads key/value heads
# For Phase 1: n_heads=8, n_kv_heads=2 (4 queries share 1 KV head)
Q           = X @ W_q      # shape: (batch, seq, n_heads, head_dim)
K, V        = X @ W_k/v    # shape: (batch, seq, n_kv_heads, head_dim)
K, V        = repeat(K, V, n_heads // n_kv_heads)  # expand to match Q
Attn        = softmax(Q @ K.T / sqrt(head_dim)) @ V
```

### 5. SwiGLU Feed-Forward

```
gate        = W_gate @ x
up          = W_up @ x
hidden      = gate * sigmoid(beta * gate) * up   # SwiGLU gating
output      = W_down @ hidden
# FFN dim = 4 * embedding_dim (standard), then projected back
```

### 6. Training Objective (Cross-Entropy)

```
L = -(1/T) * sum_t [ log P(x_t | x_{<t}; theta) ]
# Equivalently: cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

### 7. AdamW Optimiser (use PyTorch's built-in, but understand this)

```
m_t = beta1 * m_{t-1} + (1 - beta1) * grad
v_t = beta2 * v_{t-1} + (1 - beta2) * grad²
theta = theta - lr * m_t / (sqrt(v_t) + eps) - lr * weight_decay * theta
# weight_decay applied DIRECTLY to theta, not via gradient — this is AdamW vs Adam
```

---

## PHASE 1 CONFIG (source of truth for all values)

```yaml
model:
  vocab_size: 8192
  embedding_dim: 256
  num_layers: 6
  num_heads: 8
  num_kv_heads: 2
  ffn_hidden_dim: 1024
  max_seq_length: 512
  dropout: 0.1
  weight_bits: 1.58
  activation_bits: 8
  norm_eps: 1.0e-6

training:
  batch_size: 32
  learning_rate: 3.0e-4
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  warmup_steps: 500
  max_steps: 50000
  grad_clip: 1.0
  save_every: 1000
  eval_every: 500
  seed: 42

data:
  train_path: "data/phase1/train.bin"
  val_path: "data/phase1/val.bin"
  num_workers: 4
  tokenizer_path: "cognicore/tokenizer/"

cpu:
  num_threads: 0          # 0 = auto-detect all cores
  use_gradient_checkpointing: true
  compile: false          # torch.compile — disable for CPU debugging

logging:
  log_dir: "logs/"
  wandb: false
  log_every: 50
```

---

## DATASETS TO USE (Phase 1)

Pull these from Hugging Face `datasets` library. No manual downloads.

```python
# Primary training data
datasets_to_use = [
    {
        "name": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "text_column": "text",
        "max_samples": 500_000,       # ~1.5GB of text
    },
    {
        "name": "OpenAssistant/oasst2",
        "split": "train",
        "text_column": "text",
        "max_samples": None,          # use all ~160K turns
    },
    {
        "name": "databricks/databricks-dolly-15k",
        "split": "train",
        "text_column": "instruction",  # combine with "response"
        "max_samples": None,
    },
]
# Format: concatenate instruction + "\n" + response for Dolly
# Format: use 'text' field directly for Wikipedia and OASST2
# Shuffle combined corpus with seed=42 before tokenising
# 90/10 train/val split
```

---

## FRONTEND REQUIREMENTS

The React UI must feel like a **premium, minimal dark-mode AI assistant**. Not a generic chatbot. Think: clean terminal meets modern consumer app.

**Visual direction:**
- Dark background `#0D1117` (GitHub dark)
- Accent: electric blue `#1A73E8` (Google Blue) with glow effects
- Monospace font for model output (to feel technical and trustworthy)
- Sans-serif for UI chrome
- Smooth streaming — tokens appear one by one, no jarring reflows
- A status bar showing live: CPU %, tokens/sec, RAM used
- Subtle animated pulse on the avatar when model is generating

**Components to build:**
1. `ChatWindow` — scrollable message list, markdown rendered, streaming-aware
2. `QueryBar` — text input, submit on Enter, send button, web search toggle
3. `MessageBubble` — user vs assistant styling, timestamp, copy button
4. `ModelStatusBar` — live CPU/RAM/speed metrics polled from `/api/status`
5. `SettingsPanel` — slide-out drawer, model variant selector, max tokens slider

**API calls:**
- `POST /api/query` → `{ message: string, use_web_search: boolean, max_tokens: number }`
- `GET /api/stream` → Server-Sent Events, each event is `{ token: string, done: boolean }`
- `GET /api/status` → `{ cpu_percent: float, ram_mb: float, tokens_per_sec: float, model_loaded: boolean }`

---

## HOW TO PROCEED — BUILD ORDER

Follow this exact sequence. Complete each step fully before moving to the next.
Do not jump ahead. Do not skip tests.

```
STEP 1:  configs/phase1_config.yaml          — Write the config file
STEP 2:  model/config.py                     — Pydantic loader for the config
STEP 3:  model/rmsnorm.py                    — RMSNorm layer + test
STEP 4:  model/bitlinear.py                  — BitLinear layer + test (verify {-1,0,+1} weights)
STEP 5:  model/attention.py                  — GQA + RoPE + test
STEP 6:  model/ffn.py                        — SwiGLU FFN + test
STEP 7:  model/architecture.py               — Full CogniCore transformer + test
STEP 8:  model/tokenizer/train_tokenizer.py  — BPE tokeniser training script
STEP 9:  data/download.py                    — Dataset downloader
STEP 10: data/preprocess.py                  — Tokenise corpus to .bin shards
STEP 11: data/dataloader.py                  — CPU DataLoader
STEP 12: model/training.py                   — Training loop with logging + checkpointing
STEP 13: model/inference.py                  — Streaming inference engine
STEP 14: api/main.py + routes/               — FastAPI backend
STEP 15: api/rag/search.py + context.py      — Web search RAG layer
STEP 16: frontend/                           — React UI (all components)
STEP 17: docker/                             — Dockerfiles for train and serve
STEP 18: scripts/train_phase1.sh             — One-command training runner
STEP 19: README.md                           — Complete setup and run instructions
STEP 20: RUN PHASE 1 TRAINING                — Execute and report loss/perplexity
```

---

## HOW TO START

When I say **"Start"** or **"Build Step [N]"**, you build that step completely.

For each step:
1. State which file you are building
2. Write the COMPLETE file — no placeholders
3. After the file, write the corresponding test
4. Show a one-line sanity check command I can run to verify it works
5. Tell me what the next step is

If you encounter an ambiguity, make the best engineering decision and document your reasoning in a comment. Do not stop to ask unless it is a genuine blocker.

---

## CONTEXT FOR EVERY RESPONSE

This project will be **demonstrated to Google AI engineers**. The code represents:

> "A CPU-native foundational SLM — proving that AI can be trained and served on commodity hardware without a single GPU, making intelligence accessible to anyone with a laptop."

Every file you write should be something an Anthropic or Google ML engineer would be proud to review. Production quality. Not a tutorial. Not a demo script. A real system.

---

## READY

Type **"Start"** to begin with Step 1.

Or type **"Build Step [N]"** to jump to a specific step if some steps are already done.

Or type **"Explain [component]"** if you want me to walk through the mathematics or design of any part before building it.

---

*CogniCore SLM | Anti Gravity Build Prompt v1.0 | Reference: CogniCore_SLM_Blueprint.docx*
