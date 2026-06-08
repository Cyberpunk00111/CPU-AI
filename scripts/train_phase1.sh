#!/usr/bin/env bash
set -euo pipefail

python data/download.py --config configs/phase1_config.yaml
python cognicore/model/tokenizer/train_tokenizer.py --config configs/phase1_config.yaml
python data/preprocess.py --config configs/phase1_config.yaml
python cognicore/model/training.py --config configs/phase1_config.yaml
