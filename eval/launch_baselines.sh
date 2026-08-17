#!/bin/bash
# Launch the three single-model baselines + the SAHF ensemble baseline on the
# mixed 100-prompt sample, each in the background on its own GPU(s).
set -e
REPO=/storage/riya/riya_experiment/updated-shaf
cd "$REPO"
export HF_HOME=/storage/riya/bhuvi-hf-cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.hf_token 2>/dev/null)}"
export PYTHONPATH=$REPO
PY=/storage/riya/bhuvi-sahf-test/repo/venv/bin/python
mkdir -p eval/out eval/logs

nohup $PY eval/run_single.py --model mistralai/Mistral-7B-Instruct-v0.2 \
  --device cuda:0 --out eval/out/single_mistral.json > eval/logs/single_mistral.log 2>&1 &
echo "mistral  pid $!"
nohup $PY eval/run_single.py --model meta-llama/Llama-2-13b-chat-hf \
  --device cuda:4 --out eval/out/single_llama.json > eval/logs/single_llama.log 2>&1 &
echo "llama    pid $!"
nohup $PY eval/run_single.py --model 01-ai/Yi-6B-Chat \
  --device cuda:5 --out eval/out/single_yi.json > eval/logs/single_yi.log 2>&1 &
echo "yi       pid $!"
nohup $PY eval/run_ensemble.py --devices cuda:7,cuda:3,cuda:2 \
  --out eval/out/ensemble_base.json > eval/logs/ensemble_base.log 2>&1 &
echo "ensemble pid $!"
echo "all launched"
