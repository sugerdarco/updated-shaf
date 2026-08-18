#!/bin/bash
REPO=/storage/riya/riya_experiment/updated-shaf; cd $REPO
export HF_HOME=/storage/riya/bhuvi-hf-cache HUGGINGFACE_HUB_CACHE=/storage/riya/bhuvi-hf-cache/hub HF_TOKEN=$(cat ~/.hf_token) PYTHONPATH=$REPO
PY=/storage/riya/bhuvi-sahf-test/repo/venv/bin/python
for i in $(seq 1 260); do
  d=0; for j in mod_qwen mod_mistral mod_llama31; do grep -q '^SUMMARY' eval/logs/$j.log 2>/dev/null && d=$((d+1)); done
  [ $d -ge 3 ] && break; sleep 60
done
echo "=== HEADLINE MODERN trio done ($d/3) ==="
$PY eval/rescore.py eval/out/mod_qwen.json eval/out/mod_llama31.json eval/out/mod_mistral.json
echo "=== CES (qwen+llama31+mistral) ==="
$PY eval/ces_batched.py --config config_modern2.yaml --sample eval/full_eval.jsonl --devices cuda:0,cuda:1,cuda:7 --cands eval/out/mod_qwen.json eval/out/mod_llama31.json eval/out/mod_mistral.json --out eval/out/ces_modern2.json 2>&1 | grep -E 'SUMMARY|need endorsement'
echo "=== analysis (voting + significance) ==="
$PY eval/paper_analysis.py --cands eval/out/mod_qwen.json eval/out/mod_llama31.json eval/out/mod_mistral.json --ces eval/out/ces_modern2.json --names qwen25,llama31,mistral03
echo "=== HEADLINE MODERN COMPLETE ==="
