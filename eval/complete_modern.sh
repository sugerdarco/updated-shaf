#!/bin/bash
REPO=/storage/riya/riya_experiment/updated-shaf
cd $REPO
export HF_HOME=/storage/riya/bhuvi-hf-cache HUGGINGFACE_HUB_CACHE=/storage/riya/bhuvi-hf-cache/hub HF_TOKEN=$(cat ~/.hf_token) PYTHONPATH=$REPO
PY=/storage/riya/bhuvi-sahf-test/repo/venv/bin/python
for i in $(seq 1 220); do
  d=0; for j in mod_qwen mod_mistral mod_yi; do grep -q '^SUMMARY' eval/logs/$j.log 2>/dev/null && d=$((d+1)); done
  [ $d -ge 3 ] && break; sleep 60
done
echo "=== MODERN singles done ($d/3) after $(( $(date +%s) - $(cat eval/logs/tmodern_start) ))s ==="
$PY eval/rescore.py eval/out/mod_qwen.json eval/out/mod_mistral.json eval/out/mod_yi.json
echo '=== MODERN CES ==='
$PY eval/ces_batched.py --config config_modern.yaml --sample eval/full_eval.jsonl --devices cuda:0,cuda:1,cuda:2   --cands eval/out/mod_qwen.json eval/out/mod_mistral.json eval/out/mod_yi.json --out eval/out/ces_modern.json 2>&1 | grep -E 'SUMMARY|need endorsement'
echo '=== MODERN analysis (voting + significance + length) ==='
$PY eval/paper_analysis.py --cands eval/out/mod_qwen.json eval/out/mod_mistral.json eval/out/mod_yi.json --ces eval/out/ces_modern.json --names qwen25,mistral03,yi15
echo '=== MODERN COMPLETE ==='
