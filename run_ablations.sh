#!/bin/bash
# Run all 7 ablation experiments sequentially on 50k-story subsets.
# Each run: 3000 steps, same seed (42), LR, batch size and tokenizer;
# only one variable changes per run vs the baseline.
# Safe to re-run: a config whose log already has the final step 2999 is skipped.
# Logs land in logs/ablation_<name>.csv ; at the end they are copied to
# Google Drive (if mounted) alongside the full-run logs.
set -u

for cfg in ablation_baseline ablation_1head ablation_sinusoidal ablation_2layer ablation_4layer ablation_ctx128 ablation_ctx512; do
  log="logs/${cfg}.csv"
  if [ -f "$log" ] && grep -q "^2999," "$log"; then
    echo "SKIP $cfg (already finished)"
    continue
  fi
  echo "=== RUN $cfg ==="
  python3 -u -m src.train --config "configs/${cfg}.yaml" 2>&1 | tee "logs/${cfg}.stdout.log"
  echo "=== DONE $cfg ==="
done

echo "ALL ABLATIONS DONE"
ls -la logs/ablation_*.csv

if [ -d /content/drive/MyDrive/transformer-from-scratch ]; then
  cp logs/ablation_*.csv logs/ablation_*.stdout.log /content/drive/MyDrive/transformer-from-scratch/ 2>/dev/null
  echo "logs copied to Google Drive"
fi
