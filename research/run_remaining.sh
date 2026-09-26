#!/usr/bin/env bash
PY="C:/Program Files/Python310/python.exe"
cd "C:/Users/sagar/Desktop/CoinJoin/research"
echo "waiting for E05 to finish ..."
until [ -f results/e05_gnn_main_aggregate.csv ]; do sleep 30; done
echo "=== E05 AGGREGATE ==="
cat results/e05_gnn_main_aggregate.csv
sleep 10
echo; echo "=== E11 compute cost (idle device) ==="
"$PY" experiments/exp11_compute_cost.py
echo; echo "=== E10 statistical analysis ==="
"$PY" experiments/exp10_statistical_analysis.py
echo; echo "=== E09 explainability ==="
"$PY" experiments/exp09_explainability.py
echo; echo "ALL REMAINING EXPERIMENTS COMPLETE"
