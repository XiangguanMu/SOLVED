# Eval

- `test_query_splitting.py`: print offline query-decomposition results
- `test_specific_retrieval.py`: run a hand-authored split through the retrieval pipeline

Dataset queries and segment-level ground truth live in `eval_data/`.
Use `experiments/export_query_results.py` to compute IoU metrics from run logs against `ground_truth_segments`.
