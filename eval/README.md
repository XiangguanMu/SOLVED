# Eval

- `test_query_splitting.py`: print offline query-decomposition results
- `test_specific_retrieval.py`: run a hand-authored split through the retrieval pipeline

Each dataset's complete 38-query set and segment-level ground truth live directly
in `eval_data/<dataset>.py`; cached decompositions live in `eval_data/splits/`.
Use `experiments/export_query_results.py` to compute IoU metrics from run logs against `ground_truth_segments`.
