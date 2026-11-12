import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from db import cleanup_index, setup_index
from model_loader import load_models_parallel
from paths import LANCEDB_PATH, QUERY_SPLITTER_MODEL_PATH, QWEN3_VL_MODEL_PATH
from pipeline import process_query
from query_splitter.splitter import dump_split_results_txt, split_queries_offline


class _Tee:
    """Write to multiple streams (stdout + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query LanceDB for video clips with LLM query splitting and candidate restriction."
    )
    parser.add_argument("--db_path", type=str, default=LANCEDB_PATH)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--qwen3_model_path",
        type=str,
        default=QWEN3_VL_MODEL_PATH,
    )
    parser.add_argument("--print_gt_scores", action="store_true")
    parser.add_argument(
        "--model_path",
        type=str,
        default=QUERY_SPLITTER_MODEL_PATH,
    )
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--drop_index_on_exit", action="store_true")
    parser.add_argument("--num_partitions", type=int, default=1024)
    parser.add_argument("--num_sub_vectors", type=int, default=256)
    parser.add_argument("--nprobes", type=int, default=100)
    parser.add_argument(
        "--split_output_path", type=str, default="logs/split_results.txt"
    )
    parser.add_argument(
        "--candidate_log_path",
        type=str,
        default="logs/query_candidates.jsonl",
        help="JSONL file for per-query coarse top20 clips and hard-constraint candidates.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Tee all stdout/stderr to this .log file. "
        "Default: experiments/data/<dataset>.log inferred from --db_path.",
    )
    parser.add_argument(
        "--query_index",
        type=int,
        default=None,
        help="Only run a single query by 0-based index.",
    )
    parser.add_argument(
        "--query_indices",
        type=str,
        default=None,
        help="Comma-separated 0-based query indices. Overrides start/end.",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Only run the first N queries.",
    )
    parser.add_argument(
        "--query_start",
        type=int,
        default=0,
        help="First 0-based query index to run (inclusive).",
    )
    parser.add_argument(
        "--query_end",
        type=int,
        default=None,
        help="End 0-based query index (exclusive).",
    )
    parser.add_argument(
        "--eval_module",
        type=str,
        default=None,
        help="Query module, e.g. eval_data.lovr. Default: eval_data.<dataset>.",
    )
    parser.add_argument("--coarse_k", type=int, default=20)
    parser.add_argument("--beam_width", type=int, default=3)
    parser.add_argument(
        "--video_beam_width",
        type=int,
        default=5,
        help=(
            "Retain this many videos after each logical time step in cheap beam. "
            "Set to 0 to use the full-scoring path."
        ),
    )
    parser.add_argument(
        "--score_quantum",
        type=float,
        default=0.1,
        help="Round each VLM clause score to this bin size before beam ranking. "
        "Set to 0 to disable quantization.",
    )
    parser.add_argument("--penalty_weight", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=200.0)
    parser.add_argument("--skip_time_filter", action="store_true")
    parser.add_argument("--score_all_frames", action="store_true")
    parser.add_argument(
        "--append_log",
        action="store_true",
        help="Append to the log and candidate jsonl instead of truncating them.",
    )
    parser.add_argument(
        "--split_cache",
        type=str,
        default=None,
        help="JSON cache of query splits. Defaults to eval_data/splits/<dataset>.json.",
    )
    parser.add_argument(
        "--require_multiple_time_steps",
        action="store_true",
        help="Run only queries containing at least two distinct logical time steps.",
    )
    return parser.parse_args()


def _dataset_name_from_db_path(db_path: str) -> str:
    name = os.path.basename(os.path.normpath(db_path))
    if not name:
        raise ValueError(f"Cannot infer dataset name from db_path: {db_path}")
    return name


def _load_eval_data(dataset_name: str):
    module = importlib.import_module(f"eval_data.{dataset_name}")
    return module.queries, module.ground_truth_scene_names


def _load_split_cache(path, queries):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("queries") != list(queries):
        print(f"Split cache does not match the current queries: {path}")
        return None
    return payload["results"], payload["times"]


def _save_split_cache(path, queries, results, times):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {"queries": list(queries), "results": results, "times": list(times)},
            handle,
            ensure_ascii=False,
        )


def _setup_tee_logging(log_path: str, append: bool = False):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a" if append else "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_file


def _init_gt_log(gt_log_path):
    os.makedirs(os.path.dirname(gt_log_path), exist_ok=True)
    with open(gt_log_path, "w", encoding="utf-8") as f:
        f.write(
            f"=== Ground Truth Scores (generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n"
        )


def _write_gt_log_entry(f, i, query, ground_truth_names, all_seqs, time_steps):
    f.write(f"Query {i+1}: {query}\n")
    for gt_name in ground_truth_names:
        gt_base = gt_name.replace(".mp4", "")
        gt_cand, gt_rank = None, -1
        for rank, cand in enumerate(all_seqs, 1):
            if gt_base in cand["scene"]:
                gt_cand, gt_rank = cand, rank
                break
        f.write(f"Ground Truth: {gt_name}\n")
        if gt_cand:
            f.write(f"Rank globally: {gt_rank}\n")
            f.write(f"Matched Scene: {gt_cand['scene']}\n")
            f.write(
                f"Normalized Score: {gt_cand['normalized_score']:.4f} (Raw Total: {gt_cand['raw_score']:.4f})\n"
            )
            if gt_cand.get("frame_start") is not None and gt_cand.get("frame_end") is not None:
                f.write(
                    f"Segment: frame {int(gt_cand['frame_start'])} -> frame {int(gt_cand['frame_end'])}\n"
                )
            f.write("Sequence details:\n")
            for step, (s, t, col) in enumerate(gt_cand["sequence"]):
                f.write(
                    f"  Step {step+1} (time_step {time_steps[step]}): score={s:.4f}, cluster_frame_id={t}\n"
                )
        else:
            f.write(
                "Rank globally: Not Found (filtered out or failed hard constraint)\n"
            )
    f.write("-" * 40 + "\n")


def main():
    args = parse_args()
    persistent_vlm_bridge = None
    dataset_name = _dataset_name_from_db_path(args.db_path)
    video_beam_width = (
        args.video_beam_width if args.video_beam_width > 0 else None
    )
    if args.score_quantum > 0:
        os.environ["CHEAP_BEAM_SCORE_QUANTUM"] = str(args.score_quantum)
    else:
        os.environ.pop("CHEAP_BEAM_SCORE_QUANTUM", None)
    if args.split_cache is None:
        args.split_cache = str(
            _REPO_ROOT / "eval_data" / "splits" / f"{dataset_name}.json"
        )
    elif not os.path.isabs(args.split_cache):
        args.split_cache = str(_REPO_ROOT / args.split_cache)

    if args.eval_module:
        eval_module = importlib.import_module(args.eval_module)
        queries = list(eval_module.queries)
        ground_truth_scene_names = list(eval_module.ground_truth_scene_names)
    else:
        queries, ground_truth_scene_names = _load_eval_data(dataset_name)

    log_path = args.log_path
    if log_path is None:
        log_path = os.path.join(
            os.path.dirname(__file__), "experiments", "data", f"{dataset_name}.log"
        )
    elif not os.path.isabs(log_path):
        log_path = os.path.join(os.path.dirname(__file__), log_path)

    log_file = _setup_tee_logging(log_path, append=args.append_log)
    print(f"Dataset: {dataset_name}")
    print(f"Eval module: {args.eval_module or 'eval_data.' + dataset_name}")
    print(
        "Run config: "
        f"coarse_k={args.coarse_k} matrix_k={args.top_k} "
        f"frame_beam_width={args.beam_width} "
        f"video_beam_width={video_beam_width} "
        f"lambda={args.penalty_weight} "
        f"epsilon={args.epsilon} backend=vlm "
        f"score_quantum={args.score_quantum} "
        f"skip_time_filter={args.skip_time_filter} "
        f"score_all_frames={args.score_all_frames}"
    )
    print(f"All console output will also be saved to: {log_path}")

    try:
        models = load_models_parallel(args.qwen3_model_path)
        intern_model = models["intern_model"]
        qwen3_model_path = models["qwen3_model_path"]
        qwen3_bridge_script = models["qwen3_bridge_script"]
        qwen3_python = models["qwen3_python"]

        cached_splits = _load_split_cache(args.split_cache, queries)
        if cached_splits is not None:
            all_split_results, split_times = cached_splits
            print(
                f"Loaded {len(all_split_results)} cached query splits from {args.split_cache}"
            )
        else:
            split_wall_start = time.time()
            all_split_results, split_times, model_load_time = split_queries_offline(
                queries, args.model_path
            )
            split_wall_time = time.time() - split_wall_start
            print(
                f"Query decomposition summary: model_load={model_load_time:.4f}s, "
                f"per_query_sum={sum(split_times):.4f}s, wall={split_wall_time:.4f}s"
            )
            _save_split_cache(args.split_cache, queries, all_split_results, split_times)
        split_output_path = os.path.join(os.path.dirname(__file__), args.split_output_path)
        dump_split_results_txt(queries, all_split_results, split_output_path)
        print(f"Split results saved to: {split_output_path}")

        index_name = "temp_split_clip_index"
        db, clips_tbl = setup_index(
            args.db_path,
            args.build_index,
            args.num_partitions,
            args.num_sub_vectors,
            index_name,
        )
        if clips_tbl is None:
            return

        gt_log_path = os.path.join(os.path.dirname(__file__), "logs/gt_scores.txt")
        if args.print_gt_scores:
            _init_gt_log(gt_log_path)

        candidate_log_path = os.path.join(
            os.path.dirname(__file__), args.candidate_log_path
        )
        candidate_log_dir = os.path.dirname(candidate_log_path)
        if candidate_log_dir:
            os.makedirs(candidate_log_dir, exist_ok=True)
        if not args.append_log:
            with open(candidate_log_path, "w", encoding="utf-8"):
                pass
        print(f"Candidate debug log will be saved to: {candidate_log_path}")

        if video_beam_width is not None:
            from cheap_beam import PersistentVLMBridge
            import torch as _torch

            print("Starting dataset-persistent Qwen3-VL bridge...")
            intern_model.cpu()
            _torch.cuda.empty_cache()
            try:
                persistent_vlm_bridge = PersistentVLMBridge(
                    qwen3_python,
                    qwen3_bridge_script,
                    qwen3_model_path,
                )
                # Wait until model loading is complete before placing InternVideo2
                # back on the same GPU.
                persistent_vlm_bridge.score([])
            finally:
                intern_model.cuda()
                _torch.cuda.empty_cache()
            print("Dataset-persistent Qwen3-VL bridge is ready.")

        try:
            indices = list(range(len(queries)))
            if args.query_indices:
                indices = [
                    int(part)
                    for part in args.query_indices.split(",")
                    if part.strip()
                ]
                invalid = [index for index in indices if index < 0 or index >= len(queries)]
                if invalid:
                    print(
                        f"Invalid --query_indices {invalid}. "
                        f"Must be within [0, {len(queries)-1}]."
                    )
                    return
            elif args.query_index is not None:
                if args.query_index < 0 or args.query_index >= len(queries):
                    print(
                        f"Invalid --query_index {args.query_index}. Must be within [0, {len(queries)-1}]."
                    )
                    return
                indices = [args.query_index]
            else:
                end = len(queries) if args.query_end is None else args.query_end
                indices = list(range(args.query_start, min(end, len(queries))))
                if args.max_queries is not None:
                    indices = indices[: args.max_queries]
            if args.require_multiple_time_steps:
                indices = [
                    index
                    for index in indices
                    if len(
                        {
                            int(item.get("time_step", 0) or 0)
                            for item in all_split_results[index]
                            if item.get("clause")
                        }
                    )
                    > 1
                ]
                print(
                    "Multi-step-only mode: "
                    f"running {len(indices)} query/queries with >1 distinct time_step."
                )

            for i in indices:
                query = queries[i]
                atomic_items = all_split_results[i]
                all_seqs = process_query(
                    i,
                    query,
                    atomic_items,
                    intern_model,
                    clips_tbl,
                    qwen3_model_path,
                    qwen3_bridge_script,
                    qwen3_python,
                    ground_truth_scene_names[i],
                    args.nprobes,
                    args.top_k,
                    args.print_gt_scores,
                    db,
                    candidate_log_path,
                    query_split_time=split_times[i],
                    coarse_limit=args.coarse_k,
                    beam_width=args.beam_width,
                    penalty_weight=args.penalty_weight,
                    epsilon=args.epsilon,
                    skip_time_filter=args.skip_time_filter,
                    score_all_frames=args.score_all_frames,
                    score_backend="vlm",
                    text_embeddings=None,
                    video_beam_width=video_beam_width,
                    persistent_vlm_bridge=persistent_vlm_bridge,
                )
                if args.print_gt_scores and all_seqs is not None:
                    time_steps = [
                        int(item.get("time_step", 0) or 0)
                        for item in atomic_items
                        if item.get("clause", "")
                    ]
                    with open(gt_log_path, "a", encoding="utf-8") as f:
                        _write_gt_log_entry(
                            f, i, query, ground_truth_scene_names[i], all_seqs, time_steps
                        )
        finally:
            if persistent_vlm_bridge is not None:
                persistent_vlm_bridge.close()
                persistent_vlm_bridge = None
            cleanup_index(clips_tbl, args.drop_index_on_exit, index_name)
    finally:
        if persistent_vlm_bridge is not None:
            persistent_vlm_bridge.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
        print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
