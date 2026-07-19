import argparse
import importlib
import os
import sys
import time


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
        "--max_queries",
        type=int,
        default=None,
        help="Only run the first N queries.",
    )
    return parser.parse_args()


def _dataset_name_from_db_path(db_path: str) -> str:
    name = os.path.basename(os.path.normpath(db_path))
    if not name:
        raise ValueError(f"Cannot infer dataset name from db_path: {db_path}")
    return name


def _load_eval_data(dataset_name: str):
    module = importlib.import_module(f"eval_data.{dataset_name}")
    return module.queries


def _setup_tee_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_file


def main():
    args = parse_args()
    dataset_name = _dataset_name_from_db_path(args.db_path)
    queries = _load_eval_data(dataset_name)

    log_path = args.log_path
    if log_path is None:
        log_path = os.path.join(
            os.path.dirname(__file__), "experiments", "data", f"{dataset_name}.log"
        )
    elif not os.path.isabs(log_path):
        log_path = os.path.join(os.path.dirname(__file__), log_path)

    log_file = _setup_tee_logging(log_path)
    print(f"Dataset: {dataset_name}")
    print(f"All console output will also be saved to: {log_path}")

    try:
        models = load_models_parallel(args.qwen3_model_path)
        intern_model = models["intern_model"]
        qwen3_model_path = models["qwen3_model_path"]
        qwen3_bridge_script = models["qwen3_bridge_script"]
        qwen3_python = models["qwen3_python"]

        split_wall_start = time.time()
        all_split_results, split_times, model_load_time = split_queries_offline(
            queries, args.model_path
        )
        split_wall_time = time.time() - split_wall_start
        print(
            f"Query decomposition summary: model_load={model_load_time:.4f}s, "
            f"per_query_sum={sum(split_times):.4f}s, wall={split_wall_time:.4f}s"
        )
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

        frames_tbl = db.open_table("frames") if "frames" in db.table_names() else None

        candidate_log_path = os.path.join(
            os.path.dirname(__file__), args.candidate_log_path
        )
        candidate_log_dir = os.path.dirname(candidate_log_path)
        if candidate_log_dir:
            os.makedirs(candidate_log_dir, exist_ok=True)
        with open(candidate_log_path, "w", encoding="utf-8"):
            pass
        print(f"Candidate debug log will be saved to: {candidate_log_path}")

        try:
            indices = list(range(len(queries)))
            if args.query_index is not None:
                if args.query_index < 0 or args.query_index >= len(queries):
                    print(
                        f"Invalid --query_index {args.query_index}. Must be within [0, {len(queries)-1}]."
                    )
                    return
                indices = [args.query_index]
            elif args.max_queries is not None:
                indices = indices[: args.max_queries]

            for i in indices:
                query = queries[i]
                atomic_items = all_split_results[i]
                process_query(
                    i,
                    query,
                    atomic_items,
                    intern_model,
                    clips_tbl,
                    frames_tbl,
                    qwen3_model_path,
                    qwen3_bridge_script,
                    qwen3_python,
                    args.nprobes,
                    args.top_k,
                    db,
                    candidate_log_path,
                    query_split_time=split_times[i],
                )
        finally:
            cleanup_index(clips_tbl, args.drop_index_on_exit, index_name)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
        print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
