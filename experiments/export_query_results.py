import argparse
import csv
import importlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


QUERY_RE = re.compile(r"^Query\s+(\d+):\s+'(.*)'$")
SCENE_RE = re.compile(r"^\s*(\d+)\.\s+Scene:\s+(.+)$")
SCORE_RE = re.compile(
    r"^\s*Sequence Normalized Score:\s*([-+]?\d*\.?\d+)\s+\(Raw Total:\s*([-+]?\d*\.?\d+)\)"
)
SEGMENT_RE = re.compile(r"^\s*Segment:\s*frame\s+(-?\d+)\s*->\s*frame\s+(-?\d+)")
QUERY_TIME_RE = re.compile(r"^\s*Total Query Processing Time:\s*([-+]?\d*\.?\d+)s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export structured query_lancedb.py top results with GT IoU."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["lovr", "msrvtt", "videochapter"],
        help="Dataset name. Example: lovr",
    )
    return parser.parse_args()


def scene_to_video_id(scene_name: str) -> str:
    return scene_name.replace(".mp4", "").split("-Scene-")[0]


def dataset_name_from_module(dataset_module: str) -> str:
    return dataset_module.split(".")[-1]


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def resolve_output_paths(args):
    dataset_module = f"eval_data.{args.dataset}"
    dataset_name = dataset_name_from_module(dataset_module)
    result_dir = os.path.join("experiments", "results", dataset_name)
    # Indexed-run logs already print tie-aware top-5 segments; mIoU@5 uses them as-is.
    data_dir = os.path.join("experiments", "data")

    args.dataset_module = dataset_module
    args.input_log = os.path.join(data_dir, f"{dataset_name}.log")
    args.output_json = os.path.join(result_dir, "query_results.json")
    args.output_csv = os.path.join(result_dir, "query_results.csv")
    args.metrics_json = os.path.join(result_dir, "metrics.json")


def iou_frames(
    pred_start: Optional[int],
    pred_end: Optional[int],
    gt_start: Optional[int],
    gt_end: Optional[int],
) -> Optional[float]:
    if None in (pred_start, pred_end, gt_start, gt_end):
        return None
    if pred_end < pred_start or gt_end < gt_start:
        return None
    inter = max(0, min(pred_end, gt_end) - max(pred_start, gt_start) + 1)
    union = max(pred_end, gt_end) - min(pred_start, gt_start) + 1
    if union <= 0:
        return None
    return inter / union


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_segments_value(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    normalized = []
    for seg in values:
        if not isinstance(seg, dict):
            continue
        normalized.append(
            {
                "video_id": seg.get("video_id"),
                "frame_start": seg.get("frame_start"),
                "frame_end": seg.get("frame_end"),
            }
        )
    return normalized


def _normalize_gt_segments_per_query(
    gt_segments_raw: Any,
    query_count: int,
) -> List[List[Dict[str, Any]]]:
    per_query = [[] for _ in range(query_count)]

    if not gt_segments_raw:
        return per_query

    # Case 0: dict keyed by query index (1-based or 0-based, int/str)
    if isinstance(gt_segments_raw, dict):
        for q_idx in range(query_count):
            one_based_keys = [q_idx + 1, str(q_idx + 1)]
            zero_based_keys = [q_idx, str(q_idx)]
            value = None
            for key in one_based_keys + zero_based_keys:
                if key in gt_segments_raw:
                    value = gt_segments_raw[key]
                    break
            per_query[q_idx] = _normalize_segments_value(value)
        return per_query

    # Case 1: already per-query nested list format:
    # [
    #   [{"video_id": "...", "frame_start": ..., "frame_end": ...}, ...],
    #   ...
    # ]
    if isinstance(gt_segments_raw[0], list):
        for idx, query_segments in enumerate(gt_segments_raw[:query_count]):
            per_query[idx] = _normalize_segments_value(query_segments)
        return per_query

    # Case 2: flat list of dict format (one segment per query, extras ignored)
    flat_segments = [seg for seg in gt_segments_raw if isinstance(seg, dict)]
    for idx in range(min(query_count, len(flat_segments))):
        per_query[idx].extend(_normalize_segments_value(flat_segments[idx]))

    return per_query


def _finalize_result(
    current_result: Optional[Dict[str, Any]],
    current_query: Optional[Dict[str, Any]],
):
    if current_result is not None and current_query is not None:
        current_query["top_results"].append(current_result)


def parse_query_output(raw_text: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    current_query: Optional[Dict[str, Any]] = None
    current_result: Optional[Dict[str, Any]] = None

    for line in raw_text.splitlines():
        q_match = QUERY_RE.match(line)
        if q_match:
            _finalize_result(current_result, current_query)
            current_result = None
            if current_query is not None:
                results.append(current_query)
            current_query = {
                "query_number": int(q_match.group(1)),
                "query": q_match.group(2),
                "top_results": [],
                "query_time_seconds": None,
            }
            continue

        s_match = SCENE_RE.match(line)
        if s_match and current_query is not None:
            _finalize_result(current_result, current_query)
            current_result = {
                "rank": int(s_match.group(1)),
                "scene": s_match.group(2).strip(),
                "video_id": scene_to_video_id(s_match.group(2).strip()),
                "normalized_score": None,
                "raw_score": None,
                "frame_start": None,
                "frame_end": None,
            }
            continue

        score_match = SCORE_RE.match(line)
        if score_match and current_result is not None:
            current_result["normalized_score"] = float(score_match.group(1))
            current_result["raw_score"] = float(score_match.group(2))
            continue

        seg_match = SEGMENT_RE.match(line)
        if seg_match and current_result is not None:
            current_result["frame_start"] = int(seg_match.group(1))
            current_result["frame_end"] = int(seg_match.group(2))
            continue

        time_match = QUERY_TIME_RE.match(line)
        if time_match and current_query is not None:
            current_query["query_time_seconds"] = float(time_match.group(1))

    _finalize_result(current_result, current_query)
    if current_query is not None:
        results.append(current_query)
    return results


def load_dataset_annotations(
    dataset_module: str,
) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
    module = importlib.import_module(dataset_module)
    queries = list(getattr(module, "queries", []))
    gt_segments_raw = getattr(module, "ground_truth_segments", {})
    gt_segments_per_query = _normalize_gt_segments_per_query(
        gt_segments_raw, len(queries)
    )
    return queries, gt_segments_per_query


def enrich_with_ground_truth(
    parsed_results: List[Dict[str, Any]],
    dataset_queries: List[str],
    dataset_gt_segments: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    by_query_number = {r["query_number"]: r for r in parsed_results}
    merged: List[Dict[str, Any]] = []

    for idx, query_text in enumerate(dataset_queries):
        query_number = idx + 1
        parsed = by_query_number.get(
            query_number,
            {"query_number": query_number, "query": query_text, "top_results": []},
        )
        if not parsed.get("query"):
            parsed["query"] = query_text
        gt_segments = dataset_gt_segments[idx] if idx < len(dataset_gt_segments) else []

        for pred in parsed.get("top_results", []):
            best_iou = None
            best_gt = None
            for gt in gt_segments:
                if pred.get("video_id") != gt.get("video_id"):
                    continue
                cur_iou = iou_frames(
                    pred.get("frame_start"),
                    pred.get("frame_end"),
                    gt.get("frame_start"),
                    gt.get("frame_end"),
                )
                if cur_iou is None:
                    continue
                if best_iou is None or cur_iou > best_iou:
                    best_iou = cur_iou
                    best_gt = gt
            pred["iou_with_gt"] = best_iou
            pred["matched_gt"] = best_gt

        parsed["ground_truth_segments"] = gt_segments
        parsed.setdefault("query_time_seconds", None)
        merged.append(parsed)
    return merged


def _best_iou_for_top1_ties(top_results: List[Dict[str, Any]]) -> float:
    if not top_results:
        return 0.0
    scored = [r for r in top_results if r.get("normalized_score") is not None]
    if not scored:
        return 0.0
    top_score = max(_safe_float(r.get("normalized_score")) or float("-inf") for r in scored)
    eps = 1e-9
    tied_top1 = [
        r
        for r in scored
        if (abs((_safe_float(r.get("normalized_score")) or float("-inf")) - top_score) <= eps)
    ]
    if not tied_top1:
        return 0.0
    return max((_safe_float(r.get("iou_with_gt")) or 0.0) for r in tied_top1)


def _best_iou_for_topk(top_results: List[Dict[str, Any]], k: int = 5) -> float:
    """Best IoU over printed top results.

    The query log already applies tie-aware top-k truncation (include whole
    equal-score tiers until the next tier would start past ``k``). Therefore
    mIoU@5 / R5 use every printed segment as-is, without re-cutting to ``[:k]``.
    """
    if not top_results:
        return 0.0
    # ``k`` kept for call-site compatibility; cutoff already applied when printing.
    if k <= 0:
        return 0.0
    return max((_safe_float(r.get("iou_with_gt")) or 0.0) for r in top_results)


def _pred_iou_with_gt(pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
    if pred.get("video_id") != gt.get("video_id"):
        return 0.0
    return (
        iou_frames(
            pred.get("frame_start"),
            pred.get("frame_end"),
            gt.get("frame_start"),
            gt.get("frame_end"),
        )
        or 0.0
    )


def _average_precision_at_tau(
    top_results: List[Dict[str, Any]],
    gt_segments: List[Dict[str, Any]],
    tau: float,
) -> float:
    """Compute AP for one query at IoU threshold tau."""
    if not gt_segments:
        return 0.0
    if not top_results:
        return 0.0

    # Tie-aware ranking:
    # - primary key: normalized score descending
    # - tie-breaker: IoU-with-GT descending
    # This avoids penalizing a true positive just because a same-score false positive
    # appears earlier in raw log order.
    ranked_results = sorted(
        top_results,
        key=lambda pred: (
            -(_safe_float(pred.get("normalized_score")) or float("-inf")),
            -(_safe_float(pred.get("iou_with_gt")) or 0.0),
        ),
    )

    gt_used = [False] * len(gt_segments)
    tp = 0
    fp = 0
    precision_sum = 0.0

    for pred in ranked_results:
        best_iou = 0.0
        best_gt_idx = -1
        for idx, gt in enumerate(gt_segments):
            if gt_used[idx]:
                continue
            cur_iou = _pred_iou_with_gt(pred, gt)
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_gt_idx = idx

        if best_gt_idx >= 0 and best_iou >= tau:
            gt_used[best_gt_idx] = True
            tp += 1
            precision_sum += tp / (tp + fp)
        else:
            fp += 1

    return precision_sum / len(gt_segments)


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "query_count": 0,
            "R1@0.3": 0.0,
            "R1@0.5": 0.0,
            "R1@0.7": 0.0,
            "R5@0.5": 0.0,
            "R5@0.7": 0.0,
            "mIoU@1": 0.0,
            "mIoU@5": 0.0,
            "total_query_time_seconds": 0.0,
            "avg_query_time_seconds": 0.0,
            "mAP@0.5": 0.0,
            "mAP@0.7": 0.0,
            "per_query": [],
        }

    per_query = []
    top1_ious = []
    top5_ious = []
    query_times = []
    ap_05_list = []
    ap_07_list = []
    for row in rows:
        top_results = row.get("top_results", [])
        gt_segments = row.get("ground_truth_segments", [])
        best_iou_at1 = _best_iou_for_top1_ties(top_results)
        best_iou_at5 = _best_iou_for_topk(top_results, k=5)
        ap_05 = _average_precision_at_tau(top_results, gt_segments, 0.5)
        ap_07 = _average_precision_at_tau(top_results, gt_segments, 0.7)
        top1_ious.append(best_iou_at1)
        top5_ious.append(best_iou_at5)
        ap_05_list.append(ap_05)
        ap_07_list.append(ap_07)
        query_time = _safe_float(row.get("query_time_seconds"))
        if query_time is not None:
            query_times.append(query_time)
        per_query.append(
            {
                "query_number": row.get("query_number"),
                "best_iou_at1": best_iou_at1,
                "best_iou_at5": best_iou_at5,
                "AP@0.5": ap_05,
                "AP@0.7": ap_07,
                "query_time_seconds": query_time,
            }
        )

    r1_03 = sum(1 for x in top1_ious if x >= 0.3) / n
    r1_05 = sum(1 for x in top1_ious if x >= 0.5) / n
    r1_07 = sum(1 for x in top1_ious if x >= 0.7) / n
    r5_05 = sum(1 for x in top5_ious if x >= 0.5) / n
    r5_07 = sum(1 for x in top5_ious if x >= 0.7) / n
    avg_iou_1 = sum(top1_ious) / n
    avg_iou_5 = sum(top5_ious) / n
    total_query_time_seconds = sum(query_times)
    avg_query_time_seconds = (
        total_query_time_seconds / len(query_times) if query_times else 0.0
    )
    map_05 = sum(ap_05_list) / n
    map_07 = sum(ap_07_list) / n

    return {
        "query_count": n,
        "R1@0.3": r1_03,
        "R1@0.5": r1_05,
        "R1@0.7": r1_07,
        "R5@0.5": r5_05,
        "R5@0.7": r5_07,
        "mIoU@1": avg_iou_1,
        "mIoU@5": avg_iou_5,
        "total_query_time_seconds": total_query_time_seconds,
        "avg_query_time_seconds": avg_query_time_seconds,
        "mAP@0.5": map_05,
        "mAP@0.7": map_07,
        "timed_query_count": len(query_times),
        "per_query": per_query,
    }


def dump_json(rows: List[Dict[str, Any]], output_path: str):
    _ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def dump_metrics(metrics: Dict[str, Any], output_path: str):
    _ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def dump_csv(rows: List[Dict[str, Any]], output_path: str):
    _ensure_parent_dir(output_path)
    fieldnames = [
        "query_number",
        "query",
        "rank",
        "scene",
        "video_id",
        "frame_start",
        "frame_end",
        "normalized_score",
        "raw_score",
        "iou_with_gt",
        "matched_gt_video_id",
        "matched_gt_frame_start",
        "matched_gt_frame_end",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for query_row in rows:
            query_number = query_row.get("query_number")
            query_text = query_row.get("query")
            top_results = query_row.get("top_results", [])
            if not top_results:
                writer.writerow({"query_number": query_number, "query": query_text})
                continue
            for pred in top_results:
                matched_gt = pred.get("matched_gt") or {}
                writer.writerow(
                    {
                        "query_number": query_number,
                        "query": query_text,
                        "rank": pred.get("rank"),
                        "scene": pred.get("scene"),
                        "video_id": pred.get("video_id"),
                        "frame_start": pred.get("frame_start"),
                        "frame_end": pred.get("frame_end"),
                        "normalized_score": pred.get("normalized_score"),
                        "raw_score": pred.get("raw_score"),
                        "iou_with_gt": pred.get("iou_with_gt"),
                        "matched_gt_video_id": matched_gt.get("video_id"),
                        "matched_gt_frame_start": matched_gt.get("frame_start"),
                        "matched_gt_frame_end": matched_gt.get("frame_end"),
                    }
                )


def load_or_run_output(args) -> str:
    if not os.path.exists(args.input_log):
        raise FileNotFoundError(
            f"Input log not found: {args.input_log}. Please place the console log there."
        )
    with open(args.input_log, "r", encoding="utf-8") as f:
        return f.read()


def main():
    args = parse_args()
    resolve_output_paths(args)
    raw_text = load_or_run_output(args)
    parsed = parse_query_output(raw_text)
    dataset_queries, dataset_gt_segments = load_dataset_annotations(args.dataset_module)
    merged = enrich_with_ground_truth(parsed, dataset_queries, dataset_gt_segments)
    metrics = compute_metrics(merged)

    dump_json(merged, args.output_json)
    print(f"JSON exported to: {args.output_json}")
    dump_csv(merged, args.output_csv)
    print(f"CSV exported to: {args.output_csv}")
    dump_metrics(metrics, args.metrics_json)
    print(f"Metrics exported to: {args.metrics_json}")
    print(
        "Metrics | "
        f"R1@0.3={metrics['R1@0.3']:.4f}, "
        f"R1@0.5={metrics['R1@0.5']:.4f}, "
        f"R1@0.7={metrics['R1@0.7']:.4f}, "
        # f"R5@0.5={metrics['R5@0.5']:.4f}, "
        # f"R5@0.7={metrics['R5@0.7']:.4f}, "
        f"mIoU@1={metrics['mIoU@1']:.4f}, "
        f"mIoU@5={metrics['mIoU@5']:.4f}, "
        # f"mAP@0.5={metrics['mAP@0.5']:.4f}, "
        # f"mAP@0.7={metrics['mAP@0.7']:.4f}, "
        f"total_time={metrics['total_query_time_seconds']:.4f}s, "
        f"avg_time={metrics['avg_query_time_seconds']:.4f}s"
    )
    print(f"Parsed queries: {len(parsed)} | Dataset queries: {len(dataset_queries)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
