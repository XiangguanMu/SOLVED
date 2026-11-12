import concurrent.futures
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from search import (
    extract_chunk_t,
    find_candidate_sequences,
    search_clips_indexed,
    top2_time_order_ok,
)
from model_loader import query_to_embedding


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _json_safe_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if pd.isna(value):
        return None
    return value


def _df_to_light_records(df, limit=None):
    if df is None or df.empty:
        return []
    heavy_cols = {
        "vector",
        "embedding",
        "clip_embedding",
        "frame_image",
        "image",
        "frames",
    }
    rows = df.head(limit) if limit else df
    records = []
    for _, row in rows.iterrows():
        record = {}
        for key, value in row.items():
            if key in heavy_cols:
                continue
            record[key] = _json_safe_value(value)
        records.append(record)
    return records


def _build_hard_constraint_candidate_log(
    atomic_items, candidate_scenes, scene_subquery_results, top_k
):
    time_steps = [
        int(item.get("time_step", 0) or 0)
        for item in atomic_items
        if item.get("clause", "")
    ]
    valid_clauses = [
        item.get("clause", "") for item in atomic_items if item.get("clause", "")
    ]
    valid_scenes_hard = _filter_scenes_by_time_order(
        candidate_scenes, scene_subquery_results, time_steps
    )

    hard_candidates = []
    for sc in valid_scenes_hard:
        scores = scene_subquery_results[sc]["S"]
        times = scene_subquery_results[sc]["T"]
        steps = []
        for step_idx, (row_s, row_t) in enumerate(zip(scores, times)):
            top_matches = []
            for rank, (score, chunk_t) in enumerate(zip(row_s, row_t), 1):
                if rank > top_k or score == -1.0:
                    continue
                top_matches.append(
                    {
                        "rank": rank,
                        "chunk_t": _json_safe_value(chunk_t),
                        "score": _json_safe_value(score),
                    }
                )
            steps.append(
                {
                    "time_step": (
                        time_steps[step_idx] if step_idx < len(time_steps) else None
                    ),
                    "clause": (
                        valid_clauses[step_idx] if step_idx < len(valid_clauses) else ""
                    ),
                    "top1_best_t": top_matches[0]["chunk_t"] if top_matches else None,
                    "top_matches": top_matches,
                }
            )
        hard_candidates.append({"scene": sc, "steps": steps})

    return valid_scenes_hard, hard_candidates


def _append_query_candidate_log(log_path, payload):
    if not log_path:
        return
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")
    print(f"Saved query candidate debug log to: {log_path}")


def fetch_candidate_scenes(
    query, intern_model, clips_tbl, nprobes, return_debug=False, coarse_limit=20
):
    """对完整查询做粗检索，返回候选场景列表和 WHERE 子句。"""
    t0 = time.time()
    complex_query_embedding = query_to_embedding(intern_model, query)
    complex_embedding_time = time.time() - t0

    t1 = time.time()
    df_candidates = (
        clips_tbl.search(complex_query_embedding)
        .metric("cosine")
        .nprobes(nprobes)
        # .limit(100)
        .limit(coarse_limit)
        .to_pandas()
    )
    complex_search_time = time.time() - t1
    print(
        f"[{complex_search_time:.4f}s] Fetched top {coarse_limit} candidate clips using the original complex query."
    )

    if df_candidates.empty:
        print("No candidate clips found for complex query.")
        if return_debug:
            return [], "", complex_embedding_time, complex_search_time, []
        return [], "", complex_embedding_time, complex_search_time

    if "_distance" in df_candidates.columns and "score" not in df_candidates.columns:
        df_candidates["score"] = 1.0 - df_candidates["_distance"]
    df_candidates["scene_name"] = df_candidates["clip_id"].str.replace(
        r"_\d+_\d+_\d+$", "", regex=True
    )
    candidate_scenes = df_candidates["scene_name"].unique().tolist()
    print(
        f"Restricting sub-query searches to the following {len(candidate_scenes)} unique candidate scenes (and all their associated clips)."
    )

    where_clause = " OR ".join(f"clip_id LIKE '{sc}%'" for sc in candidate_scenes)
    if return_debug:
        return (
            candidate_scenes,
            where_clause,
            complex_embedding_time,
            complex_search_time,
            _df_to_light_records(df_candidates, limit=coarse_limit),
        )
    return candidate_scenes, where_clause, complex_embedding_time, complex_search_time


def _search_sub_queries(
    atomic_items,
    intern_model,
    clips_tbl,
    top_k,
    print_gt_scores,
    ground_truth,
    nprobes,
    where_clause,
    candidate_scenes,
):
    """对每个原子子查询做向量检索，返回 scene_step_results 及各阶段耗时。"""
    total_sub_embedding_time = 0.0
    total_sub_search_time = 0.0
    total_sub_processing_time = 0.0
    scene_step_results = {sc: [] for sc in candidate_scenes}

    for item in atomic_items:
        sub_query = item.get("clause", "")
        if not sub_query:
            for sc in candidate_scenes:
                scene_step_results[sc].append([])
            continue

        t0 = time.time()
        query_embedding = query_to_embedding(intern_model, sub_query)
        total_sub_embedding_time += time.time() - t0

        t0 = time.time()
        # 返回子查询item对应的top_k * 5候选片段
        all_results, _ = search_clips_indexed(
            clips_tbl,
            query_embedding,
            top_k * 5,
            nprobes,
            where_clause,
            True,
        )
        total_sub_search_time += time.time() - t0

        t0 = time.time()
        # 筛选出子查询item匹配片段中，是时间片类型的片段
        if not all_results.empty:
            chunk_results = all_results[
                all_results["clip_id"].str.match(r".*_\d+_\d+_\d+$", na=False)
            ].copy()
            if not chunk_results.empty:
                chunk_results["chunk_t"] = chunk_results["clip_id"].map(extract_chunk_t)
                chunk_results = chunk_results.dropna(subset=["chunk_t"])
        else:
            chunk_results = pd.DataFrame(columns=["scene_name", "chunk_t", "score"])

        for sc in candidate_scenes:
            scene_rows = chunk_results[chunk_results["scene_name"] == sc]
            if scene_rows.empty:
                scene_step_results[sc].append([])
                continue
            # 选出每个时间片 chunk_t 得分最高的那个片段，作为该时间片的代表分数
            dedup_chunks = (
                scene_rows.groupby("chunk_t")["score"]
                .max()
                .reset_index()
                .sort_values(by="score", ascending=False)
                .head(top_k)
            )
            scene_step_results[sc].append(
                [
                    {"t": row["chunk_t"], "score": float(row["score"])}
                    for _, row in dedup_chunks.iterrows()
                ]
            )
        total_sub_processing_time += time.time() - t0

    return (
        scene_step_results,
        total_sub_embedding_time,
        total_sub_search_time,
        total_sub_processing_time,
    )


# scene_subquery_results = {S, T}
#   `S[step][j]`：第 `step` 步第 `j` 好的候选片段的相似度分数
#   `T[step][j]`：对应片段内部的时间位置 `chunk_t`
def _build_score_matrices(scene_step_results, candidate_scenes, top_k):
    """将 scene_step_results 转换为每个场景的 S/T 得分矩阵。"""
    scene_subquery_results = {}
    for sc in candidate_scenes:
        S, T, is_valid = [], [], True
        # 对每一个逻辑时间片（子查询时间片顺序）
        for step_data in scene_step_results[sc]:
            if not step_data:
                is_valid = False
                break
            row_s = [d["score"] for d in step_data]
            row_t = [d["t"] for d in step_data]
            while len(row_s) < top_k:
                row_s.append(-1.0)
                row_t.append(row_t[-1] if row_t else 0.0)
            S.append(row_s)
            T.append(row_t)
        if is_valid:
            scene_subquery_results[sc] = {"S": S, "T": T}
    return scene_subquery_results


def process_sub_queries(
    atomic_items,
    intern_model,
    clips_tbl,
    top_k,
    print_gt_scores,
    ground_truth,
    nprobes,
    where_clause,
    candidate_scenes,
):
    """迭代处理所有原子子查询，返回各场景的 S/T 得分矩阵及耗时统计。"""
    scene_step_results, t_embed, t_search, t_proc = _search_sub_queries(
        atomic_items,
        intern_model,
        clips_tbl,
        top_k,
        print_gt_scores,
        ground_truth,
        nprobes,
        where_clause,
        candidate_scenes,
    )
    scene_subquery_results = _build_score_matrices(
        scene_step_results, candidate_scenes, top_k
    )
    return scene_subquery_results, t_embed, t_search, t_proc


def _bounds_from_member_count(cluster_frame_id, member_count):
    """Recover [start, end] for a contiguous cluster.

    The representative is ``sorted_member_ids[n // 2]``, so the inclusive
    range is ``rep - n // 2`` through ``rep - n // 2 + n - 1``.
    """
    rep = int(cluster_frame_id)
    count = int(member_count)
    if count <= 0:
        return rep, rep
    start = rep - count // 2
    return start, start + count - 1


def _cluster_centers_to_frame_bounds(cluster_centers, bounds_map):
    """Map cluster centers to [frame_start, frame_end] using cluster member ranges."""
    centers = [int(c) for c in cluster_centers if c is not None]
    if not centers:
        return None, None
    earliest_center = min(centers)
    latest_center = max(centers)
    if not bounds_map:
        return earliest_center, latest_center
    start_bounds = bounds_map.get(earliest_center)
    end_bounds = bounds_map.get(latest_center)
    frame_start = start_bounds[0] if start_bounds else earliest_center
    frame_end = end_bounds[1] if end_bounds else latest_center
    return frame_start, frame_end


def _bounds_map_from_member_counts(member_counts):
    return {
        int(cluster_id): _bounds_from_member_count(cluster_id, count)
        for cluster_id, count in (member_counts or {}).items()
    }


def _enrich_candidates_with_frame_bounds(scene_candidates, member_counts_by_scene):
    for scene, candidates in scene_candidates.items():
        bounds_map = _bounds_map_from_member_counts(
            (member_counts_by_scene or {}).get(scene, {})
        )
        for cand in candidates:
            cluster_centers = [
                step[1] for step in cand.get("sequence", []) if step[1] is not None
            ]
            frame_start, frame_end = _cluster_centers_to_frame_bounds(
                cluster_centers, bounds_map
            )
            cand["frame_start"] = frame_start
            cand["frame_end"] = frame_end


# 1 = 原始硬约束：每个子查询只留最高分时间块。2 = 软约束：前两名里存在不倒序组合即可。
TIME_ORDER_TOPK = 2


def _top2_time_options(row_s, row_t):
    """取该子查询分数最高的若干个有效切片的时间下标。"""
    pairs = sorted(
        (
            (float(score), time_idx)
            for score, time_idx in zip(row_s, row_t)
            if score is not None and float(score) != -1.0
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    options = []
    for _, time_idx in pairs:
        if time_idx not in options:
            options.append(time_idx)
        if len(options) == TIME_ORDER_TOPK:
            break
    return options


def _filter_scenes_by_time_order(candidate_scenes, scene_subquery_results, time_steps):
    """硬约束过滤：每个子查询取最高分的两个时间下标，存在非降序选择则保留该场景。"""
    del time_steps  # 时间顺序由子查询排列本身决定，不再按 time_step 分组。
    valid = []
    for sc in candidate_scenes:
        if sc not in scene_subquery_results:
            continue
        S = scene_subquery_results[sc]["S"]
        T = scene_subquery_results[sc]["T"]
        options = [_top2_time_options(row_s, row_t) for row_s, row_t in zip(S, T)]
        if top2_time_order_ok(options):
            valid.append(sc)
    return valid


def _select_tie_aware_topk(sequences, k=5, score_key="normalized_score", eps=1e-9):
    """Select score-tier groups until the next tier would start past the limit.

    While fewer than ``k`` items are selected, the entire next equal-score tier
    is included—even if that pushes the total above ``k``. Once ``k`` or more
    items are selected, remaining tiers are dropped.
    """
    if not sequences or k <= 0:
        return []
    selected = []
    i = 0
    n = len(sequences)
    while i < n and len(selected) < k:
        score = float(sequences[i].get(score_key, 0.0) or 0.0)
        j = i + 1
        while j < n:
            other = float(sequences[j].get(score_key, 0.0) or 0.0)
            if abs(other - score) > eps:
                break
            j += 1
        selected.extend(sequences[i:j])
        i = j
    return selected


def _collect_and_print_top_sequences(scene_candidates, time_steps, dedup_range_tol=60):
    """展平各场景的候选序列，排序后按同分同档打印 Top-k 并返回去重列表。"""
    all_sequences = [
        {
            "scene": scene,
            "raw_score": cand.get("raw_total_score", 0),
            "normalized_score": cand["score"],
            "frame_start": cand.get("frame_start"),
            "frame_end": cand.get("frame_end"),
            "sequence": cand["sequence"],
        }
        for scene, candidates in scene_candidates.items()
        for cand in candidates
    ]
    all_sequences.sort(key=lambda x: x["normalized_score"], reverse=True)

    def _sequence_range(seq):
        if seq.get("frame_start") is not None and seq.get("frame_end") is not None:
            return float(seq["frame_start"]), float(seq["frame_end"])
        ids = [step[1] for step in seq.get("sequence", []) if step[1] is not None]
        if not ids:
            return 0.0, 0.0
        return float(min(ids)), float(max(ids))

    deduped = []
    per_scene_ranges = {}
    for seq in all_sequences:
        scene = seq["scene"]
        seq_min, seq_max = _sequence_range(seq)
        ranges = per_scene_ranges.setdefault(scene, [])
        is_dup = False
        for r_min, r_max in ranges:
            # 起始相差2秒左右视为同一序列
            if (
                abs(seq_min - r_min) <= dedup_range_tol
                and abs(seq_max - r_max) <= dedup_range_tol
            ):
                is_dup = True
                break
        if not is_dup:
            ranges.append((seq_min, seq_max))
            deduped.append(seq)

    top_global = _select_tie_aware_topk(deduped, k=5)

    if top_global:
        print(
            f"✅ Found {len(deduped)} candidate sequence(s) after dedup. "
            f"Showing {len(top_global)} result(s) (tie-aware top-5):"
        )
        for idx, cand in enumerate(top_global, 1):
            print(f"   {idx}. Scene: {cand['scene']}")
            print(
                f"      Sequence Normalized Score: {cand['normalized_score']:.4f} (Raw Total: {cand['raw_score']:.4f})"
            )
            if cand.get("frame_start") is not None and cand.get("frame_end") is not None:
                print(
                    f"      Segment: frame {int(cand['frame_start'])} -> frame {int(cand['frame_end'])}"
                )
            for step, (s, t, col) in enumerate(cand["sequence"]):
                print(
                    f"         Step {step+1} (time_step {time_steps[step]}): score={s:.4f}, cluster_frame_id={t}"
                )
    else:
        print("❌ No candidate scenes generated a valid score sequence.")
    return deduped


def _normalize_vec(vec):
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr


def _load_frame_embeddings(table, clip_id, frame_ids):
    clip_expr = _sql_literal(str(clip_id).replace(".mp4", ""))
    frame_df = (
        table.search()
        .where(f"clip_id = {clip_expr}")
        .select(["frame_id", "frame_embedding"])
        .limit(2_000_000)
        .to_pandas()
    )
    wanted = {int(frame_id) for frame_id in frame_ids}
    embeddings = {}
    if frame_df.empty:
        return embeddings
    for frame_id, embedding in zip(
        frame_df["frame_id"].tolist(), frame_df["frame_embedding"].tolist()
    ):
        frame_id = int(frame_id)
        if frame_id in wanted and frame_id not in embeddings:
            embeddings[frame_id] = _normalize_vec(embedding)
    return embeddings


def _score_scenes_with_encoder(
    scenes, scene_results, valid_clauses, text_embeddings, db, score_backend
):
    table_name = "frames_siglip" if score_backend == "siglip" else "frames"
    embedding_table = db.open_table(table_name)
    clause_vectors = []
    for clause in valid_clauses:
        vector = None if text_embeddings is None else text_embeddings.get(clause)
        clause_vectors.append(None if vector is None else _normalize_vec(vector))
    final_scores = []
    scene_to_meta = {}
    for scene_name in scenes:
        if scene_name not in scene_results:
            continue
        frame_ids, _image_map, _member_counts = scene_results[scene_name]
        embedding_map = _load_frame_embeddings(embedding_table, scene_name, frame_ids)
        scene_to_meta[scene_name] = {
            "cluster_ids": frame_ids,
            "start_idx": len(final_scores),
        }
        for vector in clause_vectors:
            for frame_id in frame_ids:
                stored = embedding_map.get(int(frame_id))
                if vector is None or stored is None:
                    final_scores.append(0.0)
                else:
                    final_scores.append(float(np.dot(stored, vector)))
    return final_scores, scene_to_meta


def evaluate_and_rank_scenes(
    i,
    atomic_items,
    candidate_scenes,
    scene_subquery_results,
    qwen3_model_path,
    qwen3_bridge_script,
    qwen3_python,
    top_k,
    db,
    beam_width=3,
    penalty_weight=0.01,
    epsilon=200.0,
    skip_time_filter=False,
    score_all_frames=False,
    score_backend="vlm",
    text_embeddings=None,
    video_beam_width=None,
    persistent_vlm_bridge=None,
):
    import io
    import json
    import subprocess
    import tempfile
    import os as _os
    from PIL import Image

    if video_beam_width is not None:
        from cheap_beam import evaluate_and_rank_scenes_cheap

        return evaluate_and_rank_scenes_cheap(
            i,
            atomic_items,
            candidate_scenes,
            scene_subquery_results,
            qwen3_model_path,
            qwen3_bridge_script,
            qwen3_python,
            top_k,
            db,
            frame_beam_width=beam_width,
            video_beam_width=video_beam_width,
            penalty_weight=penalty_weight,
            epsilon=epsilon,
            skip_time_filter=skip_time_filter,
            persistent_bridge=persistent_vlm_bridge,
            score_all_frames=score_all_frames,
            score_backend=score_backend,
            text_embeddings=text_embeddings,
        )

    print(f"\n[Intersection Summary (Soft Constraint)] for Query {i+1}")
    timing = {
        "hard_filter": 0.0,
        "cluster_query": 0.0,
        "frames_query": 0.0,
        "image_save": 0.0,
        "prompt_build": 0.0,
        "bridge_input_write": 0.0,
        "bridge_infer": 0.0,
        "bridge_output_read": 0.0,
        "candidate_build": 0.0,
    }
    # ====================硬约束筛选=====================
    time_steps = [
        int(item.get("time_step", 0) or 0)
        for item in atomic_items
        if item.get("clause", "")
    ]
    valid_clauses = [
        item.get("clause", "") for item in atomic_items if item.get("clause", "")
    ]

    hard_filter_start = time.time()
    valid_scenes_hard = _filter_scenes_by_time_order(
        candidate_scenes, scene_subquery_results, time_steps
    )
    timing["hard_filter"] = time.time() - hard_filter_start
    if TIME_ORDER_TOPK <= 1:
        time_order_label = "Top-1 expected"
    else:
        time_order_label = "top-2 non-decreasing"
    print(
        f"Candidates passing hard time order ({time_order_label}): {len(valid_scenes_hard)}"
    )
    print(
        f"Sequence matching: r={top_k} B={beam_width} "
        f"lambda={penalty_weight} epsilon={epsilon} backend={score_backend}"
    )
    # ====================硬约束筛选=====================

    # ====================软约束筛选=====================
    if skip_time_filter:
        sc_list_to_search = list(candidate_scenes)
        print(
            "[Ablation] Skipping time-aware hard filter; "
            f"passing all {len(sc_list_to_search)} coarse candidate scene(s) to scoring."
        )
    else:
        sc_list_to_search = valid_scenes_hard if valid_scenes_hard else candidate_scenes
        if not valid_scenes_hard:
            print(
                "Warning: strict hard filter failed to find any candidate, falling back to all candidate scenes for soft search."
            )
    if score_all_frames:
        print(
            "Ablation mode: no semantic bucketing — scoring ALL frames in each candidate scene."
        )

    if not valid_clauses or not sc_list_to_search:
        print("❌ No candidate scenes to evaluate or no valid clauses.")
        return None, timing

    if score_all_frames:
        print(
            "score_all_frames reads the frames table, which query no longer uses. "
            "Scoring requires frame_cluster_reps."
        )
        return None, timing

    table_names = db.table_names()
    if "frame_cluster_reps" not in table_names:
        print("frame_cluster_reps table is missing. Cannot score scenes.")
        return None, timing
    frame_cluster_reps_tbl = db.open_table("frame_cluster_reps")
    print("Using frame_cluster_reps for bridge inputs and segment bounds.")

    # Build prompts with image paths for the bridge subprocess
    # Save images to temp files so the bridge can read them
    with tempfile.TemporaryDirectory(prefix="qwen3_images_") as tmp_img_dir:
        prompt_build_start = time.time()
        bridge_prompts = []
        scene_to_meta = {}

        # Parallelize per-scene queries (cluster + frame tables) to reduce wall time.
        def _fetch_scene_data(sc_name):
            sc_clean = sc_name.replace(".mp4", "")
            clip_expr = _sql_literal(sc_clean)
            load_images = not (text_embeddings is not None and score_backend != "vlm")
            columns = ["cluster_frame_id", "member_count"] + (
                ["frame_image"] if load_images else []
            )
            t0 = time.time()
            try:
                reps_df = (
                    frame_cluster_reps_tbl.search()
                    .where(f"clip_id = {clip_expr}")
                    .select(columns)
                    .to_pandas()
                )
            except Exception as e:
                print(f"Warning: frame_cluster_reps query failed for {sc_clean}: {e}")
                reps_df = pd.DataFrame()
            reps_elapsed = time.time() - t0
            if reps_df.empty or "cluster_frame_id" not in reps_df.columns:
                print(f"Warning: no frame_cluster_reps rows for {sc_clean}.")
                return sc_name, None, None, None, reps_elapsed, 0.0
            cluster_frame_ids = reps_df["cluster_frame_id"].tolist()
            member_counts = dict(
                zip(reps_df["cluster_frame_id"], reps_df["member_count"])
            )
            frame_image_map = (
                dict(zip(reps_df["cluster_frame_id"], reps_df["frame_image"]))
                if load_images
                else {}
            )
            return (
                sc_name,
                cluster_frame_ids,
                frame_image_map,
                member_counts,
                reps_elapsed,
                0.0,
            )

        scene_results = {}
        scene_workers = min(4, (_os.cpu_count() or 4))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=scene_workers
        ) as executor:
            futures = {
                executor.submit(_fetch_scene_data, sc): sc for sc in sc_list_to_search
            }
            for future in concurrent.futures.as_completed(futures):
                (
                    sc_name,
                    cluster_frame_ids,
                    frame_image_map,
                    member_counts,
                    cluster_elapsed,
                    frames_elapsed,
                ) = future.result()
                timing["cluster_query"] += cluster_elapsed
                timing["frames_query"] += frames_elapsed
                if cluster_frame_ids and frame_image_map is not None:
                    scene_results[sc_name] = (
                        cluster_frame_ids,
                        frame_image_map,
                        member_counts or {},
                    )

        use_encoder = text_embeddings is not None and score_backend != "vlm"
        if use_encoder:
            encoder_final_scores, scene_to_meta = _score_scenes_with_encoder(
                sc_list_to_search,
                scene_results,
                valid_clauses,
                text_embeddings,
                db,
                score_backend,
            )
        else:
            encoder_final_scores = None

        # Revert to simple loop: No filtering, no deduplication
        prompt_scenes = [] if use_encoder else sc_list_to_search
        for sc_name in prompt_scenes:
            if sc_name not in scene_results:
                continue
            cluster_frame_ids, frame_image_map, _member_counts = scene_results[sc_name]
            scene_to_meta[sc_name] = {
                "cluster_ids": cluster_frame_ids,
                "start_idx": len(bridge_prompts),
            }

            for clause in valid_clauses:
                for c_id in cluster_frame_ids:
                    img_bytes = frame_image_map.get(c_id)
                    if not img_bytes:
                        bridge_prompts.append(None)
                        continue

                    # Sequential save (as requested, no parallel/dedup)
                    idx = len(bridge_prompts)
                    img_path = _os.path.join(tmp_img_dir, f"img_{idx:06d}.jpg")
                    t_save0 = time.time()
                    try:
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        img.save(img_path, "JPEG", quality=95)
                        timing["image_save"] += time.time() - t_save0
                        bridge_prompts.append(
                            {
                                "image_path": img_path,
                                "prompt": f"Rate how well the image matches the description '{clause}'. Output exactly one float between 0 and 1.",
                            }
                        )
                    except Exception:
                        bridge_prompts.append(None)

        if use_encoder:
            final_scores = encoder_final_scores
            print(
                f"Total prompts to score ({score_backend} cosine): {len(final_scores)}"
            )
            timing["prompt_build"] = time.time() - prompt_build_start
        else:
            label = "All frames, no clustering" if score_all_frames else "All clusters"
            print(f"Total prompts to score ({label}): {len(bridge_prompts)}")
            timing["prompt_build"] = time.time() - prompt_build_start

            # Write input JSON for the bridge
            bridge_input = {
                "model_path": qwen3_model_path,
                "prompts": bridge_prompts,
            }

            t0 = time.time()
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="qwen3_input_"
            ) as f_in:
                json.dump(bridge_input, f_in)
                bridge_input_path = f_in.name
            timing["bridge_input_write"] = time.time() - t0

            bridge_output_path = bridge_input_path.replace("_input_", "_output_")

            try:
                # Free CUDA memory used by InternVideo2 before launching SGLang bridge
                import torch

                torch.cuda.empty_cache()
                print("Cleared CUDA cache before bridge call.")

                print(
                    f"Calling Qwen3-VL bridge ({len(bridge_prompts)} prompts) via {qwen3_python}..."
                )
                bridge_env = os.environ.copy()
                driver_bin = os.path.dirname(sys.executable)
                bridge_env["PATH"] = (
                    driver_bin + os.pathsep + bridge_env.get("PATH", "")
                )
                t0 = time.time()
                result = subprocess.run(
                    [
                        qwen3_python,
                        qwen3_bridge_script,
                        bridge_input_path,
                        bridge_output_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=7200 if score_all_frames else 600,
                    env=bridge_env,
                )
                timing["bridge_infer"] = time.time() - t0
                if result.returncode != 0:
                    print(f"❌ Bridge subprocess failed (rc={result.returncode}):")
                    print("--- Bridge STDERR ---")
                    print(result.stderr)
                    print("--- Bridge STDOUT ---")
                    print(result.stdout)
                    print("---------------------")
                    final_scores = [0.0] * len(bridge_prompts)
                else:
                    # NEW: Always print stderr to see DEBUG info
                    if result.stderr:
                        print("--- Bridge Logs ---")
                        print(result.stderr)
                        print("-------------------")
                    t0 = time.time()
                    with open(bridge_output_path, "r", encoding="utf-8") as f:
                        output_data = json.load(f)
                    timing["bridge_output_read"] = time.time() - t0
                    final_scores = output_data.get("scores", [0.0] * len(bridge_prompts))
                    if len(final_scores) < len(bridge_prompts):
                        final_scores += [0.0] * (len(bridge_prompts) - len(final_scores))
                    print(
                        f"Bridge returned {len(final_scores)} scores. First few: {final_scores[:5]}"
                    )
            except subprocess.TimeoutExpired:
                print("Bridge subprocess timed out after 600s.")
                final_scores = [0.0] * len(bridge_prompts)
            except Exception as e:
                print(f"Bridge subprocess error: {e}")
                final_scores = [0.0] * len(bridge_prompts)
            finally:
                # Clean up temp files
                for p in (bridge_input_path, bridge_output_path):
                    if _os.path.exists(p):
                        _os.unlink(p)

    candidate_build_start = time.time()
    # Build scene candidates from scores
    scene_candidates = {}
    for sc_name, meta in scene_to_meta.items():
        base_idx = meta["start_idx"]
        cluster_ids = meta["cluster_ids"]

        S_new = []
        T_new = []

        for step_idx in range(len(valid_clauses)):
            row_s_raw = []
            row_t_raw = []
            for i_c, c_id in enumerate(cluster_ids):
                score = final_scores[base_idx + step_idx * len(cluster_ids) + i_c]
                row_s_raw.append(score)
                row_t_raw.append(c_id)

            sorted_pairs = sorted(
                zip(row_s_raw, row_t_raw), key=lambda x: x[0], reverse=True
            )[:top_k]
            row_s = [float(p[0]) for p in sorted_pairs]
            row_t = [float(p[1]) for p in sorted_pairs]
            while len(row_s) < top_k:
                row_s.append(-1.0)
                row_t.append(row_t[-1] if row_t else 0.0)

            S_new.append(row_s)
            T_new.append(row_t)

        candidates = find_candidate_sequences(
            S_new,
            T_new,
            penalty_weight=penalty_weight,
            top_k=beam_width,
            epsilon=epsilon,
        )
        if candidates:
            scene_candidates[sc_name] = candidates

    if score_all_frames:
        for _scene_name, candidates in scene_candidates.items():
            for cand in candidates:
                matched_ids = [
                    int(step[1])
                    for step in cand.get("sequence", [])
                    if step[1] is not None
                ]
                if matched_ids:
                    cand["frame_start"] = min(matched_ids)
                    cand["frame_end"] = max(matched_ids)
                else:
                    cand["frame_start"] = None
                    cand["frame_end"] = None
    else:
        _enrich_candidates_with_frame_bounds(
            scene_candidates,
            {
                scene_name: counts
                for scene_name, (_ids, _images, counts) in scene_results.items()
            },
        )
    all_sequences = _collect_and_print_top_sequences(scene_candidates, time_steps)
    timing["candidate_build"] = time.time() - candidate_build_start

    return all_sequences, timing


def process_query(
    i,
    query,
    atomic_items,
    intern_model,
    clips_tbl,
    qwen3_model_path,
    qwen3_bridge_script,
    qwen3_python,
    ground_truth,
    nprobes,
    top_k,
    print_gt_scores,
    db,
    candidate_log_path=None,
    query_split_time=0.0,
    coarse_limit=20,
    beam_width=3,
    penalty_weight=0.01,
    epsilon=200.0,
    skip_time_filter=False,
    score_all_frames=False,
    score_backend="vlm",
    text_embeddings=None,
    video_beam_width=None,
    persistent_vlm_bridge=None,
):
    """单条查询的端到端流程：粗检索 → 子查询对齐 → 序列排序。"""
    query_start_time = time.time()

    print("=" * 60)
    print(f"Query {i+1}: '{query}'")
    print("\n[LLM 扁平化扩写打标结果]:")
    for item in atomic_items:
        conf = item.get("confidence", "N/A")
        remark = f" ({item.get('remarks')})" if item.get("remarks") else ""
        print(
            f"  Step {item.get('time_step')} [Conf: {conf}]: {item.get('clause')}{remark}"
        )

    (
        candidate_scenes,
        where_clause,
        complex_embedding_time,
        complex_search_time,
        coarse_top20_clips,
    ) = fetch_candidate_scenes(
        query, intern_model, clips_tbl, nprobes, True, coarse_limit=coarse_limit
    )
    if not candidate_scenes:
        _append_query_candidate_log(
            candidate_log_path,
            {
                "query_index": i,
                "query_number": i + 1,
                "query": query,
                "coarse_k": coarse_limit,
                "candidate_scenes": [],
                "coarse_top20_clips": coarse_top20_clips,
                "hard_constraint_passed_scenes": [],
                "hard_constraint_candidates": [],
                "time_aware_filtering_ablated": skip_time_filter,
                "scenes_sent_to_scoring": [],
            },
        )
        return []

    (
        scene_subquery_best_t,
        total_sub_embedding_time,
        total_sub_search_time,
        total_sub_processing_time,
    ) = process_sub_queries(
        atomic_items,
        intern_model,
        clips_tbl,
        top_k,
        print_gt_scores,
        ground_truth,
        nprobes,
        where_clause,
        candidate_scenes,
    )

    valid_scenes_hard, hard_candidates = _build_hard_constraint_candidate_log(
        atomic_items, candidate_scenes, scene_subquery_best_t, top_k
    )
    _append_query_candidate_log(
        candidate_log_path,
        {
            "query_index": i,
            "query_number": i + 1,
            "query": query,
            "coarse_k": coarse_limit,
            "n_unique_scenes": len(candidate_scenes),
            "candidate_scenes": list(candidate_scenes),
            "coarse_top20_clips": coarse_top20_clips,
            "hard_constraint_passed_scenes": valid_scenes_hard,
            "hard_constraint_candidates": hard_candidates,
            "time_aware_filtering_ablated": skip_time_filter,
            "scenes_sent_to_scoring": (
                list(candidate_scenes) if skip_time_filter else list(valid_scenes_hard)
            ),
        },
    )

    intersection_start = time.time()

    import torch as _torch

    moved_intern_to_cpu = persistent_vlm_bridge is None
    if moved_intern_to_cpu:
        # One-shot bridge mode needs the encoder memory while Qwen is loading.
        intern_model.cpu()
        _torch.cuda.empty_cache()

    all_sequences, intersection_timing = evaluate_and_rank_scenes(
        i,
        atomic_items,
        candidate_scenes,
        scene_subquery_best_t,
        qwen3_model_path,
        qwen3_bridge_script,
        qwen3_python,
        top_k,
        db,
        beam_width=beam_width,
        penalty_weight=penalty_weight,
        epsilon=epsilon,
        skip_time_filter=skip_time_filter,
        score_all_frames=score_all_frames,
        score_backend=score_backend,
        text_embeddings=text_embeddings,
        video_beam_width=video_beam_width,
        persistent_vlm_bridge=persistent_vlm_bridge,
    )

    if moved_intern_to_cpu:
        # Move InternVideo2 back to GPU for next query.
        intern_model.cuda()
        _torch.cuda.empty_cache()
    intersection_time = time.time() - intersection_start
    retrieval_time = time.time() - query_start_time
    total_time = retrieval_time + float(query_split_time or 0.0)

    print(f"\n--- Execution Time Breakdown ---")
    print(f"0. Query Decomposition (LLM): {float(query_split_time or 0.0):.4f}s")
    print(f"1. Complex Query Embedding:   {complex_embedding_time:.4f}s")
    print(f"2. Complex Query Search:      {complex_search_time:.4f}s")
    print(f"3. Sub-queries Embedding:     {total_sub_embedding_time:.4f}s")
    print(f"4. Sub-queries Search:        {total_sub_search_time:.4f}s")
    print(f"5. Sub-queries Processing:    {total_sub_processing_time:.4f}s")
    print(f"6. Intersection & Ranking:    {intersection_time:.4f}s")
    if intersection_timing:
        print(
            f"   6.1 Hard time-order filter: {intersection_timing['hard_filter']:.4f}s"
        )
        print(
            f"   6.2 Build bridge inputs:    {intersection_timing['prompt_build']:.4f}s"
        )
        print(
            f"      6.2.1 Cluster query:     {intersection_timing['cluster_query']:.4f}s"
        )
        print(
            f"      6.2.2 Frame query:       {intersection_timing['frames_query']:.4f}s"
        )
        print(
            f"      6.2.3 Image decode/save: {intersection_timing['image_save']:.4f}s"
        )
        print(
            f"   6.3 Write bridge input:     {intersection_timing['bridge_input_write']:.4f}s"
        )
        print(
            f"   6.4 Bridge inference:       {intersection_timing['bridge_infer']:.4f}s"
        )
        print(
            f"   6.5 Read bridge output:     {intersection_timing['bridge_output_read']:.4f}s"
        )
        print(
            f"   6.6 Build candidates:       {intersection_timing['candidate_build']:.4f}s"
        )
    print(f"--------------------------------")
    print(f"Retrieval Processing Time:    {retrieval_time:.4f}s")
    print(f"Total Query Processing Time:  {total_time:.4f}s")

    return all_sequences
