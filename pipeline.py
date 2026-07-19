import concurrent.futures
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from search import (
    extract_chunk_t,
    find_candidate_sequences,
    search_clips_indexed,
    time_order_ok,
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


def fetch_candidate_scenes(query, intern_model, clips_tbl, nprobes, return_debug=False):
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
        .limit(20)
        .to_pandas()
    )
    complex_search_time = time.time() - t1
    print(
        f"[{complex_search_time:.4f}s] Fetched top 20 candidate clips using the original complex query."
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
            _df_to_light_records(df_candidates, limit=20),
        )
    return candidate_scenes, where_clause, complex_embedding_time, complex_search_time


def _search_sub_queries(
    atomic_items,
    intern_model,
    clips_tbl,
    top_k,
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
        nprobes,
        where_clause,
        candidate_scenes,
    )
    scene_subquery_results = _build_score_matrices(
        scene_step_results, candidate_scenes, top_k
    )
    return scene_subquery_results, t_embed, t_search, t_proc


def _load_cluster_frame_bounds(frames_clusters_tbl, clip_id):
    """Return {cluster_frame_id: (min_frame_id, max_frame_id)} for a scene clip."""
    if frames_clusters_tbl is None:
        return {}
    clip_expr = _sql_literal(str(clip_id).replace(".mp4", ""))
    try:
        df = (
            frames_clusters_tbl.search()
            .where(f"clip_id = {clip_expr}")
            .select(["cluster_frame_id", "frame_id"])
            .to_pandas()
        )
    except Exception:
        return {}
    if df.empty or "cluster_frame_id" not in df.columns or "frame_id" not in df.columns:
        return {}
    grouped = df.groupby("cluster_frame_id")["frame_id"]
    return {
        int(cluster_id): (int(group.min()), int(group.max()))
        for cluster_id, group in grouped
    }


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


def _enrich_candidates_with_frame_bounds(scene_candidates, frames_clusters_tbl):
    bounds_cache = {}
    for scene, candidates in scene_candidates.items():
        clip_id = scene.replace(".mp4", "")
        if clip_id not in bounds_cache:
            bounds_cache[clip_id] = _load_cluster_frame_bounds(
                frames_clusters_tbl, clip_id
            )
        bounds_map = bounds_cache[clip_id]
        for cand in candidates:
            cluster_centers = [
                step[1] for step in cand.get("sequence", []) if step[1] is not None
            ]
            frame_start, frame_end = _cluster_centers_to_frame_bounds(
                cluster_centers, bounds_map
            )
            cand["frame_start"] = frame_start
            cand["frame_end"] = frame_end


def _filter_scenes_by_time_order(candidate_scenes, scene_subquery_results, time_steps):
    """硬约束过滤：保留 Top-1 子查询结果满足时间顺序的场景。"""
    valid = []
    for sc in candidate_scenes:
        if sc not in scene_subquery_results:
            continue
        S = scene_subquery_results[sc]["S"]
        T = scene_subquery_results[sc]["T"]
        entries = [
            {"time_step": time_steps[i], "best_t": T[i][0] if S[i][0] != -1.0 else None}
            for i in range(len(S))
        ]
        if time_order_ok(entries):
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


def evaluate_and_rank_scenes(
    i,
    atomic_items,
    candidate_scenes,
    scene_subquery_results,
    frames_tbl,
    qwen3_model_path,
    qwen3_bridge_script,
    qwen3_python,
    top_k,
    db,
):
    import io
    import json
    import subprocess
    import tempfile
    import os as _os
    from PIL import Image

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
    print(
        f"Candidates passing hard time order explicitly (Top-1 expected): {len(valid_scenes_hard)}"
    )
    # ====================硬约束筛选=====================

    # ====================软约束筛选=====================
    sc_list_to_search = valid_scenes_hard if valid_scenes_hard else candidate_scenes
    if not valid_scenes_hard:
        print(
            "Warning: strict hard filter failed to find any candidate, falling back to all candidate scenes for soft search."
        )

    if not valid_clauses or not sc_list_to_search:
        print("❌ No candidate scenes to evaluate or no valid clauses.")
        return None, timing

    table_names = db.table_names()
    frame_cluster_reps_tbl = (
        db.open_table("frame_cluster_reps")
        if "frame_cluster_reps" in table_names
        else None
    )
    frames_clusters_tbl = (
        db.open_table("frame_clusters")
        if "frame_clusters" in table_names
        else frames_tbl
    )
    if frame_cluster_reps_tbl is not None:
        print("Using precomputed frame_cluster_reps table for bridge inputs.")

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

            if frame_cluster_reps_tbl is not None:
                t0 = time.time()
                try:
                    reps_df = (
                        frame_cluster_reps_tbl.search()
                        .where(f"clip_id = {clip_expr}")
                        .select(["cluster_frame_id", "frame_image"])
                        .to_pandas()
                    )
                except Exception as e:
                    print(
                        f"Warning: frame_cluster_reps query failed for {sc_clean}: {e}"
                    )
                    reps_df = pd.DataFrame()
                reps_elapsed = time.time() - t0
                if not reps_df.empty:
                    cluster_frame_ids = reps_df["cluster_frame_id"].tolist()
                    frame_image_map = dict(
                        zip(reps_df["cluster_frame_id"], reps_df["frame_image"])
                    )
                    return (
                        sc_name,
                        cluster_frame_ids,
                        frame_image_map,
                        reps_elapsed,
                        0.0,
                    )

            t0 = time.time()
            try:
                df_clusters = (
                    frames_clusters_tbl.search()
                    .where(f"clip_id = {clip_expr}")
                    .select(["cluster_frame_id"])
                    .to_pandas()
                )
            except Exception:
                df_clusters = (
                    frames_clusters_tbl.search()
                    .where(f"clip_id LIKE '{sc_clean}%'")
                    .to_pandas()
                )
            cluster_elapsed = time.time() - t0
            if df_clusters.empty or "cluster_frame_id" not in df_clusters.columns:
                return sc_name, None, None, cluster_elapsed, 0.0
            unique_cluster_frames = df_clusters.drop_duplicates(
                subset=["cluster_frame_id"]
            ).copy()
            if unique_cluster_frames.empty:
                return sc_name, None, None, cluster_elapsed, 0.0

            cluster_frame_ids = unique_cluster_frames["cluster_frame_id"].tolist()

            t0 = time.time()
            frames_df = (
                frames_tbl.search()
                .where(f"clip_id = {clip_expr}")
                .select(["frame_id", "frame_image"])
                .to_pandas()
            )
            frames_elapsed = time.time() - t0
            frame_image_map = dict(zip(frames_df["frame_id"], frames_df["frame_image"]))
            return (
                sc_name,
                cluster_frame_ids,
                frame_image_map,
                cluster_elapsed,
                frames_elapsed,
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
                    cluster_elapsed,
                    frames_elapsed,
                ) = future.result()
                timing["cluster_query"] += cluster_elapsed
                timing["frames_query"] += frames_elapsed
                if cluster_frame_ids and frame_image_map is not None:
                    scene_results[sc_name] = (
                        cluster_frame_ids,
                        frame_image_map,
                    )

        # Revert to simple loop: No filtering, no deduplication
        for sc_name in sc_list_to_search:
            if sc_name not in scene_results:
                continue
            cluster_frame_ids, frame_image_map = scene_results[sc_name]
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

        print(f"Total prompts to score (All clusters): {len(bridge_prompts)}")
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
                timeout=600,
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
            S_new, T_new, penalty_weight=0.01, top_k=3, epsilon=200
        )
        if candidates:
            scene_candidates[sc_name] = candidates

    _enrich_candidates_with_frame_bounds(scene_candidates, frames_clusters_tbl)
    all_sequences = _collect_and_print_top_sequences(scene_candidates, time_steps)
    timing["candidate_build"] = time.time() - candidate_build_start

    return all_sequences, timing


def process_query(
    i,
    query,
    atomic_items,
    intern_model,
    clips_tbl,
    frames_tbl,
    qwen3_model_path,
    qwen3_bridge_script,
    qwen3_python,
    nprobes,
    top_k,
    db,
    candidate_log_path=None,
    query_split_time=0.0,
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
    ) = fetch_candidate_scenes(query, intern_model, clips_tbl, nprobes, True)
    if not candidate_scenes:
        _append_query_candidate_log(
            candidate_log_path,
            {
                "query_index": i,
                "query_number": i + 1,
                "query": query,
                "coarse_top20_clips": coarse_top20_clips,
                "hard_constraint_passed_scenes": [],
                "hard_constraint_candidates": [],
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
            "coarse_top20_clips": coarse_top20_clips,
            "hard_constraint_passed_scenes": valid_scenes_hard,
            "hard_constraint_candidates": hard_candidates,
        },
    )

    intersection_start = time.time()

    # Move InternVideo2 to CPU to free GPU memory for Qwen3-VL bridge
    intern_model_cpu = intern_model.cpu()
    import torch as _torch

    _torch.cuda.empty_cache()

    all_sequences, intersection_timing = evaluate_and_rank_scenes(
        i,
        atomic_items,
        candidate_scenes,
        scene_subquery_best_t,
        frames_tbl,
        qwen3_model_path,
        qwen3_bridge_script,
        qwen3_python,
        top_k,
        db,
    )

    # Move InternVideo2 back to GPU for next query
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
