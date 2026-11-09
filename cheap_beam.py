"""Incremental video-aware VLM beam search."""

from __future__ import annotations

import concurrent.futures
import io
import json
import os
import select
import subprocess
import sys
import time
from collections import OrderedDict

import pandas as pd
from PIL import Image

import pipeline as base


def _quantize_clause_score(score: float) -> float:
    """Snap a clause score to ``CHEAP_BEAM_SCORE_QUANTUM`` when that env var is set.

    Full scoring often left the true span tied with a one-tick-higher fragment.
    Quantizing to one decimal recreates that tie before the beam ranks paths.
    """
    quantum = os.environ.get("CHEAP_BEAM_SCORE_QUANTUM", "").strip()
    if not quantum:
        return float(score)
    from decimal import Decimal, ROUND_HALF_UP

    return float(
        Decimal(f"{float(score):.6f}").quantize(Decimal(quantum), rounding=ROUND_HALF_UP)
    )


def scene_to_video_id(scene_name: str) -> str:
    return str(scene_name).replace(".mp4", "").split("-Scene-")[0]


class PersistentVLMBridge:
    """Keep the bridge model loaded while logical-time-step batches are scored."""

    def __init__(
        self,
        python_path: str,
        script_path: str,
        model_path: str,
        timeout_seconds: float = 7200,
    ):
        env = os.environ.copy()
        driver_bin = os.path.dirname(sys.executable)
        env["PATH"] = driver_bin + os.pathsep + env.get("PATH", "")
        self.timeout_seconds = timeout_seconds
        self.next_request_id = 1
        self.process = subprocess.Popen(
            [python_path, script_path, "--stream", model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

    def score(self, prompts):
        if self.process.poll() is not None:
            raise RuntimeError(
                f"Persistent VLM bridge exited with code {self.process.returncode}"
            )
        request_id = self.next_request_id
        self.next_request_id += 1
        payload = {
            "request_id": request_id,
            "prompts": prompts,
        }
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

        deadline = time.monotonic() + self.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Persistent VLM bridge timed out on request {request_id}"
                )
            ready, _, _ = select.select(
                [self.process.stdout], [], [], min(remaining, 5.0)
            )
            if not ready:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "Persistent VLM bridge exited before returning scores "
                        f"(code {self.process.returncode})"
                    )
                continue
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Persistent VLM bridge closed stdout")
            if not line.startswith("BRIDGE_RESULT\t"):
                print(f"[Bridge stdout] {line.rstrip()}")
                continue
            response = json.loads(line.split("\t", 1)[1])
            if response.get("request_id") != request_id:
                continue
            scores = response.get("scores", [])
            if len(scores) < len(prompts):
                scores.extend([0.0] * (len(prompts) - len(scores)))
            return scores

    def close(self):
        if self.process.poll() is not None:
            return
        try:
            self.process.stdin.write(json.dumps({"command": "close"}) + "\n")
            self.process.stdin.flush()
            self.process.wait(timeout=120)
        except Exception:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=30)


def _logical_groups(atomic_items):
    groups = OrderedDict()
    for item in atomic_items:
        clause = item.get("clause", "")
        if not clause:
            continue
        time_step = int(item.get("time_step", 0) or 0)
        groups.setdefault(time_step, []).append(clause)
    return list(groups.items())


def _fetch_scene_results(
    scenes,
    frame_cluster_reps_tbl,
    frames_clusters_tbl,
    frames_tbl,
    timing,
    score_all_frames=False,
    load_images=True,
):
    def fetch_one(scene_name):
        scene_clean = scene_name.replace(".mp4", "")
        clip_expr = base._sql_literal(scene_clean)
        if score_all_frames:
            started = time.time()
            columns = ["frame_id"] + (["frame_image"] if load_images else [])
            try:
                frames_df = (
                    frames_tbl.search()
                    .where(f"clip_id = {clip_expr}")
                    .select(columns)
                    .limit(2_000_000)
                    .to_pandas()
                )
            except Exception as exc:
                print(f"Warning: frames query failed for {scene_clean}: {exc}")
                frames_df = pd.DataFrame()
            elapsed = time.time() - started
            if frames_df.empty or "frame_id" not in frames_df.columns:
                return scene_name, None, None, 0.0, elapsed
            frames_df = frames_df.drop_duplicates(subset=["frame_id"]).sort_values(
                "frame_id"
            )
            frame_ids = frames_df["frame_id"].tolist()
            image_map = (
                dict(zip(frames_df["frame_id"], frames_df["frame_image"]))
                if load_images
                else {}
            )
            return scene_name, frame_ids, image_map, 0.0, elapsed
        if not load_images:
            id_table = frame_cluster_reps_tbl or frames_clusters_tbl
            started = time.time()
            try:
                id_df = (
                    id_table.search()
                    .where(f"clip_id = {clip_expr}")
                    .select(["cluster_frame_id"])
                    .limit(2_000_000)
                    .to_pandas()
                )
            except Exception as exc:
                print(f"Warning: cluster id query failed for {scene_clean}: {exc}")
                id_df = pd.DataFrame()
            elapsed = time.time() - started
            if id_df.empty or "cluster_frame_id" not in id_df.columns:
                return scene_name, None, None, elapsed, 0.0
            frame_ids = id_df["cluster_frame_id"].drop_duplicates().tolist()
            return scene_name, frame_ids, {}, elapsed, 0.0
        if frame_cluster_reps_tbl is not None:
            started = time.time()
            try:
                reps_df = (
                    frame_cluster_reps_tbl.search()
                    .where(f"clip_id = {clip_expr}")
                    .select(["cluster_frame_id", "frame_image"])
                    .to_pandas()
                )
            except Exception as exc:
                print(
                    f"Warning: frame_cluster_reps query failed for "
                    f"{scene_clean}: {exc}"
                )
                reps_df = pd.DataFrame()
            elapsed = time.time() - started
            if not reps_df.empty:
                cluster_ids = reps_df["cluster_frame_id"].tolist()
                image_map = dict(
                    zip(reps_df["cluster_frame_id"], reps_df["frame_image"])
                )
                return scene_name, cluster_ids, image_map, elapsed, 0.0

        started = time.time()
        try:
            cluster_df = (
                frames_clusters_tbl.search()
                .where(f"clip_id = {clip_expr}")
                .select(["cluster_frame_id"])
                .to_pandas()
            )
        except Exception:
            cluster_df = (
                frames_clusters_tbl.search()
                .where(f"clip_id LIKE '{scene_clean}%'")
                .to_pandas()
            )
        cluster_elapsed = time.time() - started
        if cluster_df.empty or "cluster_frame_id" not in cluster_df.columns:
            return scene_name, None, None, cluster_elapsed, 0.0
        cluster_ids = (
            cluster_df["cluster_frame_id"].drop_duplicates().sort_values().tolist()
        )

        started = time.time()
        frames_df = (
            frames_tbl.search()
            .where(f"clip_id = {clip_expr}")
            .select(["frame_id", "frame_image"])
            .to_pandas()
        )
        frames_elapsed = time.time() - started
        image_map = dict(zip(frames_df["frame_id"], frames_df["frame_image"]))
        return (
            scene_name,
            cluster_ids,
            image_map,
            cluster_elapsed,
            frames_elapsed,
        )

    results = {}
    workers = min(4, os.cpu_count() or 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_one, scene): scene for scene in scenes}
        for future in concurrent.futures.as_completed(futures):
            scene_name, cluster_ids, image_map, cluster_time, frames_time = (
                future.result()
            )
            timing["cluster_query"] += cluster_time
            timing["frames_query"] += frames_time
            if cluster_ids and image_map is not None:
                results[scene_name] = (cluster_ids, image_map)
    return results


def _save_cluster_images(scene_results, directory, timing):
    paths = {}
    for scene_name, (cluster_ids, image_map) in scene_results.items():
        for cluster_id in cluster_ids:
            image_bytes = image_map.get(cluster_id)
            if not image_bytes:
                paths[(scene_name, cluster_id)] = None
                continue
            path = os.path.join(directory, f"img_{len(paths):06d}.jpg")
            started = time.time()
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image.save(path, "JPEG", quality=95)
                paths[(scene_name, cluster_id)] = path
            except Exception:
                paths[(scene_name, cluster_id)] = None
            timing["image_save"] += time.time() - started
    return paths


def _score_group(
    bridge,
    clauses,
    scenes,
    scene_results,
    image_paths,
    top_k,
    timing,
):
    prompts = []
    prompt_keys = []
    for clause_index, clause in enumerate(clauses):
        for scene_name in scenes:
            if scene_name not in scene_results:
                continue
            cluster_ids, _ = scene_results[scene_name]
            for cluster_id in cluster_ids:
                image_path = image_paths.get((scene_name, cluster_id))
                prompt_keys.append((clause_index, scene_name, cluster_id))
                prompts.append(
                    None
                    if image_path is None
                    else {
                        "image_path": image_path,
                        "prompt": (
                            "Rate how well the image matches the description "
                            f"'{clause}'. Output exactly one float between 0 and 1."
                        ),
                    }
                )

    print(
        f"Cheap beam batch: {len(clauses)} clause(s), {len(scenes)} scene(s), "
        f"{len(prompts)} VLM prompt(s)"
    )
    started = time.time()
    scores = bridge.score(prompts)
    timing["bridge_infer"] += time.time() - started
    timing["vlm_prompts"] += len(prompts)

    by_clause_scene = {}
    for (clause_index, scene_name, cluster_id), score in zip(prompt_keys, scores):
        by_clause_scene.setdefault((clause_index, scene_name), []).append(
            (float(score), float(cluster_id), scene_name)
        )

    rows = []
    for clause_index in range(len(clauses)):
        by_scene = {}
        for scene_name in scenes:
            candidates = sorted(
                by_clause_scene.get((clause_index, scene_name), []),
                key=lambda item: item[0],
                reverse=True,
            )[:top_k]
            by_scene[scene_name] = [
                (_quantize_clause_score(score), timestamp, scene)
                for score, timestamp, scene in candidates
            ]
        rows.append(by_scene)
    return rows


def _cluster_embedding_table(score_backend):
    if score_backend == "siglip":
        return "frames_siglip_clusters"
    if score_backend == "fgclip":
        return "frame_clusters"
    raise ValueError(f"Unsupported encoder backend: {score_backend}")


def _load_cluster_centroids(table, clip_id):
    """Load one stored centroid per cluster, not every member frame."""
    clip_expr = base._sql_literal(str(clip_id).replace(".mp4", ""))
    frame_df = (
        table.search()
        .where(f"clip_id = {clip_expr} AND frame_id = cluster_frame_id")
        .select(["cluster_frame_id", "frame_cluster_centroid_embedding"])
        .limit(2_000_000)
        .to_pandas()
    )
    embeddings = {}
    if frame_df.empty:
        return embeddings
    for cluster_id, embedding in zip(
        frame_df["cluster_frame_id"].tolist(),
        frame_df["frame_cluster_centroid_embedding"].tolist(),
    ):
        cluster_id = int(cluster_id)
        if cluster_id not in embeddings:
            embeddings[cluster_id] = base._normalize_vec(embedding)
    return embeddings


def _score_group_encoder(
    clauses,
    scenes,
    scene_results,
    top_k,
    text_embeddings,
    db,
    score_backend,
    timing,
    embedding_cache,
):
    table_name = _cluster_embedding_table(score_backend)
    embedding_table = embedding_cache.get("_table")
    if embedding_table is None:
        embedding_table = db.open_table(table_name)
        embedding_cache["_table"] = embedding_table
    clause_vectors = [
        None
        if text_embeddings.get(clause) is None
        else base._normalize_vec(text_embeddings[clause])
        for clause in clauses
    ]
    comparisons = 0
    started = time.time()
    for scene_name in scenes:
        if scene_name not in scene_results or scene_name in embedding_cache:
            continue
        embedding_cache[scene_name] = _load_cluster_centroids(
            embedding_table, scene_name
        )
    rows = []
    for clause_index, _clause in enumerate(clauses):
        vector = clause_vectors[clause_index]
        by_scene = {}
        for scene_name in scenes:
            if scene_name not in scene_results:
                continue
            embeddings = embedding_cache.get(scene_name, {})
            candidates = []
            for cluster_id, stored in embeddings.items():
                comparisons += 1
                score = 0.0 if vector is None else float(stored.dot(vector))
                candidates.append((score, float(cluster_id), scene_name))
            by_scene[scene_name] = [
                (_quantize_clause_score(score), timestamp, scene)
                for score, timestamp, scene in sorted(
                    candidates, key=lambda item: item[0], reverse=True
                )[:top_k]
            ]
        rows.append(by_scene)
    timing["bridge_infer"] += time.time() - started
    timing["vlm_prompts"] += comparisons
    print(
        f"Cheap beam encoder batch ({table_name}): {len(clauses)} clause(s), "
        f"{len(scenes)} scene(s), {comparisons} cosine comparison(s)"
    )
    return rows


def _advance_beam(beam, candidates, frame_beam_width, penalty_weight, epsilon):
    if not candidates:
        return []
    if not beam:
        initialized = [
            (
                semantic_score,
                [(semantic_score, timestamp, column)],
                [scene_name],
            )
            for column, (semantic_score, timestamp, scene_name) in enumerate(candidates)
        ]
        return sorted(initialized, key=lambda state: state[0], reverse=True)[
            :frame_beam_width
        ]

    expanded = []
    for total_score, sequence, sequence_scenes in beam:
        previous_time = sequence[-1][1]
        for column, (semantic_score, timestamp, scene_name) in enumerate(candidates):
            violation = max(0.0, (previous_time - timestamp) - epsilon)
            expanded.append(
                (
                    total_score + semantic_score - penalty_weight * violation,
                    sequence + [(semantic_score, timestamp, column)],
                    sequence_scenes + [scene_name],
                )
            )
    return sorted(expanded, key=lambda state: state[0], reverse=True)[
        :frame_beam_width
    ]


def _advance_scenes(
    beams_by_scene,
    rows,
    frame_beam_width,
    penalty_weight,
    epsilon,
):
    scenes = set(beams_by_scene)
    if not scenes:
        scenes = {scene_name for row in rows for scene_name in row}
    updated = {}
    for scene_name in scenes:
        beam = beams_by_scene.get(scene_name, [])
        for row in rows:
            beam = _advance_beam(
                beam,
                row.get(scene_name, []),
                frame_beam_width,
                penalty_weight,
                epsilon,
            )
            if not beam:
                break
        if beam:
            updated[scene_name] = beam
    return updated


def _select_tie_aware_videos(beams_by_scene, video_beam_width, eps=1e-9):
    video_scores = {}
    for scene_name, beam in beams_by_scene.items():
        if not beam:
            continue
        video_id = scene_to_video_id(scene_name)
        video_scores[video_id] = max(
            video_scores.get(video_id, float("-inf")), beam[0][0]
        )
    ranked = sorted(
        video_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    selected = []
    index = 0
    while index < len(ranked) and len(selected) < video_beam_width:
        tier_score = ranked[index][1]
        tier_end = index + 1
        while (
            tier_end < len(ranked)
            and abs(ranked[tier_end][1] - tier_score) <= eps
        ):
            tier_end += 1
        selected.extend(video_id for video_id, _score in ranked[index:tier_end])
        index = tier_end
    return set(selected)


def _build_scene_candidates(
    beams_by_scene,
    clause_count,
    frames_clusters_tbl,
    score_all_frames=False,
):
    scene_candidates = {}
    for scene_name, beam in beams_by_scene.items():
        bounds_map = {}
        if not score_all_frames:
            bounds_map = base._load_cluster_frame_bounds(
                frames_clusters_tbl, scene_name.replace(".mp4", "")
            )
        candidates = []
        for total_score, sequence, _sequence_scenes in beam:
            centers = [step[1] for step in sequence if step[1] is not None]
            if score_all_frames:
                frame_start = min(centers) if centers else None
                frame_end = max(centers) if centers else None
            else:
                frame_start, frame_end = base._cluster_centers_to_frame_bounds(
                    centers, bounds_map
                )
            candidates.append(
                {
                    "score": total_score / clause_count,
                    "raw_total_score": total_score,
                    "sequence": sequence,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                }
            )
        scene_candidates[scene_name] = candidates
    return scene_candidates


def evaluate_and_rank_scenes_cheap(
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
    frame_beam_width=3,
    video_beam_width=5,
    penalty_weight=0.01,
    epsilon=200.0,
    skip_time_filter=False,
    persistent_bridge=None,
    score_all_frames=False,
    score_backend="vlm",
    text_embeddings=None,
):
    """Gate videos after step one while keeping every beam scene-local."""
    print(f"\n[Cheap Video Beam] for Query {i + 1}")
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
        "vlm_prompts": 0,
    }
    groups = _logical_groups(atomic_items)
    valid_clauses = [clause for _, clauses in groups for clause in clauses]
    time_steps = [
        time_step for time_step, clauses in groups for _clause in clauses
    ]
    if not groups:
        print("No valid clauses to evaluate.")
        return None, timing

    started = time.time()
    valid_scenes = base._filter_scenes_by_time_order(
        candidate_scenes, scene_subquery_results, time_steps
    )
    timing["hard_filter"] = time.time() - started
    if skip_time_filter:
        scenes = list(candidate_scenes)
    else:
        scenes = valid_scenes if valid_scenes else list(candidate_scenes)
    score_quantum = os.environ.get("CHEAP_BEAM_SCORE_QUANTUM", "").strip()
    print(
        f"Cheap beam config: B_video={video_beam_width}, "
        f"B_frame={frame_beam_width}, logical_steps={len(groups)}"
        + (f", clause_score_quantum={score_quantum}" if score_quantum else "")
    )
    print(f"Scenes entering first VLM step: {len(scenes)}")
    if not scenes:
        return None, timing

    table_names = db.table_names()
    reps_table = (
        db.open_table("frame_cluster_reps")
        if "frame_cluster_reps" in table_names
        else None
    )
    clusters_table = (
        db.open_table("frame_clusters")
        if "frame_clusters" in table_names
        else frames_tbl
    )
    use_encoder = text_embeddings is not None and score_backend != "vlm"
    scene_results = _fetch_scene_results(
        scenes,
        reps_table,
        clusters_table,
        frames_tbl,
        timing,
        score_all_frames=score_all_frames,
        load_images=not use_encoder,
    )
    scenes = [scene for scene in scenes if scene in scene_results]
    if not scenes:
        return None, timing

    import tempfile

    bridge = persistent_bridge
    owns_bridge = bridge is None
    beams_by_scene = {}
    try:
        with tempfile.TemporaryDirectory(prefix="qwen3_cheap_beam_") as image_dir:
            image_paths = {}
            if not use_encoder:
                started = time.time()
                image_paths = _save_cluster_images(scene_results, image_dir, timing)
                timing["prompt_build"] += time.time() - started
                if bridge is None:
                    bridge = PersistentVLMBridge(
                        qwen3_python,
                        qwen3_bridge_script,
                        qwen3_model_path,
                        timeout_seconds=21600 if score_all_frames else 7200,
                    )
            selected_videos = None
            embedding_cache = {}
            for group_index, (logical_step, clauses) in enumerate(groups):
                if selected_videos is None:
                    step_scenes = scenes
                else:
                    step_scenes = [
                        scene
                        for scene in scenes
                        if scene_to_video_id(scene) in selected_videos
                    ]
                print(
                    f"Logical step {logical_step}: scoring "
                    f"{len(step_scenes)} scene(s)"
                )
                if use_encoder:
                    rows = _score_group_encoder(
                        clauses,
                        step_scenes,
                        scene_results,
                        top_k,
                        text_embeddings,
                        db,
                        score_backend,
                        timing,
                        embedding_cache,
                    )
                else:
                    rows = _score_group(
                        bridge,
                        clauses,
                        step_scenes,
                        scene_results,
                        image_paths,
                        top_k,
                        timing,
                    )
                beams_by_scene = _advance_scenes(
                    beams_by_scene,
                    rows,
                    frame_beam_width,
                    penalty_weight,
                    epsilon,
                )
                if group_index == 0 and len(groups) > 1:
                    selected_videos = _select_tie_aware_videos(
                        beams_by_scene, video_beam_width
                    )
                    beams_by_scene = {
                        scene_name: beam
                        for scene_name, beam in beams_by_scene.items()
                        if scene_to_video_id(scene_name) in selected_videos
                    }
                    print(
                        "Videos retained after first logical step "
                        f"(tie-aware target={video_beam_width}, "
                        f"actual={len(selected_videos)}): "
                        + ", ".join(sorted(selected_videos))
                    )
                elif group_index == 0:
                    print(
                        "Single logical step: preserving baseline scene-local "
                        "ranking without video pruning."
                    )
    except Exception as exc:
        import traceback

        print(f"Cheap beam scoring failed: {exc}")
        traceback.print_exc()
        return None, timing
    finally:
        if owns_bridge and bridge is not None:
            bridge.close()

    started = time.time()
    scene_candidates = _build_scene_candidates(
        beams_by_scene,
        len(valid_clauses),
        clusters_table,
        score_all_frames=score_all_frames,
    )
    all_sequences = base._collect_and_print_top_sequences(
        scene_candidates, time_steps
    )
    timing["candidate_build"] = time.time() - started
    print(f"Total VLM prompts scored by cheap beam: {timing['vlm_prompts']}")
    return all_sequences, timing
