import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import cv2
import lancedb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from paths import (
    FEATURE_CACHE_ROOT,
    FGCLIP_MODEL_PATH,
    INTERNVIDEO2_ROOT,
    LANCEDB_PATH,
    SCENE_ROOT,
    TEST_LANCEDB_PATH,
    VIDEOS_ROOT,
)

project_root = INTERNVIDEO2_ROOT
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "multi_modality"))

# Moved to load_query_encoder to avoid direct import errors in environments that only run FG-CLIP
# from demo.utils import setup_internvideo2, normalize
# from utils.config import Config, eval_dict_leaf

CACHE_VERSION = 1


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _try_create_scalar_index(tbl, column: str):
    try:
        tbl.create_scalar_index(column, replace=True)
        print(f"Scalar index ready: {column}")
    except Exception as e:
        print(f"Warning: Could not create scalar index on '{column}': {e}")


def resolve_internvideo_config_paths(config):
    mm_root = os.path.join(project_root, "multi_modality")
    text_cfg = config.get("model", {}).get("text_encoder", {}).get("config")
    if isinstance(text_cfg, str) and not os.path.isabs(text_cfg):
        config.model.text_encoder.config = os.path.join(mm_root, text_cfg)


def list_video_files(video_root: str):
    video_files = []
    for name in os.listdir(video_root):
        full_path = os.path.join(video_root, name)
        if os.path.isfile(full_path) and name.lower().endswith(".mp4"):
            video_files.append(full_path)
    return sorted(video_files)


def list_video_dirs(video_root: str):
    video_dirs = []
    for name in os.listdir(video_root):
        full_path = os.path.join(video_root, name)
        if os.path.isdir(full_path):
            video_dirs.append(full_path)
    return sorted(video_dirs)


def list_scene_videos(video_dir: str):
    scene_videos = []
    for name in os.listdir(video_dir):
        full_path = os.path.join(video_dir, name)
        if os.path.isfile(full_path) and name.endswith(".mp4") and "-Scene-" in name:
            scene_videos.append(full_path)

    def scene_sort_key(path: str):
        stem = Path(path).stem
        scene_num = stem.split("-Scene-")[-1]
        try:
            return int(scene_num)
        except ValueError:
            return scene_num

    return sorted(scene_videos, key=scene_sort_key)


def get_scene_cache_path(
    scene_path: Path, num_frames: int, size_t: int, feature_cache_root: str
):
    video_id = os.path.basename(os.path.dirname(scene_path))
    scene_stem = Path(scene_path).stem
    cache_name = f"{scene_stem}_nf{num_frames}_sz{size_t}.pt"
    return os.path.join(feature_cache_root, video_id, cache_name)


def try_load_scene_feature_cache(
    scene_path: Path, num_frames: int, size_t: int, feature_cache_root: str
):
    cache_path = get_scene_cache_path(
        scene_path, num_frames, size_t, feature_cache_root
    )
    # print(f"+++++{cache_path}+++++")
    if not os.path.exists(cache_path):
        return None

    payload = torch.load(cache_path, map_location="cpu")
    meta = payload.get("meta", {})

    stat = os.stat(scene_path)
    if meta.get("cache_version") != CACHE_VERSION:
        return None
    if meta.get("scene_size") != stat.st_size:
        return None
    # if meta.get("scene_mtime_ns") != stat.st_mtime_ns:
    #     return None
    if meta.get("num_frames") != num_frames:
        return None
    if meta.get("size_t") != size_t:
        return None

    return payload


def load_query_encoder():
    from demo.utils import setup_internvideo2, normalize
    from utils.config import Config, eval_dict_leaf

    config = Config.from_file(
        os.path.join(
            project_root, "multi_modality", "demo/internvideo2_stage2_config.py"
        )
    )
    config = eval_dict_leaf(config)
    resolve_internvideo_config_paths(config)

    intern_model, _ = setup_internvideo2(config)
    return intern_model, config


from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)


def determine_max_value(image):
    # Removed as FG-CLIP base model logic doesn't require this (uses simple processor)
    pass


def load_fgclip_model(model_root=None):
    if model_root is None:
        model_root = FGCLIP_MODEL_PATH
    print(f"Loading FG-CLIP model from {model_root}...")
    model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True)
    model.eval()
    image_processor = AutoImageProcessor.from_pretrained(model_root)
    return model, image_processor


def extract_batch_frame_embeddings(frame_imgs, fgclip_model, fgclip_processor, device):
    """
    Convert a batch of frame images to FG-CLIP embeddings.
    """
    images = [Image.fromarray(img).resize((224, 224)) for img in frame_imgs]

    image_inputs = fgclip_processor.preprocess(images, return_tensors="pt")[
        "pixel_values"
    ].to(device)

    with torch.no_grad():
        image_features = fgclip_model.get_image_features(image_inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    embs = image_features.cpu().numpy().astype(np.float32)
    return embs


def extract_all_frames_and_embeddings(
    scene_path: str,
    fgclip_model,
    fgclip_processor,
    device,
    size_t: int,
    batch_size: int = 128,
):
    """
    Reads all frames from a video using OpenCV, extracts 1D embedding for every frame using batching,
    and also returns the frame image data (JPEG bytes).
    """
    cap = cv2.VideoCapture(scene_path)
    if not cap.isOpened():
        print(f"Failed to open video: {scene_path}")
        return [], [], 0

    frames_bytes = []
    all_embeddings = []

    batch_rgb = []
    batch_bytes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (size_t, size_t))
        frame_rgb = frame_resized[:, :, ::-1]

        # Convert to bytes (JPEG) to save space in LanceDB
        _, buffer = cv2.imencode(".jpg", frame_resized)

        batch_rgb.append(frame_rgb)
        batch_bytes.append(buffer.tobytes())

        if len(batch_rgb) >= batch_size:
            embs = extract_batch_frame_embeddings(
                batch_rgb, fgclip_model, fgclip_processor, device
            )
            all_embeddings.extend(embs)
            frames_bytes.extend(batch_bytes)
            batch_rgb = []
            batch_bytes = []

    # Process remaining frames
    if len(batch_rgb) > 0:
        embs = extract_batch_frame_embeddings(
            batch_rgb, fgclip_model, fgclip_processor, device
        )
        all_embeddings.extend(embs)
        frames_bytes.extend(batch_bytes)

    cap.release()
    scene_total_frames = len(frames_bytes)

    return all_embeddings, frames_bytes, scene_total_frames


def _collect_scene_videos(scene_root: str):
    if not os.path.isdir(scene_root):
        raise ValueError(f"Expected scene_root directory, but got: {scene_root}")

    video_dirs = list_video_dirs(scene_root)
    if len(video_dirs) == 0:
        raise ValueError(f"No video folders found under scene_root: {scene_root}")

    scene_videos = []
    for video_dir in tqdm(video_dirs, desc="Collecting scene videos"):
        scene_videos.extend(list_scene_videos(video_dir))

    if len(scene_videos) == 0:
        raise ValueError(f"No scene clips found under scene_root: {scene_root}")
    return scene_videos


def _count_missing_scene_feature_cache(
    scene_videos, num_frames: int, size_t: int, feature_cache_root: str
):
    missing = 0
    for scene_video in scene_videos:
        cache_path = get_scene_cache_path(
            Path(scene_video), num_frames, size_t, feature_cache_root
        )
        if not os.path.exists(cache_path):
            missing += 1
    return missing


def ensure_scene_feature_cache(
    scene_root: str,
    feature_cache_root: str,
    num_frames: int,
    size_t: int,
    extractor_python_cmd: str = "conda run -n internvideo python",
    extractor_script_path: str = None,
):
    scene_videos = _collect_scene_videos(scene_root)
    missing_before = _count_missing_scene_feature_cache(
        scene_videos, num_frames, size_t, feature_cache_root
    )
    print(
        f"Scene feature cache check: total={len(scene_videos)}, "
        f"missing={missing_before}, root={feature_cache_root}"
    )
    if missing_before == 0:
        print("All required clip feature cache files exist. Skipping extraction.")
        return

    if extractor_script_path is None:
        extractor_script_path = os.path.join(
            project_root, "multi_modality", "extract_lovr_features.py"
        )
    if not os.path.exists(extractor_script_path):
        raise FileNotFoundError(f"Extractor script not found: {extractor_script_path}")

    os.makedirs(feature_cache_root, exist_ok=True)
    cmd = shlex.split(extractor_python_cmd) + [
        extractor_script_path,
        "--video_root",
        scene_root,
        "--feature_cache_root",
        feature_cache_root,
    ]
    print("Missing cache detected. Running extractor:")
    print(" ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"Feature extraction failed with return code {rc}")

    missing_after = _count_missing_scene_feature_cache(
        scene_videos, num_frames, size_t, feature_cache_root
    )
    print(f"Cache check after extraction: missing={missing_after}")
    if missing_after > 0:
        print(
            "Warning: Some cache files are still missing. "
            "Build will continue and those scenes may be skipped in clips table."
        )


def build_videos_table(db, videos_root: str):
    print("Building videos table...")
    if not os.path.isdir(videos_root):
        raise ValueError(f"Expected videos_root directory, but got: {videos_root}")

    video_files = list_video_files(videos_root)
    if len(video_files) == 0:
        raise ValueError(f"No .mp4 files found under videos_root: {videos_root}")

    videos_data = []
    for path in video_files:
        video_id = Path(path).stem
        videos_data.append({"video_id": video_id})

    db.create_table("videos", data=pd.DataFrame(videos_data), mode="overwrite")
    print(f"Built videos table with {len(videos_data)} records.")


def build_clips_table(
    db, intern_model, config, scene_root: str, feature_cache_root: str
):
    print("Building clips table...")
    scene_videos = _collect_scene_videos(scene_root)

    device = torch.device(config.device)
    T = intern_model.vision_encoder.patch_embed.grid_size[0]
    H = W = intern_model.vision_encoder.patch_embed.grid_size[1]
    t_size = T // 4
    temporal_chunk_count = T // t_size

    num_frames = config.get("num_frames", 8)
    size_t = config.get("size_t", 224)

    clips_data = []
    cache_hits = 0
    cache_misses = 0

    current_video_id = None
    global_frame_offset = 0

    for scene_video in tqdm(scene_videos, desc="Loading embeddings for clips"):
        video_id = os.path.basename(os.path.dirname(scene_video))
        if video_id.endswith(".mp4"):
            video_id = video_id[:-4]

        if video_id != current_video_id:
            current_video_id = video_id
            global_frame_offset = 0

        payload = try_load_scene_feature_cache(
            scene_video,
            num_frames=num_frames,
            size_t=size_t,
            feature_cache_root=feature_cache_root,
        )

        scene_total_frames = 0
        if payload is not None:
            scene_total_frames = payload.get("scene_total_frames", 0)
        else:
            try:
                cap = cv2.VideoCapture(scene_video)
                if cap.isOpened():
                    scene_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
            except Exception as e:
                print(
                    f"Failed to read frame count for missing cache {scene_video}: {e}"
                )

        if payload is not None:
            cache_hits += 1
            clip_feat = payload["clip_feat"].to(device)
            clip_feat_pooled = payload["clip_feat_pooled"].to(device)
            with torch.no_grad():
                clip_feat_chunked_pooled, clip_feat_chunked_pooled_idxs = (
                    intern_model.pool_chunk_feat(clip_feat, T, H, W, t_size)
                )

            base_clip_id = str(payload.get("scene_name", os.path.basename(scene_video)))
            if base_clip_id.endswith(".mp4"):
                base_clip_id = base_clip_id[:-4]
            if video_id.endswith(".mp4"):
                video_id = video_id[:-4]

            # Store the overall pooled feature
            clips_data.append(
                {
                    "clip_id": base_clip_id,
                    "video_id": video_id,
                    "clip_frame_start": float(global_frame_offset),
                    "clip_frame_end": float(global_frame_offset + scene_total_frames),
                    "clip_embedding": clip_feat_pooled.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    .reshape(-1),
                }
            )

            for feat, idx_tuple in zip(
                clip_feat_chunked_pooled, clip_feat_chunked_pooled_idxs
            ):
                t_idx = idx_tuple[0]
                local_start_idx = int(
                    (t_idx * scene_total_frames) // temporal_chunk_count
                )
                local_end_idx = int(
                    ((t_idx + 1) * scene_total_frames) // temporal_chunk_count
                )

                global_start_idx = global_frame_offset + local_start_idx
                global_end_idx = global_frame_offset + local_end_idx

                chunk_id = (
                    f"{base_clip_id}_{idx_tuple[0]}_{idx_tuple[1]}_{idx_tuple[2]}"
                )
                clip_embedding = (
                    feat.detach().cpu().numpy().astype(np.float32).reshape(-1)
                )

                clips_data.append(
                    {
                        "clip_id": chunk_id,
                        "video_id": video_id,
                        "clip_frame_start": float(global_start_idx),
                        "clip_frame_end": float(global_end_idx),
                        "clip_embedding": clip_embedding,
                    }
                )
        else:
            cache_misses += 1
            print(
                f"Warning: Cache missing for {scene_video}. Extracted frames won't have clip counterpart."
            )

        global_frame_offset += scene_total_frames

    if len(clips_data) > 0:
        db.create_table("clips", data=pd.DataFrame(clips_data), mode="overwrite")
        print(
            f"Built clips table with {len(clips_data)} records. (Cache hits: {cache_hits}, misses: {cache_misses})"
        )
    else:
        print("Warning: No clips data extracted.")


def build_frames_table(
    db,
    intern_model,
    config,
    scene_root: str,
    feature_cache_root: str,
    fgclip_model=None,
    fgclip_processor=None,
):

    # =================================暂时测试用====================================
    print("Building frames table...")
    if (
        "frames" in db.table_names()
        and db.open_table("frames").count_rows() >= 19787876
    ):
        print(
            "Frames table already fully populated (>=19787876 rows). Skipping frame extraction to save time."
        )
        return
    # =================================暂时测试用====================================

    scene_videos = _collect_scene_videos(scene_root)

    device = torch.device(config.device)
    num_frames = config.get("num_frames", 8)
    size_t = config.get("size_t", 224)

    frames_data = []
    current_video_id = None
    global_frame_offset = 0
    table_created = False

    # === Resume Logic (断点恢复) ===
    processed_clips = []
    clip_counts = {}
    if "frames" in db.table_names():
        print("Found existing 'frames' table. Loading processed clip IDs for resume...")
        table_created = True
        try:
            # Safely fetch all existing clip_ids and their actual frame counts
            import pyarrow as pa

            tbl = db.open_table("frames")
            existing_clips = (
                tbl.to_lance()
                .to_table(columns=["clip_id"])
                .column("clip_id")
                .to_pylist()
            )
            from collections import Counter

            clip_counts = Counter(existing_clips)
            processed_clips = set(clip_counts.keys())
            print(
                f"Resuming operation... Already processed {len(processed_clips)} clips."
            )
        except Exception as e:
            print(f"Warning: Could not read existing table for resume: {e}")
            table_created = False
    # ===============================

    for scene_video in tqdm(
        scene_videos, desc="Loading embeddings & extracting frames"
    ):
        video_id = os.path.basename(os.path.dirname(scene_video))
        if video_id.endswith(".mp4"):
            video_id = video_id[:-4]

        if video_id != current_video_id:
            current_video_id = video_id
            global_frame_offset = 0

        base_clip_id = os.path.basename(scene_video)
        if base_clip_id.endswith(".mp4"):
            base_clip_id = base_clip_id[:-4]

        # Fast path to skip already processed clips
        if base_clip_id in processed_clips:
            try:
                cap = cv2.VideoCapture(scene_video)
                if cap.isOpened():
                    scene_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                else:
                    scene_total_frames = clip_counts[base_clip_id]
            except Exception:
                scene_total_frames = clip_counts[base_clip_id]
            global_frame_offset += scene_total_frames
            continue

        frame_embeddings, frames_bytes, actual_total_frames = (
            extract_all_frames_and_embeddings(
                scene_video, fgclip_model, fgclip_processor, device, size_t
            )
        )

        scene_total_frames = actual_total_frames

        for i, (f_emb, f_bytes) in enumerate(zip(frame_embeddings, frames_bytes)):
            frames_data.append(
                {
                    "frame_id": int(global_frame_offset + i),
                    "video_id": video_id,
                    "clip_id": base_clip_id,
                    "frame_embedding": f_emb,
                    "frame_image": f_bytes,
                }
            )

        global_frame_offset += scene_total_frames

        # flush to lanceDB progressively to prevent Host Memory (RAM) OOM
        if len(frames_data) >= 2000:
            if not table_created:
                db.create_table(
                    "frames", data=pd.DataFrame(frames_data), mode="overwrite"
                )
                table_created = True
            else:
                db.open_table("frames").add(pd.DataFrame(frames_data))
            frames_data.clear()

    if len(frames_data) > 0:
        if not table_created:
            db.create_table("frames", data=pd.DataFrame(frames_data), mode="overwrite")
        else:
            db.open_table("frames").add(pd.DataFrame(frames_data))
        frames_data.clear()
        print("Finished building frames table.")
    elif not table_created:
        print("Warning: No frames data extracted.")


def print_table_summary(db, table_name: str):
    if table_name not in db.table_names():
        print(f"Table '{table_name}' does not exist in the database.")
        return

    try:
        tbl = db.open_table(table_name)

        count = "Unknown"
        if hasattr(tbl, "count_rows"):
            count = tbl.count_rows()  # lancedb 0.5+
        else:
            try:
                count = getattr(
                    tbl.to_lance(), "count_rows"
                )()  # fallback via core lance pyarrow dataset
            except Exception:
                count = (
                    len(tbl.search().limit(0).to_pandas())
                    if hasattr(tbl, "search")
                    else len(tbl.to_pandas())
                )

        print(f"Table '{table_name}' row count: {count}")
        # 打印第一行记录以查看字段和数据类型
        first_row = tbl.search().limit(1).to_pandas()
        print(f"Sample record from '{table_name}':\n{first_row}\n")
    except Exception as e:
        print(f"Failed to read table '{table_name}': {e}")


def build_frames_clusters(db, threshold=0.92):
    print("Building clusters for frames...")
    if "frames" not in db.table_names():
        print("Table 'frames' not found. Cannot build clusters.")
        return

    tbl = db.open_table("frames")

    # We always rebuild frame_clusters from scratch for deterministic results.
    if "frame_clusters" in db.table_names():
        print("Table 'frame_clusters' already exists. Dropping it before rebuild.")
        db.drop_table("frame_clusters")

    print("Counting frames per clip to optimize memory usage...")
    try:
        ds = tbl.to_lance()
        df_meta = ds.to_table(columns=["clip_id"]).to_pandas()
        clip_counts = df_meta["clip_id"].value_counts().to_dict()
    except Exception as e:
        print(f"Failed to load frame metadata for clustering: {e}")
        return

    all_clusters_data = []

    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

    processed_clip_cnt = 0
    temp_clusters_batch = []
    clusters_table_created = False

    def process_clip(rows, current_clip_id):
        nonlocal processed_clip_cnt, temp_clusters_batch, clusters_table_created
        if not rows:
            return
            return
        rows.sort(key=lambda x: x["frame_id"])

        first_frame_id = rows[0]["frame_id"]
        last_frame_id = rows[-1]["frame_id"]
        first_cluster_id = None

        current_cluster = [rows[0]]
        current_centroid = rows[0]["frame_embedding"].copy()

        for row in rows[1:]:
            emb = row["frame_embedding"]
            sim = cosine_similarity(current_centroid, emb)

            if sim >= threshold:
                current_cluster.append(row)
                all_embs = np.array([r["frame_embedding"] for r in current_cluster])
                mean_emb = np.mean(all_embs, axis=0)
                current_centroid = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
            else:
                cluster_frame_ids = sorted([r["frame_id"] for r in current_cluster])
                c_frame_id = cluster_frame_ids[len(cluster_frame_ids) // 2]

                if first_cluster_id is None:
                    first_cluster_id = c_frame_id

                for r in current_cluster:
                    temp_clusters_batch.append(
                        {
                            "frame_id": int(r["frame_id"]),
                            "clip_id": current_clip_id,
                            "frame_cluster_centroid_embedding": current_centroid.astype(
                                np.float32
                            ),
                            "cluster_frame_id": int(c_frame_id),
                        }
                    )

                current_cluster = [row]
                current_centroid = row["frame_embedding"].copy()

        if current_cluster:
            cluster_frame_ids = sorted([r["frame_id"] for r in current_cluster])
            c_frame_id = cluster_frame_ids[len(cluster_frame_ids) // 2]

            if first_cluster_id is None:
                first_cluster_id = c_frame_id

            for r in current_cluster:
                temp_clusters_batch.append(
                    {
                        "frame_id": int(r["frame_id"]),
                        "clip_id": current_clip_id,
                        "frame_cluster_centroid_embedding": current_centroid.astype(
                            np.float32
                        ),
                        "cluster_frame_id": int(c_frame_id),
                    }
                )

        # 打印部分调试信息: 每 100 个 clip 打印一次
        processed_clip_cnt += 1
        if processed_clip_cnt % 100 == 0:
            tqdm.write(
                f"Clip: {current_clip_id} | Frames: {first_frame_id} -> {last_frame_id} | "
                f"Clusters: First={first_cluster_id}, Last={c_frame_id}"
            )

        if len(temp_clusters_batch) >= 100000:
            df_write = pd.DataFrame(temp_clusters_batch)
            if not clusters_table_created:
                db.create_table("frame_clusters", data=df_write, mode="overwrite")
                clusters_table_created = True
            else:
                db.open_table("frame_clusters").add(df_write)
            temp_clusters_batch.clear()

    from collections import defaultdict

    clip_buffers = defaultdict(list)

    print(
        "Processing frame batches and streaming inserts to new 'frame_clusters' table..."
    )
    with tqdm(total=len(df_meta), desc="Clustering frames") as pbar:
        # We don't need _rowid anymore since we are not merging, just making a new table
        for batch in ds.scanner(
            columns=["frame_id", "clip_id", "frame_embedding"]
        ).to_reader():
            df_batch = batch.to_pandas()
            for clip_id, group in df_batch.groupby("clip_id"):
                clip_buffers[clip_id].extend(group.to_dict("records"))
                if len(clip_buffers[clip_id]) == clip_counts[clip_id]:
                    process_clip(clip_buffers[clip_id], clip_id)
                    del clip_buffers[clip_id]
            pbar.update(len(df_batch))

    # 写出所有残留缓存
    if len(temp_clusters_batch) > 0:
        df_write = pd.DataFrame(temp_clusters_batch)
        if not clusters_table_created:
            db.create_table("frame_clusters", data=df_write, mode="overwrite")
        else:
            db.open_table("frame_clusters").add(df_write)
        temp_clusters_batch.clear()

    print("Finished building independent 'frame_clusters' table!")


def build_frame_cluster_reps_table(
    db,
    scene_root: str,
    batch_size: int = 5000,
    overwrite: bool = False,
    frame_cluster_threshold: float = 0.92,
):
    """Build one lightweight row per frame cluster representative.

    The query pipeline only needs representative frame ids and image bytes for
    bridge scoring. Materializing them avoids scanning frame-level cluster rows
    and then joining back to the heavy frames table on every query.
    """
    print("Building frame_cluster_reps table...")
    table_names = db.table_names()
    if "frames" not in table_names:
        print("Table 'frames' not found. Cannot build frame_cluster_reps.")
        return
    if "frame_clusters" not in table_names:
        print("Table 'frame_clusters' not found. Building it from existing frames first...")
        build_frames_clusters(db, threshold=frame_cluster_threshold)
        table_names = db.table_names()
        if "frame_clusters" not in table_names:
            print("Table 'frame_clusters' still not found. Cannot build frame_cluster_reps.")
            return

    # Always rebuild frame_cluster_reps from scratch to stay in sync
    # with the current frame_clusters result.
    if "frame_cluster_reps" in table_names:
        print("Table 'frame_cluster_reps' already exists. Dropping it before rebuild.")
        db.drop_table("frame_cluster_reps")

    frames_tbl = db.open_table("frames")
    clusters_tbl = db.open_table("frame_clusters")
    print("Ensuring source scalar indexes for fast per-scene lookups...")
    _try_create_scalar_index(frames_tbl, "clip_id")
    _try_create_scalar_index(clusters_tbl, "clip_id")
    scene_videos = _collect_scene_videos(scene_root)

    reps_batch = []
    table_created = False
    total_reps = 0
    missing_images = 0
    empty_clusters = 0

    def flush_batch():
        nonlocal reps_batch, table_created, total_reps
        if not reps_batch:
            return
        df_write = pd.DataFrame(reps_batch)
        if not table_created:
            db.create_table("frame_cluster_reps", data=df_write, mode="overwrite")
            table_created = True
        else:
            db.open_table("frame_cluster_reps").add(df_write)
        total_reps += len(reps_batch)
        reps_batch = []

    for scene_video in tqdm(scene_videos, desc="Building frame_cluster_reps"):
        clip_id = Path(scene_video).stem
        clip_expr = _sql_literal(clip_id)
        try:
            df_clusters = (
                clusters_tbl.search()
                .where(f"clip_id = {clip_expr}")
                .select(["clip_id", "cluster_frame_id"])
                .to_pandas()
            )
        except Exception as e:
            print(f"Warning: Failed to read clusters for {clip_id}: {e}")
            continue

        if df_clusters.empty:
            empty_clusters += 1
            continue

        member_counts = (
            df_clusters.groupby("cluster_frame_id").size().rename("member_count")
        )
        cluster_frame_ids = set(int(x) for x in member_counts.index.tolist())

        try:
            frames_df = (
                frames_tbl.search()
                .where(f"clip_id = {clip_expr}")
                .select(["video_id", "clip_id", "frame_id", "frame_image"])
                .to_pandas()
            )
        except Exception as e:
            print(f"Warning: Failed to read frames for {clip_id}: {e}")
            continue

        if frames_df.empty:
            missing_images += len(cluster_frame_ids)
            continue

        frames_df = frames_df[frames_df["frame_id"].isin(cluster_frame_ids)]
        image_by_frame_id = dict(zip(frames_df["frame_id"], frames_df["frame_image"]))
        video_by_frame_id = dict(zip(frames_df["frame_id"], frames_df["video_id"]))

        for cluster_frame_id, member_count in member_counts.items():
            cluster_frame_id = int(cluster_frame_id)
            frame_image = image_by_frame_id.get(cluster_frame_id)
            if frame_image is None:
                missing_images += 1
                continue
            reps_batch.append(
                {
                    "clip_id": clip_id,
                    "video_id": video_by_frame_id.get(cluster_frame_id),
                    "cluster_frame_id": cluster_frame_id,
                    "frame_image": frame_image,
                    "member_count": int(member_count),
                }
            )
            if len(reps_batch) >= batch_size:
                flush_batch()

    flush_batch()

    if table_created:
        reps_tbl = db.open_table("frame_cluster_reps")
        _try_create_scalar_index(reps_tbl, "clip_id")
        _try_create_scalar_index(reps_tbl, "cluster_frame_id")
        print(
            "Finished building frame_cluster_reps table: "
            f"{total_reps} reps, {missing_images} missing images, "
            f"{empty_clusters} scenes without clusters."
        )
    else:
        print("Warning: No frame_cluster_reps data was written.")


def build_index(
    intern_model,
    config,
    videos_root: str,
    scene_root: str,
    feature_cache_root: str,
    db_path: str,
    build_videos: bool = True,
    build_clips: bool = True,
    build_frames: bool = True,
    build_frame_cluster_reps: bool = False,
    rebuild_frame_cluster_reps: bool = False,
    frame_cluster_threshold: float = 0.92,
    auto_extract_missing_cache: bool = True,
    extractor_python_cmd: str = "conda run -n internvideo python",
    extractor_script_path: str = None,
    fgclip_model=None,
    fgclip_processor=None,
):
    db = lancedb.connect(db_path)
    existing_tables = db.table_names()

    if build_videos and "videos" in existing_tables:
        print("Table 'videos' already exists, dropping it to rebuild.")
        db.drop_table("videos")

    if build_clips and "clips" in existing_tables:
        print(
            "Table 'clips' already exists. Rebuilding is currently set to drop. Adjust manually if needed."
        )
        if os.path.normpath(db_path) != os.path.normpath(TEST_LANCEDB_PATH):
            db.drop_table("clips")

    if build_frames and "frames" in existing_tables:
        print(
            "Table 'frames' already exists! Entering append/resume mode instead of dropping."
        )
        # db.drop_table("frames") # commented out for resume

    if not build_videos and not build_clips and not build_frames and not build_frame_cluster_reps:
        print("No tables requested to be built.")
        return db

    if build_videos:
        build_videos_table(db, videos_root)
        print_table_summary(db, "videos")

    if build_clips:
        if auto_extract_missing_cache:
            num_frames = int(config.get("num_frames", 8))
            size_t = int(config.get("size_t", 224))
            ensure_scene_feature_cache(
                scene_root=scene_root,
                feature_cache_root=feature_cache_root,
                num_frames=num_frames,
                size_t=size_t,
                extractor_python_cmd=extractor_python_cmd,
                extractor_script_path=extractor_script_path,
            )
        build_clips_table(db, intern_model, config, scene_root, feature_cache_root)
        print_table_summary(db, "clips")

    if build_frames:
        build_frames_table(
            db,
            intern_model,
            config,
            scene_root,
            feature_cache_root,
            fgclip_model,
            fgclip_processor,
        )
        print_table_summary(db, "frames")
        build_frames_clusters(db, threshold=frame_cluster_threshold)

    if build_frame_cluster_reps:
        build_frame_cluster_reps_table(
            db,
            scene_root,
            overwrite=rebuild_frame_cluster_reps,
            frame_cluster_threshold=frame_cluster_threshold,
        )
        print_table_summary(db, "frame_cluster_reps")

    return db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos_root",
        type=str,
        default=VIDEOS_ROOT,
        help="Directory containing merged videos for videos table.",
    )
    parser.add_argument(
        "--scene_root",
        type=str,
        default=SCENE_ROOT,
        help="Directory containing scene clips grouped by video id.",
    )
    parser.add_argument(
        "--feature_cache_root",
        type=str,
        default=FEATURE_CACHE_ROOT,
        help="Feature cache root used by demo_lovr_split.py.",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=LANCEDB_PATH,
        help="LanceDB path.",
    )
    parser.add_argument(
        "--build_videos", action="store_true", help="Build videos table."
    )
    parser.add_argument("--build_clips", action="store_true", help="Build clips table.")
    parser.add_argument(
        "--build_frames", action="store_true", help="Build frames table."
    )
    parser.add_argument(
        "--build_frame_cluster_reps",
        action="store_true",
        help="Build frame_cluster_reps table from existing frames/frame_clusters.",
    )
    parser.add_argument(
        "--rebuild_frame_cluster_reps",
        action="store_true",
        help="Overwrite frame_cluster_reps if it already exists.",
    )
    parser.add_argument(
        "--fgclip_model_path",
        type=str,
        default=FGCLIP_MODEL_PATH,
        help="HuggingFace path for FG-CLIP model.",
    )
    parser.add_argument(
        "--frame_cluster_threshold",
        type=float,
        default=0.92,
        help="Cosine similarity threshold for frame clustering (higher = more, finer clusters).",
    )
    parser.add_argument(
        "--disable_auto_extract_missing_cache",
        action="store_true",
        help="Disable automatic extraction for missing clip feature cache files before building clips.",
    )
    parser.add_argument(
        "--extractor_python_cmd",
        type=str,
        default="conda run -n internvideo python",
        help="Python command used to run the extractor script.",
    )
    parser.add_argument(
        "--extractor_script_path",
        type=str,
        default=None,
        help="Path to extract_lovr_features.py. Defaults to InternVideo project path.",
    )

    args = parser.parse_args()

    # If no flags are provided, run all by default to keep backward compatibility
    if not any([args.build_videos, args.build_clips, args.build_frames, args.build_frame_cluster_reps]):
        args.build_videos = True
        args.build_clips = True
        args.build_frames = True
        args.build_frame_cluster_reps = True

    intern_model, config = None, None
    if args.build_clips:
        intern_model, config = load_query_encoder()
        device = torch.device(config.device)
    else:
        # Default fallback for standalone frame building
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class DummyConfig:
            pass

        config = DummyConfig()
        config.device = device
        config.get = lambda k, d=None: (
            d if d is not None else {"num_frames": 8, "size_t": 224}.get(k)
        )

    fgclip_model, fgclip_processor = None, None
    if args.build_frames:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        fgclip_model, fgclip_processor = load_fgclip_model(args.fgclip_model_path)
        fgclip_model = fgclip_model.to(device)

    db = build_index(
        intern_model=intern_model,
        config=config,
        videos_root=args.videos_root,
        scene_root=args.scene_root,
        feature_cache_root=args.feature_cache_root,
        db_path=args.db_path,
        build_videos=args.build_videos,
        build_clips=args.build_clips,
        build_frames=args.build_frames,
        build_frame_cluster_reps=args.build_frame_cluster_reps,
        rebuild_frame_cluster_reps=args.rebuild_frame_cluster_reps,
        frame_cluster_threshold=args.frame_cluster_threshold,
        auto_extract_missing_cache=not args.disable_auto_extract_missing_cache,
        extractor_python_cmd=args.extractor_python_cmd,
        extractor_script_path=args.extractor_script_path,
        fgclip_model=fgclip_model,
        fgclip_processor=fgclip_processor,
    )


if __name__ == "__main__":
    main()
