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



def _cluster_representatives(frame_ids, embeddings, images, threshold):
    """Greedy contiguous clusters. The representative is the median member."""
    if not frame_ids:
        return []

    def cosine_similarity(left, right):
        return np.dot(left, right) / (
            np.linalg.norm(left) * np.linalg.norm(right) + 1e-10
        )

    representatives = []
    current_ids = [int(frame_ids[0])]
    current_embs = [embeddings[0]]
    current_images = [images[0]]
    current_centroid = np.array(embeddings[0], copy=True)

    def close_cluster():
        order = sorted(range(len(current_ids)), key=lambda index: current_ids[index])
        sorted_ids = [current_ids[index] for index in order]
        median_index = len(sorted_ids) // 2
        representatives.append(
            {
                "cluster_frame_id": int(sorted_ids[median_index]),
                "frame_image": current_images[order[median_index]],
                "member_count": len(sorted_ids),
            }
        )

    for frame_id, embedding, image in zip(
        frame_ids[1:], embeddings[1:], images[1:]
    ):
        if cosine_similarity(current_centroid, embedding) >= threshold:
            current_ids.append(int(frame_id))
            current_embs.append(embedding)
            current_images.append(image)
            mean_emb = np.mean(np.stack(current_embs), axis=0)
            current_centroid = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
        else:
            close_cluster()
            current_ids = [int(frame_id)]
            current_embs = [embedding]
            current_images = [image]
            current_centroid = np.array(embedding, copy=True)

    close_cluster()
    return representatives


def build_frame_cluster_reps_table(
    db,
    config,
    scene_root: str,
    fgclip_model,
    fgclip_processor,
    batch_size: int = 5000,
    frame_cluster_threshold: float = 0.92,
):
    """Cluster each scene in memory and store only representative frames."""
    print("Building frame_cluster_reps table...")
    if "frame_cluster_reps" in db.table_names():
        print("Table 'frame_cluster_reps' already exists. Dropping it before rebuild.")
        db.drop_table("frame_cluster_reps")

    scene_videos = _collect_scene_videos(scene_root)
    device = torch.device(config.device)
    size_t = int(config.get("size_t", 224))
    reps_batch = []
    table_created = False
    total_reps = 0
    empty_scenes = 0
    current_video_id = None
    global_frame_offset = 0

    def flush_batch():
        nonlocal reps_batch, table_created, total_reps
        if not reps_batch:
            return
        frame = pd.DataFrame(reps_batch)
        if not table_created:
            db.create_table("frame_cluster_reps", data=frame, mode="overwrite")
            table_created = True
        else:
            db.open_table("frame_cluster_reps").add(frame)
        total_reps += len(reps_batch)
        reps_batch = []

    for scene_video in tqdm(scene_videos, desc="Building frame_cluster_reps"):
        video_id = os.path.basename(os.path.dirname(scene_video))
        if video_id.endswith(".mp4"):
            video_id = video_id[:-4]
        if video_id != current_video_id:
            current_video_id = video_id
            global_frame_offset = 0

        clip_id = os.path.basename(scene_video)
        if clip_id.endswith(".mp4"):
            clip_id = clip_id[:-4]

        embeddings, frame_bytes, scene_total_frames = extract_all_frames_and_embeddings(
            scene_video, fgclip_model, fgclip_processor, device, size_t
        )
        frame_ids = list(
            range(global_frame_offset, global_frame_offset + scene_total_frames)
        )
        global_frame_offset += scene_total_frames
        if scene_total_frames == 0:
            empty_scenes += 1
            continue

        for rep in _cluster_representatives(
            frame_ids, embeddings, frame_bytes, frame_cluster_threshold
        ):
            reps_batch.append(
                {
                    "clip_id": clip_id,
                    "video_id": video_id,
                    "cluster_frame_id": rep["cluster_frame_id"],
                    "frame_image": rep["frame_image"],
                    "member_count": rep["member_count"],
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
            f"{total_reps} reps, {empty_scenes} empty scenes."
        )
    else:
        print("Warning: No frame_cluster_reps data was written.")


def build_index(
    intern_model,
    config,
    scene_root: str,
    feature_cache_root: str,
    db_path: str,
    build_clips: bool = True,
    build_frame_cluster_reps: bool = False,
    frame_cluster_threshold: float = 0.92,
    auto_extract_missing_cache: bool = True,
    extractor_python_cmd: str = "conda run -n internvideo python",
    extractor_script_path: str = None,
    fgclip_model=None,
    fgclip_processor=None,
):
    db = lancedb.connect(db_path)

    if not build_clips and not build_frame_cluster_reps:
        print("No tables requested to be built.")
        return db

    if build_clips and "clips" in db.table_names():
        print(
            "Table 'clips' already exists. Rebuilding is currently set to drop. Adjust manually if needed."
        )
        if os.path.normpath(db_path) != os.path.normpath(TEST_LANCEDB_PATH):
            db.drop_table("clips")

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

    if build_frame_cluster_reps:
        build_frame_cluster_reps_table(
            db,
            config,
            scene_root,
            fgclip_model,
            fgclip_processor,
            frame_cluster_threshold=frame_cluster_threshold,
        )
        print_table_summary(db, "frame_cluster_reps")

    return db


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--build_clips", action="store_true", help="Build clips table.")
    parser.add_argument(
        "--build_frame_cluster_reps",
        action="store_true",
        help="Build frame_cluster_reps directly from scene clips.",
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

    if not any([args.build_clips, args.build_frame_cluster_reps]):
        args.build_clips = True
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
    if args.build_frame_cluster_reps:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        fgclip_model, fgclip_processor = load_fgclip_model(args.fgclip_model_path)
        fgclip_model = fgclip_model.to(device)

    build_index(
        intern_model=intern_model,
        config=config,
        scene_root=args.scene_root,
        feature_cache_root=args.feature_cache_root,
        db_path=args.db_path,
        build_clips=args.build_clips,
        build_frame_cluster_reps=args.build_frame_cluster_reps,
        frame_cluster_threshold=args.frame_cluster_threshold,
        auto_extract_missing_cache=not args.disable_auto_extract_missing_cache,
        extractor_python_cmd=args.extractor_python_cmd,
        extractor_script_path=args.extractor_script_path,
        fgclip_model=fgclip_model,
        fgclip_processor=fgclip_processor,
    )


if __name__ == "__main__":
    main()
