# SOLVED

A prototype video retrieval system for compositional natural-language queries over long videos.
SOLVED indexes clips and frames in LanceDB, decomposes complex queries into atomic sub-queries with an LLM, then ranks temporally consistent candidate sequences with InternVideo2 coarse retrieval and Qwen3-VL fine-grained scoring.

## Setup Instructions

1. The project uses `conda` (recommended) or a Python virtual environment. Clone the repo and install Python dependencies:

```sh
git clone https://github.com/XiangguanMu/SOLVED.git
cd SOLVED

conda create -n solved python=3.10 -y
conda activate solved
pip install -r requirements.txt
```

2. Export the project root and path overrides (copy `.env.example` as a starting point):

```sh
export SOLVED_ROOT="$(pwd)"

# Externally installed InternVideo2 checkout (required for encoding)
export INTERNVIDEO2_ROOT="/path/to/InternVideo2"

# Local models
export FGCLIP_MODEL_PATH="$(pwd)/models/fg-clip-base"
export QWEN3_VL_MODEL_PATH="$(pwd)/models/Qwen3-VL-2B-Instruct-FP8"
export QUERY_SPLITTER_MODEL_PATH="$(pwd)/query_splitter/qwen-7b-local"

# Optional: Python interpreter for the Qwen3-VL SGLang bridge
# export QWEN3_PYTHON="/path/to/conda/envs/sglang-qwen3/bin/python"

# LanceDB + dataset roots used by the table builder
export LANCEDB_PATH="$(pwd)/lancedb_data/lovr"
export VIDEOS_ROOT="/path/to/merged_videos"
export SCENE_ROOT="/path/to/scene_clips"
export FEATURE_CACHE_ROOT="/path/to/feature_cache"
```

Defaults are centralized in `paths.py`. Any of the variables above can be overridden without editing source files.

3. Prepare external models and code that are **not** shipped in this repository:

| Component | Role | Notes |
|---|---|---|
| InternVideo2 | Text / clip encoder | Point `INTERNVIDEO2_ROOT` at a local checkout |
| FG-CLIP | Frame embeddings | Save under `models/fg-clip-base` or set `FGCLIP_MODEL_PATH` |
| Qwen3-VL (FP8) | Soft-constraint scoring via `qwen3_bridge.py` | Set `QWEN3_VL_MODEL_PATH` |
| Qwen query splitter | Offline query decomposition | Set `QUERY_SPLITTER_MODEL_PATH` |

## Prepare Data

SOLVED currently evaluates on LoVR, MSRVTT, and VideoChapters-style subsets.
Place (or symlink) your prepared videos / scene clips / feature caches, then point the env vars above at those directories.

Expected layout for table construction:

```
$VIDEOS_ROOT/                 # merged source videos: <video_id>.mp4
$SCENE_ROOT/<video_id>/       # scene clip mp4s for each video
$FEATURE_CACHE_ROOT/          # precomputed InternVideo2 clip features (*.pt)
$LANCEDB_PATH/                # LanceDB directory (created by the builder)
```

Build LanceDB tables:

```sh
# Build all tables (videos / clips / frames / frame_cluster_reps)
python build_lancedb_tables.py \
  --videos_root "$VIDEOS_ROOT" \
  --scene_root "$SCENE_ROOT" \
  --feature_cache_root "$FEATURE_CACHE_ROOT" \
  --db_path "$LANCEDB_PATH" \
  --fgclip_model_path "$FGCLIP_MODEL_PATH"

# Or build selectively
python build_lancedb_tables.py --build_videos --db_path "$LANCEDB_PATH"
python build_lancedb_tables.py --build_clips --db_path "$LANCEDB_PATH"
python build_lancedb_tables.py --build_frames --db_path "$LANCEDB_PATH"
python build_lancedb_tables.py --build_frame_cluster_reps --db_path "$LANCEDB_PATH"
```

### LanceDB Schema (brief)

- `clips`: base clips + temporal chunks with `clip_embedding` for coarse / sub-query retrieval
- `frames`: per-frame embeddings and JPEG bytes
- `frame_clusters` / `frame_cluster_reps`: cluster representatives used by Qwen3-VL soft scoring

See the comments in `build_lancedb_tables.py` and the previous schema notes in git history for field-level details.

## Sanity Checks

After building tables, verify schemas and spot-check clip consistency:

```sh
python sanity_checks/check_lancedb_tables.py --db_path "$LANCEDB_PATH"

python sanity_checks/extract_video_clip.py \
  --video_name <video_id> \
  --start_frame 100 \
  --end_frame 150 \
  --videos_root "$VIDEOS_ROOT"
```

## Example Usage

1. Run end-to-end retrieval (dataset name is inferred from the basename of `--db_path`, e.g. `lovr`):

```sh
python query_lancedb.py \
  --db_path "$LANCEDB_PATH" \
  --qwen3_model_path "$QWEN3_VL_MODEL_PATH" \
  --model_path "$QUERY_SPLITTER_MODEL_PATH" \
  --top_k 5
```

Useful flags:

```sh
# Only run query index 0
python query_lancedb.py --db_path "$LANCEDB_PATH" --query_index 0

# Only run the first N queries
python query_lancedb.py --db_path "$LANCEDB_PATH" --max_queries 2

# Build / drop a temporary IVF-PQ index around the run
python query_lancedb.py --db_path "$LANCEDB_PATH" --build_index --drop_index_on_exit
```

Console output is tee'd to `experiments/data/<dataset>.log` by default.

2. Export structured results and compute IoU against `ground_truth_segments`:

```sh
python experiments/export_query_results.py lovr
# also supports: msrvtt | videochapter
```

Outputs are written under `experiments/results/<dataset>/`.

## Project Layout

```
SOLVED/
├── query_lancedb.py          # CLI entry: split queries + run retrieval
├── pipeline.py               # End-to-end ranking pipeline
├── search.py                 # Indexed clip search + beam search
├── model_loader.py           # InternVideo2 loading helpers
├── build_lancedb_tables.py   # LanceDB table construction
├── paths.py                  # Path defaults (overridable via env)
├── qwen3_bridge.py           # Subprocess bridge for Qwen3-VL scoring
├── query_splitter/           # Offline LLM query decomposition
├── eval_data/                # queries + ground_truth_segments
├── eval/                     # Small retrieval / splitting tests
├── sanity_checks/            # Schema / consistency check scripts
├── tools/                    # Maintenance utilities
└── experiments/              # Result export + plotting
```

## Eval Queries

Hand-authored evaluation queries and segment-level ground truth live in:

- `eval_data/lovr.py`
- `eval_data/msrvtt.py`
- `eval_data/videochapter.py`

Ground truth is stored as `ground_truth_segments` (`video_id`, `frame_start`, `frame_end`) and consumed by `experiments/export_query_results.py` for IoU metrics.
