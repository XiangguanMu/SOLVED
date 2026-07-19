"""Central path defaults for SOLVED.

Override any value via environment variables (see `.env.example` / README).
"""

from __future__ import annotations

import os

PROJECT_ROOT = os.path.abspath(
    os.environ.get("SOLVED_ROOT", os.path.dirname(__file__))
)


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


INTERNVIDEO2_ROOT = _env(
    "INTERNVIDEO2_ROOT",
    os.path.join(PROJECT_ROOT, "third_party", "InternVideo2"),
)

LANCEDB_PATH = _env(
    "LANCEDB_PATH",
    os.path.join(PROJECT_ROOT, "lancedb_data", "lovr"),
)

TEST_LANCEDB_PATH = _env(
    "TEST_LANCEDB_PATH",
    os.path.join(PROJECT_ROOT, "test_lancedb_data"),
)

FGCLIP_MODEL_PATH = _env(
    "FGCLIP_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "fg-clip-base"),
)

QWEN3_VL_MODEL_PATH = _env(
    "QWEN3_VL_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "Qwen3-VL-2B-Instruct-FP8"),
)

QUERY_SPLITTER_MODEL_PATH = _env(
    "QUERY_SPLITTER_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "query_splitter", "qwen-7b-local"),
)

# Python interpreter used by the Qwen3-VL SGLang bridge subprocess.
# Defaults to the current interpreter so the project runs without a custom env.
QWEN3_PYTHON = _env("QWEN3_PYTHON", os.environ.get("PYTHON", "python"))

VIDEOS_ROOT = _env(
    "VIDEOS_ROOT",
    os.path.join(PROJECT_ROOT, "data", "videos"),
)

SCENE_ROOT = _env(
    "SCENE_ROOT",
    os.path.join(PROJECT_ROOT, "data", "scenes"),
)

FEATURE_CACHE_ROOT = _env(
    "FEATURE_CACHE_ROOT",
    os.path.join(PROJECT_ROOT, "data", "feature_cache"),
)
