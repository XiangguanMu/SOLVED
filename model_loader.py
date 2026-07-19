import os
import sys
import threading
import time

import numpy as np
import torch

from paths import INTERNVIDEO2_ROOT, QWEN3_PYTHON

project_root = INTERNVIDEO2_ROOT
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "multi_modality"))

from demo.utils import setup_internvideo2
from utils.config import Config, eval_dict_leaf


def _resolve_internvideo_config_paths(config):
    """将 InternVideo2 配置中的相对路径修正为绝对路径。"""
    mm_root = os.path.join(project_root, "multi_modality")
    text_cfg = config.get("model", {}).get("text_encoder", {}).get("config")
    if isinstance(text_cfg, str) and not os.path.isabs(text_cfg):
        config.model.text_encoder.config = os.path.join(mm_root, text_cfg)


def load_query_encoder():
    """加载 InternVideo2 模型，用于将文本编码为特征向量。"""
    config = Config.from_file(
        os.path.join(
            project_root, "multi_modality", "demo/internvideo2_stage2_config.py"
        )
    )
    config = eval_dict_leaf(config)
    _resolve_internvideo_config_paths(config)
    intern_model, _ = setup_internvideo2(config)
    return intern_model, config


def query_to_embedding(intern_model, query: str) -> np.ndarray:
    """将自然语言查询转化为归一化的稠密特征向量。"""
    with torch.no_grad():
        text_feat = intern_model.get_txt_feat(query)
    return text_feat.detach().cpu().numpy().astype(np.float32).reshape(-1)


def load_models_parallel(qwen3_model_path: str) -> dict:
    """并发加载 InternVideo2 和 Qwen3-VL，返回包含所有模型的字典。
    Qwen3-VL 不再在当前进程加载；改为通过子进程 bridge (qwen3_bridge.py) 调用。
    """
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading retrieval models (InternVideo2 only; Qwen3-VL via subprocess bridge)..."
    )
    results = {}
    t1 = threading.Thread(
        target=lambda: results.update({"intern_model": load_query_encoder()[0]})
    )
    t1.start()
    t1.join()

    # Store only the model path for the subprocess bridge
    results["qwen3_model_path"] = qwen3_model_path
    results["qwen3_bridge_script"] = os.path.join(
        os.path.dirname(__file__), "qwen3_bridge.py"
    )
    results["qwen3_python"] = QWEN3_PYTHON
    return results
