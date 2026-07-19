import os
import sys
import time
import torch
import lancedb
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# 确保能导入 query_lancedb 中的函数
sys.path.insert(0, os.path.dirname(__file__))
from query_lancedb import (
    load_query_encoder,
    setup_index,
    fetch_candidate_scenes,
    process_sub_queries,
    evaluate_and_rank_scenes,
)


def test_specific_case():
    # 1. 配置参数
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from paths import FGCLIP_MODEL_PATH, LANCEDB_PATH

    db_path = LANCEDB_PATH
    fgclip_path = FGCLIP_MODEL_PATH
    top_k = 5
    nprobes = 100

    # 定义特定的查询和手动拆分结果 (三个动作均为 Step 1，即并行关系)
    complex_query = "At a round-table political studio discussion, a woman in glasses and a brown blazer gestures while a man is speaking, and two men listen."
    atomic_items = [
        {
            "time_step": 1,
            "clause": "At a round-table political studio discussion, a woman in glasses and a brown blazer gestures.",
        },
        {
            "time_step": 1,
            "clause": "At a round-table political studio discussion, a man is speaking.",
        },
        {
            "time_step": 1,
            "clause": "At a round-table political studio discussion, two men listen.",
        },
    ]

    print(f"\n[Test Case] Query: {complex_query}")
    print("[Manual Split Items]:")
    for item in atomic_items:
        print(f"  Step {item['time_step']}: {item['clause']}")

    # 2. 加载模型
    print("\n--- Loading Models ---")
    intern_model, config = load_query_encoder()

    fgclip_model = (
        AutoModelForCausalLM.from_pretrained(fgclip_path, trust_remote_code=True)
        .eval()
        .to("cuda")
    )
    fgclip_tokenizer = AutoTokenizer.from_pretrained(
        fgclip_path, trust_remote_code=True
    )

    # 3. 连接数据库
    print("\n--- Connecting to DB ---")
    db, clips_tbl = setup_index(db_path, False, 1024, 256)
    frames_tbl = db.open_table("frames")

    # 4. 执行宏观检索 (粗筛)
    print("\n--- Step 1: Macro Retrieval (InternVideo2) ---")
    candidate_scenes, where_clause, _, _ = fetch_candidate_scenes(
        complex_query, intern_model, clips_tbl, nprobes
    )

    if not candidate_scenes:
        print("No candidate scenes found in macro stage.")
        return

    # 5. 执行子查询打分
    print("\n--- Step 2: Sub-query Clip Scoring ---")
    scene_subquery_best_t, _, _, _ = process_sub_queries(
        atomic_items,
        intern_model,
        clips_tbl,
        top_k,
        nprobes,
        where_clause,
        candidate_scenes,
    )

    # 6. 执行细粒度帧搜索与排序 (软约束 Beam Search)
    print("\n--- Step 3: Fine-grained Frame Sequence Search (FGCLIP) ---")
    # evaluate_and_rank_scenes 内部会打印最终的帧中心序列结果
    evaluate_and_rank_scenes(
        0,  # query index
        atomic_items,
        candidate_scenes,
        scene_subquery_best_t,
        frames_tbl,
        fgclip_model,
        fgclip_tokenizer,
        top_k,
        db,
    )


if __name__ == "__main__":
    # 建议在 FGCLIP2 conda 环境下运行
    test_specific_case()
