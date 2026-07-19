import re

import numpy as np
import pandas as pd


def extract_chunk_t(clip_id: str):
    """从 clip_id 后缀中提取时间块序号 chunk_t。"""
    match = re.search(r"_(\d+)_\d+_\d+$", clip_id)
    if match is None:
        return None
    return int(match.group(1))


def time_order_ok(entries):
    """校验各时间步的 best_t 是否满足严格递增的时序约束。"""
    if not entries or any(e.get("best_t") is None for e in entries):
        return False
    step_to_ts = {}
    for e in entries:
        step_to_ts.setdefault(e["time_step"], []).append(e["best_t"])
    prev_max_t = None
    for step in sorted(step_to_ts.keys()):
        cur_min_t = min(step_to_ts[step])
        cur_max_t = max(step_to_ts[step])
        if prev_max_t is not None and cur_min_t < prev_max_t:
            return False
        prev_max_t = cur_max_t
    return True


def search_clips_indexed(
    clips_tbl,
    query_embedding: np.ndarray,
    top_k: int = 5,
    nprobes: int = 100,
    where_clause: str = None,
    return_raw: bool = False,
):
    """在候选集合上做余弦向量检索，返回按场景去重后的 top-k 结果。

    return_raw=True 时返回 (df_all, df_dedup)，否则只返回 df_dedup。
    """
    fetch_limit = 2000 if where_clause else top_k * 10
    query_obj = (
        clips_tbl.search(query_embedding)
        .metric("cosine")
        .nprobes(nprobes)
        .limit(fetch_limit)
    )
    if where_clause:
        query_obj = query_obj.where(where_clause)
    df = query_obj.to_pandas()
    if df.empty:
        print("No results found.")
        return (df, df) if return_raw else df
    df["score"] = 1.0 - df["_distance"]
    df["scene_name"] = df["clip_id"].str.replace(r"_\d+_\d+_\d+$", "", regex=True)
    idx_of_max_per_scene = df.groupby("scene_name")["score"].idxmax()
    df_dedup = df.loc[idx_of_max_per_scene].copy()
    df_dedup = df_dedup.sort_values(by="score", ascending=False).reset_index(drop=True)
    if return_raw:
        return df, df_dedup.head(top_k)
    return df_dedup.head(top_k)


def find_candidate_sequences(S, T, penalty_weight=1.0, top_k=3, epsilon=0.0):
    """Beam Search：在得分矩阵 S/T 上搜索最优时序序列，对时间倒流施加惩罚。"""
    if not S or not S[0]:
        return []
    L = len(S)
    k = len(S[0])
    beam = sorted(
        [(S[0][j], [(S[0][j], T[0][j], j)]) for j in range(k)],
        key=lambda x: x[0], reverse=True,
    )[:top_k]
    for i in range(1, L):
        next_beam = []
        for score, seq in beam:
            prev_t = seq[-1][1]
            for j in range(k):
                curr_s, curr_t = S[i][j], T[i][j]
                violation = max(0, (prev_t - curr_t) - epsilon)
                next_beam.append((score + curr_s - penalty_weight * violation, seq + [(curr_s, curr_t, j)]))
        beam = sorted(next_beam, key=lambda x: x[0], reverse=True)[:top_k]
    return [{"score": b[0] / L, "raw_total_score": b[0], "sequence": b[1]} for b in beam]
