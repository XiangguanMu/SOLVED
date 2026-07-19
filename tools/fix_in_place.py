import argparse
import os
import sys

import lancedb
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import LANCEDB_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Fix frames.frame_id to global video frame indices in-place."
    )
    parser.add_argument("--db_path", type=str, default=LANCEDB_PATH)
    args = parser.parse_args()

    db = lancedb.connect(args.db_path)

    print("Loading clips...")
    clips_tbl = db.open_table("clips").to_pandas()
    base_clips = clips_tbl[~clips_tbl["clip_id"].str.contains(r"_\d+_\d+_\d+$")]
    clip_start_map = dict(zip(base_clips["clip_id"], base_clips["clip_frame_start"]))

    print("Scanning clip IDs to build index map...")
    frames_tbl = db.open_table("frames")
    meta_df = frames_tbl.to_lance().to_table(columns=["clip_id", "frame_id"]).to_pandas()
    meta_df["orig_idx"] = np.arange(len(meta_df))

    meta_df.sort_values(by=["clip_id", "frame_id"], inplace=True)
    meta_df["relative_idx"] = meta_df.groupby("clip_id").cumcount()
    meta_df["true_start"] = meta_df["clip_id"].map(clip_start_map).fillna(0).astype("int64")
    meta_df["new_frame_id"] = meta_df["true_start"] + meta_df["relative_idx"]
    meta_df["offset"] = (meta_df["new_frame_id"] - meta_df["frame_id"]).astype("int64")

    affected = meta_df[meta_df["offset"] != 0]
    diffs = len(affected)
    print(f"Total rows requiring update: {diffs}")

    if diffs == 0:
        print("No errors found. Table is correct.")
        sys.exit(0)

    # Validate that all rows inside a clip have the EXACT SAME offset
    issue_check = affected.groupby("clip_id")["offset"].nunique()
    if (issue_check > 1).any():
        print("CRITICAL ERROR: Some clips have non-uniform offsets. Update logic will fail.")
        print("Clips with varying offsets:", issue_check[issue_check > 1])
        sys.exit(1)

    # Group clips by their offset
    offset_map = (
        affected.groupby("offset")["clip_id"].apply(lambda x: list(set(x))).to_dict()
    )
    print(f"Total distinct offset groups: {len(offset_map)}")

    for offset_val, clips in tqdm(offset_map.items(), desc="Updating groups"):
        if offset_val == 0:
            continue

        clips_str = ", ".join([f"'{c}'" for c in clips])
        where_clause = f"clip_id IN ({clips_str})"

        op = "+" if offset_val > 0 else "-"
        abs_val = abs(offset_val)
        val_expr = f"frame_id {op} {abs_val}"

        frames_tbl.update(where=where_clause, values_sql={"frame_id": val_expr})

    print("All updates completed successfully in-place!")


if __name__ == "__main__":
    main()
