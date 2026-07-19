# Sanity Checks

Lightweight scripts for verifying LanceDB table construction.

- `check_lancedb_tables.py`: print schema, row counts, and sample rows for `videos` / `clips` / `frames` / `frame_clusters`
- `extract_video_clip.py`: extract a frame range from a source video to visually compare against stored clip metadata

```sh
# Inspect table schemas
python sanity_checks/check_lancedb_tables.py --db_path "$LANCEDB_PATH"

# Extract a short clip for consistency checking
python sanity_checks/extract_video_clip.py \
  --video_name <video_id> \
  --start_frame 100 \
  --end_frame 150 \
  --videos_root "$VIDEOS_ROOT"
```

Heavier interactive notebooks (`verify_*.ipynb`, etc.) are kept locally and ignored by git.
