# Tools

- `fix_in_place.py`: legacy repair for an old `frames` table. The current builder does not create that table.

```sh
python tools/fix_in_place.py --db_path "$LANCEDB_PATH"
```
