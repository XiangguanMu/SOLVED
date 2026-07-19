import lancedb


def setup_index(
    db_path,
    build_index,
    num_partitions,
    num_sub_vectors,
    index_name="temp_split_clip_index",
    index_column="clip_embedding",
):
    """初始化 LanceDB 连接，可选创建 IVF-PQ 索引。"""
    print(f"Connecting to LanceDB at '{db_path}'...")
    db = lancedb.connect(db_path)
    try:
        table_names = db.table_names()
    except Exception:
        table_names = db.list_tables()
    if "clips" not in table_names:
        print("Error: 'clips' table not found in the database. Please build it first.")
        return None, None
    clips_tbl = db.open_table("clips")

    has_index = False
    try:
        existing_indices = clips_tbl.list_indices()
        has_index = any(
            index_name == (getattr(idx, "name", None) or getattr(idx, "index_name", None))
            for idx in existing_indices
        )
    except Exception:
        pass

    if build_index:
        if not has_index:
            print(
                f"===== Index '{index_name}' not found. Creating IVFPQ index with "
                f"{num_partitions} partitions and {num_sub_vectors} sub-vectors..."
            )
            clips_tbl.create_index(
                vector_column_name=index_column,
                metric="cosine",
                name=index_name,
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
            )
        else:
            print(f"Index '{index_name}' already exists.")
    else:
        print("Index building is disabled. Proceeding with full-scan search...")
    return db, clips_tbl


def cleanup_index(clips_tbl, drop_index_on_exit, index_name):
    """若启用退出时删除索引，则移除临时向量索引。"""
    if not drop_index_on_exit or clips_tbl is None:
        return
    print(f"\nCleaning up: Checking for index '{index_name}' to remove...")
    try:
        indices = clips_tbl.list_indices()
        dropped = False
        for idx in indices:
            idx_name = getattr(idx, "name", None) or getattr(idx, "index_name", None)
            if idx_name == index_name:
                print(f"Dropping the specific index: '{idx_name}'...")
                clips_tbl.drop_index(idx_name)
                dropped = True
        print("Index dropped successfully." if dropped else "No matching index found to drop.")
    except Exception as e:
        print(f"Note: Index cleanup skipped or not supported: {e}")
