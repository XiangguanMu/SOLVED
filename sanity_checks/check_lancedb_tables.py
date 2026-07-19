import argparse
import os
import sys

import lancedb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import LANCEDB_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Check length and attributes (schema) of LanceDB tables."
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=LANCEDB_PATH,
        help="Path to LanceDB database.",
    )

    args = parser.parse_args()

    print(f"Connecting to LanceDB at '{args.db_path}'...\n")
    try:
        db = lancedb.connect(args.db_path)
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        sys.exit(1)

    existing_tables = db.table_names()

    target_tables = ["videos", "clips", "frames", "frame_clusters"]

    for table_name in target_tables:
        print(f"================ Table: {table_name} ================")
        if table_name not in existing_tables:
            print(f"> Table '{table_name}' does not exist in the database.\n")
            continue

        try:
            tbl = db.open_table(table_name)

            # Fast row counting logic
            count = "Unknown"
            if hasattr(tbl, "count_rows"):
                count = tbl.count_rows()  # lancedb 0.5+
            else:
                try:
                    count = getattr(
                        tbl.to_lance(), "count_rows"
                    )()  # fallback via core lance pyarrow dataset
                except Exception:
                    # Last resort fallback if count_rows is totally unavailable
                    count = (
                        len(tbl.search().limit(0).to_pandas())
                        if hasattr(tbl, "search")
                        else len(tbl.to_pandas())
                    )

            import pandas as pd
            import numpy as np

            first_row = tbl.search().limit(17).to_pandas()

            # 将长特征如图像字节和冗长的 embedding 转化为人类可读的短形式，以免 CSV 爆炸
            for col in first_row.columns:
                if "embedding" in col:
                    first_row[col] = first_row[col].apply(
                        lambda x: (
                            f"[{x[0]:.4f}, {x[1]:.4f}, ..., {x[-1]:.4f}]"
                            if isinstance(x, (list, np.ndarray)) and len(x) > 2
                            else x
                        )
                    )
                elif "image" in col or (
                    len(first_row) > 0 and isinstance(first_row[col].iloc[0], bytes)
                ):
                    first_row[col] = "<binary_data>"

            out_text = f"> Length (Row Count): {count}\n"

            unique_cluster_text = ""
            if table_name == "frame_clusters":
                try:
                    df_all = tbl.to_pandas()
                    if "cluster_frame_id" in df_all.columns:
                        unique_clusters = df_all["cluster_frame_id"].nunique()
                        unique_cluster_text = (
                            f"> Unique cluster_frame_ids: {unique_clusters}\n"
                        )
                except Exception as e:
                    unique_cluster_text = (
                        f"> Failed to count unique cluster_frame_ids: {e}\n"
                    )

            out_text += unique_cluster_text
            out_text += f"> Schema (Attributes):\n{tbl.schema}\n\n"
            out_text += f"Sample record from '{table_name}' (CSV Format):\n{first_row.to_csv(index=False)}\n"

            print(f"> Length (Row Count): {count}")
            if unique_cluster_text:
                print(unique_cluster_text.strip())
            print(f"> Schema (Attributes):\n{tbl.schema}\n")
            print(
                f"Detailed sample records have been cleanly saved as CSV format into 'check_output.txt'."
            )

            with open("logs/check_output.txt", "a", encoding="utf-8") as f:
                f.write(f"================ Table: {table_name} ================\n")
                f.write(out_text)

        except Exception as e:
            print(f"> Error reading table '{table_name}': {e}\n")


if __name__ == "__main__":
    main()
