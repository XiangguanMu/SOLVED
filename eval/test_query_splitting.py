import os
import sys

# 将项目根目录添加到 sys.path 以便导入 query_splitter
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from query_splitter.splitter import split_queries_offline, dump_split_results_txt


def test_splitting():
    # 测试用的复杂查询列表
    test_queries = [
        "A man in black stands behind the counter and speaks while a woman in a light blue shirt holds a red phone and a man in a gray T-shirt leans forward to listen.",
        "A young man in a pink hoodie speaks to a woman seated at a patio table",
        "In a dim room, a woman with a shoulder bag enters and walks toward a man in a gray hoodie; he turns his head to acknowledge her, and they stop face-to-face for a serious conversation.",
        "At a round-table political studio discussion, a woman in glasses and a brown blazer gestures while a man is speaking, and two men listen.",
        "A woman at a table raises both hands while speaking, then the video switches to a four-window split screen where remote participants continue the discussion.",
        "A boy in a green T-shirt stands by a whiteboard and explains with hand gestures; then another boy in a black hoodie talks and smiles, and a girl in a red hoodie talks with hand gestures.",
        "In an informal office meeting, a standing manager holding a coffee cup gestures while presenting, one seated attendee raises both hands enthusiastically, and another in a pink hoodie listens attentively.",
        "In a split-format talk show, two studio hosts at a desk with microphones speak and gesture, while two remote guests in separate windows watch, listen, and respond during the ongoing discussion.",
    ]

    model_path = os.path.join(project_root, "query_splitter/qwen-7b-local")
    output_path = os.path.join(project_root, "logs/split_results.txt")

    print(f"Starting query splitting test with {len(test_queries)} queries...")
    print(f"Model: {model_path}")

    # 执行分解
    # 注意：此函数内部会加载模型并在处理完后卸载
    all_split_results, split_times, model_load_time = split_queries_offline(
        test_queries, model_path
    )
    print(
        f"Split timings: model_load={model_load_time:.4f}s, "
        f"per_query={', '.join(f'{t:.4f}s' for t in split_times)}"
    )

    # 导出结果
    dump_split_results_txt(test_queries, all_split_results, output_path)

    print("\n" + "=" * 50)
    print(f"Test complete! Results saved to: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    test_splitting()
