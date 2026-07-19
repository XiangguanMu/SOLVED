import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from query_splitter.prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES


def split_queries_offline(queries, model_path):
    """Offline LLM query splitting.

    Returns:
        results: list of per-query split item lists
        split_times: list of per-query LLM split wall times (seconds)
        model_load_time: wall time to load the splitter model (seconds)
    """
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading Qwen model into GPU from {model_path}..."
    )
    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
    ).eval()
    model_load_time = time.time() - load_start
    print(f"Splitter model load time: {model_load_time:.4f}s")

    results = []
    split_times = []
    for i, q in enumerate(queries):
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing query {i+1}/{len(queries)} with LLM..."
        )
        query_split_start = time.time()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": FEW_SHOT_EXAMPLES
                + f'\\n现在，请处理以下输入：\\n输入: "{q}"\\n输出:\\n',
            },
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        resp_clean = response.strip()
        if resp_clean.startswith("```json"):
            resp_clean = resp_clean[7:]
        if resp_clean.endswith("```"):
            resp_clean = resp_clean[:-3]
        resp_clean = resp_clean.strip()
        try:
            parsed = json.loads(resp_clean)
            items = parsed.get("items", [])
            results.append(items)
        except Exception as e:
            print(f"Failed to parse JSON: {e}\\nRaw output: {response}")
            results.append([])
        elapsed = time.time() - query_split_start
        split_times.append(elapsed)
        print(f"Query {i+1} split time: {elapsed:.4f}s")
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Unloading Qwen model to free GPU memory..."
    )
    del model
    del tokenizer
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return results, split_times, model_load_time


def dump_split_results_txt(queries, all_split_results, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Query Split Results ===\n")
        f.write(f"generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"total_queries: {len(queries)}\n\n")
        for i, query in enumerate(queries, start=1):
            items = all_split_results[i - 1] if i - 1 < len(all_split_results) else []
            f.write(f"[Query {i}]\n")
            f.write(f"original: {query}\n")
            if items:
                f.write("split_items:\n")
                for j, item in enumerate(items, start=1):
                    f.write(
                        f"  {j}. time_step={item.get('time_step')} | confidence={item.get('confidence', 'N/A')} | clause={item.get('clause', '')}\n"
                    )
                    if item.get("remarks"):
                        f.write(f"     remarks: {item.get('remarks')}\n")
            else:
                f.write("split_items: (empty or parse failed)\n")
            f.write("\n")
