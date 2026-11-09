"""
Standalone Qwen3-VL inference script using SGLang Engine.
Runs in sglang-qwen3 environment as a subprocess.
Reads prompts and images from a JSON file, writes scores back.
"""

import json
import sys
import os
import traceback
import re
from sglang import Engine
from transformers import AutoProcessor


def load_model(model_path: str):
    print(f"Loading Qwen3-VL model via SGLang from {model_path}...", file=sys.stderr)

    # Load processor for chat template
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # mem_fraction_static=0.3 is conservative and safe
    llm = Engine(
        model_path=model_path,
        enable_multimodal=True,
        mem_fraction_static=0.3,
        disable_cuda_graph=True,
    )
    return processor, llm


def score_prompts(processor, llm, prompts_data):
    """Score one prompt batch using an already loaded model."""

    scores = [0.0] * len(prompts_data)
    active_indices = [i for i, p in enumerate(prompts_data) if p is not None]

    if not active_indices:
        print("No active prompts to process.", file=sys.stderr)
        return scores

    print(
        f"SGLang Engine loaded. Batch processing {len(active_indices)} active prompts...",
        file=sys.stderr,
    )

    batch_inputs = []
    sampling_params = {"max_new_tokens": 16, "temperature": 0.0}

    for idx in active_indices:
        p = prompts_data[idx]
        img_path = p.get("image_path")

        messages = [
            {
                "role": "system",
                "content": "You are an AI that evaluates image-text matching. Respond ONLY with a single float number between 0 and 1.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": p["prompt"]},
                ],
            },
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Use 'prompt' instead of 'text' as key
        batch_inputs.append({"prompt": text, "image_data": img_path})

    try:
        # Mini-batching within the script to ensure stability
        mini_batch_size = 10
        all_outputs = []

        for i in range(0, len(batch_inputs), mini_batch_size):
            chunk = batch_inputs[i : i + mini_batch_size]

            # FIX: Separate prompts and image_data into two lists
            prompts = [c["prompt"] for c in chunk]
            images = [c["image_data"] for c in chunk]

            chunk_outputs = llm.generate(
                prompts, image_data=images, sampling_params=sampling_params
            )
            all_outputs.extend(chunk_outputs)

            # # Critical Debug: See what the model is saying
            # if chunk_outputs:
            #     raw_out = chunk_outputs[0]["text"].strip()
            #     print(f"DEBUG: Model response for prompt {i}: '{raw_out}'", file=sys.stderr)

        # Robust regex for float: matches 0.5, .5, 1.0, 1 etc.
        val_pattern = re.compile(r"(\d*\.\d+|\d+)")

        for i, idx in enumerate(active_indices):
            out_text = all_outputs[i]["text"].strip()
            match = val_pattern.search(out_text)
            if match:
                try:
                    scores[idx] = float(match.group(1))
                except ValueError:
                    scores[idx] = 0.0
            else:
                # If regex fails, score is 0.0
                pass

    except Exception as e:
        print(f"Batch inference failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    return scores


def run_inference(input_path: str, output_path: str):
    """Load model, score a single JSON batch, and write its scores."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts_data = data["prompts"]
    processor, llm = load_model(data["model_path"])
    scores = score_prompts(processor, llm, prompts_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"scores": scores}, f)
    print(f"Done. Wrote {len(scores)} scores to {output_path}", file=sys.stderr)


def run_stream(model_path: str):
    """Keep one model resident and process newline-delimited prompt batches."""
    processor, llm = load_model(model_path)
    print("SGLang Engine ready for streaming batches.", file=sys.stderr)
    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        request = json.loads(raw_line)
        if request.get("command") == "close":
            break
        request_id = request.get("request_id")
        scores = score_prompts(processor, llm, request.get("prompts", []))
        response = {"request_id": request_id, "scores": scores}
        print(
            "BRIDGE_RESULT\t" + json.dumps(response, ensure_ascii=False),
            flush=True,
        )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--stream":
        run_stream(sys.argv[2])
    elif len(sys.argv) == 3:
        run_inference(sys.argv[1], sys.argv[2])
    else:
        sys.exit(1)
