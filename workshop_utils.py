"""Model helpers for the Gemma 4 GRPO Sudoku notebook.

Loading the model, generating text, building the prompt dataset, saving and
exporting: the plumbing around training, kept here so the notebook stays on the
RL task itself.
"""
import os

from unsloth import FastVisionModel
from datasets import Dataset
from safetensors import safe_open
from transformers import TextStreamer

gemma4_models = [
    # Gemma-4 instruct models:
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    #"unsloth/gemma-4-31B-it",
    #"unsloth/gemma-4-26B-A4B-it",
    # Gemma-4 base models:
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E4B",
    #"unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B",
] # More models at https://huggingface.co/unsloth


def load_model(model_name = "unsloth/gemma-4-E2B-it", max_seq_length = 4096, lora_rank = 32):
    """Load Gemma 4 in 16-bit and attach LoRA adapters. Returns (model, tokenizer)."""
    # The workshop image ships the model under /models; elsewhere, download it from Hugging Face.
    if os.path.isdir("/models/" + model_name):
        model_name = "/models/" + model_name

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = False, # False for LoRA 16bit
        fast_inference = False, # Enable vLLM fast inference
    )
    model = FastVisionModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = lora_rank*2, # *2 speeds up training
        use_gradient_checkpointing = "unsloth", # Reduces memory usage
        random_state = 3407,
    )
    model.config._attn_implementation = "sdpa" # Use this to disable flash attention to avoid OOM
    return model, tokenizer


def generate(model, tokenizer, question, max_new_tokens = 512, temperature = 1.0):
    """Stream the model's answer to one user message and return it as text."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize = False,
        add_generation_prompt = True,
    )
    inputs = tokenizer(text = text, add_special_tokens = False, return_tensors = "pt").to("cuda")
    output = model.generate(
        **inputs,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
        max_new_tokens = max_new_tokens,
        use_cache = True,
        temperature = temperature,
        top_p = 0.95,
        top_k = 64,
    )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens = True)


def make_dataset(tokenizer, prompt, size = 1000):
    """Build the GRPO dataset: the same prompt repeated, with no answers. Returns (dataset, prompt_length)."""
    messages = [{"role": "user", "content": prompt}]
    dataset = Dataset.from_list([{"prompt": messages, "answer": 0}] * size)
    prompt_length = len(tokenizer.apply_chat_template(messages, add_generation_prompt = True))
    return dataset, prompt_length


def save_adapters(model, tokenizer, path = "grpo_saved_lora"):
    """Save the LoRA adapters and check that training actually changed them."""
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    # LoRA B matrices start at zero, so any non-zero B means training updated the adapters.
    with safe_open(os.path.join(path, "adapter_model.safetensors"), framework = "pt") as f:
        b_keys = [key for key in f.keys() if "lora_B" in key]
        updated = sum(bool((f.get_tensor(key) != 0).any()) for key in b_keys)
    assert updated > 0, "Every LoRA B matrix is still zero: training did not update the adapters"
    print(f"Saved LoRA adapters to {path}/ ({updated} of {len(b_keys)} LoRA B matrices updated)")


def export_model(model, tokenizer, method, save_dir = "model", repo_id = None, token = ""):
    """Save or upload the trained model in another format.

    method is "merged_16bit" or "merged_4bit" (adapters folded into the base weights),
    "lora" (adapters only), or a GGUF quantization for llama.cpp: "q8_0", "f16",
    "q4_k_m", "q5_k_m", or a list of these. With repo_id (for example
    "your-hf-username/model") and a token from https://huggingface.co/settings/tokens,
    the result is pushed to the Hugging Face Hub instead of saved to save_dir.
    """
    if method in ("merged_16bit", "merged_4bit"):
        if repo_id: model.push_to_hub_merged(repo_id, tokenizer, save_method = method, token = token)
        else: model.save_pretrained_merged(save_dir, tokenizer, save_method = method)
    elif method == "lora":
        if repo_id:
            model.push_to_hub(repo_id, token = token)
            tokenizer.push_to_hub(repo_id, token = token)
        else:
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
    else:
        if repo_id: model.push_to_hub_gguf(repo_id, tokenizer, quantization_method = method, token = token)
        else: model.save_pretrained_gguf(save_dir, tokenizer, quantization_method = method)
