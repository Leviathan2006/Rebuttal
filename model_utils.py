"""
Model loading and chain generation.
Handles standard instruct models and reasoning models (DeepSeek-R1, Qwen3 thinking).
Strips <think>...</think> blocks from reasoning models before step segmentation.
"""

import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from typing import Optional


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_model_and_tokenizer(model_cfg: dict, cache_dir: Optional[str] = None):
    """
    Load a model in 4-bit NF4 quantization, matching the paper's setup.
    Returns (model, tokenizer).
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["hf_id"],
        cache_dir=cache_dir,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["hf_id"],
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────
# Chain generation
# ─────────────────────────────────────────────

def generate_chain(
    prompt: str,
    model,
    tokenizer,
    model_cfg: dict,
    gen_cfg: dict,
) -> str:
    """
    Generate a reasoning chain for a single prompt.
    Returns the raw generated text (thinking tags stripped for reasoning models).
    """
    # Build chat messages
    messages = [{"role": "user", "content": prompt}]

    # Qwen3 thinking mode: pass enable_thinking via generation_config
    extra_kwargs = {}
    if model_cfg.get("enable_thinking"):
        extra_kwargs["enable_thinking"] = True

    chat_input = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        **extra_kwargs,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            chat_input,
            do_sample=gen_cfg["do_sample"],
            repetition_penalty=gen_cfg["repetition_penalty"],
            max_new_tokens=gen_cfg["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][chat_input.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Strip <think>...</think> for reasoning models — we only keep the final answer portion
    if model_cfg.get("strip_thinking_tags"):
        text = strip_thinking(text)

    return text.strip()


def strip_thinking(text: str) -> str:
    """
    Remove <think>...</think> blocks produced by R1/Qwen3 thinking models.
    If no closing tag, drops everything after <think>.
    """
    # Complete block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Unclosed block (model stopped mid-think)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


# ─────────────────────────────────────────────
# Step segmentation  (matches paper heuristics)
# ─────────────────────────────────────────────

def segment_steps(text: str) -> list[str]:
    """
    Segment a reasoning chain into individual steps.
    Priority:
      1. Explicit numbered list (1. ... 2. ... or Step 1: ...)
      2. Sentence-boundary heuristic (fallback)
    Returns list of non-empty step strings.
    """
    # Try numbered list pattern
    numbered = re.split(r"\n\s*(?:Step\s*)?\d+[\.\)]\s*", text)
    numbered = [s.strip() for s in numbered if s.strip()]
    if len(numbered) >= 2:
        return numbered

    # Try newline-separated blocks
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 2:
        return lines

    # Sentence boundary fallback
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences if sentences else [text.strip()]
