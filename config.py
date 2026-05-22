"""
Configuration for error propagation rebuttal experiments.
Edit MODEL_CONFIGS and BENCHMARK_CONFIGS to control what runs.
"""

# ─────────────────────────────────────────────
# Models to evaluate
# ─────────────────────────────────────────────
MODEL_CONFIGS = {
    # Your original baseline (already in paper — used to verify reproducibility)
    "mistral_7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "quantization": "4bit",
        "is_reasoning_model": False,
        "chat_template": "mistral",
    },
    # Reasoning-tuned — addresses Reviewer GEJd / yhU1 / fq4L main concern
    "deepseek_r1_distill_qwen_7b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "quantization": "4bit",
        "is_reasoning_model": True,
        "chat_template": "chatml",
        # R1 wraps thinking in <think>...</think>; we strip it for step segmentation
        "strip_thinking_tags": True,
    },
    # Optional second reasoning model — comment out if GPU memory is tight
    "qwen3_8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "quantization": "4bit",
        "is_reasoning_model": True,
        "chat_template": "chatml",
        "enable_thinking": True,          # pass enable_thinking=True in generation
        "strip_thinking_tags": True,
    },
}

# ─────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────
BENCHMARK_CONFIGS = {
    "gsm8k": {
        "hf_dataset": "gsm8k",
        "hf_split": "test",
        "hf_subset": "main",
        "question_field": "question",
        "answer_field": "answer",           # raw answer string, e.g. "#### 42"
        "answer_parse": "gsm8k",            # strip #### prefix
        "reasoning_type": "mathematical",
        "n_samples": 500,                   # set to None for full test set
        "prompt_template": (
            "Solve step by step. Be concise. "
            "End with 'The answer is [X].' where X is a number.\n\nProblem: {question}"
        ),
    },
    "strategyqa": {
        "hf_dataset": "wics/strategy-qa",
        "hf_split": "test",
        "hf_subset": None,
        "question_field": "question",
        "answer_field": "answer",           # True / False
        "answer_parse": "bool",
        "reasoning_type": "boolean_multihop",
        "n_samples": 300,
        "prompt_template": (
            "Answer yes or no. Think step by step. "
            "End with 'The answer is yes.' or 'The answer is no.'\n\nQuestion: {question}"
        ),
    },
    "csqa": {
        "hf_dataset": "tau/commonsense_qa",
        "hf_split": "validation",
        "hf_subset": None,
        "question_field": "question",
        "answer_field": "answerKey",        # A/B/C/D/E
        "answer_parse": "choice_letter",
        "reasoning_type": "commonsense",
        "n_samples": 500,
        "prompt_template": (
            "Choose the best answer (A/B/C/D/E). Think step by step. "
            "End with 'The answer is [X].' where X is the letter.\n\n"
            "Question: {question}\nChoices: {choices}"
        ),
    },
}

# ─────────────────────────────────────────────
# Verifier model (same as paper)
# ─────────────────────────────────────────────
VERIFIER_CONFIG = {
    "hf_id": "Qwen/Qwen2.5-7B-Instruct",
    "quantization": "4bit",
    "chat_template": "chatml",
}

# ─────────────────────────────────────────────
# Generation settings (match paper exactly)
# ─────────────────────────────────────────────
GENERATION_CONFIG = {
    "do_sample": False,           # greedy decoding
    "repetition_penalty": 1.2,
    "max_new_tokens": 512,
    "temperature": None,          # ignored when do_sample=False
}

# ─────────────────────────────────────────────
# Causal injection experiment settings
# ─────────────────────────────────────────────
INJECTION_CONFIG = {
    "benchmark": "gsm8k",
    "n_chains": 94,               # match paper sample size
    "min_steps": 3,               # only chains with ≥3 steps
    "perturbation": "number_swap",  # change one integer in step 1
    "paraphrase_control": True,   # NEW: also run correct-paraphrase control
    "paraphrase_prompt": (
        "Rewrite the following reasoning step in different words "
        "without changing its meaning or any numbers. "
        "Return only the rewritten step.\n\nStep: {step}"
    ),
}

# ─────────────────────────────────────────────
# Output / paths
# ─────────────────────────────────────────────
OUTPUT_DIR = "results"
CACHE_DIR = "cache"               # HuggingFace model cache
RANDOM_SEED = 42
