"""
Step-level verifier using Qwen2.5-7B-Instruct (matches paper setup).
Annotates each step with:
  - correct: bool
  - propagated: bool (True if error came from a prior error, not fresh mistake)
Returns structured JSON matching the paper's annotation schema.
"""

import json
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


VERIFIER_SYSTEM_PROMPT = """You are a step-by-step reasoning verifier.
Given a question, the gold answer, and a reasoning chain split into numbered steps,
annotate each step.

Respond ONLY with valid JSON in this exact format:
{
  "steps": [
    {
      "step_index": 1,
      "correct": true,
      "propagated": false,
      "explanation": "brief reason"
    },
    ...
  ]
}

Rules:
- "correct": true if the step is logically and factually correct given the question.
- "propagated": true if the step is INCORRECT AND its error was caused by an error in a
  previous step (i.e., it inherited a wrong value or conclusion). false if the step
  introduces a NEW independent error, or if it is correct.
- Do not include any text outside the JSON object.
"""

VERIFIER_USER_TEMPLATE = """Question: {question}
Gold answer: {gold_answer}

Reasoning steps:
{steps_text}

Annotate each step."""


def load_verifier(verifier_cfg: dict, cache_dir: Optional[str] = None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        verifier_cfg["hf_id"],
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        verifier_cfg["hf_id"],
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def annotate_chain(
    question: str,
    gold_answer: str,
    steps: list[str],
    verifier_model,
    verifier_tokenizer,
    max_retries: int = 2,
) -> Optional[list[dict]]:
    """
    Annotate a list of steps. Returns list of annotation dicts, or None on parse failure.
    Each dict: {"step_index": int, "correct": bool, "propagated": bool}
    """
    steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    user_content = VERIFIER_USER_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        steps_text=steps_text,
    )

    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(max_retries + 1):
        inputs = verifier_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(verifier_model.device)

        with torch.no_grad():
            output_ids = verifier_model.generate(
                inputs,
                do_sample=False,
                max_new_tokens=1024,
                pad_token_id=verifier_tokenizer.pad_token_id,
                eos_token_id=verifier_tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        new_tokens = output_ids[0][inputs.shape[1]:]
        raw = verifier_tokenizer.decode(new_tokens, skip_special_tokens=True)

        annotations = _parse_verifier_output(raw, n_steps=len(steps))
        if annotations is not None:
            return annotations

        # On retry, add a nudge
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": "Please respond with only the JSON object."})

    return None


def _parse_verifier_output(raw: str, n_steps: int) -> Optional[list[dict]]:
    """Extract and validate JSON from verifier output."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", raw).strip()

    # Find JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None

    try:
        parsed = json.loads(m.group(0))
        steps_annot = parsed.get("steps", [])
        if not steps_annot:
            return None

        result = []
        for ann in steps_annot:
            result.append({
                "step_index": int(ann.get("step_index", 0)),
                "correct": bool(ann.get("correct", True)),
                "propagated": bool(ann.get("propagated", False)),
            })

        # Must cover all steps (allow off-by-one from segmentation)
        if abs(len(result) - n_steps) > 1:
            return None

        return result

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
