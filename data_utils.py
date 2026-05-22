"""
Dataset loading and answer parsing.
Handles GSM8K, StrategyQA, CommonsenseQA answer formats.
"""

import re
import random
from typing import Optional
from datasets import load_dataset


def load_benchmark(cfg: dict, seed: int = 42) -> list[dict]:
    """
    Load a benchmark from HuggingFace and return a list of
    {"question": str, "gold_answer": str, "choices": str | None} dicts.
    """
    kwargs = {"split": cfg["hf_split"]}
    if cfg["hf_subset"]:
        kwargs["name"] = cfg["hf_subset"]

    ds = load_dataset(cfg["hf_dataset"], **kwargs)

    samples = []
    for item in ds:
        question = item[cfg["question_field"]]
        raw_answer = item[cfg["answer_field"]]
        gold = parse_answer(raw_answer, cfg["answer_parse"])

        # Build choices string for CSQA
        choices_str = None
        if "choices" in item:
            labels = item["choices"]["label"]
            texts = item["choices"]["text"]
            choices_str = " ".join(f"{l}) {t}" for l, t in zip(labels, texts))

        samples.append({
            "question": question,
            "gold_answer": gold,
            "choices": choices_str,
        })

    # Deterministic subsample
    if cfg["n_samples"] and cfg["n_samples"] < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, cfg["n_samples"])

    return samples


def parse_answer(raw: str, fmt: str) -> str:
    """Normalise raw answer fields into a canonical string."""
    if fmt == "gsm8k":
        # "#### 42" → "42"
        m = re.search(r"####\s*(-?\d[\d,]*)", str(raw))
        if m:
            return m.group(1).replace(",", "")
        # Fallback: last number in string
        nums = re.findall(r"-?\d[\d,]*", str(raw))
        return nums[-1].replace(",", "") if nums else str(raw).strip()

    elif fmt == "bool":
        v = str(raw).strip().lower()
        if v in ("true", "yes", "1"):
            return "yes"
        if v in ("false", "no", "0"):
            return "no"
        return v

    elif fmt == "choice_letter":
        return str(raw).strip().upper()

    return str(raw).strip()


def extract_predicted_answer(text: str, fmt: str) -> Optional[str]:
    """
    Extract predicted answer from generated chain text.
    Looks for 'The answer is X' pattern first, then falls back to heuristics.
    """
    # Primary: "The answer is X."
    m = re.search(r"[Tt]he answer is\s+([A-Za-z0-9,\.\-\s]+?)[\.\n]", text)
    if m:
        raw = m.group(1).strip()
        return normalise_prediction(raw, fmt)

    # Fallback by format
    if fmt == "gsm8k":
        nums = re.findall(r"-?\d[\d,]*", text)
        return nums[-1].replace(",", "") if nums else None

    if fmt == "bool":
        text_lower = text.lower()
        if "yes" in text_lower:
            return "yes"
        if "no" in text_lower:
            return "no"
        return None

    if fmt == "choice_letter":
        m2 = re.search(r"\b([A-E])\b", text[::-1])   # last letter mentioned
        return m2.group(1) if m2 else None

    return None


def normalise_prediction(raw: str, fmt: str) -> str:
    if fmt == "gsm8k":
        nums = re.findall(r"-?\d[\d,]*", raw)
        return nums[0].replace(",", "") if nums else raw.strip()
    if fmt == "bool":
        raw_l = raw.strip().lower()
        if raw_l in ("yes", "true"):
            return "yes"
        if raw_l in ("no", "false"):
            return "no"
        return raw_l
    if fmt == "choice_letter":
        m = re.search(r"[A-E]", raw.upper())
        return m.group(0) if m else raw.strip()
    return raw.strip()


def format_prompt(template: str, sample: dict) -> str:
    """Fill prompt template with sample fields."""
    return template.format(
        question=sample["question"],
        choices=sample.get("choices") or "",
    )
