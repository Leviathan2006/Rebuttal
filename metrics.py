"""
Error propagation metrics as defined in the paper.

Metrics:
  EPR  - Error Propagation Rate:    P(c_{i+1}=0 | c_i=0)
  SCR  - Self-Correction Rate:      1 - EPR
  FEF  - First-Error Fatality:      P(wrong final answer | first error at position i)
  EMR  - Error Masking Rate:        fraction of correct-answer chains with ≥1 wrong step
  EID  - Error Influence Decay:     δ(k) = E_ctrl(k) - E_base(k)

Extended:
  Markov order tests (likelihood-ratio)
  EPM  - Error Propagation Matrix   P(c_j=0 | c_i=0) for j > i
"""

import numpy as np
from scipy.stats import chi2
from typing import Optional
from collections import defaultdict
import warnings


# ─────────────────────────────────────────────
# Chain-level data structure
# ─────────────────────────────────────────────

class Chain:
    """Represents one annotated reasoning chain."""

    def __init__(
        self,
        question: str,
        gold_answer: str,
        predicted_answer: Optional[str],
        steps: list[str],
        annotations: list[dict],   # from verifier.annotate_chain()
    ):
        self.question = question
        self.gold_answer = gold_answer
        self.predicted_answer = predicted_answer
        self.steps = steps
        self.annotations = annotations

        # correctness vector: True=correct, False=error
        self.correctness = [a["correct"] for a in annotations[: len(steps)]]
        self.propagated = [a.get("propagated", False) for a in annotations[: len(steps)]]
        self.n_steps = len(self.correctness)

    @property
    def final_answer_correct(self) -> bool:
        if self.predicted_answer is None or self.gold_answer is None:
            return False
        return str(self.predicted_answer).strip().lower() == str(self.gold_answer).strip().lower()

    @property
    def has_error(self) -> bool:
        return any(not c for c in self.correctness)

    @property
    def first_error_position(self) -> Optional[int]:
        """1-indexed position of first error, or None."""
        for i, c in enumerate(self.correctness):
            if not c:
                return i + 1
        return None


# ─────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────

def compute_epr(chains: list[Chain]) -> dict:
    """
    Error Propagation Rate: P(c_{i+1}=0 | c_i=0)
    Returns {"epr": float, "n_pairs": int, "n_propagated": int}
    """
    n_error_followed = 0   # c_i=0 and c_{i+1}=0
    n_error_total = 0      # c_i=0 (with a next step)

    for chain in chains:
        for i in range(chain.n_steps - 1):
            if not chain.correctness[i]:       # step i is wrong
                n_error_total += 1
                if not chain.correctness[i + 1]:   # step i+1 also wrong
                    n_error_followed += 1

    epr = n_error_followed / n_error_total if n_error_total > 0 else float("nan")
    return {
        "epr": epr,
        "scr": 1.0 - epr if not np.isnan(epr) else float("nan"),
        "n_error_pairs": n_error_total,
        "n_propagated_pairs": n_error_followed,
    }


def compute_fef(chains: list[Chain], max_position: int = 6) -> dict:
    """
    First-Error Fatality by position.
    FEF(i) = P(final_answer_wrong | first_error_at_position_i)
    Returns {"fef_by_position": {pos: float}, "fef_1": float}
    """
    pos_wrong = defaultdict(int)
    pos_total = defaultdict(int)

    for chain in chains:
        pos = chain.first_error_position
        if pos is None:
            continue
        # Bin position 6+ together
        bucket = min(pos, max_position)
        pos_total[bucket] += 1
        if not chain.final_answer_correct:
            pos_wrong[bucket] += 1

    fef_by_position = {}
    for pos in sorted(pos_total.keys()):
        fef_by_position[pos] = pos_wrong[pos] / pos_total[pos] if pos_total[pos] > 0 else float("nan")

    fef_1 = fef_by_position.get(1, float("nan"))
    return {
        "fef_by_position": fef_by_position,
        "fef_1": fef_1,
        "n_by_position": dict(pos_total),
    }


def compute_emr(chains: list[Chain]) -> dict:
    """
    Error Masking Rate: fraction of correct-answer chains with ≥1 wrong step.
    """
    correct_answer_chains = [c for c in chains if c.final_answer_correct]
    if not correct_answer_chains:
        return {"emr": float("nan"), "n_correct_chains": 0}

    masked = sum(1 for c in correct_answer_chains if c.has_error)
    return {
        "emr": masked / len(correct_answer_chains),
        "n_correct_chains": len(correct_answer_chains),
        "n_masked": masked,
    }


def compute_accuracy(chains: list[Chain]) -> dict:
    """Final-answer accuracy and step-level accuracy."""
    final_acc = np.mean([c.final_answer_correct for c in chains])

    # Step-level accuracy: fraction of all steps that are correct
    all_correct = [c for chain in chains for c in chain.correctness]
    step_acc = np.mean(all_correct) if all_correct else float("nan")

    # Propagated error fraction
    all_errors = [not c for chain in chains for c in chain.correctness]
    all_prop = [p for chain in chains for p in chain.propagated]
    n_errors = sum(all_errors)
    n_propagated = sum(p and not c for chain in chains for c, p in zip(chain.correctness, chain.propagated))
    prop_fraction = n_propagated / n_errors if n_errors > 0 else float("nan")

    return {
        "final_accuracy": float(final_acc),
        "step_accuracy": float(step_acc),
        "n_chains": len(chains),
        "n_errors": n_errors,
        "propagated_error_fraction": float(prop_fraction),
    }


# ─────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────

def bootstrap_ci(
    chains: list[Chain],
    metric_fn,
    metric_key: str,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap CI for any scalar metric.
    metric_fn(chains) must return a dict with metric_key.
    """
    rng = np.random.default_rng(seed)
    n = len(chains)
    samples = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_chains = [chains[i] for i in idx]
        val = metric_fn(boot_chains).get(metric_key, float("nan"))
        if not np.isnan(val):
            samples.append(val)

    if not samples:
        return (float("nan"), float("nan"))

    alpha = (1 - ci) / 2
    lower = float(np.percentile(samples, 100 * alpha))
    upper = float(np.percentile(samples, 100 * (1 - alpha)))
    return lower, upper


# ─────────────────────────────────────────────
# Markov order likelihood-ratio test
# ─────────────────────────────────────────────

def _transition_counts(sequences: list[list[bool]], order: int) -> dict:
    """Count transitions for a Markov model of given order."""
    counts = defaultdict(lambda: defaultdict(int))
    for seq in sequences:
        # Convert bool to int (0=correct, 1=error)
        s = [0 if c else 1 for c in seq]
        for i in range(order, len(s)):
            context = tuple(s[i - order: i])
            counts[context][s[i]] += 1
    return counts


def _log_likelihood(counts: dict) -> float:
    """Log-likelihood of observed transition counts under MLE parameters."""
    ll = 0.0
    for context, outcomes in counts.items():
        total = sum(outcomes.values())
        for state, cnt in outcomes.items():
            if cnt > 0 and total > 0:
                ll += cnt * np.log(cnt / total)
    return ll


def markov_order_test(chains: list[Chain]) -> dict:
    """
    Likelihood-ratio test: order 1 vs 2, order 2 vs 3.
    Returns dict with Lambda statistics and p-values.
    """
    sequences = [chain.correctness for chain in chains if chain.n_steps >= 3]

    results = {}
    for (order_null, order_alt) in [(1, 2), (2, 3)]:
        counts_null = _transition_counts(sequences, order_null)
        counts_alt = _transition_counts(sequences, order_alt)

        ll_null = _log_likelihood(counts_null)
        ll_alt = _log_likelihood(counts_alt)

        lambda_stat = -2 * (ll_null - ll_alt)
        # Degrees of freedom: difference in free parameters
        # For binary chain: 2^order parameters per order
        df = 2 ** order_alt - 2 ** order_null
        p_value = float(1 - chi2.cdf(lambda_stat, df=df))

        results[f"{order_null}v{order_alt}"] = {
            "lambda": float(lambda_stat),
            "df": df,
            "p_value": p_value,
            "reject_h0": p_value < 0.001,
        }

    return results


# ─────────────────────────────────────────────
# Error Influence Decay (EID)
# ─────────────────────────────────────────────

def compute_eid(chains: list[Chain], max_k: int = 6) -> dict:
    """
    δ(k) = E_ctrl(k) - E_base(k)

    E_ctrl(k): error rate at distance k from an error, given all intermediate steps correct.
    E_base(k): error rate at distance k from a CORRECT step.
    """
    ctrl_errors = defaultdict(int)
    ctrl_total = defaultdict(int)
    base_errors = defaultdict(int)
    base_total = defaultdict(int)

    for chain in chains:
        c = chain.correctness
        n = chain.n_steps

        for i in range(n):
            if c[i]:   # correct step at i
                for k in range(1, max_k + 1):
                    j = i + k
                    if j >= n:
                        break
                    # Check all intermediate steps are correct
                    if all(c[i + 1: j]):
                        base_total[k] += 1
                        if not c[j]:
                            base_errors[k] += 1

            else:   # error at i
                for k in range(1, max_k + 1):
                    j = i + k
                    if j >= n:
                        break
                    # All intermediate steps must be correct
                    if all(c[i + 1: j]):
                        ctrl_total[k] += 1
                        if not c[j]:
                            ctrl_errors[k] += 1

    delta = {}
    for k in range(1, max_k + 1):
        e_ctrl = ctrl_errors[k] / ctrl_total[k] if ctrl_total[k] > 0 else float("nan")
        e_base = base_errors[k] / base_total[k] if base_total[k] > 0 else float("nan")
        delta[k] = {
            "e_ctrl": e_ctrl,
            "e_base": e_base,
            "delta": e_ctrl - e_base if not (np.isnan(e_ctrl) or np.isnan(e_base)) else float("nan"),
            "n_ctrl": ctrl_total[k],
            "n_base": base_total[k],
        }

    return delta


# ─────────────────────────────────────────────
# Full metrics summary
# ─────────────────────────────────────────────

def compute_all_metrics(chains: list[Chain], n_bootstrap: int = 10_000) -> dict:
    """Compute all paper metrics for a list of chains. Returns nested dict."""
    acc = compute_accuracy(chains)
    epr_result = compute_epr(chains)
    fef_result = compute_fef(chains)
    emr_result = compute_emr(chains)
    markov = markov_order_test(chains)
    eid = compute_eid(chains)

    # Bootstrap CIs for key scalars
    epr_ci = bootstrap_ci(chains, lambda c: compute_epr(c), "epr", n_bootstrap)
    emr_ci = bootstrap_ci(chains, lambda c: compute_emr(c), "emr", n_bootstrap)
    fef1_ci = bootstrap_ci(chains, lambda c: compute_fef(c), "fef_1", n_bootstrap)

    return {
        "accuracy": acc,
        "epr": epr_result,
        "epr_ci_95": epr_ci,
        "fef": fef_result,
        "fef1_ci_95": fef1_ci,
        "emr": emr_result,
        "emr_ci_95": emr_ci,
        "markov_tests": markov,
        "eid": eid,
    }
