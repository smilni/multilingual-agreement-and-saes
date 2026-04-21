"""Model competence evaluation on MultiBLiMP minimal pairs.

Computes conditional log-probability of grammatical vs ungrammatical continuations
(verb span, first diverging token, or full sentence). Verb and first-token scoring
condition on the first-divergence context shared with feature discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.multiblimp import MinimalPair
from ..utils.logging import get_logger
from .feature_discovery import minimal_pair_divergence_context

logger = get_logger(__name__)


def _join_prefix_continuation(prefix: str, continuation: str) -> str:
    return f"{prefix.rstrip()} {continuation.strip()}"


def _verb_token_ids_after_prefix(
    tokenizer: AutoTokenizer,
    prefix: str,
    verb: str,
) -> list[int]:
    """Return token ids for `verb` in context after `prefix`."""
    verb_clean = verb.strip()
    prefix_stripped = prefix.rstrip()
    context_str = f"{prefix_stripped} {verb_clean}" if prefix_stripped else verb_clean
    context_ids = tokenizer.encode(prefix_stripped, add_special_tokens=True)
    full_ids = tokenizer.encode(context_str, add_special_tokens=True)
    if len(full_ids) <= len(context_ids):
        return []
    return full_ids[len(context_ids):]


def _logprob_of_verb_tokens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    verb_token_ids: list[int],
    device: torch.device,
) -> float:
    """Sum of log P(t_i | prefix, t_0, ..., t_{i-1}) over all verb subtokens."""
    if not verb_token_ids:
        return 0.0
    context_ids = tokenizer.encode(prefix.rstrip(), add_special_tokens=True)
    input_ids = context_ids + verb_token_ids
    ids_tensor = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        logits = model(ids_tensor).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for i, tid in enumerate(verb_token_ids):
        pos = len(context_ids) - 1 + i
        total += log_probs[pos, tid].item()
    return total


def _logprob_of_verb_tokens_from_context_ids(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context_ids: list[int],
    verb_token_ids: list[int],
    device: torch.device,
) -> float:
    """Like _logprob_of_verb_tokens but conditions on explicit context_ids."""
    if not verb_token_ids:
        return 0.0
    input_ids = context_ids + verb_token_ids
    ids_tensor = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        logits = model(ids_tensor).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for i, tid in enumerate(verb_token_ids):
        pos = len(context_ids) - 1 + i
        total += log_probs[pos, tid].item()
    return total


def _first_token_logprobs_from_divergence_context(
    model: AutoModelForCausalLM,
    divergence_context_ids: list[int],
    good_full_ids: list[int],
    bad_full_ids: list[int],
    device: torch.device,
) -> tuple[float, float]:
    """Log-prob of the first differing continuation token after divergence_context_ids."""
    n = len(divergence_context_ids)
    g_tok = good_full_ids[n]
    b_tok = bad_full_ids[n]
    input_ids = torch.tensor([divergence_context_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1]
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[g_tok].item(), log_probs[b_tok].item()


def _minimal_pair_full_ids(
    tokenizer: AutoTokenizer,
    pair: MinimalPair,
    *,
    good: bool,
) -> list[int]:
    cont = pair.good_continuation if good else pair.bad_continuation
    text = _join_prefix_continuation(pair.prefix, cont)
    return tokenizer.encode(text, add_special_tokens=True)


@dataclass
class PairResult:
    uid: str
    language: str
    prefix: str
    good_continuation: str
    bad_continuation: str
    good_verb: str
    bad_verb: str
    good_logprob: float
    bad_logprob: float
    correct: bool
    logprob_gap: float


def _logprob_of_sentence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sentence: str,
    device: torch.device,
) -> float:
    """Sum of log-probs of all tokens in the sentence."""
    input_ids = tokenizer.encode(sentence, add_special_tokens=True)
    ids_tensor = torch.tensor([input_ids], device=device)
    with torch.no_grad():
        logits = model(ids_tensor).logits
    log_probs = torch.log_softmax(logits[0], dim=-1)
    total = 0.0
    for pos in range(1, len(input_ids)):
        total += log_probs[pos - 1, input_ids[pos]].item()
    return total


def _first_token_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    good_continuation: str,
    bad_continuation: str,
    device: torch.device,
) -> tuple[float, float]:
    """Log-prob of the first continuation token for good and bad, given prefix."""
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
    good_ids = tokenizer.encode(f"{prefix} {good_continuation}", add_special_tokens=True)
    bad_ids = tokenizer.encode(f"{prefix} {bad_continuation}", add_special_tokens=True)
    cont_start = len(prefix_ids)
    input_ids = torch.tensor([prefix_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits[0, -1], dim=-1)
    return log_probs[good_ids[cont_start]].item(), log_probs[bad_ids[cont_start]].item()


def _verb_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    good_verb: str,
    bad_verb: str,
    device: torch.device,
) -> tuple[float, float]:
    """Sum of log-probs of all subtokens of good_verb and bad_verb given prefix."""
    good_ids = _verb_token_ids_after_prefix(tokenizer, prefix, good_verb)
    bad_ids = _verb_token_ids_after_prefix(tokenizer, prefix, bad_verb)
    good_lp = _logprob_of_verb_tokens(model, tokenizer, prefix, good_ids, device)
    bad_lp = _logprob_of_verb_tokens(model, tokenizer, prefix, bad_ids, device)
    return good_lp, bad_lp


def evaluate_pair(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pair: MinimalPair,
    device: torch.device,
    scoring: Literal["verb", "first_token", "full_sentence"] = "verb",
    debug: bool = False,
    divergence_context_ids: list[int] | None = None,
    use_first_divergence: bool = True,
) -> PairResult:
    """Evaluate the model on a single minimal pair.

    scoring:
        "verb": sum of log-probs of verb subtokens (falls back to first diverging token
            if both verbs are empty).
        "first_token": log-prob of the first diverging continuation token.
        "full_sentence": sum of log-probs of the full sentence.
    divergence_context_ids: explicit context (e.g. for causal ablation); overrides
        use_first_divergence.
    use_first_divergence: build context via minimal_pair_divergence_context; falls back
        to prefix-only if that fails.
    """
    aligned: list[int] | None = divergence_context_ids
    if aligned is None and use_first_divergence:
        try:
            aligned = minimal_pair_divergence_context(tokenizer, pair)
        except ValueError as err:
            logger.warning(
                "pair %s: first-divergence context unavailable (%s); using prefix-only scoring",
                pair.uid, err,
            )
            aligned = None

    if aligned is not None:
        g_full = _minimal_pair_full_ids(tokenizer, pair, good=True)
        b_full = _minimal_pair_full_ids(tokenizer, pair, good=False)

        if scoring == "verb":
            if pair.good_verb.strip() or pair.bad_verb.strip():
                if debug:
                    print(f"prefix: {pair.prefix!r}")
                    print(f"  good_verb={pair.good_verb!r}  bad_verb={pair.bad_verb!r}")
                g_ref = _verb_token_ids_after_prefix(tokenizer, pair.prefix.rstrip(), pair.good_verb)
                b_ref = _verb_token_ids_after_prefix(tokenizer, pair.prefix.rstrip(), pair.bad_verb)
                g_chunk = g_full[len(aligned): len(aligned) + len(g_ref)]
                b_chunk = b_full[len(aligned): len(aligned) + len(b_ref)]
                g_lp = _logprob_of_verb_tokens_from_context_ids(
                    model, tokenizer, aligned, g_chunk, device
                )
                b_lp = _logprob_of_verb_tokens_from_context_ids(
                    model, tokenizer, aligned, b_chunk, device
                )
            else:
                g_lp, b_lp = _first_token_logprobs_from_divergence_context(
                    model, aligned, g_full, b_full, device
                )
        elif scoring == "first_token":
            g_lp, b_lp = _first_token_logprobs_from_divergence_context(
                model, aligned, g_full, b_full, device
            )
        else:
            g_lp = _logprob_of_sentence(model, tokenizer, pair.good_sentence, device)
            b_lp = _logprob_of_sentence(model, tokenizer, pair.bad_sentence, device)
    elif scoring == "verb":
        if pair.good_verb.strip() or pair.bad_verb.strip():
            if debug:
                print(f"prefix: {pair.prefix!r}")
                print(f"  good_verb={pair.good_verb!r}  bad_verb={pair.bad_verb!r}")
            g_lp, b_lp = _verb_logprobs(
                model, tokenizer, pair.prefix, pair.good_verb, pair.bad_verb, device,
            )
        else:
            g_lp, b_lp = _first_token_logprobs(
                model, tokenizer, pair.prefix, pair.good_continuation, pair.bad_continuation, device,
            )
    elif scoring == "first_token":
        g_lp, b_lp = _first_token_logprobs(
            model, tokenizer, pair.prefix, pair.good_continuation, pair.bad_continuation, device,
        )
    else:
        g_lp = _logprob_of_sentence(model, tokenizer, pair.good_sentence, device)
        b_lp = _logprob_of_sentence(model, tokenizer, pair.bad_sentence, device)

    return PairResult(
        uid=pair.uid,
        language=pair.language,
        prefix=pair.prefix,
        good_continuation=pair.good_continuation,
        bad_continuation=pair.bad_continuation,
        good_verb=pair.good_verb,
        bad_verb=pair.bad_verb,
        good_logprob=g_lp,
        bad_logprob=b_lp,
        correct=g_lp > b_lp,
        logprob_gap=g_lp - b_lp,
    )


def evaluate_all(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pairs: list[MinimalPair],
    device: torch.device | None = None,
    scoring: Literal["verb", "first_token", "full_sentence"] = "verb",
    debug: bool = False,
    use_first_divergence: bool = True,
) -> list[PairResult]:
    """Evaluate the model on every pair and return per-pair results."""
    if device is None:
        device = next(model.parameters()).device
    results = []
    for pair in tqdm(pairs, desc="Evaluating competence"):
        results.append(
            evaluate_pair(
                model, tokenizer, pair, device, scoring,
                debug=debug,
                use_first_divergence=use_first_divergence,
            )
        )
    return results


def summarize_results(results: list[PairResult]) -> dict:
    """Compute aggregate accuracy and mean log-prob gap."""
    n = len(results)
    n_correct = sum(1 for r in results if r.correct)
    mean_gap = sum(r.logprob_gap for r in results) / n
    return {
        "n": n,
        "accuracy": n_correct / n,
        "n_correct": n_correct,
        "mean_logprob_gap": mean_gap,
    }


def results_to_records(results: list[PairResult]) -> list[dict]:
    return [
        {
            "uid": r.uid,
            "language": r.language,
            "prefix": r.prefix,
            "good_continuation": r.good_continuation,
            "bad_continuation": r.bad_continuation,
            "good_verb": r.good_verb,
            "bad_verb": r.bad_verb,
            "good_logprob": r.good_logprob,
            "bad_logprob": r.bad_logprob,
            "correct": r.correct,
            "logprob_gap": r.logprob_gap,
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Number-pair competence evaluation
# ---------------------------------------------------------------------------


def load_number_pairs(lang_code: str):
    """Load the same-verb number-pairs TSV for a language."""
    import pandas as pd
    from ..utils.config import DATA_DIR

    path = DATA_DIR / "number_pairs" / f"{lang_code}_same_verb.tsv"
    if not path.exists():
        logger.warning("No number-pairs file at %s", path)
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _verb_logprob_given_prefix(model, tokenizer, prefix, verb, device):
    """Log-prob of *verb* (first word) in context after *prefix*. (Legacy scoring.)"""
    ids = _verb_token_ids_after_prefix(tokenizer, prefix, verb)
    return _logprob_of_verb_tokens(model, tokenizer, prefix, ids, device)


def _logprobs_of_two_tokens_at_last_pos(
    model: AutoModelForCausalLM,
    context_ids: list[int],
    tok_a: int,
    tok_b: int,
    device: torch.device,
) -> tuple[float, float]:
    """Log-probs of two candidate next-token ids given an aligned context."""
    input_ids = torch.tensor([context_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1]
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[tok_a].item(), log_probs[tok_b].item()


def _first_div_logprobs_under_prefix(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    sg_continuation: str,
    pl_continuation: str,
    device: torch.device,
) -> tuple[float, float] | None:
    """Score V_sg vs V_pl on their first diverging subtoken, given one prefix.
    """
    from .feature_discovery import _aligned_context_ids, _continuation_divergence_info

    info = _continuation_divergence_info(tokenizer, prefix, sg_continuation, pl_continuation)
    if not info["diverges"]:
        return None
    if info["first_div_sg_id"] is None or info["first_div_pl_id"] is None:
        return None
    aligned = _aligned_context_ids(tokenizer, prefix, info["sg_ids"], info["lcp_len"])
    return _logprobs_of_two_tokens_at_last_pos(
        model, aligned, info["first_div_sg_id"], info["first_div_pl_id"], device,
    )


def evaluate_all_number_pairs(
    model,
    tokenizer,
    df,
    device,
    scoring: Literal["first_token", "verb"] = "first_token",
) -> list[dict]:
    """Evaluate every number pair in *df*.

    scoring="first_token" (default): for each prefix (SG-prefix and PL-prefix),
        score V_sg vs V_pl on their *first diverging subtoken*.
    scoring="verb": legacy behaviour - sum log-probs of all subtokens of the
        verb and compare the same verb under good vs bad prefix.
    """
    results: list[dict] = []
    for pair_id in tqdm(df["pair_id"].unique(), desc="Evaluating number pairs"):
        pair_df = df[df["pair_id"] == pair_id]
        sg = pair_df[pair_df["target_number"] == "SG"]
        pl = pair_df[pair_df["target_number"] == "PL"]
        if sg.empty or pl.empty:
            continue
        sg = sg.iloc[0]
        pl = pl.iloc[0]

        sg_cont = sg["continuation"]
        pl_cont = pl["continuation"]
        if not isinstance(sg_cont, str) or not isinstance(pl_cont, str):
            logger.warning(
                "Skipping pair_id=%s: missing/non-string continuation (sg=%r, pl=%r)",
                pair_id, sg_cont, pl_cont,
            )
            continue

        sg_cont_clean = sg_cont.strip()
        pl_cont_clean = pl_cont.strip()
        if not sg_cont_clean or not pl_cont_clean:
            logger.warning(
                "Skipping pair_id=%s: empty continuation after strip()", pair_id,
            )
            continue

        sg_verb = sg_cont_clean.split()[0]
        pl_verb = pl_cont_clean.split()[0]

        sg_prefix = sg["good_prefix"]   # SG-subject prefix (agrees with V_sg)
        pl_prefix = pl["good_prefix"]   # PL-subject prefix (agrees with V_pl)

        if scoring == "first_token":
            scored_sg = _first_div_logprobs_under_prefix(
                model, tokenizer, sg_prefix, sg_cont_clean, pl_cont_clean, device,
            )
            scored_pl = _first_div_logprobs_under_prefix(
                model, tokenizer, pl_prefix, sg_cont_clean, pl_cont_clean, device,
            )
            if scored_sg is None or scored_pl is None:
                logger.warning(
                    "Skipping pair_id=%s: SG/PL continuations do not diverge in tokens",
                    pair_id,
                )
                continue
            sg_under_sg_lp, pl_under_sg_lp = scored_sg
            sg_under_pl_lp, pl_under_pl_lp = scored_pl

            sg_good_lp, sg_bad_lp = sg_under_sg_lp, pl_under_sg_lp
            pl_good_lp, pl_bad_lp = pl_under_pl_lp, sg_under_pl_lp
        elif scoring == "verb":
            sg_good_lp = _verb_logprob_given_prefix(model, tokenizer, sg_prefix, sg_verb, device)
            sg_bad_lp  = _verb_logprob_given_prefix(model, tokenizer, pl_prefix, sg_verb, device)
            pl_good_lp = _verb_logprob_given_prefix(model, tokenizer, pl_prefix, pl_verb, device)
            pl_bad_lp  = _verb_logprob_given_prefix(model, tokenizer, sg_prefix, pl_verb, device)
        else:
            raise ValueError(f"unknown scoring: {scoring!r}")

        sg_correct = sg_good_lp > sg_bad_lp
        pl_correct = pl_good_lp > pl_bad_lp

        results.append({
            "pair_id": sg["pair_id"],
            "language": sg["language"],
            "sg_verb": sg_verb,
            "pl_verb": pl_verb,
            "sg_prefix": sg_prefix,
            "pl_prefix": pl_prefix,
            "sg_good_logprob": sg_good_lp,
            "sg_bad_logprob": sg_bad_lp,
            "pl_good_logprob": pl_good_lp,
            "pl_bad_logprob": pl_bad_lp,
            "sg_correct": sg_correct,
            "pl_correct": pl_correct,
            "both_correct": sg_correct and pl_correct,
            "sg_logprob_gap": sg_good_lp - sg_bad_lp,
            "pl_logprob_gap": pl_good_lp - pl_bad_lp,
            "scoring": scoring,
        })
    return results


def summarize_number_pair_results(results: list[dict]) -> dict:
    """Aggregate accuracy stats for number-pair evaluation."""
    n = len(results)
    if n == 0:
        return {"n": 0, "sg_accuracy": 0, "pl_accuracy": 0, "both_accuracy": 0,
                "n_sg_correct": 0, "n_pl_correct": 0, "n_both_correct": 0,
                "mean_sg_logprob_gap": 0, "mean_pl_logprob_gap": 0}
    n_sg = sum(r["sg_correct"] for r in results)
    n_pl = sum(r["pl_correct"] for r in results)
    n_both = sum(r["both_correct"] for r in results)
    return {
        "n": n,
        "sg_accuracy": n_sg / n,
        "pl_accuracy": n_pl / n,
        "both_accuracy": n_both / n,
        "n_sg_correct": n_sg,
        "n_pl_correct": n_pl,
        "n_both_correct": n_both,
        "mean_sg_logprob_gap": sum(r["sg_logprob_gap"] for r in results) / n,
        "mean_pl_logprob_gap": sum(r["pl_logprob_gap"] for r in results) / n,
    }
