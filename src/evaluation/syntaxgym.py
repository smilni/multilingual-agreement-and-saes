"""SyntaxGym evaluation for ablation specificity testing.

Each SyntaxGym suite tests a syntactic phenomenon by comparing
log P(critical_region | prefix) across minimal-pair conditions.
We evaluate the model clean and then with number-agreement features
zeroed out to check that the ablation is specific to number agreement.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from ..utils.config import DATA_DIR
from ..utils.logging import get_logger
from ._syntaxgym_prediction import METRICS as _SG_METRICS
from ._syntaxgym_prediction import Prediction as _SgPrediction

logger = get_logger(__name__)

SYNTAXGYM_DIR = DATA_DIR / "syntaxgym"

# test suites from Gauthier et al. 2020 (cpllab/syntactic-generalization)
_GITHUB_BASE = (
    "https://raw.githubusercontent.com/cpllab/syntactic-generalization"
    "/master/test_suites/json/{name}.json"
)


SPECIFICITY_SUITES: dict[str, list[str]] = {
    # Positive control - ablation should specifically hurt these
    "number_agreement": ["number_orc", "number_src", "number_prep"],
    # Specificity controls - ablation should leave these roughly intact
    "filler_gap":       [
        "fgd_object", "fgd_subject", "fgd_pp",
        "fgd-embed3", "fgd-embed4", "fgd_hierarchy",
    ],
    "npi":              ["npi_orc_any", "npi_src_any", "npi_orc_ever", "npi_src_ever"],
    "reflexive":        [
        "reflexive_orc_masc", "reflexive_src_masc",
        "reflexive_orc_fem",  "reflexive_src_fem",
        "reflexive_prep_masc", "reflexive_prep_fem",
    ],
    "center_embed":     ["center_embed", "center_embed_mod"],
    "cleft":            ["cleft", "cleft_modifier"],
    "garden_path":      ["npz_ambig", "npz_ambig_mod", "npz_obj", "npz_obj_mod", "mvrr", "mvrr_mod"],
    "subordination":    [
        "subordination", "subordination_orc-orc",
        "subordination_pp-pp", "subordination_src-src",
    ],
    # "nn_nv_rpl":        ["nn-nv-rpl"],
}

_TOKEN = re.compile(r"\(([\d*]+);%([A-Za-z0-9_-]+)%\)")


def load_or_download_suite(name: str, cache_dir: Path | None = None) -> dict[str, Any]:
    cache_dir = Path(cache_dir) if cache_dir else SYNTAXGYM_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{name}.json"
    if not dest.exists():
        url = _GITHUB_BASE.format(name=name)
        logger.info("Downloading %s", url)
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download '{name}'. Place the JSON manually at {dest}."
            ) from exc
    return json.loads(dest.read_text(encoding="utf-8"))


def _region_token_span(
    tokenizer: Any,
    prefix_st: str,
    region_clean: str,
) -> tuple[list[int], list[int], int]:
    """Tokenize *prefix_st* + region as one string; return (full_ids, region_idx, hook_pos).
    """
    prefix_st = prefix_st.strip()
    if not region_clean:
        return [], [], -1

    full_text = f"{prefix_st} {region_clean}".strip() if prefix_st else region_clean
    region_char_start = len(prefix_st) + (1 if prefix_st else 0)
    region_char_end = region_char_start + len(region_clean)

    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(
            full_text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        ids: list[int] = list(enc["input_ids"])
        offsets = enc["offset_mapping"]
        region_positions: list[int] = []
        for i, (cs, ce) in enumerate(offsets):
            if cs is None or ce is None:
                continue
            # HF uses (0, 0) for special/padding tokens - skip for region span
            if cs == ce == 0:
                continue
            if cs < region_char_end and ce > region_char_start:
                region_positions.append(i)
        if not region_positions:
            return ids, [], -1
        # Last token wholly before the critical region (character-aligned), so we
        # do not accidentally hook on BOS when the first content token is region 1.
        hook_pos = -1
        for i, (cs, ce) in enumerate(offsets):
            if cs is None or ce is None or (cs == ce == 0):
                continue
            if ce <= region_char_start:
                hook_pos = i
        if hook_pos < 0:
            hook_pos = region_positions[0] - 1
        return ids, region_positions, hook_pos

    # Slow tokenizer: fall back to split encode (legacy behaviour).
    prefix_ids = tokenizer.encode(prefix_st, add_special_tokens=True)
    region_ids = tokenizer.encode(" " + region_clean, add_special_tokens=False)
    if not region_ids:
        return prefix_ids, [], -1
    full_ids = prefix_ids + region_ids
    start = len(prefix_ids)
    region_positions = list(range(start, start + len(region_ids)))
    hook_pos = start - 1
    return full_ids, region_positions, hook_pos


def _region_logprob_maybe_ablate(
    model: Any,
    tokenizer: Any,
    prefix: str,
    region: str,
    device: Any,
    ablation_hook: Any = None,
) -> float:
    """Same as log-prob region sum, optionally applying ``ablation_hook`` at the
    last token before the first region token.
    """
    region_clean = region.strip()
    if not region_clean:
        return 0.0

    full_ids_list, region_positions, hook_pos = _region_token_span(
        tokenizer, prefix.strip(), region_clean,
    )
    if not full_ids_list or not region_positions:
        return 0.0

    if ablation_hook is not None and hook_pos >= 0:
        ablation_hook.set_position(hook_pos)
        ablation_hook.active = True

    try:
        full_ids = torch.tensor([full_ids_list], device=device)
        with torch.no_grad():
            log_probs = torch.log_softmax(model(full_ids).logits[0], dim=-1)
        total = 0.0
        for pos in region_positions:
            tok_id = full_ids_list[pos]
            if pos > 0:
                total += log_probs[pos - 1, tok_id].item()
            else:
                total += log_probs[0, tok_id].item()
        return total
    finally:
        if ablation_hook is not None:
            ablation_hook.active = False


def _parse_predictions(
    predictions: list[dict],
    metric: str,
) -> list[_SgPrediction | None]:
    """Parse each prediction formula once per suite.
    """
    parsed: list[_SgPrediction | None] = []
    for i, pred in enumerate(predictions):
        formula = pred.get("formula", "")
        try:
            parsed.append(_SgPrediction(idx=i, formula=formula, metric=metric))
        except ValueError as exc:
            logger.warning(
                "Failed to parse SyntaxGym formula %r (metric=%s): %s",
                formula, metric, exc,
            )
            parsed.append(None)
    return parsed


def _collect_required_regions(
    predictions: list[dict],
    items: list[dict],
) -> tuple[set[int], bool]:
    """Return the set of integer region numbers referenced by any formula.
    """
    wildcard = False
    crit_regions: set[int] = set()
    for pred in predictions:
        formula = pred.get("formula", "")
        for m in _TOKEN.finditer(formula):
            region_tok = m.group(1)
            if region_tok == "*":
                wildcard = True
            else:
                crit_regions.add(int(region_tok))
    if wildcard:
        for item in items:
            for cond in item.get("conditions", []):
                for r in cond.get("regions", []):
                    crit_regions.add(int(r["region_number"]))
    return crit_regions, wildcard


def _item_from_logprobs(
    logprobs: dict[str, dict[int, float]],
    metric: str,
) -> dict:
    """Build the ``item`` dict shape expected by ``_SgPrediction.__call__``.
    """
    return {
        "conditions": [
            {
                "condition_name": cond_name,
                "regions": [
                    {
                        "region_number": region_num,
                        "metric_value": {metric: -float(lp)},
                    }
                    for region_num, lp in region_map.items()
                ],
            }
            for cond_name, region_map in logprobs.items()
        ],
    }


@dataclass
class SuiteResult:
    """Results for one suite run.
    """

    suite_name: str
    n_items: int
    n_predictions: int
    n_trials: int
    n_correct: int
    accuracy: float
    item_results: list[dict] = field(default_factory=list)
    prediction_pass_count: list[int] = field(default_factory=list)


def evaluate_suite(model, tokenizer, suite: dict, device, ablation_hook=None) -> SuiteResult:
    """Evaluate a SyntaxGym suite, optionally with an SaeLatentAblationHook.
    """
    name        = suite.get("meta", {}).get("name", "unknown")
    predictions = suite.get("predictions", [])
    items       = suite.get("items", [])

    # SyntaxGym suites declare the within-region aggregation metric (``sum``
    # for all published suites, but honour whatever the JSON says).  Unknown
    # metrics would crash upstream's ``Prediction`` constructor, so fall back
    # to ``sum`` with a warning instead of aborting the whole run.
    metric = suite.get("meta", {}).get("metric", "sum")
    if metric not in _SG_METRICS:
        logger.warning(
            "Suite %r declares unknown metric %r; falling back to 'sum'",
            name, metric,
        )
        metric = "sum"

    parsed_preds = _parse_predictions(predictions, metric)
    crit_regions, _uses_wildcard = _collect_required_regions(predictions, items)

    if ablation_hook is not None:
        ablation_hook.register()

    n_preds = len(predictions)
    prediction_pass_count = [0] * n_preds if n_preds else []
    n_correct = 0
    item_results: list[dict] = []
    try:
        for item in tqdm(items, desc=name, leave=False):
            # logprobs[condition][region_number] = log-prob value
            logprobs: dict[str, dict[int, float]] = {}

            for cond in item.get("conditions", []):
                cond_name = cond["condition_name"]
                regions   = sorted(cond["regions"], key=lambda r: r["region_number"])
                logprobs[cond_name] = {}

                for crit_r in sorted(crit_regions):
                    prefix = " ".join(r["content"] for r in regions if r["region_number"] < crit_r).strip()
                    crit_texts = [r["content"] for r in regions if r["region_number"] == crit_r]
                    if not crit_texts:
                        continue

                    region_text = crit_texts[0].strip()
                    # Gap conditions (e.g. fgd suites) have empty region content.
                    # Upstream assigns surprisal 0 to those - matches us here
                    # because ``-log_prob`` of an empty region is 0 by
                    # convention.
                    if not region_text:
                        logprobs[cond_name][crit_r] = 0.0
                        continue

                    logprobs[cond_name][crit_r] = _region_logprob_maybe_ablate(
                        model, tokenizer, prefix, region_text, device, ablation_hook=ablation_hook,
                    )

            inum = item.get("item_number")
            if n_preds:
                sg_item = _item_from_logprobs(logprobs, metric)
                # SyntaxGym evaluates each prediction separately; accuracy is
                # over all (item, prediction_index) pairs, not AND across
                # predictions.
                per_pred: list[bool] = []
                for parsed in parsed_preds:
                    if parsed is None:
                        per_pred.append(False)
                        continue
                    try:
                        per_pred.append(bool(parsed(sg_item)))
                    except KeyError as exc:
                        logger.warning(
                            "Suite %r item %s: surprisal missing for %s "
                            "(did _collect_required_regions miss a region?)",
                            name, inum, exc,
                        )
                        per_pred.append(False)
                    except Exception as exc:  # defensive: upstream bugs, nan
                        logger.warning(
                            "Suite %r item %s: formula eval error: %s",
                            name, inum, exc,
                        )
                        per_pred.append(False)

                for pi, ok in enumerate(per_pred):
                    if ok:
                        prediction_pass_count[pi] += 1
                    n_correct += int(ok)
                    item_results.append({
                        "trial_id": f"{inum}:{pi}",
                        "item_number": inum,
                        "prediction_index": pi,
                        "correct": ok,
                    })
            else:
                # No prediction block - nothing to score.
                item_results.append({"trial_id": str(inum), "item_number": inum, "correct": False})

    finally:
        if ablation_hook is not None:
            ablation_hook.remove()

    n_items = len(items)
    if n_preds:
        n_trials = n_items * n_preds
        acc = n_correct / n_trials if n_trials else float("nan")
    else:
        n_trials = n_items
        acc = float("nan")

    return SuiteResult(
        name,
        n_items,
        n_preds,
        n_trials,
        n_correct,
        acc,
        item_results,
        prediction_pass_count,
    )
