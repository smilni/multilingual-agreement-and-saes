"""Feature discovery for subject-number SAE latents (Experiment 5.2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.discovery_pairs import load_discovery_pairs_from_path
from ..data.multiblimp import MinimalPair
from ..model.loading import register_residual_hooks
from ..utils.config import DATA_DIR
from ..utils.logging import get_logger

logger = get_logger(__name__)


FEATURE_DISCOVERY_DIR = DATA_DIR / "feature_discovery"
ACTIVATIONS_DIR = FEATURE_DISCOVERY_DIR / "activations"
LATENTS_DIR = FEATURE_DISCOVERY_DIR / "latents"
CANDIDATES_DIR = FEATURE_DISCOVERY_DIR / "candidates"


@dataclass
class LayerActivationBatch:
    language: str
    layer: int
    pair_ids: list[str]
    sg_positions: list[int]
    pl_positions: list[int]
    pair_position_info: list[dict[str, Any]]
    extraction_counts: dict[str, int]
    sg: torch.Tensor           # (n, d_model) - decision-point activations
    pl: torch.Tensor           # (n, d_model) - decision-point activations
    all_sg: torch.Tensor       # (total_sg_tokens, d_model) - all token positions
    all_pl: torch.Tensor       # (total_pl_tokens, d_model) - all token positions


@dataclass
class LayerLatentBatch:
    language: str
    layer: int
    pair_ids: list[str]
    z_sg: torch.Tensor         # (n, d_sae) - decision-point SAE latents
    z_pl: torch.Tensor         # (n, d_sae) - decision-point SAE latents
    theta: torch.Tensor        # (d_sae,) - per-feature median over all token positions


def load_discovery_pairs(lang_code: str) -> pd.DataFrame:
    """Load SG/PL examples from `data/number_pairs/{lang_code}_same_verb.tsv`."""
    path = DATA_DIR / "number_pairs" / f"{lang_code}_same_verb.tsv"
    return load_discovery_pairs_from_path(path)


def _join_prefix_continuation(prefix: str, continuation: str) -> str:
    return f"{prefix.rstrip()} {continuation.strip()}"


def _encode_no_special(tokenizer: AutoTokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _with_bos(tokenizer: AutoTokenizer, token_ids: list[int]) -> list[int]:
    if tokenizer.bos_token_id is None:
        return token_ids
    return [tokenizer.bos_token_id] + token_ids


def _longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _continuation_token_ids_after_prefix(
    tokenizer: AutoTokenizer,
    prefix: str,
    continuation: str,
) -> list[int]:
    """Return continuation token ids in exact prefix context (no special tokens)."""
    clean_prefix = prefix.rstrip()
    full_text = _join_prefix_continuation(clean_prefix, continuation)

    prefix_ids = _encode_no_special(tokenizer, clean_prefix)
    full_ids = _encode_no_special(tokenizer, full_text)

    lcp = _longest_common_prefix_len(prefix_ids, full_ids)
    return full_ids[lcp:]


def _continuation_divergence_info(
    tokenizer: AutoTokenizer,
    prefix: str,
    continuation_sg: str,
    continuation_pl: str,
) -> dict[str, Any]:
    """Compute where SG and PL continuations start to differ."""
    sg_ids = _continuation_token_ids_after_prefix(tokenizer, prefix, continuation_sg)
    pl_ids = _continuation_token_ids_after_prefix(tokenizer, prefix, continuation_pl)
    lcp_len = _longest_common_prefix_len(sg_ids, pl_ids)

    diverges = lcp_len < min(len(sg_ids), len(pl_ids)) or len(sg_ids) != len(pl_ids)
    first_div_sg_id = sg_ids[lcp_len] if lcp_len < len(sg_ids) else None
    first_div_pl_id = pl_ids[lcp_len] if lcp_len < len(pl_ids) else None
    return {
        "sg_ids": sg_ids,
        "pl_ids": pl_ids,
        "lcp_len": lcp_len,
        "diverges": diverges,
        "first_div_sg_id": first_div_sg_id,
        "first_div_pl_id": first_div_pl_id,
    }


def _aligned_context_ids(
    tokenizer: AutoTokenizer,
    prefix: str,
    continuation_ids: list[int],
    lcp_len: int,
) -> list[int]:
    """Build model context: BOS + prefix + shared continuation."""
    prefix_ids = _encode_no_special(tokenizer, prefix.rstrip())
    shared = continuation_ids[:lcp_len] if lcp_len > 0 else []
    return _with_bos(tokenizer, prefix_ids + shared)


def minimal_pair_divergence_context(
    tokenizer: AutoTokenizer,
    pair: MinimalPair,
) -> list[int]:
    """BOS + prefix + shared continuation up to where good/bad diverge.

    Used by competence scoring and causal validation so hooks and
    log-probs align on one position.
    """
    info = _continuation_divergence_info(
        tokenizer, pair.prefix, pair.good_continuation, pair.bad_continuation,
    )
    if not info["diverges"]:
        raise ValueError(
            f"{pair.uid}: good vs bad continuations do not diverge as token sequences"
        )
    return _aligned_context_ids(
        tokenizer, pair.prefix, info["sg_ids"], info["lcp_len"],
    )


def sample_alignment_debug(
    tokenizer: AutoTokenizer,
    pairs_df: pd.DataFrame,
    sample_size: int = 10,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Return decoded examples to sanity-check token alignment."""
    sample_df = pairs_df.sample(n=min(sample_size, len(pairs_df)), random_state=seed)
    samples: list[dict[str, Any]] = []

    for row in sample_df.itertuples(index=False):
        sg_info = _continuation_divergence_info(
            tokenizer, row.prefix_sg, row.continuation_sg, row.continuation_pl,
        )
        pl_info = _continuation_divergence_info(
            tokenizer, row.prefix_pl, row.continuation_sg, row.continuation_pl,
        )

        samples.append(
            {
                "pair_id": row.pair_id,
                "prefix_sg_decoded": tokenizer.decode(
                    _encode_no_special(tokenizer, row.prefix_sg.rstrip())
                ),
                "prefix_pl_decoded": tokenizer.decode(
                    _encode_no_special(tokenizer, row.prefix_pl.rstrip())
                ),
                "sg_shared_tokens_decoded": tokenizer.decode(
                    sg_info["sg_ids"][: sg_info["lcp_len"]]
                ),
                "pl_shared_tokens_decoded": tokenizer.decode(
                    pl_info["pl_ids"][: pl_info["lcp_len"]]
                ),
                "sg_first_div_sg_decoded": (
                    tokenizer.decode([sg_info["first_div_sg_id"]])
                    if sg_info["first_div_sg_id"] is not None
                    else None
                ),
                "sg_first_div_pl_decoded": (
                    tokenizer.decode([sg_info["first_div_pl_id"]])
                    if sg_info["first_div_pl_id"] is not None
                    else None
                ),
                "pl_first_div_sg_decoded": (
                    tokenizer.decode([pl_info["first_div_sg_id"]])
                    if pl_info["first_div_sg_id"] is not None
                    else None
                ),
                "pl_first_div_pl_decoded": (
                    tokenizer.decode([pl_info["first_div_pl_id"]])
                    if pl_info["first_div_pl_id"] is not None
                    else None
                ),
                "sg_lcp_len": sg_info["lcp_len"],
                "pl_lcp_len": pl_info["lcp_len"],
            }
        )
    return samples


def extract_decision_point_residual_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pairs_df: pd.DataFrame,
    language: str,
    layers: list[int],
    device: torch.device | str | None = None,
) -> dict[int, LayerActivationBatch]:
    """Extract residual activations right before SG/PL continuations diverge."""
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    store = register_residual_hooks(model, layers)
    per_layer_sg: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    per_layer_pl: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    per_layer_all_sg: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    per_layer_all_pl: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    pair_ids: list[str] = []
    sg_positions: list[int] = []
    pl_positions: list[int] = []
    pair_position_info: list[dict[str, Any]] = []
    n_nondiverging = 0

    try:
        for row in tqdm(
            pairs_df.itertuples(index=False),
            total=len(pairs_df),
            desc=f"Extracting activations ({language})",
        ):
            sg_info = _continuation_divergence_info(
                tokenizer, row.prefix_sg, row.continuation_sg, row.continuation_pl,
            )
            pl_info = _continuation_divergence_info(
                tokenizer, row.prefix_pl, row.continuation_sg, row.continuation_pl,
            )

            if not sg_info["diverges"] or not pl_info["diverges"]:
                n_nondiverging += 1
                continue

            ids_ctx_sg = _aligned_context_ids(
                tokenizer, row.prefix_sg, sg_info["sg_ids"], sg_info["lcp_len"],
            )
            ids_ctx_pl = _aligned_context_ids(
                tokenizer, row.prefix_pl, pl_info["pl_ids"], pl_info["lcp_len"],
            )

            pos_sg = len(ids_ctx_sg) - 1
            pos_pl = len(ids_ctx_pl) - 1

            with torch.no_grad():
                store.clear()
                model(torch.tensor([ids_ctx_sg], device=device))
                sg_hidden = {
                    layer: store.activations[layer][0, pos_sg, :].detach().cpu()
                    for layer in layers
                }
                sg_all_hidden = {
                    layer: store.activations[layer][0, :, :].detach().cpu()
                    for layer in layers
                }

                store.clear()
                model(torch.tensor([ids_ctx_pl], device=device))
                pl_hidden = {
                    layer: store.activations[layer][0, pos_pl, :].detach().cpu()
                    for layer in layers
                }
                pl_all_hidden = {
                    layer: store.activations[layer][0, :, :].detach().cpu()
                    for layer in layers
                }

            for layer in layers:
                per_layer_sg[layer].append(sg_hidden[layer])
                per_layer_pl[layer].append(pl_hidden[layer])
                per_layer_all_sg[layer].append(sg_all_hidden[layer])
                per_layer_all_pl[layer].append(pl_all_hidden[layer])

            pair_ids.append(row.pair_id)
            sg_positions.append(pos_sg)
            pl_positions.append(pos_pl)
            pair_position_info.append(
                {
                    "pair_id": row.pair_id,
                    "sg_position": pos_sg,
                    "pl_position": pos_pl,
                    "sg_lcp_len": sg_info["lcp_len"],
                    "pl_lcp_len": pl_info["lcp_len"],
                    "sg_context_token_count": len(ids_ctx_sg),
                    "pl_context_token_count": len(ids_ctx_pl),
                    "sg_first_div_token_id": sg_info["first_div_sg_id"],
                    "pl_first_div_token_id": pl_info["first_div_pl_id"],
                }
            )
    finally:
        store.remove_hooks()

    extraction_counts = {
        "n_total": len(pairs_df),
        "n_kept": len(pair_ids),
        "n_nondiverging": n_nondiverging,
    }
    logger.info("Extraction (%s): %d/%d kept", language, len(pair_ids), len(pairs_df))

    output: dict[int, LayerActivationBatch] = {}
    for layer in layers:
        if not per_layer_sg[layer]:
            continue
        output[layer] = LayerActivationBatch(
            language=language,
            layer=layer,
            pair_ids=pair_ids,
            sg_positions=sg_positions,
            pl_positions=pl_positions,
            pair_position_info=pair_position_info,
            extraction_counts=extraction_counts,
            sg=torch.stack(per_layer_sg[layer]),
            pl=torch.stack(per_layer_pl[layer]),
            all_sg=torch.cat(per_layer_all_sg[layer], dim=0),
            all_pl=torch.cat(per_layer_all_pl[layer], dim=0),
        )
    return output


def save_activation_batch(batch: LayerActivationBatch, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "language": batch.language,
            "layer": batch.layer,
            "pair_ids": batch.pair_ids,
            "sg_positions": batch.sg_positions,
            "pl_positions": batch.pl_positions,
            "pair_position_info": batch.pair_position_info,
            "extraction_counts": batch.extraction_counts,
            "sg": batch.sg,
            "pl": batch.pl,
            "all_sg": batch.all_sg,
            "all_pl": batch.all_pl,
        },
        path,
    )
    return path


def save_scores_table(scores_df: pd.DataFrame, path: Path) -> Path:
    """Save full feature score table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(path, index=False)
    return path


def load_scores_table(path: Path | str) -> pd.DataFrame:
    """Load a scores CSV, fixing up column whitespace and ci_excludes_zero dtype."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "ci_excludes_zero" in df.columns:
        col = df["ci_excludes_zero"]
        if pd.api.types.is_object_dtype(col):
            df["ci_excludes_zero"] = col.astype(str).str.strip().str.lower().isin(("true", "1", "t", "yes"))
        else:
            df["ci_excludes_zero"] = col.fillna(False).astype(bool)
    return df


def warn_if_scores_roundtrip_drift(
    scores_df: pd.DataFrame,
    path: Path | str,
    *,
    atol: float = 1e-6,
) -> None:
    """Warn if the CSV at path does not round-trip scores_df faithfully."""
    try:
        reloaded = load_scores_table(Path(path))
    except Exception as exc:
        logger.warning("Could not re-load scores CSV for round-trip check %s: %s", path, exc)
        return
    if len(reloaded) != len(scores_df):
        logger.warning(
            "Scores CSV row count mismatch: disk=%d in_mem=%d (%s)",
            len(reloaded), len(scores_df), path,
        )
        return
    for col in ("abs_eale", "frc_score", "sign_consistency", "frc_pvalue", "frc_qvalue", "mean_delta"):
        if col not in scores_df.columns or col not in reloaded.columns:
            continue
        if not np.allclose(
            scores_df[col].astype(float).values,
            reloaded[col].astype(float).values,
            rtol=0, atol=atol, equal_nan=True,
        ):
            logger.warning("Scores CSV round-trip drift on column %r (%s)", col, path)


def encode_with_sae(
    sae: Any,
    activation_batch: LayerActivationBatch,
    sae_device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> LayerLatentBatch:
    """Encode SG/PL residual activations with an SAE.

    Decision-point latents (z_sg, z_pl) are used for FRC and EALE scoring.
    All-position latents are encoded transiently to compute the per-feature
    median threshold theta as described in the thesis.
    """
    sg = activation_batch.sg.to(sae_device)
    pl = activation_batch.pl.to(sae_device)
    all_sg = activation_batch.all_sg.to(sae_device)
    all_pl = activation_batch.all_pl.to(sae_device)

    z_sg_chunks: list[torch.Tensor] = []
    z_pl_chunks: list[torch.Tensor] = []
    all_latents_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, sg.shape[0], batch_size):
            end = start + batch_size
            z_sg_chunks.append(sae.encode(sg[start:end]).detach().cpu())
            z_pl_chunks.append(sae.encode(pl[start:end]).detach().cpu())

        all_positions = torch.cat([all_sg, all_pl], dim=0)
        for start in range(0, all_positions.shape[0], batch_size):
            end = start + batch_size
            all_latents_chunks.append(sae.encode(all_positions[start:end]).detach().cpu())

    all_latents = torch.cat(all_latents_chunks, dim=0)
    theta = torch.median(all_latents, dim=0).values

    return LayerLatentBatch(
        language=activation_batch.language,
        layer=activation_batch.layer,
        pair_ids=activation_batch.pair_ids,
        z_sg=torch.cat(z_sg_chunks),
        z_pl=torch.cat(z_pl_chunks),
        theta=theta,
    )


def save_latent_batch(batch: LayerLatentBatch, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "language": batch.language,
            "layer": batch.layer,
            "pair_ids": batch.pair_ids,
            "z_sg": batch.z_sg,
            "z_pl": batch.z_pl,
            "theta": batch.theta,
        },
        path,
    )
    return path


def _bootstrap_ci_mean(
    values: torch.Tensor,
    n_samples: int = 400,
    alpha: float = 0.05,
) -> tuple[float, float]:
    n = values.shape[0]
    if n == 1:
        v = float(values[0].item())
        return v, v

    idx = torch.randint(0, n, (n_samples, n))
    means = values[idx].mean(dim=1)
    low = float(torch.quantile(means, alpha / 2).item())
    high = float(torch.quantile(means, 1 - alpha / 2).item())
    return low, high


def _compute_directional_metrics(
    active_a: torch.Tensor,
    active_b: torch.Tensor,
) -> tuple[int, int, int, float, float, float]:
    joint = (active_a & (~active_b)).sum().item()
    denom_ps = (~active_b).sum().item()
    denom_pn = active_a.sum().item()

    ps = joint / denom_ps if denom_ps > 0 else 0.0
    pn = joint / denom_pn if denom_pn > 0 else 0.0
    frc = 0.0 if (ps + pn) == 0.0 else 2 * ps * pn / (ps + pn)
    return int(joint), int(denom_ps), int(denom_pn), ps, pn, frc


def score_number_features(
    latent_batch: LayerLatentBatch,
    bootstrap_samples: int = 400,
    *,
    bootstrap_seed: int | None = None,
) -> pd.DataFrame:
    """Compute per-feature EALE, sign consistency, bootstrap CI, and FRC."""
    if bootstrap_seed is not None:
        torch.manual_seed(int(bootstrap_seed))

    z_sg = latent_batch.z_sg
    z_pl = latent_batch.z_pl
    n_examples, d_sae = z_sg.shape

    delta = z_pl - z_sg
    mean_delta = delta.mean(dim=0)
    abs_eale = mean_delta.abs()

    pos_frac = (delta > 0).float().mean(dim=0)
    neg_frac = (delta < 0).float().mean(dim=0)
    sign_consistency = torch.maximum(pos_frac, neg_frac)

    records: list[dict[str, Any]] = []
    for feat_idx in range(d_sae):
        feat_delta = delta[:, feat_idx]
        ci_low, ci_high = _bootstrap_ci_mean(feat_delta, n_samples=bootstrap_samples)
        ci_excludes_zero = (ci_low > 0) or (ci_high < 0)
        mean_direction = "PL>SG" if mean_delta[feat_idx].item() >= 0 else "SG>PL"

        theta = latent_batch.theta[feat_idx]
        active_sg = z_sg[:, feat_idx] > theta
        active_pl = z_pl[:, feat_idx] > theta

        (
            _, denom_ps_pl, denom_pn_pl, ps_pl, pn_pl, frc_pl,
        ) = _compute_directional_metrics(active_pl, active_sg)
        (
            _, denom_ps_sg, denom_pn_sg, ps_sg, pn_sg, frc_sg,
        ) = _compute_directional_metrics(active_sg, active_pl)

        if frc_pl >= frc_sg:
            frc_direction = "PL>SG"
            frc_score = frc_pl
        else:
            frc_direction = "SG>PL"
            frc_score = frc_sg

        records.append(
            {
                "language": latent_batch.language,
                "layer": latent_batch.layer,
                "feature_id": feat_idx,
                "n_examples": n_examples,
                "mean_direction": mean_direction,
                "mean_delta": float(mean_delta[feat_idx].item()),
                "abs_eale": float(abs_eale[feat_idx].item()),
                "sign_consistency": float(sign_consistency[feat_idx].item()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_excludes_zero": ci_excludes_zero,
                "theta": float(theta.item()),
                "frc_direction": frc_direction,
                "frc_pl": frc_pl,
                "frc_sg": frc_sg,
                "frc_score": frc_score,
                "ps_pl": ps_pl,
                "pn_pl": pn_pl,
                "ps_sg": ps_sg,
                "pn_sg": pn_sg,
                "denom_ps_pl": denom_ps_pl,
                "denom_pn_pl": denom_pn_pl,
                "denom_ps_sg": denom_ps_sg,
                "denom_pn_sg": denom_pn_sg,
            }
        )
    return pd.DataFrame.from_records(records)


def eale_permutation_threshold(
    z_sg: torch.Tensor,
    z_pl: torch.Tensor,
    n_permutations: int = 1000,
    quantiles: tuple[float, ...] = (0.95, 0.99),
    seed: int = 0,
) -> dict[str, object]:
    """Null distribution of abs_eale via label permutation (SG/PL swapped per pair).

    Returns dict with pooled scalar thresholds ("p95", "p99", …), per-feature
    tensor thresholds ("feature_p95", …), and "null_abs_eale" (n_permutations, d_sae).
    Use the per-feature tensors for selection; pooled scalars are for plotting only.
    """
    torch.manual_seed(seed)
    n = z_sg.shape[0]
    device = z_sg.device

    # Batched permutations: process PERM_BATCH permutations at once.
    # Shape per batch: (B, n, d_sae) - faster than a Python loop over 10k iterations.
    PERM_BATCH = 500
    chunks = []
    remaining = n_permutations
    while remaining > 0:
        B = min(PERM_BATCH, remaining)
        swap = torch.randint(0, 2, (B, n, 1), dtype=torch.bool, device=device)
        z_a = torch.where(swap, z_pl.unsqueeze(0), z_sg.unsqueeze(0))
        z_b = torch.where(swap, z_sg.unsqueeze(0), z_pl.unsqueeze(0))
        chunks.append((z_a - z_b).mean(dim=1).abs().float())  # (B, d_sae)
        remaining -= B

    null_mat = torch.cat(chunks, dim=0)  # (n_permutations, d_sae)

    result: dict[str, object] = {"null_abs_eale": null_mat}
    # pooled_np: used only for the scalar summary - numpy has no size limit
    pooled_np = null_mat.cpu().numpy().ravel()
    for q in quantiles:
        pct = q * 100
        key = f"p{int(pct)}" if pct == int(pct) else f"p{pct:.1f}".replace(".", "")
        result[key] = float(np.quantile(pooled_np, q))
        # Per-feature: each feature compared against its own null distribution
        result[f"feature_{key}"] = torch.quantile(null_mat, q, dim=0)  # (d_sae,)
    return result


def frc_permutation_threshold(
    z_sg: torch.Tensor,
    z_pl: torch.Tensor,
    theta: torch.Tensor,
    n_permutations: int = 1000,
    quantiles: tuple[float, ...] = (0.95, 0.99),
    seed: int = 0,
) -> dict[str, object]:
    """Null distribution of FRC scores via label permutation (theta held fixed).

    Returns dict with pooled scalar thresholds ("p95", "p99", …), per-feature
    tensor thresholds ("feature_p95", …), and "null_frc" (n_permutations, d_sae).
    Use per-feature thresholds for selection - FRC null is firing-rate-dependent
    so each feature needs to be compared to its own null distribution.
    """
    torch.manual_seed(seed)
    n = z_sg.shape[0]
    device = z_sg.device

    # Binary activation matrices - theta is fixed across permutations
    active_sg = z_sg > theta.unsqueeze(0)   # (n, d_sae)
    active_pl = z_pl > theta.unsqueeze(0)   # (n, d_sae)

    def _frc_scores_batched(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Batched FRC for direction a>b. a, b: (B, n, d_sae) bool."""
        joint    = (a & ~b).float().sum(1)           # (B, d_sae)
        denom_ps = (~b).float().sum(1)
        denom_pn = a.float().sum(1)
        ps = torch.where(denom_ps > 0, joint / denom_ps, torch.zeros_like(joint))
        pn = torch.where(denom_pn > 0, joint / denom_pn, torch.zeros_like(joint))
        denom = ps + pn
        return torch.where(denom > 0, 2 * ps * pn / denom, torch.zeros_like(denom))

    # Batched permutations: process PERM_BATCH at once instead of looping 10k times
    PERM_BATCH = 500
    chunks = []
    remaining = n_permutations
    while remaining > 0:
        B = min(PERM_BATCH, remaining)
        swap  = torch.randint(0, 2, (B, n, 1), dtype=torch.bool, device=device)
        a_pl  = torch.where(swap, active_sg.unsqueeze(0), active_pl.unsqueeze(0))
        a_sg  = torch.where(swap, active_pl.unsqueeze(0), active_sg.unsqueeze(0))
        frc_pl = _frc_scores_batched(a_pl, a_sg)
        frc_sg = _frc_scores_batched(a_sg, a_pl)
        chunks.append(torch.maximum(frc_pl, frc_sg).float())  # (B, d_sae)
        remaining -= B

    null_mat = torch.cat(chunks, dim=0)  # (n_permutations, d_sae)

    result: dict[str, object] = {"null_frc": null_mat}
    # pooled_np: used only for the scalar summary - numpy has no size limit
    pooled_np = null_mat.cpu().numpy().ravel()
    for q in quantiles:
        pct = q * 100
        key = f"p{int(pct)}" if pct == int(pct) else f"p{pct:.1f}".replace(".", "")
        result[key] = float(np.quantile(pooled_np, q))
        # Per-feature: each feature compared against its own null distribution
        result[f"feature_{key}"] = torch.quantile(null_mat, q, dim=0)  # (d_sae,)
    return result


def compute_empirical_permutation_pvalues(
    observed: np.ndarray | torch.Tensor,
    null_matrix: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Empirical one-sided p-values: p_f = (1 + #{b : null[b,f] >= obs[f]}) / (B+1)."""
    if isinstance(observed, torch.Tensor):
        observed = observed.detach().float().cpu().numpy()
    if isinstance(null_matrix, torch.Tensor):
        null_matrix = null_matrix.detach().float().cpu().numpy()
    obs = np.asarray(observed, dtype=np.float64).ravel()
    null = np.asarray(null_matrix, dtype=np.float64)
    counts = (null >= obs.reshape(1, -1)).sum(axis=0)
    return (1.0 + counts.astype(np.float64)) / (float(null.shape[0]) + 1.0)


def benjamini_hochberg(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """BH FDR step-up procedure. Returns (reject_mask, qvalues) in original order."""
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    m = p.size
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float64)
    p = np.where(np.isfinite(p), p, 1.0)
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    raw = p_sorted * (m / ranks)
    q_sorted = np.empty(m, dtype=np.float64)
    q_sorted[m - 1] = float(min(raw[m - 1], 1.0))
    for i in range(m - 2, -1, -1):
        q_sorted[i] = float(min(raw[i], q_sorted[i + 1], 1.0))
    qvalues = np.empty(m, dtype=np.float64)
    qvalues[order] = q_sorted
    reject = qvalues <= alpha
    return reject, qvalues


def add_frc_fdr_columns(
    scores_df: pd.DataFrame,
    null_frc: np.ndarray | torch.Tensor,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Add frc_pvalue, frc_qvalue, frc_fdr_reject via permutation test + BH FDR."""
    out = scores_df.sort_values("feature_id", kind="mergesort").reset_index(drop=True)
    obs = out["frc_score"].to_numpy(dtype=np.float64)
    if isinstance(null_frc, torch.Tensor):
        null_np = null_frc.detach().float().cpu().numpy()
    else:
        null_np = np.asarray(null_frc, dtype=np.float64)
    pvals = compute_empirical_permutation_pvalues(obs, null_np)
    reject, qvals = benjamini_hochberg(pvals, alpha=fdr_alpha)
    out["frc_pvalue"] = pvals
    out["frc_qvalue"] = qvals
    out["frc_fdr_reject"] = reject
    return out


def add_eale_fdr_columns(
    scores_df: pd.DataFrame,
    null_abs_eale: np.ndarray | torch.Tensor,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Add eale_pvalue, eale_qvalue, eale_fdr_reject via permutation test + BH FDR.

    Mirror of :func:`add_frc_fdr_columns`, but operating on ``abs_eale`` and
    the label-permutation null produced by :func:`eale_permutation_threshold`
    (keyed ``null_abs_eale``).  BH is applied across features exactly as for
    FRC-FDR, so the two selectors are directly comparable.
    """
    out = scores_df.sort_values("feature_id", kind="mergesort").reset_index(drop=True)
    obs = out["abs_eale"].to_numpy(dtype=np.float64)
    if isinstance(null_abs_eale, torch.Tensor):
        null_np = null_abs_eale.detach().float().cpu().numpy()
    else:
        null_np = np.asarray(null_abs_eale, dtype=np.float64)
    pvals = compute_empirical_permutation_pvalues(obs, null_np)
    reject, qvals = benjamini_hochberg(pvals, alpha=fdr_alpha)
    out["eale_pvalue"] = pvals
    out["eale_qvalue"] = qvals
    out["eale_fdr_reject"] = reject
    return out


def _apply_min_frc_denom_mask(
    scores_df: pd.DataFrame,
    mask: np.ndarray,
    min_frc_denom: int,
) -> np.ndarray:
    """Intersect ``mask`` with the per-direction denominator support floor.

    Factored out so FDR and raw-pvalue selectors share exactly the same
    denominator-floor logic (a feature is kept only if the PAIR counts in the
    direction that matches its ``frc_direction`` are ≥ ``min_frc_denom``).
    """
    if min_frc_denom <= 0:
        return mask
    denom_ok_pl = (
        (scores_df["frc_direction"] == "PL>SG")
        & (scores_df["denom_ps_pl"] >= min_frc_denom)
        & (scores_df["denom_pn_pl"] >= min_frc_denom)
    ).to_numpy()
    denom_ok_sg = (
        (scores_df["frc_direction"] == "SG>PL")
        & (scores_df["denom_ps_sg"] >= min_frc_denom)
        & (scores_df["denom_pn_sg"] >= min_frc_denom)
    ).to_numpy()
    return mask & (denom_ok_pl | denom_ok_sg)


def select_candidate_features_frc_raw(
    scores_df: pd.DataFrame,
    pvalue_alpha: float = 0.05,
    min_frc_denom: int = 5,
) -> pd.DataFrame:
    """Select FRC candidates by RAW empirical p-value (no BH correction).
    """
    p = scores_df["frc_pvalue"].to_numpy(dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    mask = p <= float(pvalue_alpha)
    mask = _apply_min_frc_denom_mask(scores_df, mask, min_frc_denom)

    selected = scores_df.loc[mask].copy()
    selected["_sort_p"] = p[mask]
    selected = selected.sort_values(
        by=["_sort_p", "frc_score", "abs_eale"],
        ascending=[True, False, False],
    )
    return selected.drop(columns=["_sort_p"]).reset_index(drop=True)


def select_candidate_features_eale_raw(
    scores_df: pd.DataFrame,
    pvalue_alpha: float = 0.05,
    min_frc_denom: int = 0,
) -> pd.DataFrame:
    """Select EALE candidates by RAW empirical p-value (no BH correction).
    """
    p = scores_df["eale_pvalue"].to_numpy(dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    mask = p <= float(pvalue_alpha)
    mask = _apply_min_frc_denom_mask(scores_df, mask, min_frc_denom)

    selected = scores_df.loc[mask].copy()
    selected["_sort_p"] = p[mask]
    selected = selected.sort_values(
        by=["_sort_p", "abs_eale", "frc_score"],
        ascending=[True, False, False],
    )
    return selected.drop(columns=["_sort_p"]).reset_index(drop=True)


def select_candidate_features_eale_fdr(
    scores_df: pd.DataFrame,
    fdr_alpha: float = 0.05,
    min_frc_denom: int = 0,
) -> pd.DataFrame:
    """Select EALE candidates by BH FDR at fdr_alpha.

    Mirror of :func:`select_candidate_features_frc_fdr`.  BH is recomputed
    here so changing ``fdr_alpha`` in config does not require rerunning the
    permutation draws.  ``min_frc_denom`` defaults to 0 because EALE is a
    mean difference over all paired observations and does not depend on the
    FRC ratio denominators; pass a positive value if you want to additionally
    require the same denominator floor as FRC-FDR (e.g. for head-to-head
    selector comparisons at matched support).  Sort order: BH q ascending,
    then |EALE| descending, then FRC descending.
    """
    reject, sort_q = benjamini_hochberg(
        scores_df["eale_pvalue"].to_numpy(dtype=float),
        alpha=fdr_alpha,
    )
    mask = reject.astype(bool)

    if min_frc_denom > 0:
        denom_ok_pl = (
            (scores_df["frc_direction"] == "PL>SG")
            & (scores_df["denom_ps_pl"] >= min_frc_denom)
            & (scores_df["denom_pn_pl"] >= min_frc_denom)
        ).to_numpy()
        denom_ok_sg = (
            (scores_df["frc_direction"] == "SG>PL")
            & (scores_df["denom_ps_sg"] >= min_frc_denom)
            & (scores_df["denom_pn_sg"] >= min_frc_denom)
        ).to_numpy()
        mask = mask & (denom_ok_pl | denom_ok_sg)

    selected = scores_df.loc[mask].copy()
    selected["_sort_bh_q"] = sort_q[mask]
    selected = selected.sort_values(
        by=["_sort_bh_q", "abs_eale", "frc_score"],
        ascending=[True, False, False],
    )
    return selected.drop(columns=["_sort_bh_q"]).reset_index(drop=True)


def select_candidate_features_frc_fdr(
    scores_df: pd.DataFrame,
    fdr_alpha: float = 0.05,
    min_frc_denom: int = 5,
) -> pd.DataFrame:
    """Select FRC candidates by BH FDR at fdr_alpha with minimum denominator support.

    Requires add_frc_fdr_columns to have been called so frc_pvalue is present.
    BH is recomputed here so changing fdr_alpha in config doesn't require rerunning
    the permutation draws.
    """
    reject, sort_q = benjamini_hochberg(
        scores_df["frc_pvalue"].to_numpy(dtype=float),
        alpha=fdr_alpha,
    )
    mask = reject.astype(bool)

    if min_frc_denom > 0:
        denom_ok_pl = (
            (scores_df["frc_direction"] == "PL>SG")
            & (scores_df["denom_ps_pl"] >= min_frc_denom)
            & (scores_df["denom_pn_pl"] >= min_frc_denom)
        ).to_numpy()
        denom_ok_sg = (
            (scores_df["frc_direction"] == "SG>PL")
            & (scores_df["denom_ps_sg"] >= min_frc_denom)
            & (scores_df["denom_pn_sg"] >= min_frc_denom)
        ).to_numpy()
        mask = mask & (denom_ok_pl | denom_ok_sg)

    selected = scores_df.loc[mask].copy()
    selected["_sort_bh_q"] = sort_q[mask]
    selected = selected.sort_values(
        by=["_sort_bh_q", "frc_score", "abs_eale"],
        ascending=[True, False, False],
    )
    return selected.drop(columns=["_sort_bh_q"]).reset_index(drop=True)


def summarize_discovery(
    scores_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    *,
    fdr_alpha: float | None = None,
    min_frc_denom: int | None = None,
) -> dict[str, Any]:
    """Return a small diagnostics summary."""
    sample_size = min(256, len(scores_df))
    random_baseline_mean = float(scores_df["mean_delta"].sample(sample_size).mean())
    out: dict[str, Any] = {
        "n_features": len(scores_df),
        "n_selected": len(selected_df),
        "mean_abs_eale_all": float(scores_df["abs_eale"].mean()),
        "mean_frc_all": float(scores_df["frc_score"].mean()),
        "random_signed_delta_mean": random_baseline_mean,
    }
    if "frc_fdr_reject" in scores_df.columns:
        out["n_frc_fdr_reject"] = int(scores_df["frc_fdr_reject"].sum())
    if "frc_pvalue" in scores_df.columns:
        out["mean_frc_pvalue"] = float(np.nanmean(scores_df["frc_pvalue"].to_numpy(dtype=float)))
        out["mean_frc_qvalue"] = float(np.nanmean(scores_df["frc_qvalue"].to_numpy(dtype=float)))
    if fdr_alpha is not None:
        out["fdr_alpha"] = float(fdr_alpha)
    if min_frc_denom is not None:
        out["min_frc_denom"] = int(min_frc_denom)
    return out


def discovery_pair_context_and_tokens(
    tokenizer: AutoTokenizer,
    prefix: str,
    continuation_sg: str,
    continuation_pl: str,
) -> dict[str, Any] | None:
    """Return aligned context and first-diverging token IDs for a discovery pair.

    Used by the latent patch sweep to measure the log-prob gap at the decision
    point without requiring the pair to be a full ``MinimalPair`` object.

    Returns
    -------
    dict with keys:
    - ``"context_ids"`` (list[int]): BOS + prefix + shared continuation up to where
      SG and PL continuations first differ.
    - ``"sg_token_id"`` (int): first diverging SG continuation token.
    - ``"pl_token_id"`` (int): first diverging PL continuation token.

    Returns ``None`` if the continuations do not diverge (pair cannot be scored).
    """
    info = _continuation_divergence_info(tokenizer, prefix, continuation_sg, continuation_pl)
    if not info["diverges"]:
        return None
    if info["first_div_sg_id"] is None or info["first_div_pl_id"] is None:
        return None
    context_ids = _aligned_context_ids(tokenizer, prefix, info["sg_ids"], info["lcp_len"])
    return {
        "context_ids": context_ids,
        "sg_token_id": int(info["first_div_sg_id"]),
        "pl_token_id": int(info["first_div_pl_id"]),
    }


def export_candidates_json(
    selected_df: pd.DataFrame,
    summary: dict[str, Any],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "candidates": selected_df.to_dict(orient="records")}
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return output_path
