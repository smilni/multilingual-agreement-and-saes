"""Causal validation of discovered SAE features via latent ablation (Experiment 5.3).

Zeros out selected SAE latents at the first-divergence residual position, decodes back
into residual space, and measures the change in log-probability gap on MultiBLiMP pairs.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.multiblimp import MinimalPair
from ..model.loading import register_residual_hooks
from ..utils.logging import get_logger
from .competence import PairResult, evaluate_pair
from .feature_discovery import minimal_pair_divergence_context

logger = get_logger(__name__)


def load_candidate_records(
    candidates_json: Path | str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load candidate rows from feature-discovery JSON."""
    path = Path(candidates_json)
    payload = json.loads(path.read_text())
    rows = payload["candidates"]
    meta = {
        "summary": payload.get("summary"),
        "n_candidates": len(rows),
        "path": str(path),
    }
    return rows, meta


def load_candidate_feature_ids(
    candidates_json: Path | str,
) -> tuple[list[int], dict[str, Any]]:
    """Load `feature_id` values from a feature-discovery candidates JSON."""
    rows, meta = load_candidate_records(candidates_json)
    ids = [int(r["feature_id"]) for r in rows]
    meta = {**meta, "n_candidates": len(ids)}
    return ids, meta


class SaeLatentAblationHook:
    """Forward hook: apply feature ablation while preserving SAE reconstruction error.

    Computes x_new = x + (decode(z_ab) - decode(z)) so that only the
    selected latent contributions are removed and the original error term
    e = x - decode(z) is kept intact.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        layer: int,
        sae: Any,
        feature_indices: list[int],
        sae_device: str | torch.device,
    ) -> None:
        self.model = model
        self.layer = layer
        self.sae = sae
        self.feature_indices = sorted(set(feature_indices))
        self.sae_device = torch.device(sae_device)
        self.position: int | None = None
        self.active: bool = True
        self._handle: Any = None

    def set_position(self, position: int) -> None:
        self.position = position

    def _hook_fn(self, module, inputs, output):
        if not self.active or self.position is None:
            return output
        pos = self.position
        hidden = output[0] if isinstance(output, tuple) else output

        x = hidden[0, pos, :].float().to(self.sae_device)
        with torch.no_grad():
            z = self.sae.encode(x.unsqueeze(0))
            z_ab = z.clone()
            for j in self.feature_indices:
                z_ab[:, j] = 0.0
            x_rec = self.sae.decode(z).squeeze(0)
            x_rec_ab = self.sae.decode(z_ab).squeeze(0)
            x_new = x + (x_rec_ab - x_rec)

        hidden[0, pos, :] = x_new.to(device=hidden.device, dtype=hidden.dtype)
        return output

    def register(self) -> None:
        layer_module = self.model.model.layers[self.layer]
        self._handle = layer_module.register_forward_hook(self._hook_fn)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class SaeLatentPatchHook:
    """Forward hook: replace selected SAE latents with donor values at a fixed position.

    x_new = x + (decode(z_patched) - decode(z)), preserving the reconstruction error.
    Set donor_values before each forward pass; re-register only once per feature.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        layer: int,
        sae: Any,
        feature_indices: list[int],
        sae_device: str | torch.device,
    ) -> None:
        self.model = model
        self.layer = layer
        self.sae = sae
        self.feature_indices = sorted(set(feature_indices))
        self.sae_device = torch.device(sae_device)
        self.position: int | None = None
        self.active: bool = True
        self.donor_values: torch.Tensor | None = None  # set per-pair before running
        self._handle: Any = None

    def set_position(self, position: int) -> None:
        self.position = position

    def set_donor(self, donor_values: torch.Tensor) -> None:
        """Set the donor latent vector (full d_sae) for the next forward pass."""
        self.donor_values = donor_values

    def _hook_fn(self, module, inputs, output):
        if not self.active or self.position is None or self.donor_values is None:
            return output
        pos = self.position
        hidden = output[0] if isinstance(output, tuple) else output

        x = hidden[0, pos, :].float().to(self.sae_device)
        donor = self.donor_values.to(self.sae_device)
        with torch.no_grad():
            z = self.sae.encode(x.unsqueeze(0))
            z_pat = z.clone()
            for j in self.feature_indices:
                z_pat[:, j] = donor[j]
            x_rec = self.sae.decode(z).squeeze(0)
            x_rec_pat = self.sae.decode(z_pat).squeeze(0)
            x_new = x + (x_rec_pat - x_rec)

        hidden[0, pos, :] = x_new.to(device=hidden.device, dtype=hidden.dtype)
        return output

    def register(self) -> None:
        layer_module = self.model.model.layers[self.layer]
        self._handle = layer_module.register_forward_hook(self._hook_fn)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@dataclass
class SingleFeaturePatchResult:
    """Causal patch effect for one SAE feature across all discovery pairs.

    patch_direction is the FRC-aligned direction ("sg_to_pl" or "pl_to_sg").
    mean_patch_effect is the mean logprob gap reduction (positive = moved model
    toward donor's number). switch_rate is the fraction of pairs where the model's
    grammatical preference actually flipped.
    """

    feature_id: int
    patch_direction: str
    mean_patch_effect: float
    switch_rate: float
    n_pairs: int


@torch.no_grad()
def run_single_feature_patch_sweep_on_discovery_pairs(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    pairs_df: Any,
    latent_batch: Any,
    sae: Any,
    layer: int,
    device: torch.device | str,
    sae_device: str | torch.device,
    candidate_feature_ids: list[int],
    frc_directions: dict[int, str],
    desc: str = "Patch sweep",
) -> list[SingleFeaturePatchResult]:
    """FRC-direction-aligned single-feature latent patching on discovery pairs.

    For each candidate feature, transplants its latent value in the FRC-indicated
    direction (SG→PL for PL>SG features, PL→SG for SG>PL features) and records
    mean_patch_effect (logprob gap reduction) and switch_rate (fraction of pairs
    where the model's grammatical preference flipped). Clean baselines are computed
    once before the feature loop; results are sorted by mean_patch_effect descending.
    """
    from .feature_discovery import discovery_pair_context_and_tokens

    device = torch.device(device)
    sae_device = torch.device(sae_device)

    # Align pairs_df rows with latent_batch entries via pair_id.
    id_to_batch_idx = {pid: i for i, pid in enumerate(latent_batch.pair_ids)}
    rows_df = pairs_df[pairs_df["pair_id"].isin(id_to_batch_idx)].copy()

    # Pre-compute per-pair context info and clean log-prob gaps.
    pair_contexts: list[dict[str, Any]] = []
    for row in rows_df.itertuples(index=False):
        ctx_sg = discovery_pair_context_and_tokens(
            tokenizer, row.prefix_sg, row.continuation_sg, row.continuation_pl,
        )
        ctx_pl = discovery_pair_context_and_tokens(
            tokenizer, row.prefix_pl, row.continuation_sg, row.continuation_pl,
        )
        if ctx_sg is None or ctx_pl is None:
            continue
        batch_idx = id_to_batch_idx[row.pair_id]
        pair_contexts.append(
            {
                "pair_id": row.pair_id,
                "batch_idx": batch_idx,
                "ctx_sg": ctx_sg["context_ids"],
                "ctx_pl": ctx_pl["context_ids"],
                "sg_tok": ctx_sg["sg_token_id"],
                "pl_tok": ctx_sg["pl_token_id"],
                "pos_sg": len(ctx_sg["context_ids"]) - 1,
                "pos_pl": len(ctx_pl["context_ids"]) - 1,
            }
        )

    if not pair_contexts:
        logger.warning("Patch sweep: no pairs with valid divergence context; returning empty.")
        return []

    def _gap(context_ids: list[int], sg_tok: int, pl_tok: int) -> float:
        """log P(sg_tok | context) − log P(pl_tok | context) at last position."""
        ids_t = torch.tensor([context_ids], device=device)
        logits = model(ids_t).logits[0, -1]
        lp = torch.log_softmax(logits.float(), dim=-1)
        return (lp[sg_tok] - lp[pl_tok]).item()

    # Clean baseline gaps (one forward pass per context per pair, hook inactive).
    clean_sg: dict[str, float] = {}
    clean_pl: dict[str, float] = {}
    for pc in tqdm(pair_contexts, desc=f"{desc} – clean baseline"):
        clean_sg[pc["pair_id"]] = _gap(pc["ctx_sg"], pc["sg_tok"], pc["pl_tok"])
        clean_pl[pc["pair_id"]] = _gap(pc["ctx_pl"], pc["sg_tok"], pc["pl_tok"])

    z_sg_all = latent_batch.z_sg  # (n, d_sae) CPU tensor
    z_pl_all = latent_batch.z_pl  # (n, d_sae) CPU tensor

    results: list[SingleFeaturePatchResult] = []

    for feat_id in tqdm(candidate_feature_ids, desc=desc):
        # FRC direction determines which single direction to test.
        frc_dir = frc_directions.get(feat_id, "PL>SG")
        patch_dir = "sg_to_pl" if frc_dir == "PL>SG" else "pl_to_sg"

        hook = SaeLatentPatchHook(
            model=model,
            layer=layer,
            sae=sae,
            feature_indices=[feat_id],
            sae_device=sae_device,
        )
        hook.register()

        effects: list[float] = []
        switched: list[bool] = []

        try:
            for pc in pair_contexts:
                bidx = pc["batch_idx"]
                sg_tok = pc["sg_tok"]
                pl_tok = pc["pl_tok"]

                if patch_dir == "sg_to_pl":
                    # Inject z_pl value into SG context: does it push toward PL?
                    hook.set_position(pc["pos_sg"])
                    hook.set_donor(z_pl_all[bidx])
                    gap_patched = _gap(pc["ctx_sg"], sg_tok, pl_tok)
                    # Positive effect = gap shrank (model prefers SG less, or now prefers PL).
                    effects.append(clean_sg[pc["pair_id"]] - gap_patched)
                    # Switched = model flipped from SG preference to PL preference.
                    switched.append(gap_patched < 0)
                else:
                    # Inject z_sg value into PL context: does it push toward SG?
                    hook.set_position(pc["pos_pl"])
                    hook.set_donor(z_sg_all[bidx])
                    gap_patched = _gap(pc["ctx_pl"], sg_tok, pl_tok)
                    # Positive effect = gap rose (model prefers PL less, or now prefers SG).
                    effects.append(gap_patched - clean_pl[pc["pair_id"]])
                    # Switched = model flipped from PL preference to SG preference.
                    switched.append(gap_patched > 0)

        finally:
            hook.remove()

        n = len(effects)
        results.append(
            SingleFeaturePatchResult(
                feature_id=feat_id,
                patch_direction=patch_dir,
                mean_patch_effect=float(sum(effects) / n) if n else 0.0,
                switch_rate=float(sum(switched) / n) if n else 0.0,
                n_pairs=n,
            )
        )

    results.sort(key=lambda r: r.mean_patch_effect, reverse=True)
    return results


def patch_sweep_to_df(results: list[SingleFeaturePatchResult]) -> Any:
    """Convert a list of :class:`SingleFeaturePatchResult` to a pandas DataFrame."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "feature_id": r.feature_id,
                "patch_direction": r.patch_direction,
                "mean_patch_effect": r.mean_patch_effect,
                "switch_rate": r.switch_rate,
                "n_pairs": r.n_pairs,
            }
            for r in results
        ]
    )


class MultiLayerSaeLatentAblation:
    """Ablation hooks at multiple layers sharing the same first-divergence position.

    Each layer uses its own SAE and feature-id list.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        layer_sae_feats: list[tuple[int, Any, list[int]]],
        sae_device: str | torch.device,
    ) -> None:
        self.hooks: list[SaeLatentAblationHook] = []
        for layer, sae, feats in layer_sae_feats:
            if not feats:
                continue
            self.hooks.append(
                SaeLatentAblationHook(
                    model=model,
                    layer=layer,
                    sae=sae,
                    feature_indices=feats,
                    sae_device=sae_device,
                )
            )

    def register(self) -> None:
        for h in self.hooks:
            h.register()

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()

    def set_position(self, position: int) -> None:
        for h in self.hooks:
            h.set_position(position)

    @property
    def active(self) -> bool:
        return self.hooks[0].active if self.hooks else False

    @active.setter
    def active(self, value: bool) -> None:
        for h in self.hooks:
            h.active = value


@torch.no_grad()
def calibrate_mean_abs_latents(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pairs: list[MinimalPair],
    layer: int,
    sae: Any,
    device: torch.device,
    sae_device: str | torch.device,
    max_pairs: int = 64,
) -> torch.Tensor:
    """Mean |z_f| per latent at first-divergence positions."""
    store = register_residual_hooks(model, [layer])
    acc = torch.zeros(sae.cfg.d_sae, device=sae_device)
    n = 0
    try:
        for pair in pairs[:max_pairs]:
            try:
                aligned = minimal_pair_divergence_context(tokenizer, pair)
            except ValueError:
                continue
            pos = len(aligned) - 1
            store.clear()
            model(torch.tensor([aligned], device=device))
            h = store.activations[layer][0, pos, :].float().to(sae_device)
            acc += sae.encode(h.unsqueeze(0)).squeeze(0).abs()
            n += 1
    finally:
        store.remove_hooks()
    return acc / n if n > 0 else acc


def sample_mean_activation_matched_random_features(
    candidate_ids: list[int],
    mu_abs: torch.Tensor,
    k: int,
    n_draws: int = 400,
    seed: int = 0,
) -> list[int]:
    """Sample k non-candidate features whose sum(mu_abs) is closest to the candidate set."""
    cand = set(candidate_ids)
    pool = [i for i in range(mu_abs.shape[0]) if i not in cand]
    target = float(sum(mu_abs[i].item() for i in candidate_ids))
    rng = random.Random(seed)
    best: list[int] = []
    best_diff = float("inf")
    for _ in range(n_draws):
        sample = rng.sample(pool, k)
        diff = abs(sum(mu_abs[i].item() for i in sample) - target)
        if diff < best_diff:
            best_diff = diff
            best = sample
    return best


@dataclass
class AblationRunSummary:
    name: str
    mean_logprob_gap: float
    accuracy: float
    n: int
    mean_gap_delta_vs_clean: float | None
    feature_indices: list[int]
    multi_layer_specs: list[dict[str, Any]] | None = None


def evaluate_pairs_with_ablation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pairs: list[MinimalPair],
    device: torch.device,
    layer: int,
    sae: Any,
    sae_device: str | torch.device,
    feature_indices: list[int] | None,
    scoring: Literal["verb", "first_token", "full_sentence"] = "verb",
    desc: str = "Ablating",
    run_label: str | None = None,
) -> tuple[list[PairResult], AblationRunSummary]:
    """Evaluate minimal pairs with optional SAE latent ablation at first-divergence."""
    hook = SaeLatentAblationHook(
        model=model,
        layer=layer,
        sae=sae,
        feature_indices=feature_indices or [],
        sae_device=sae_device,
    )
    gap_deltas: list[float] = []
    results: list[PairResult] = []

    if feature_indices:
        hook.register()
    try:
        for pair in tqdm(pairs, desc=desc):
            try:
                aligned = minimal_pair_divergence_context(tokenizer, pair)
            except ValueError:
                continue
            pos = len(aligned) - 1
            hook.set_position(pos)

            hook.active = False
            with torch.no_grad():
                clean = evaluate_pair(
                    model, tokenizer, pair, device,
                    scoring=scoring,
                    divergence_context_ids=aligned,
                )

            if feature_indices:
                hook.active = True
                with torch.no_grad():
                    ablated = evaluate_pair(
                        model, tokenizer, pair, device,
                        scoring=scoring,
                        divergence_context_ids=aligned,
                    )
                gap_deltas.append(clean.logprob_gap - ablated.logprob_gap)
                results.append(ablated)
            else:
                results.append(clean)
    finally:
        if feature_indices:
            hook.remove()

    n = len(results)
    mean_gap = sum(r.logprob_gap for r in results) / n if n else 0.0
    acc = sum(1 for r in results if r.correct) / n if n else 0.0
    mean_delta = sum(gap_deltas) / len(gap_deltas) if gap_deltas else None
    summary_name = run_label or ("ablation" if feature_indices else "clean")

    return results, AblationRunSummary(
        name=summary_name,
        mean_logprob_gap=mean_gap,
        accuracy=acc,
        n=n,
        mean_gap_delta_vs_clean=mean_delta,
        feature_indices=list(feature_indices or []),
    )


def run_single_feature_ablation_sweep(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pairs: list[MinimalPair],
    device: torch.device,
    layer: int,
    sae: Any,
    sae_device: str | torch.device,
    feature_ids: list[int],
    scoring: Literal["verb", "first_token", "full_sentence"] = "verb",
) -> dict[str, Any]:
    """Ablate each feature individually and rank by causal effect (mean gap delta)."""
    per_feature: list[AblationRunSummary] = []
    for fid in feature_ids:
        _, s = evaluate_pairs_with_ablation(
            model, tokenizer, pairs, device, layer, sae, sae_device,
            feature_indices=[fid],
            scoring=scoring,
            desc=f"Single f{fid}",
            run_label=f"single_feature_{fid}",
        )
        per_feature.append(s)

    ranked = sorted(
        per_feature,
        key=lambda x: x.mean_gap_delta_vs_clean or 0.0,
        reverse=True,
    )
    return {
        "per_feature": per_feature,
        "ranked_by_mean_gap_delta": ranked,
        "layer": layer,
        "n_features": len(feature_ids),
        "n_pairs": len(pairs),
    }


def top_k_ablation_schedule(
    n_features: int,
    ks: Sequence[int] | None = None,
) -> list[int]:
    """Unique sorted k values in (0, n], always including n.

    Default template is 1, 2, 5, 10 (clipped to n).
    """
    template = (1, 2, 5, 10) if ks is None else tuple(int(k) for k in ks)
    out = {k for k in template if 0 < k <= n_features}
    out.add(n_features)
    return sorted(out)


def run_prefix_ablation_curve(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pairs: list[MinimalPair],
    device: torch.device,
    layer: int,
    sae: Any,
    sae_device: str | torch.device,
    ordered_feature_ids: list[int],
    scoring: Literal["verb", "first_token", "full_sentence"] = "verb",
    k_schedule: Sequence[int] | None = None,
    ordering_label: str = "discovery",
) -> dict[str, Any]:
    """Ablate the first k features for each k in the schedule."""
    n = len(ordered_feature_ids)
    schedule = top_k_ablation_schedule(n, k_schedule)
    points: list[dict[str, Any]] = []
    for k in schedule:
        subset = ordered_feature_ids[:k]
        _, s = evaluate_pairs_with_ablation(
            model, tokenizer, pairs, device, layer, sae, sae_device,
            feature_indices=subset,
            scoring=scoring,
            desc=f"{ordering_label} |k|={k}",
            run_label=f"{ordering_label}_top{k}",
        )
        points.append({"k": k, "feature_ids": list(subset), "summary": s})
    return {
        "ordering_label": ordering_label,
        "schedule": schedule,
        "points": points,
        "layer": layer,
        "n_features": n,
    }


def _finite_float(x: Any) -> float | None:
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def join_discovery_with_single_feature_causality(
    candidate_rows: list[dict[str, Any]],
    single_feature_payload: dict[str, Any],
    discovery_score_key: str = "frc_score",
) -> dict[str, Any]:
    """Merge discovery rows with single-feature ablation results and compute Spearman rho."""
    by_id: dict[int, AblationRunSummary] = {}
    for s in single_feature_payload["per_feature"]:
        by_id[s.feature_indices[0]] = s

    joined: list[dict[str, Any]] = []
    for row in candidate_rows:
        fid = int(row["feature_id"])
        summ = by_id.get(fid)
        if summ is None:
            continue
        joined.append(
            {
                "feature_id": fid,
                discovery_score_key: row.get(discovery_score_key),
                "mean_gap_delta_vs_clean": summ.mean_gap_delta_vs_clean,
                "mean_logprob_gap_ablated": summ.mean_logprob_gap,
                "accuracy_ablated": summ.accuracy,
            }
        )

    xs: list[float] = []
    ys: list[float] = []
    for r in joined:
        sx = _finite_float(r.get(discovery_score_key))
        sy = _finite_float(r.get("mean_gap_delta_vs_clean"))
        if sx is not None and sy is not None:
            xs.append(sx)
            ys.append(sy)

    rho: float | None = None
    p_value: float | None = None
    if len(xs) >= 2:
        res = spearmanr(xs, ys)
        rho = float(res.statistic)
        p_value = float(res.pvalue)

    return {
        "joined_rows": joined,
        "discovery_score_key": discovery_score_key,
        "n_joined": len(joined),
        "n_for_correlation": len(xs),
        "spearman_rho": rho,
        "spearman_pvalue": p_value,
    }


def run_causal_ablation_experiment_bundle(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pairs: list[MinimalPair],
    device: torch.device,
    layer: int,
    sae: Any,
    sae_device: str | torch.device,
    candidates_json: Path | str,
    scoring: Literal["verb", "first_token", "full_sentence"] = "verb",
    k_schedule: Sequence[int] | None = None,
    discovery_score_key: str = "frc_score",
    do_single_feature_sweep: bool = True,
    do_discovery_prefix_curve: bool = True,
    do_causal_prefix_curve: bool = True,
    do_score_correlation: bool = True,
) -> dict[str, Any]:
    """Run single-feature sweep, top-k curves, and score correlation."""
    rows, meta = load_candidate_records(candidates_json)
    ordered_ids = [int(r["feature_id"]) for r in rows]

    out: dict[str, Any] = {
        "meta": meta,
        "ordered_feature_ids": ordered_ids,
        "discovery_score_key": discovery_score_key,
    }

    run_sweep = (
        do_single_feature_sweep or do_causal_prefix_curve or do_score_correlation
    )
    single: dict[str, Any] | None = None
    if run_sweep:
        if not ordered_ids:
            logger.warning("No feature_id rows in candidates JSON; skipping sweep.")
        else:
            single = run_single_feature_ablation_sweep(
                model, tokenizer, pairs, device, layer, sae, sae_device,
                ordered_ids, scoring=scoring,
            )
    out["single_feature"] = single

    if do_discovery_prefix_curve and ordered_ids:
        out["discovery_prefix_curve"] = run_prefix_ablation_curve(
            model, tokenizer, pairs, device, layer, sae, sae_device,
            ordered_ids,
            scoring=scoring,
            k_schedule=k_schedule,
            ordering_label="discovery",
        )
    else:
        out["discovery_prefix_curve"] = None

    if do_causal_prefix_curve and single and ordered_ids:
        causal_order = [s.feature_indices[0] for s in single["ranked_by_mean_gap_delta"]]
        out["causal_rank_prefix_curve"] = run_prefix_ablation_curve(
            model, tokenizer, pairs, device, layer, sae, sae_device,
            causal_order,
            scoring=scoring,
            k_schedule=k_schedule,
            ordering_label="causal_rank",
        )
    else:
        out["causal_rank_prefix_curve"] = None

    if do_score_correlation and single:
        out["discovery_causal_correlation"] = join_discovery_with_single_feature_causality(
            rows, single, discovery_score_key=discovery_score_key,
        )
    else:
        out["discovery_causal_correlation"] = None

    return out


def evaluate_pairs_with_multi_layer_ablation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pairs: list[MinimalPair],
    device: torch.device,
    layer_sae_feats: list[tuple[int, Any, list[int]]],
    sae_device: str | torch.device,
    scoring: Literal["verb", "first_token", "full_sentence"] = "verb",
    desc: str = "Multi-layer ablation",
) -> tuple[list[PairResult], AblationRunSummary]:
    """Like evaluate_pairs_with_ablation but zeros latents at multiple layers at once."""
    multi = MultiLayerSaeLatentAblation(model, layer_sae_feats, sae_device)
    gap_deltas: list[float] = []
    results: list[PairResult] = []

    multi_specs = [
        {"layer": layer, "feature_ids": list(feats)}
        for layer, _, feats in layer_sae_feats
        if feats
    ]
    if multi.hooks:
        multi.register()
    try:
        for pair in tqdm(pairs, desc=desc):
            try:
                aligned = minimal_pair_divergence_context(tokenizer, pair)
            except ValueError:
                continue
            pos = len(aligned) - 1
            multi.set_position(pos)

            multi.active = False
            with torch.no_grad():
                clean = evaluate_pair(
                    model, tokenizer, pair, device,
                    scoring=scoring,
                    divergence_context_ids=aligned,
                )

            if multi.hooks:
                multi.active = True
                with torch.no_grad():
                    ablated = evaluate_pair(
                        model, tokenizer, pair, device,
                        scoring=scoring,
                        divergence_context_ids=aligned,
                    )
                gap_deltas.append(clean.logprob_gap - ablated.logprob_gap)
                results.append(ablated)
            else:
                results.append(clean)
    finally:
        if multi.hooks:
            multi.remove()

    n = len(results)
    mean_gap = sum(r.logprob_gap for r in results) / n if n else 0.0
    acc = sum(1 for r in results if r.correct) / n if n else 0.0
    mean_delta = sum(gap_deltas) / len(gap_deltas) if gap_deltas else None

    return results, AblationRunSummary(
        name="multi_layer_ablation" if multi.hooks else "clean",
        mean_logprob_gap=mean_gap,
        accuracy=acc,
        n=n,
        mean_gap_delta_vs_clean=mean_delta,
        feature_indices=[],
        multi_layer_specs=multi_specs or None,
    )


def run_causal_validation_suite(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pairs: list[MinimalPair],
    device: torch.device,
    layer: int,
    sae: Any,
    sae_device: str | torch.device,
    candidate_feature_ids: list[int],
    scoring: Literal["verb", "first_token", "full_sentence"] = "verb",
    n_random_trials: int = 5,
    random_seed: int = 0,
    calibrate_max_pairs: int = 64,
) -> dict[str, Any]:
    """Clean baseline, candidate ablation, and mean-matched random baselines."""
    logger.info("Causal validation: clean baseline (%d pairs)", len(pairs))
    clean_results, clean_sum = evaluate_pairs_with_ablation(
        model, tokenizer, pairs, device, layer, sae, sae_device,
        feature_indices=None,
        scoring=scoring,
        desc="Clean",
    )

    k = len(candidate_feature_ids)
    cand_results: list[PairResult] = []
    cand_sum: AblationRunSummary | None = None
    random_summaries: list[AblationRunSummary] = []

    if k == 0:
        logger.warning(
            "No candidate features; skipping candidate ablation and random baselines."
        )
    else:
        mu_abs = calibrate_mean_abs_latents(
            model, tokenizer, pairs, layer, sae, device, sae_device,
            max_pairs=calibrate_max_pairs,
        )

        logger.info("Causal validation: ablating %d candidate features", k)
        cand_results, cand_sum = evaluate_pairs_with_ablation(
            model, tokenizer, pairs, device, layer, sae, sae_device,
            feature_indices=candidate_feature_ids,
            scoring=scoring,
            desc="Candidates",
        )

        for t in range(n_random_trials):
            rid = sample_mean_activation_matched_random_features(
                candidate_feature_ids,
                mu_abs.cpu(),
                k=k,
                n_draws=500,
                seed=random_seed + t,
            )
            _, rs = evaluate_pairs_with_ablation(
                model, tokenizer, pairs, device, layer, sae, sae_device,
                feature_indices=rid,
                scoring=scoring,
                desc=f"Random {t+1}/{n_random_trials}",
            )
            random_summaries.append(rs)

    mean_random_gap = sum(s.mean_logprob_gap for s in random_summaries) / max(
        len(random_summaries), 1
    )
    return {
        "clean": clean_sum,
        "candidates": cand_sum,
        "random_trials": random_summaries,
        "mean_random_gap": mean_random_gap,
        "layer": layer,
        "n_random_trials": n_random_trials,
        "k_features": k,
        # Pair-level results for significance testing
        "clean_pair_results": clean_results,
        "cand_pair_results": cand_results,
    }


def causal_significance_tests(
    clean_results: list[PairResult],
    ablated_results: list[PairResult],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired significance tests for clean vs. ablated results.

    Runs bootstrap stability (n_bootstrap resamples, same indices per condition),
    McNemar's exact test on paired binary accuracy, and a one-sided Wilcoxon
    signed-rank test on per-pair Δgap. Both result lists must be aligned
    (same pair order, from run_causal_validation_suite).
    """
    import numpy as np
    from scipy.stats import wilcoxon
    from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test

    n = min(len(clean_results), len(ablated_results))
    if n == 0:
        return {"error": "No paired results available."}

    clean_correct   = np.array([r.correct   for r in clean_results[:n]], dtype=float)
    ablated_correct = np.array([r.correct   for r in ablated_results[:n]], dtype=float)
    clean_gaps      = np.array([r.logprob_gap for r in clean_results[:n]])
    ablated_gaps    = np.array([r.logprob_gap for r in ablated_results[:n]])

    # --- Bootstrap stability ---
    rng = np.random.default_rng(seed)
    boot_delta_acc: list[float] = []
    boot_delta_gap: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_delta_acc.append(float(clean_correct[idx].mean() - ablated_correct[idx].mean()))
        boot_delta_gap.append(float(clean_gaps[idx].mean() - ablated_gaps[idx].mean()))

    da = np.array(boot_delta_acc)
    dg = np.array(boot_delta_gap)

    # --- McNemar's test on paired data ---
    # b: clean correct, ablated wrong (ablation-caused failures)
    # c: clean wrong, ablated correct (ablation-caused recoveries)
    clean_c   = clean_correct.astype(bool)
    ablated_c = ablated_correct.astype(bool)
    mcnemar_b = int((clean_c & ~ablated_c).sum())
    mcnemar_c = int((~clean_c & ablated_c).sum())
    # 2×2 table arranged as [[a, b], [c, d]] for statsmodels
    mcnemar_a = int((clean_c & ablated_c).sum())
    mcnemar_d = int((~clean_c & ~ablated_c).sum())
    mc_table  = [[mcnemar_a, mcnemar_b], [mcnemar_c, mcnemar_d]]
    mc_result = mcnemar_test(mc_table, exact=True)
    mcnemar_stat = float(mc_result.statistic)
    mcnemar_p    = float(mc_result.pvalue)

    # --- Wilcoxon signed-rank test on per-pair Δgap ---
    # Tests H0: median(clean_gap[i] − ablated_gap[i]) = 0.
    # Non-parametric, paired, no normality assumption.
    # alternative="greater" because we expect clean_gap > ablated_gap
    # (ablation should reduce the model's preference for the grammatical sentence).
    delta_gaps = clean_gaps - ablated_gaps
    if (delta_gaps != 0).any():
        wilcoxon_result = wilcoxon(delta_gaps, alternative="greater")
        wilcoxon_stat = float(wilcoxon_result.statistic)
        wilcoxon_p    = float(wilcoxon_result.pvalue)
    else:
        wilcoxon_stat = float("nan")
        wilcoxon_p    = float("nan")

    def _r(x: float) -> float:
        return round(float(x), 5)

    return {
        "n_pairs":            n,
        # Point estimates on full data
        "mean_clean_acc":     _r(clean_correct.mean()),
        "mean_ablated_acc":   _r(ablated_correct.mean()),
        "mean_delta_acc":     _r((clean_correct - ablated_correct).mean()),
        "mean_clean_gap":     _r(clean_gaps.mean()),
        "mean_ablated_gap":   _r(ablated_gaps.mean()),
        "mean_delta_gap":     _r((clean_gaps - ablated_gaps).mean()),
        # Bootstrap stability
        "boot_mean_delta_acc":  _r(da.mean()),
        "boot_std_delta_acc":   _r(da.std()),
        "boot_ci95_delta_acc":  (_r(np.percentile(da, 2.5)), _r(np.percentile(da, 97.5))),
        "boot_mean_delta_gap":  _r(dg.mean()),
        "boot_std_delta_gap":   _r(dg.std()),
        "boot_ci95_delta_gap":  (_r(np.percentile(dg, 2.5)), _r(np.percentile(dg, 97.5))),
        # McNemar's test (paired binary outcomes)
        "mcnemar_stat":    round(mcnemar_stat, 4),
        "mcnemar_pvalue":  mcnemar_p,    # full float - do not round (tiny values become 0.0)
        "mcnemar_b":       mcnemar_b,    # clean correct → ablated wrong
        "mcnemar_c":       mcnemar_c,    # clean wrong   → ablated correct
        # Wilcoxon signed-rank test (paired continuous Δgap, one-sided: clean > ablated)
        "wilcoxon_stat":   round(wilcoxon_stat, 4),
        "wilcoxon_pvalue": wilcoxon_p,   # full float - do not round
    }


def _align_number_pair_result_rows(
    clean_rows: list[dict[str, Any]],
    ablated_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair rows by ``pair_id`` (order follows *clean_rows*)."""
    by_id = {r["pair_id"]: r for r in ablated_rows}
    c_out: list[dict[str, Any]] = []
    a_out: list[dict[str, Any]] = []
    for r in clean_rows:
        pid = r["pair_id"]
        if pid in by_id:
            c_out.append(r)
            a_out.append(by_id[pid])
    return c_out, a_out


def number_pair_dicts_to_pair_results(rows: list[dict[str, Any]]) -> list[PairResult]:
    out: list[PairResult] = []
    for r in rows:
        gap = (float(r["sg_logprob_gap"]) + float(r["pl_logprob_gap"])) / 2.0
        out.append(
            PairResult(
                uid=str(r["pair_id"]),
                language=str(r.get("language", "")),
                prefix="",
                good_continuation="",
                bad_continuation="",
                good_verb="",
                bad_verb="",
                good_logprob=0.0,
                bad_logprob=0.0,
                correct=bool(r["both_correct"]),
                logprob_gap=gap,
            )
        )
    return out


def number_pair_causal_significance_tests(
    clean_rows: list[dict[str, Any]],
    ablated_rows: list[dict[str, Any]],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """McNemar + Wilcoxon + bootstrap for clean vs ablated.
    """
    c_alg, a_alg = _align_number_pair_result_rows(clean_rows, ablated_rows)
    if not c_alg:
        return {"error": "No aligned number-pair rows for testing."}
    clean_pr = number_pair_dicts_to_pair_results(c_alg)
    abl_pr = number_pair_dicts_to_pair_results(a_alg)
    return causal_significance_tests(clean_pr, abl_pr, n_bootstrap=n_bootstrap, seed=seed)


def causal_logistic_regression(
    clean_results: list[PairResult],
    ablated_results: list[PairResult],
    pairs: list[MinimalPair],
) -> dict[str, Any]:
    """Logistic regression of correct ~ is_ablated + has_attractor + distance.

    Falls back to per-subset chi-square tests if statsmodels is not available.
    Zero-variance covariates are dropped to avoid singularity.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import chi2_contingency

    # Build uid → MinimalPair lookup
    pair_lookup: dict[str, MinimalPair] = {p.uid: p for p in pairs}

    # Build pooled DataFrame - each pair appears twice (clean and ablated)
    rows: list[dict[str, Any]] = []
    for is_ab, results in ((0, clean_results), (1, ablated_results)):
        for r in results:
            p = pair_lookup.get(r.uid)
            rows.append({
                "correct":       int(r.correct),
                "is_ablated":    is_ab,
                "has_attractor": int(p.has_attractor) if p is not None else 0,
                "distance":      p.distance if p is not None else 0,
                "logprob_gap":   r.logprob_gap,
            })
    df = pd.DataFrame(rows)

    try:
        import statsmodels.formula.api as smf
        import numpy as _np

        # Drop zero-variance covariates - they make the design matrix singular.
        has_attractor_varies = df["has_attractor"].nunique() > 1
        distance_varies      = df["distance"].nunique() > 1

        if has_attractor_varies and distance_varies:
            first_formula = (
                "correct ~ is_ablated + has_attractor + distance "
                "+ is_ablated:has_attractor"
            )
        elif has_attractor_varies:
            first_formula = "correct ~ is_ablated + has_attractor + is_ablated:has_attractor"
        elif distance_varies:
            first_formula = "correct ~ is_ablated + distance"
        else:
            first_formula = "correct ~ is_ablated"

        # Progressively simpler fallbacks for singularity / separation / nan pvalues.
        candidates = list(dict.fromkeys([
            first_formula,
            "correct ~ is_ablated + has_attractor" if has_attractor_varies else None,
            "correct ~ is_ablated + distance"      if distance_varies      else None,
            "correct ~ is_ablated",
        ]))
        candidates = [f for f in candidates if f is not None]

        fit = None
        formula = first_formula
        fit_error: str = ""
        for attempt in candidates:
            try:
                candidate_fit = smf.logit(attempt, data=df).fit(disp=0)
                # Reject if any p-value is NaN - sign of separation in a term
                if _np.isnan(candidate_fit.pvalues.values).any():
                    fit_error = f"NaN p-values in formula '{attempt}'"
                    continue
                fit = candidate_fit
                formula = attempt
                break
            except Exception as exc:
                fit_error = f"{type(exc).__name__}: {exc}"

        if fit is None:
            return {"method": "logistic_regression", "error": fit_error, "n_obs": len(df)}

        return {
            "method":       "logistic_regression",
            "formula":      formula,
            "n_obs":        int(fit.nobs),
            "params":       {k: round(float(v), 5) for k, v in fit.params.items()},
            "pvalues":      {k: round(float(v), 6) for k, v in fit.pvalues.items()},
            "conf_int":     {
                k: (round(float(lo), 5), round(float(hi), 5))
                for k, (lo, hi) in fit.conf_int().iterrows()
            },
            "aic":          round(float(fit.aic), 3),
            "pseudo_r2":    round(float(fit.prsquared), 5),
            "summary_text": fit.summary().as_text(),
            "dropped_covariates": [
                c for c in ["has_attractor", "distance", "is_ablated:has_attractor"]
                if c not in formula
            ],
        }

    except ImportError:
        # Fallback: per-subset chi-square
        results_out: dict[str, Any] = {"method": "chi2_subsets"}
        subsets = [
            ("all",          np.ones(len(df), dtype=bool)),
            ("attractor",    df["has_attractor"].astype(bool).values),
            ("no_attractor", ~df["has_attractor"].astype(bool).values),
        ]
        for label, mask in subsets:
            sub = df[mask]
            clean_sub   = sub[sub["is_ablated"] == 0]
            ablated_sub = sub[sub["is_ablated"] == 1]
            table = [
                [int(clean_sub["correct"].sum()),   int((~clean_sub["correct"].astype(bool)).sum())],
                [int(ablated_sub["correct"].sum()), int((~ablated_sub["correct"].astype(bool)).sum())],
            ]
            chi2, p, dof, _ = chi2_contingency(table)
            results_out[label] = {
                "n": len(sub) // 2,
                "clean_acc":   round(float(clean_sub["correct"].mean()), 4),
                "ablated_acc": round(float(ablated_sub["correct"].mean()), 4),
                "chi2":        round(float(chi2), 4),
                "pvalue":      round(float(p), 6),
            }
        return results_out


def save_causal_validation_report(payload: dict[str, Any], path: Path) -> Path:
    # Keys holding raw PairResult lists - large and re-derivable, excluded from JSON
    _EXCLUDE_KEYS = {"clean_pair_results", "cand_pair_results"}

    def _serialize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items() if k not in _EXCLUDE_KEYS}
        if isinstance(obj, AblationRunSummary):
            return asdict(obj)
        if isinstance(obj, list):
            return [_serialize(x) for x in obj]
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_serialize(payload), indent=2, ensure_ascii=False))
    return path
