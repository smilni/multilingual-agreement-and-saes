"""Utilities for loading Gemma 3 models and Gemma Scope 2 SAEs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.config import ModelConfig, SAEConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ActivationStore:
    """Collects residual-stream activations via forward hooks.

    After a forward pass, activations[layer_idx] holds the residual-stream
    output tensor of shape (batch, seq_len, d_model) for each hooked layer.
    """

    activations: dict[int, torch.Tensor] = field(default_factory=dict)
    _handles: list[Any] = field(default_factory=list, repr=False)

    def clear(self) -> None:
        self.activations.clear()

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def load_model(
    model_cfg: ModelConfig,
    device: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    logger.info("Loading model %s ...", model_cfg.hf_id)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.hf_id,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Model loaded on %s", next(model.parameters()).device)
    return model, tokenizer


def register_residual_hooks(
    model: AutoModelForCausalLM,
    layers: list[int],
) -> ActivationStore:
    """Register forward hooks that capture post-layer residual-stream activations.

    Hooks are placed on model.model.layers[i] which outputs the residual
    stream after the full transformer block (attention + MLP + residual).
    """
    store = ActivationStore()

    def _make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # Gemma layers return a tuple; element 0 is the hidden state
            hidden = output[0] if isinstance(output, tuple) else output
            store.activations[layer_idx] = hidden.detach()
        return hook_fn

    for idx in layers:
        layer_module = model.model.layers[idx]
        handle = layer_module.register_forward_hook(_make_hook(idx))
        store._handles.append(handle)

    logger.info("Registered residual-stream hooks on layers %s", layers)
    return store


def load_sae(
    sae_cfg: SAEConfig,
    model_key: str,
    layer: int,
    width: str | None = None,
    device: str = "cpu",
):
    """Load a Gemma Scope 2 SAE for a given model, layer, and width.

    Returns an sae_lens.SAE object with .encode() and .decode() methods.
    """
    from sae_lens import SAE

    release = sae_cfg.release_name(model_key)
    sae_id = sae_cfg.sae_id(layer, width)
    logger.info("Loading SAE  release=%s  sae_id=%s ...", release, sae_id)
    sae = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    logger.info(
        "SAE loaded: d_in=%d, dict_size=%d",
        sae.cfg.d_in,
        sae.cfg.d_sae,
    )
    return sae
