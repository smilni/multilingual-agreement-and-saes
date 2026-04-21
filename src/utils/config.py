from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass
class LanguageConfig:
    code: str
    name: str


@dataclass
class ModelConfig:
    key: str
    hf_id: str
    num_layers: int
    scan_layers: list[int]


@dataclass
class SAEConfig:
    widths: list[str]
    l0: str
    default_width: str
    # Top-level folder inside google/gemma-scope-2-{size}-pt (e.g. resid_post, resid_post_all).
    site: str = "resid_post"
    # SAELens release alias suffix (e.g. res, res-all, transcoders, transcoders-all).
    # If set, we use SAELens named-release mode; if omitted, we fall back to raw repo mode.
    release_alias: str | None = "res"
    # Map config width labels to folder suffixes in the repo (e.g. 256k -> 262k).
    width_folder_names: dict[str, str] | None = None

    def repo_id(self, model_key: str) -> str:
        size = model_key.replace("gemma-3-", "")
        return f"google/gemma-scope-2-{size}-pt"

    def release_name(self, model_key: str) -> str:
        """SAELens release name when release_alias is set, else raw HF repo id."""
        size = model_key.replace("gemma-3-", "")
        if not self.release_alias:
            return self.repo_id(model_key)
        return f"gemma-scope-2-{size}-pt-{self.release_alias}"

    def _width_in_folder(self, width: str) -> str:
        m = self.width_folder_names or {}
        return m.get(width, width)

    def sae_id(self, layer: int, width: str | None = None) -> str:
        width = width or self.default_width
        w = self._width_in_folder(width)
        if self.release_alias:
            # SAELens named releases expect ID without site prefix.
            return f"layer_{layer}_width_{w}_l0_{self.l0}"
        # Raw repo mode expects the full path with site prefix.
        return f"{self.site}/layer_{layer}_width_{w}_l0_{self.l0}"


@dataclass
class ExperimentConfig:
    languages: list[LanguageConfig]
    models: dict[str, ModelConfig]
    sae: SAEConfig
    competence_min_accuracy: float = 0.95
    feature_discovery: dict = field(default_factory=dict)
    ablation: dict = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    path = Path(path) if path else CONFIGS_DIR / "default.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)

    languages = [LanguageConfig(**lang) for lang in raw["languages"]]
    models = {
        key: ModelConfig(key=key, **val)
        for key, val in raw["models"].items()
    }
    sae = SAEConfig(**raw["sae"])

    return ExperimentConfig(
        languages=languages,
        models=models,
        sae=sae,
        competence_min_accuracy=raw.get("competence", {}).get("min_accuracy", 0.95),
        feature_discovery=raw.get("feature_discovery", {}),
        ablation=raw.get("ablation", {}),
    )
