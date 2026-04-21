"""Load generated SG/PL discovery pairs from TSV for the feature discovery experiment.

The TSV format is long-form (two rows per pair_id, one SG and one PL).
Required columns: pair_id, language, target_number, continuation, good_prefix, bad_prefix.
Language codes should be eng/spa/deu (not ISO en); older files with "en" are silently remapped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

_REQUIRED_COLS = {"pair_id", "language", "target_number", "continuation", "good_prefix", "bad_prefix"}
_EXPECTED_LANGS = {"eng", "spa", "deu"}
_LANG_ALIASES = {"en": "eng"}  # older generated files used ISO "en"


def _check_tsv(df: pd.DataFrame, path: Path) -> None:
    missing = sorted(_REQUIRED_COLS - set(df.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if df.empty:
        raise ValueError(f"{path}: file is empty")
    unknown = set(df["language"].unique()) - _EXPECTED_LANGS
    if unknown:
        raise ValueError(f"{path}: unexpected language codes {sorted(unknown)}")


def load_discovery_pairs_from_path(path: Path | str) -> pd.DataFrame:
    """Load the long-form discovery TSV and return one row per SG/PL pair."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing discovery pair file: {path}")

    df = pd.read_csv(path, sep="\t")
    df["language"] = df["language"].astype(str).str.strip().replace(_LANG_ALIASES)
    _check_tsv(df, path)

    sg = df[df["target_number"].str.upper() == "SG"]
    pl = df[df["target_number"].str.upper() == "PL"]
    merged = sg.merge(pl, on="pair_id", how="inner", suffixes=("_sg", "_pl"))

    pairs = pd.DataFrame({
        "pair_id": merged["pair_id"],
        "language": merged["language_sg"],
        "prefix_sg": merged["good_prefix_sg"],
        "prefix_pl": merged["good_prefix_pl"],
        "continuation_sg": merged["continuation_sg"],
        "continuation_pl": merged["continuation_pl"],
        "source_idx_sg": merged.get("source_idx_sg"),
        "source_idx_pl": merged.get("source_idx_pl"),
        "distance": merged.get("distance_sg"),
        "has_attractor": merged.get("has_attractor_sg"),
    })

    pairs = pairs.dropna(subset=["prefix_sg", "prefix_pl", "continuation_sg", "continuation_pl"])
    pairs = pairs[(pairs["continuation_sg"].str.len() > 0) & (pairs["continuation_pl"].str.len() > 0)]
    pairs = pairs.reset_index(drop=True)

    logger.info("Loaded %d discovery pairs from %s", len(pairs), path)
    return pairs
