"""Load and preprocess MultiBLiMP minimal pairs for subject-verb number agreement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from ..utils.config import DATA_DIR
from ..utils.logging import get_logger

logger = get_logger(__name__)

REPO_ID = "jumelet/multiblimp"
CACHE_DIR = DATA_DIR / "multiblimp"

LANG_CODES = {
    "eng": "English",
    "deu": "German",
    "spa": "Spanish",
}

NUMBER_FEATURE_VALUES = {"SG", "PL"}

# SV-#  = subject-verb number agreement  (all three languages)
# SP-#  = subject-predicate number agreement (Spanish; maybe we will want to include it later)
NUMBER_PHENOMENA = {"SV-#"}


@dataclass
class MinimalPair:
    """A single MultiBLiMP minimal pair.

    word_order: 'SV' (subject before verb, usable for causal analysis) or 'VS'.
    child_idx/verb_idx: 0-based word indices of subject head and agreeing verb.
    """

    uid: str
    good_sentence: str
    bad_sentence: str
    # Shared prefix up to (but not including) the diverging verb token.
    prefix: str
    good_continuation: str
    bad_continuation: str
    language: str
    paradigm: str
    field: str
    # Verb form in the grammatical sentence (dataset column: head).
    good_verb: str = ""
    # Verb form in the ungrammatical sentence (dataset column: swap_head).
    bad_verb: str = ""
    word_order: str = ""
    child_idx: int = -1
    verb_idx: int = -1
    has_attractor: bool = False
    distance: int = 0


def download_language(lang_code: str) -> Path:
    """Download the MultiBLiMP data file for a language and return the local path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = CACHE_DIR / f"{lang_code}.tsv"
    if local_path.exists():
        logger.info("Using cached MultiBLiMP data for %s", lang_code)
        return local_path

    logger.info("Downloading MultiBLiMP data for %s ...", lang_code)
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{lang_code}/data.tsv",
        repo_type="dataset",
        local_dir=str(CACHE_DIR / "_hf_cache"),
    )
    import shutil

    shutil.copy(downloaded, local_path)
    logger.info("Saved to %s", local_path)
    return local_path


def load_raw(lang_code: str) -> pd.DataFrame:
    """Load the raw TSV for a language into a DataFrame."""
    path = download_language(lang_code)
    df = pd.read_csv(path, sep="\t")
    df["language"] = lang_code
    return df


def inspect_paradigms(lang_code: str) -> pd.Series:
    """Return value counts of phenomenon types (use to verify filter coverage)."""
    df = load_raw(lang_code)
    col = "phenomenon" if "phenomenon" in df.columns else _uid_column(df)
    return df[col].value_counts()


def _uid_column(df: pd.DataFrame) -> str:
    """Identify the paradigm/UID column."""
    for col in ("phenomenon", "uid", "UID", "paradigm", "paradigm_id", "id"):
        if col in df.columns:
            return col
    raise ValueError(f"Cannot identify UID column in MultiBLiMP. Columns: {list(df.columns)}")


def _sentence_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return (grammatical_col, ungrammatical_col) for this DataFrame."""
    for good, bad in [
        ("sen", "wrong_sen"),           # MultiBLiMP v1.0
        ("good_sentence", "bad_sentence"),
        ("sentence_good", "sentence_bad"),
        ("good", "bad"),
        ("correct", "incorrect"),
    ]:
        if good in df.columns and bad in df.columns:
            return good, bad
    raise ValueError(
        f"Cannot identify sentence columns in MultiBLiMP. Columns: {list(df.columns)}"
    )


def _extract_continuations(row: pd.Series, good_col: str, bad_col: str) -> tuple[str, str, str]:
    """Return (prefix, good_continuation, bad_continuation) for a single row.

    Uses the dataset's own `prefix` column when present (most reliable).
    Falls back to word-level diffing when the column is absent.
    """
    good_sentence = str(row[good_col]).strip()
    bad_sentence = str(row[bad_col]).strip()

    # Prefer the dataset-provided prefix column
    if "prefix" in row.index and pd.notna(row["prefix"]) and str(row["prefix"]).strip():
        prefix = str(row["prefix"]).strip()
        good_cont = good_sentence[len(prefix):].strip()
        bad_cont = bad_sentence[len(prefix):].strip()
        # Fall through to diffing if extraction looks wrong
        if good_cont and bad_cont:
            return prefix, good_cont, bad_cont

    # Fallback: word-level diff
    good_tokens = good_sentence.split()
    bad_tokens = bad_sentence.split()
    shared: list[str] = []
    for g, b in zip(good_tokens, bad_tokens):
        if g == b:
            shared.append(g)
        else:
            break
    prefix = " ".join(shared)
    n = len(shared)
    return prefix, " ".join(good_tokens[n:]), " ".join(bad_tokens[n:])


def filter_number_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose phenomenon is in NUMBER_PHENOMENA."""
    filtered = df[df["phenomenon"].isin(NUMBER_PHENOMENA)].copy()
    logger.info(
        "Filtered %d -> %d number-agreement pairs (lang=%s)",
        len(df),
        len(filtered),
        df["language"].iloc[0],
    )
    return filtered


def load_pairs(
    lang_code: str,
    filter_agreement: bool = True,
    svo_only: bool = False,
) -> list[MinimalPair]:
    """Load MultiBLiMP minimal pairs for a language.

    svo_only keeps only SV-order pairs (subject visible before verb), needed
    for causal analysis. VS pairs are fine for competence evaluation only.
    """
    df = load_raw(lang_code)
    if filter_agreement:
        df = filter_number_agreement(df)
    if svo_only and "wo" in df.columns:
        before = len(df)
        df = df[df["wo"] == "SV"].copy()
        logger.info("SVO filter: %d -> %d pairs (lang=%s)", before, len(df), lang_code)
    if len(df) == 0:
        logger.warning("No pairs found for lang=%s after filtering", lang_code)
        return []

    uid_col = _uid_column(df)
    good_col, bad_col = _sentence_columns(df)

    pairs: list[MinimalPair] = []
    for _, row in df.iterrows():
        prefix, good_cont, bad_cont = _extract_continuations(row, good_col, bad_col)
        if not good_cont or not bad_cont:
            continue

        paradigm_val = str(row.get("phenomenon", row[uid_col]))
        grammatical_feature = str(row.get("grammatical_feature", ""))
        ungrammatical_feature = str(row.get("ungrammatical_feature", ""))

        # Good verb is always from head (grammatical), bad verb from swap_head (ungrammatical).
        good_verb = (
            str(row["head"]).strip()
            if "head" in row.index and pd.notna(row["head"])
            else (good_cont.split()[0] if good_cont else "")
        )
        bad_verb = (
            str(row["swap_head"]).strip()
            if "swap_head" in row.index and pd.notna(row["swap_head"])
            else (bad_cont.split()[0] if bad_cont else "")
        )
        pairs.append(
            MinimalPair(
                uid=str(row[uid_col]),
                good_sentence=str(row[good_col]).strip(),
                bad_sentence=str(row[bad_col]).strip(),
                prefix=prefix,
                good_continuation=good_cont,
                bad_continuation=bad_cont,
                good_verb=good_verb,
                bad_verb=bad_verb,
                language=lang_code,
                paradigm=paradigm_val,
                field=f"{grammatical_feature} vs {ungrammatical_feature}",
                word_order=str(row.get("wo", "")),
                child_idx=int(row["child_idx"]) if "child_idx" in row.index and pd.notna(row["child_idx"]) else -1,
                verb_idx=int(row["verb_idx"]) if "verb_idx" in row.index and pd.notna(row["verb_idx"]) else -1,
                has_attractor=bool(row["has_attractors"]) if "has_attractors" in row.index and pd.notna(row["has_attractors"]) else False,
                distance=int(row["distance"]) if "distance" in row.index and pd.notna(row["distance"]) else 0,
            )
        )
    logger.info("Loaded %d minimal pairs for %s", len(pairs), lang_code)
    return pairs


def load_all_languages(
    lang_codes: list[str] | None = None,
    filter_agreement: bool = True,
    svo_only: bool = False,
) -> dict[str, list[MinimalPair]]:
    """Load minimal pairs for multiple languages."""
    if lang_codes is None:
        lang_codes = list(LANG_CODES.keys())
    return {code: load_pairs(code, filter_agreement, svo_only=svo_only) for code in lang_codes}


def pairs_to_dataframe(pairs: list[MinimalPair]) -> pd.DataFrame:
    """Convert a list of MinimalPair objects to a DataFrame for analysis."""
    return pd.DataFrame([vars(p) for p in pairs])
