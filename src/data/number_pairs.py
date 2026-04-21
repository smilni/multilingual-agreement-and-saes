"""Generate SG/PL prefix pairs from MultiBLiMP via LLM number-flipping.

Each MultiBLiMP SV-agreement item gets its prefix rewritten with the
opposite subject number.  The LLM also fixes participle / auxiliary
agreement in the continuations (needed for Spanish & German compound forms).
"""

from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..utils.config import DATA_DIR
from ..utils.logging import get_logger

logger = get_logger(__name__)

PAIRS_DIR = DATA_DIR / "number_pairs"
LANG_NAMES = {"eng": "English", "deu": "German", "spa": "Spanish"}

SYSTEM_PROMPT = (
    "You are a precise linguist specialising in morphosyntactic agreement. "
    "You rewrite sentence fragments by changing ONLY the grammatical number "
    "of the subject head noun and elements that DIRECTLY agree with it. "
    "Other nouns, proper nouns, and unrelated phrases must stay unchanged. "
    "You always return valid JSON and nothing else."
)

# One-shot demonstrations per language.  The Spanish and German examples
# deliberately include a proper-noun PP modifier ("Pierre Martin") that
# must NOT be pluralised, plus a compound-tense continuation whose
# participle needs agreement repair.
_EXAMPLES = {
    "eng": {
        "user": (
            'Task: rewrite this English prefix so that the noun controlling '
            'verb agreement has plural number (currently singular).\n\n'
            'Prefix: "The report that the new manager submitted"\n'
            'Agreement controller (or its antecedent): "that" (singular)\n\n'
            'The sentence has two possible continuations after the prefix:\n'
            '  A) SG-verb continuation: "contains several errors."\n'
            '  B) PL-verb continuation: "contain several errors."\n'
        ),
        "assistant": (
            '{"new_prefix": "The reports that the new manager submitted", '
            '"new_subject": "reports", '
            '"continuation_sg_verb": "contains several errors.", '
            '"continuation_pl_verb": "contain several errors.", '
            '"impossible": false}'
        ),
    },
    "spa": {
        "user": (
            'Task: rewrite this Spanish prefix so that the noun controlling '
            'verb agreement has plural number (currently singular).\n\n'
            'Prefix: "La carta del abogado francés Pierre Martin, que fue enviada ayer,"\n'
            'Agreement controller (or its antecedent): "carta" (singular)\n\n'
            'The sentence has two possible continuations after the prefix:\n'
            '  A) SG-verb continuation: "fue recibida por el juez."\n'
            '  B) PL-verb continuation: "fueron recibida por el juez."\n'
        ),
        "assistant": (
            '{"new_prefix": "Las cartas del abogado francés Pierre Martin, que fueron enviadas ayer,", '
            '"new_subject": "cartas", '
            '"continuation_sg_verb": "fue recibida por el juez.", '
            '"continuation_pl_verb": "fueron recibidas por el juez.", '
            '"impossible": false}'
        ),
    },
    "deu": {
        "user": (
            'Task: rewrite this German prefix so that the noun controlling '
            'verb agreement has plural number (currently singular).\n\n'
            'Prefix: "Der Brief des französischen Anwalts Pierre Martin, der gestern geschickt wurde,"\n'
            'Agreement controller (or its antecedent): "Brief" (singular)\n\n'
            'The sentence has two possible continuations after the prefix:\n'
            '  A) SG-verb continuation: "wurde vom Richter gelesen."\n'
            '  B) PL-verb continuation: "wurden vom Richter gelesen."\n'
        ),
        "assistant": (
            '{"new_prefix": "Die Briefe des französischen Anwalts Pierre Martin, die gestern geschickt wurden,", '
            '"new_subject": "Briefe", '
            '"continuation_sg_verb": "wurde vom Richter gelesen.", '
            '"continuation_pl_verb": "wurden vom Richter gelesen.", '
            '"impossible": false}'
        ),
    },
}


# -- loading ----------------------------------------------------------------

def load_sv_number_items(lang_code: str) -> pd.DataFrame:
    """Load MultiBLiMP SV-agreement items (SV word order only)."""
    from .multiblimp import load_raw, filter_number_agreement

    df = filter_number_agreement(load_raw(lang_code))
    if "wo" in df.columns:
        df = df[df["wo"] == "SV"].copy()
    logger.info("Loaded %d SV-# (SV order) items for %s", len(df), lang_code)
    return df.reset_index(drop=True)


def extract_continuations(row: pd.Series) -> tuple[str, str] | None:
    """Return (sg_verb_cont, pl_verb_cont) from a MultiBLiMP row, or None."""
    prefix = str(row["prefix"]).rstrip()
    good = str(row["sen"]).strip()[len(prefix):].strip()
    bad = str(row["wrong_sen"]).strip()[len(prefix):].strip()
    if not good or not bad:
        return None

    feat = str(row["grammatical_feature"]).strip().upper()
    if feat in ("SG", "SING"):
        return good, bad
    if feat in ("PL", "PLUR"):
        return bad, good
    return None


# -- prompt -----------------------------------------------------------------

def _lang_notes(lang: str) -> str:
    if lang == "deu":
        return (
            "\n   - Update articles/determiners for number (der→die, ein→∅/einige)."
            "\n   - Update adjective endings that agree with the subject."
            "\n   - In continuations: fix ALL elements that agree with the subject - "
            "auxiliary verbs, resumptive/relative pronouns, copulas, participles "
            "(e.g. 'möchten, der ist' → 'möchten, die sind')."
        )
    if lang == "spa":
        return (
            "\n   - Update articles/determiners for number+gender (el→los, la→las)."
            "\n   - Update adjective endings that agree with the subject."
            "\n   - In continuations: fix participle/adjective agreement that depends "
            "on subject number+gender (e.g. 'será retirada' → 'serán retiradas')."
        )
    return ""


def build_messages(
    prefix: str, child: str,
    orig_number: str, target_number: str,
    lang: str,
    continuation_sg_verb: str, continuation_pl_verb: str,
    verb_sg: str = "", verb_pl: str = "",
) -> list[dict]:
    """Assemble [system, one-shot example, user] message list."""
    lang_name = LANG_NAMES.get(lang, lang)

    verb_hint = ""
    if verb_sg and verb_pl:
        verb_hint = (
            f'Target verb: "{verb_sg}" (SG) / "{verb_pl}" (PL) - this is the '
            f"verb whose agreement you are changing. Only change the noun that "
            f"controls THIS verb; leave other subjects/verbs in the prefix alone.\n\n"
        )

    user_prompt = (
        f"Task: rewrite this {lang_name} prefix so that the noun controlling "
        f"verb agreement has {target_number} number (currently {orig_number}).\n\n"
        f'Prefix: "{prefix}"\n'
        f'Agreement controller (or its antecedent): "{child}" ({orig_number})\n'
        f"{verb_hint}\n"
        f"The sentence has two possible continuations after the prefix:\n"
        f'  A) SG-verb continuation: "{continuation_sg_verb}"\n'
        f'  B) PL-verb continuation: "{continuation_pl_verb}"\n\n'
        f"Rules:\n"
        f"1. Identify the noun that controls the target verb. If "
        f'"{child}" is a pronoun or relative, change its antecedent instead. '
        f"Update only that noun and words that directly agree with it "
        f"(its determiner, its adjectives, relative pronouns referring to it). "
        f"Do NOT change other subjects or verbs in the prefix.\n"
        f"2. Return new_subject as the BARE {target_number} content noun "
        f"(no articles, no determiners).\n"
        f"3. Each continuation must be internally consistent: fix ALL "
        f"elements that agree with the subject - coordinated verbs, "
        f"auxiliaries, participles, predicate adjectives.{_lang_notes(lang)}\n"
        f"   The two continuations must remain distinct (one SG, one PL).\n"
        f"4. Minimise changes. Preserve punctuation and capitalisation.\n"
        f"5. If the rewrite is impossible (mass nouns, proper nouns, named "
        f"entities), set impossible=true.\n\n"
        f"Return ONLY valid JSON:\n"
        f'{{"new_prefix": "…", "new_subject": "…", '
        f'"continuation_sg_verb": "…", "continuation_pl_verb": "…", '
        f'"impossible": false}}'
    )

    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    ex = _EXAMPLES.get(lang)
    if ex:
        msgs.append({"role": "user", "content": ex["user"]})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


# -- OpenAI -----------------------------------------------------------------

_client = None

def _openai():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client


def call_openai(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    *,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Dispatch to the Responses API (gpt-5*) or Chat Completions.

    For gpt-5* models, ``max_output_tokens`` caps total output (including reasoning).
    Long JSON outputs or heavy reasoning can hit this and return ``status=incomplete``
    with empty ``output_text`` - raise the limit for big prompts (e.g. 8192).
    ``reasoning_effort`` controls the reasoning budget ("low", "medium", "high").
    Note: gpt-5* reasoning models do not support the temperature parameter.
    """
    if model.startswith("gpt-5"):
        return _call_responses(
            messages, model,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort or "medium",
        )
    return _call_chat(messages, model, temperature, max_output_tokens=max_output_tokens)


def _call_responses(
    messages: list[dict],
    model: str,
    max_output_tokens: int | None = None,
    reasoning_effort: str = "medium",
) -> str:
    input_msgs = [
        {"role": ("developer" if m["role"] == "system" else m["role"]),
         "content": m["content"]}
        for m in messages
    ]
    cap = max_output_tokens if max_output_tokens is not None else 2048
    resp = _openai().responses.create(
        model=model,
        input=input_msgs,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=cap,
    )
    text = resp.output_text
    if text:
        return text.strip()
    # some SDK versions put content in nested items
    for item in resp.output:
        for attr in ("text", "content"):
            val = getattr(item, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, list):
                for part in val:
                    t = getattr(part, "text", None)
                    if t:
                        return t.strip()
    logger.warning("Empty API response (model=%s, status=%s)", model, resp.status)
    return ""


def _call_chat(
    messages: list[dict],
    model: str,
    temperature: float,
    *,
    max_output_tokens: int | None = None,
) -> str:
    resp = _openai().chat.completions.create(
        model=model, messages=messages,
        temperature=temperature,
        max_completion_tokens=max_output_tokens if max_output_tokens is not None else 4096,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    if not content:
        logger.warning("Empty chat response (finish=%s)", resp.choices[0].finish_reason)
        return ""
    return content.strip()


# -- parsing & validation ---------------------------------------------------

def parse_response(text: str) -> dict | None:
    """Extract a JSON object from an LLM response string."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # try the whole text first, then fall back to extracting the first {...} block
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for m in re.finditer(r"\{.*\}", text, re.DOTALL):
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue

    logger.warning("Unparseable LLM response: %.200s", text)
    return None


def _validate(orig_prefix, new_prefix, orig_subj, new_subj, cont_sg, cont_pl):
    """Quick sanity checks. Returns (ok, reason)."""
    if not new_prefix or not new_prefix.strip():
        return False, "empty_prefix"
    if new_prefix.strip() == orig_prefix.strip():
        return False, "identical_to_original"
    if new_subj.lower() == orig_subj.lower():
        return False, "subject_unchanged"
    if len(new_subj.split()) > 3:
        return False, "subject_too_long"
    if new_subj.lower() not in new_prefix.lower():
        return False, f"subject_missing:{new_subj}"
    if not cont_sg or not cont_pl:
        return False, "empty_continuation"
    if cont_sg.strip() == cont_pl.strip():
        return False, "identical_continuations"
    if abs(len(orig_prefix.split()) - len(new_prefix.split())) > 3:
        return False, "word_count_mismatch"
    return True, "ok"


# -- main generation loop --------------------------------------------------

def generate_prefix_pairs(df, lang_code, llm_fn=None,
                          model="gpt-4o-mini", delay=0.1, tag="raw"):
    """For each row, flip the subject number and collect results.

    Saves incrementally after every item so progress survives interrupts.
    On restart, already-processed source_idx values are skipped.
    """
    if llm_fn is None:
        llm_fn = lambda msgs: call_openai(msgs, model=model)

    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PAIRS_DIR / f"{lang_code}_{tag}.json"

    # resume: load existing records and figure out what's already done
    records = []
    done_idx: set[int] = set()
    if out_path.exists():
        try:
            prev = pd.read_json(out_path)
            records = prev.to_dict("records")
            done_idx = {r["source_idx"] for r in records if "source_idx" in r}
            logger.info("Resuming %s: %d items already on disk", lang_code, len(done_idx))
        except Exception:
            pass

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating {lang_code}"):
        if i in done_idx:
            continue

        prefix = str(row["prefix"]).rstrip()
        child = str(row.get("child", "")).strip()
        feat = str(row.get("grammatical_feature", "")).strip().upper()

        if feat in ("SG", "SING"):
            orig_label, tgt_label, orig_code = "singular", "plural", "SG"
        elif feat in ("PL", "PLUR"):
            orig_label, tgt_label, orig_code = "plural", "singular", "PL"
        else:
            records.append(_fail(i, lang_code, f"bad_feature:{feat}"))
            _save_incremental(records, out_path)
            continue

        conts = extract_continuations(row)
        if conts is None:
            records.append(_fail(i, lang_code, "no_continuations"))
            _save_incremental(records, out_path)
            continue
        cont_sg, cont_pl = conts

        verb = str(row.get("verb", "")).strip()
        swap_verb = str(row.get("swap_head", "")).strip()
        if orig_code == "SG":
            v_sg, v_pl = verb, swap_verb
        else:
            v_sg, v_pl = swap_verb, verb

        msgs = build_messages(
            prefix, child, orig_label, tgt_label, lang_code,
            continuation_sg_verb=cont_sg, continuation_pl_verb=cont_pl,
            verb_sg=v_sg, verb_pl=v_pl,
        )

        try:
            parsed = parse_response(llm_fn(msgs))
        except Exception as e:
            records.append(_fail(i, lang_code, str(e)))
            _save_incremental(records, out_path)
            if delay > 0:
                time.sleep(delay)
            continue

        if parsed is None:
            records.append(_fail(i, lang_code, "json_parse_fail"))
        elif parsed.get("impossible"):
            records.append(_fail(i, lang_code, "impossible"))
        else:
            new_prefix = parsed.get("new_prefix", "")
            new_subj = parsed.get("new_subject", "")
            out_sg = parsed.get("continuation_sg_verb", cont_sg)
            out_pl = parsed.get("continuation_pl_verb", cont_pl)

            ok, reason = _validate(prefix, new_prefix, child, new_subj,
                                   out_sg, out_pl)

            if orig_code == "SG":
                rec = dict(prefix_sg=prefix, prefix_pl=new_prefix,
                           subject_sg=child, subject_pl=new_subj)
            else:
                rec = dict(prefix_sg=new_prefix, prefix_pl=prefix,
                           subject_sg=new_subj, subject_pl=child)

            lemma = _lemma(str(row.get("child_features", "")), child)
            dist = int(row.get("distance", 0)) if pd.notna(row.get("distance")) else 0

            rec.update(
                continuation_sg_verb=out_sg, continuation_pl_verb=out_pl,
                verb_sg=v_sg, verb_pl=v_pl,
                source_idx=i, language=lang_code, original_number=orig_code,
                subject_lemma=lemma, auto_valid=ok, validation_reason=reason,
                has_attractor=bool(row.get("has_attractors", False)),
                distance=dist,
            )
            records.append(rec)

        _save_incremental(records, out_path)
        if delay > 0:
            time.sleep(delay)

    return pd.DataFrame(records)


def _save_incremental(records, path):
    pd.DataFrame(records).to_json(path, orient="records", force_ascii=False, indent=2)


def _fail(idx, lang, reason):
    return dict(source_idx=idx, language=lang, prefix_sg=None, prefix_pl=None,
                auto_valid=False, validation_reason=reason)


def _lemma(feat_str, fallback):
    try:
        d = ast.literal_eval(feat_str)
        if isinstance(d, dict):
            return d.get("lemma", fallback)
    except (ValueError, SyntaxError):
        pass
    return fallback


# -- post-generation --------------------------------------------------------

def reconstruct_sentences(row):
    """Build all four grammatical / ungrammatical sentence variants."""
    p_sg = str(row["prefix_sg"]).rstrip()
    p_pl = str(row["prefix_pl"]).rstrip()
    c_sg = str(row["continuation_sg_verb"]).strip()
    c_pl = str(row["continuation_pl_verb"]).strip()
    return {
        "sg_correct":   f"{p_sg} {c_sg}",
        "sg_incorrect": f"{p_sg} {c_pl}",
        "pl_correct":   f"{p_pl} {c_pl}",
        "pl_incorrect": f"{p_pl} {c_sg}",
    }


def summarize_generation(df):
    """Return a dict of summary stats for a raw generation DataFrame."""
    total = len(df)
    n_valid = int(df["auto_valid"].sum()) if "auto_valid" in df.columns else 0
    out = {
        "total": total, "auto_valid": n_valid,
        "auto_invalid": total - n_valid,
        "valid_rate": round(n_valid / total, 3) if total else 0,
    }
    if "validation_reason" in df.columns:
        out["rejection_reasons"] = (
            df.loc[~df["auto_valid"], "validation_reason"]
            .value_counts().to_dict()
        )
    if "original_number" in df.columns:
        out["direction_counts"] = df["original_number"].value_counts().to_dict()
    return out


def save_pairs(df, lang_code, tag="generated"):
    """Write pairs DataFrame to JSON and return the path."""
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    path = PAIRS_DIR / f"{lang_code}_{tag}.json"
    df.to_json(path, orient="records", force_ascii=False, indent=2)
    logger.info("Saved %d pairs → %s", len(df), path)
    return path


_SAME_VERB_COLS = [
    "pair_id", "language", "target_number", "continuation",
    "good_prefix_type", "good_prefix", "bad_prefix_type", "bad_prefix",
    "good_sentence", "bad_sentence",
    "source_idx", "has_attractor", "distance", "subject_lemma",
]


def export_same_verb_prefix_pairs(
    languages: list[str],
    generated_pairs: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    """Export same-verb prefix pairs as TSV (two rows per pair: SG and PL).

    Each row holds one verb continuation fixed and contrasts the agreeing
    prefix (good) against the disagreeing prefix (bad).
    """
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for lang in languages:
        df = generated_pairs[lang]
        rows: list[dict] = []

        for _, r in df.iterrows():
            prefix_sg = str(r["prefix_sg"]).rstrip()
            prefix_pl = str(r["prefix_pl"]).rstrip()
            cont_sg = str(r["continuation_sg_verb"]).strip()
            cont_pl = str(r["continuation_pl_verb"]).strip()

            subj_sg = str(r["subject_sg"]) if "subject_sg" in r.index and pd.notna(r.get("subject_sg")) else ""
            subj_pl = str(r["subject_pl"]) if "subject_pl" in r.index and pd.notna(r.get("subject_pl")) else ""
            lemma = str(r["subject_lemma"]) if "subject_lemma" in r.index and pd.notna(r.get("subject_lemma")) else ""

            common = {
                "pair_id": r["pair_id"],
                "language": r.get("language", lang),
                "source_idx": r.get("source_idx"),
                "has_attractor": r.get("has_attractor"),
                "distance": r.get("distance"),
                "subject_lemma": lemma,
            }

            rows.append({
                **common,
                "target_number": "SG",
                "continuation": cont_sg,
                "good_prefix_type": "SG",
                "good_prefix": prefix_sg,
                "bad_prefix_type": "PL",
                "bad_prefix": prefix_pl,
                "good_sentence": f"{prefix_sg} {cont_sg}",
                "bad_sentence": f"{prefix_pl} {cont_sg}",
                "subject_good": subj_sg,
                "subject_bad": subj_pl,
            })

            rows.append({
                **common,
                "target_number": "PL",
                "continuation": cont_pl,
                "good_prefix_type": "PL",
                "good_prefix": prefix_pl,
                "bad_prefix_type": "SG",
                "bad_prefix": prefix_sg,
                "good_sentence": f"{prefix_pl} {cont_pl}",
                "bad_sentence": f"{prefix_sg} {cont_pl}",
                "subject_good": subj_pl,
                "subject_bad": subj_sg,
            })

        out_df = pd.DataFrame(rows, columns=_SAME_VERB_COLS)
        path = PAIRS_DIR / f"{lang}_same_verb.tsv"
        out_df.to_csv(path, sep="\t", index=False)
        logger.info("Wrote %d same-verb prefix rows → %s", len(out_df), path)
        paths[lang] = path

    return paths
