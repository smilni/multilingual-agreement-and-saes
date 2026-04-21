"""Generate candidate discovery pairs with an LLM.

The templates are hand-written but loosely inspired by subject–verb agreement
environments in MultiBLiMP (SV-# / SV-P).

This module only generates candidate pairs. They still need filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from ..utils.logging import get_logger
from .number_pairs import call_openai, parse_response

logger = get_logger(__name__)

DiscoveryLangKey = Literal["en", "spa", "deu"]


@dataclass(frozen=True)
class DiscoveryTemplate:
    id: str
    instruction: str
    suggested_distance: int | None = None
    suggested_has_attractor: int | None = None
    multiblimp_phenomenon: str | None = None


@dataclass(frozen=True)
class DiscoveryLanguageConfig:
    key: DiscoveryLangKey
    multiblimp_tsv: str
    tsv_language: str
    pair_id_prefix: str
    system_prompt: str
    templates: tuple[DiscoveryTemplate, ...]


# -----------------------------------------------------------------------------
# English
# -----------------------------------------------------------------------------

DISCOVERY_TEMPLATES_EN_CORE: tuple[DiscoveryTemplate, ...] = (
    DiscoveryTemplate(
        id="simple_np",
        instruction=(
            "Simple local agreement with a bare subject DP before a finite lexical verb. "
            "No PP, no relative clause, no inversion."
        ),
        suggested_distance=1,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="adj_np",
        instruction=(
            "Subject DP with one adjective before the noun head, still local agreement, "
            "and no extra noun phrase inside the subject."
        ),
        suggested_distance=2,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="long_subject_no_attractor",
        instruction=(
            "Subject DP with at least two genuine content modifiers before the head noun - the article or "
            "possessive determiner does NOT count as one of them. "
            "Required patterns (choose different ones across pairs): "
            "(a) two adjectives, e.g. 'The tall ancient towers', 'Several bright young doctors'; "
            "(b) numeral + adjective, e.g. 'Three broken windows', 'Five eager volunteers'; "
            "(c) participial adjective + adjective, e.g. 'The recently upgraded old servers'; "
            "(d) possessive + two adjectives, e.g. 'My old reliable car'. "
            "A single adjective with just a determiner ('The old clocks') does NOT qualify - that is adj_np. "
            "No post-nominal PP, relative clause, or any other noun phrase inside the subject."
        ),
        suggested_distance=4,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="pp_attractor",
        instruction=(
            "Subject DP with a PP after the head noun. The noun inside the PP may look like an attractor, "
            "but agreement stays with the head noun."
        ),
        suggested_distance=4,
        suggested_has_attractor=1,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="quantifier_head_attractor",
        instruction=(
            "The subject is headed by a cardinal or indefinite quantifier followed by 'of + plural NP'. "
            "The quantifier head determines agreement: singular quantifiers (one, each) take singular agreement; "
            "plural quantifiers (several, many, most, few, two, three, …) take plural agreement. "
            "The 'of + plural NP' postmodifier must always be overtly present and must always contain a plural noun - "
            "it acts as a permanent attractor that the model must resist. "
            "SG examples: 'One of the students', 'Each of the delegates', 'One of her colleagues'. "
            "PL examples: 'Several of the students', 'Many of the delegates', 'Three of her colleagues'. "
            "The postmodifier noun should be the same (or semantically parallel) between the SG and PL member of each pair. "
            "Vary the quantifier head, the postmodifier noun, and any surrounding sentence material across pairs."
        ),
        suggested_distance=4,
        suggested_has_attractor=1,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="perfect_aux",
        instruction=(
            "Agreement on a finite auxiliary, e.g. has/have + participle. "
            "The prefix should contain the full subject DP, and the continuation should start with the auxiliary."
        ),
        suggested_distance=3,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="irregular_plural",
        instruction=(
            "The subject head noun must have an irregular singular/plural contrast, e.g. child/children, "
            "person/people, foot/feet, tooth/teeth, mouse/mice, goose/geese, man/men, woman/women, ox/oxen. "
            "Wrap the head noun in a varied subject DP: use different determiners (the, a, this, that, my, their, "
            "several, these), optionally one adjective before the noun, and optionally a short post-nominal PP or "
            "participial phrase. Do not produce bare-noun subjects every time. "
            "Vary the complexity of the subject DP across pairs."
        ),
        suggested_distance=2,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="relative_clause_subject",
        instruction=(
            "Subject DP contains a relative clause. The matrix finite verb must agree with the head noun "
            "of the full subject DP, and the continuation should start with the matrix agreeing form."
        ),
        suggested_distance=5,
        suggested_has_attractor=1,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="non_initial_subject",
        instruction=(
            "The agreeing subject DP is not sentence-initial. Use either (a) a short initial adverbial or finite "
            "subordinate clause, then a matrix clause where the target subject precedes its finite lexical verb or auxiliary, "
            "or (b) a first coordinate clause with its own subject and verb, then a coordinator (and/but) and the target "
            "subject before the agreeing finite verb. No inversion and no existential there in the matrix; the prefix must "
            "contain the full matrix subject DP and everything before the agreeing finite form."
        ),
        suggested_distance=6,
        suggested_has_attractor=1,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="pronoun_subject",
        instruction=(
            "The grammatical subject is a third-person personal pronoun: singular (he, she, or it) vs. plural (they). "
            "The prefix must contain the pronoun but does not have to start with it - optionally add a short "
            "sentence-initial adverbial (e.g. 'Yesterday,', 'Every morning,', 'In that case,') or a brief subordinate "
            "clause before the pronoun, and optionally add material after the pronoun before the agreeing verb "
            "(e.g. an adverb, a time phrase, a PP). Vary which singular pronoun is used (he/she/it) across pairs, "
            "vary the surrounding material, and use natural, varied verb choices."
        ),
        suggested_distance=1,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
)

# -----------------------------------------------------------------------------
# Spanish
# -----------------------------------------------------------------------------

DISCOVERY_TEMPLATES_SPA_CORE: tuple[DiscoveryTemplate, ...] = (
    DiscoveryTemplate(
        id="simple_np",
        instruction=(
            "Simple local agreement in Spanish with a bare subject DP before a finite lexical verb. "
            "Use correct determiner number and gender."
        ),
        suggested_distance=1,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="adj_np",
        instruction=(
            "Subject DP with one adjective before or after the noun, still without a competing noun phrase. "
            "Determiner and adjective should agree normally."
        ),
        suggested_distance=2,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="long_subject_no_attractor",
        instruction=(
            "Subject DP with at least two genuine content modifiers before the head noun - the article or "
            "possessive determiner does NOT count as one of them. "
            "Required patterns (choose different ones across pairs): "
            "(a) two adjectives, e.g. 'Los altos antiguos edificios', 'Varias nuevas pequeñas empresas'; "
            "(b) numeral + adjective, e.g. 'Tres viejos edificios', 'Cinco jóvenes voluntarios'; "
            "(c) participial adjective + adjective, e.g. 'Los recientemente renovados viejos museos'; "
            "(d) possessive + two adjectives, e.g. 'Mi viejo y confiable coche'. "
            "Un solo adjetivo con artículo ('Las viejas fábricas') no es suficiente - eso es adj_np. "
            "No PP postnominal, cláusula relativa, ni otro sintagma nominal dentro del sujeto."
        ),
        suggested_distance=4,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="pp_attractor",
        instruction=(
            "Subject DP with a PP after the head noun. The noun inside the PP may behave like an attractor, "
            "but agreement stays with the subject head."
        ),
        suggested_distance=4,
        suggested_has_attractor=1,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="quantifier_head_attractor",
        instruction=(
            "The subject is headed by a cardinal or indefinite quantifier followed by 'de + plural NP'. "
            "The quantifier head determines agreement: singular quantifiers (uno/una) take singular agreement; "
            "plural quantifiers (varios/varias, muchos/muchas, la mayoría de, dos, tres, …) take plural agreement. "
            "The 'de + plural NP' postmodifier must always be overtly present and must always contain a plural noun - "
            "it acts as a permanent attractor. "
            "SG examples: 'Uno de los estudiantes', 'Una de las delegadas', 'Uno de sus colegas'. "
            "PL examples: 'Varios de los estudiantes', 'Muchas de las delegadas', 'Tres de sus colegas'. "
            "The postmodifier noun should be the same (or semantically parallel) between the SG and PL member of each pair. "
            "Vary the quantifier head, the postmodifier noun, and any surrounding sentence material across pairs. "
            "Ensure full gender and number agreement on determiners and adjectives inside the subject DP."
        ),
        suggested_distance=4,
        suggested_has_attractor=1,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="perfect_aux",
        instruction=(
            "Agreement on a finite auxiliary, for example ha/han + participio. "
            "The continuation should begin with the auxiliary."
        ),
        suggested_distance=3,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="irregular_plural",
        instruction=(
            "The subject head noun must have a non-trivial singular/plural contrast - prefer nouns with stem changes, "
            "accent changes, or irregular plural formation (e.g. el hombre/los hombres, el pie/los pies, "
            "el joven/los jóvenes, el carácter/los caracteres) rather than simple -s/-es plurals. "
            "Wrap the head noun in a varied subject DP: use different determiners (el/la/los/las, un/una, este/esta/estos/estas, "
            "mi/mis, su/sus, varios/varias), optionally one adjective agreeing with the noun, and optionally a short "
            "post-nominal PP. Do not produce bare-noun or always-bare-determiner subjects. "
            "Vary the complexity of the subject DP across pairs."
        ),
        suggested_distance=2,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="relative_clause_subject",
        instruction=(
            "Subject DP contains a relative clause. The matrix finite verb or auxiliary agrees with the head noun "
            "of the subject DP."
        ),
        suggested_distance=5,
        suggested_has_attractor=1,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="non_initial_subject",
        instruction=(
            "The agreeing subject DP is not the first phrase of the sentence. Use (a) a short initial subordinate or "
            "adverbial clause, then a matrix clause in canonical subject–verb order with the target subject before the "
            "finite verb, or (b) a first coordinated clause with a different subject, then y/pero and the target subject "
            "before the agreeing verb. The prefix must include the full matrix subject DP; continuation starts with the "
            "finite verb or auxiliary."
        ),
        suggested_distance=6,
        suggested_has_attractor=1,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="pronoun_subject",
        instruction=(
            "The grammatical subject is an explicit third-person personal pronoun: singular (él or ella) vs. "
            "plural (ellos or ellas). Since Spanish is pro-drop, the explicit pronoun must appear in the prefix. "
            "The prefix does not have to start with the pronoun - optionally add a short sentence-initial adverbial "
            "(e.g. 'Ayer,', 'Por la mañana,', 'Después de cenar,') or a brief subordinate clause before the pronoun. "
            "After the pronoun, do NOT insert a PP or locative phrase - at most add a single short adverb "
            "(e.g. 'también', 'ya', 'ahora', 'aún', 'siempre') directly before the verb. "
            "The resulting prefix must sound natural in everyday Spanish; if it sounds awkward read aloud, rewrite it. "
            "Vary which singular pronoun is used (él/ella) and vary gender across pairs. "
            "Use natural, varied verb choices: mix actions, states, auxiliaries, and modals."
        ),
        suggested_distance=1,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
)

# -----------------------------------------------------------------------------
# German
# -----------------------------------------------------------------------------

DISCOVERY_TEMPLATES_DEU_CORE: tuple[DiscoveryTemplate, ...] = (
    DiscoveryTemplate(
        id="simple_np",
        instruction=(
            "Simple local agreement in German with a bare nominative subject DP before a finite lexical verb. "
            "No inversion."
        ),
        suggested_distance=1,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="adj_np",
        instruction=(
            "Subject DP with one adjective and no extra noun phrase inside the subject. "
            "Use normal nominative article/adjective agreement."
        ),
        suggested_distance=2,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="long_subject_no_attractor",
        instruction=(
            "Nominative subject DP with at least two genuine content modifiers before the head noun - "
            "the article or possessive determiner does NOT count as one of them. "
            "Required patterns (choose different ones across pairs): "
            "(a) two adjectives, e.g. 'Die hohen alten Türme', 'Mehrere neue kleine Firmen'; "
            "(b) numeral + adjective, e.g. 'Drei kaputte Fenster', 'Fünf eifrige Freiwillige'; "
            "(c) participial adjective + adjective, e.g. 'Die kürzlich renovierten alten Museen'; "
            "(d) possessive + two adjectives, e.g. 'Mein altes zuverlässiges Auto'. "
            "A single adjective with just an article ('Die alten Türme') does NOT qualify - that is adj_np. "
            "No post-nominal PP, relative clause, or any other noun phrase inside the subject."
        ),
        suggested_distance=4,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="pp_attractor",
        instruction=(
            "Subject DP with a PP after the head noun. The noun inside the PP may look like an attractor, "
            "but agreement stays with the head noun."
        ),
        suggested_distance=4,
        suggested_has_attractor=1,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="quantifier_head_attractor",
        instruction=(
            "The subject is headed by a cardinal or indefinite quantifier followed by a genitive or 'von + plural NP'. "
            "The quantifier head determines agreement: singular quantifiers (einer/eine/eines, jeder/jede/jedes) "
            "take singular agreement; plural quantifiers (mehrere, viele, die meisten, zwei, drei, …) take plural agreement. "
            "The genitive/von postmodifier must always be overtly present and must always contain a plural noun - "
            "it acts as a permanent attractor. "
            "SG examples: 'Einer der Studenten', 'Eine der Delegierten', 'Jeder von ihren Kollegen'. "
            "PL examples: 'Mehrere der Studenten', 'Viele der Delegierten', 'Drei von ihren Kollegen'. "
            "The postmodifier noun should be the same (or semantically parallel) between the SG and PL member of each pair. "
            "Vary the quantifier head, the postmodifier noun, and surrounding sentence material across pairs. "
            "Ensure correct nominative case on the quantifier head and genitive/dative case inside the postmodifier."
        ),
        suggested_distance=4,
        suggested_has_attractor=1,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="aux_or_modal",
        instruction=(
            "Agreement on a finite auxiliary or modal, e.g. hat/haben + participle, ist/sind + participle, "
            "kann/können, soll/sollen, wird/werden + infinitive/participle. "
            "The continuation should begin with the agreeing finite form."
        ),
        suggested_distance=3,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="irregular_plural",
        instruction=(
            "The subject head noun must have an irregular or umlaut plural, e.g. Kind/Kinder, Maus/Mäuse, Buch/Bücher, "
            "Mann/Männer, Frau/Frauen, Fuß/Füße, Zahn/Zähne, Baum/Bäume, Haus/Häuser, Vogel/Vögel. "
            "Wrap the head noun in a varied nominative subject DP: use different determiners (der/die/das/ein/eine/kein, "
            "dieser/diese/dieses, mein/meine, ihr/ihre, mehrere, einige), optionally one adjective with correct "
            "nominative agreement, and optionally a short post-nominal PP or genitive attribute. "
            "Do not produce always-bare or always-minimal subjects. "
            "Vary the complexity of the subject DP across pairs."
        ),
        suggested_distance=2,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="relative_clause_subject",
        instruction=(
            "Subject DP contains a relative clause. The matrix finite verb or auxiliary agrees with the head noun "
            "of the subject DP."
        ),
        suggested_distance=5,
        suggested_has_attractor=1,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="non_initial_subject",
        instruction=(
            "The agreeing nominative subject is not the first constituent. Prefer a compound sentence: first clause with "
            "its own subject and verb, then und/aber/doch and a second clause where the target subject immediately precedes "
            "the agreeing finite verb (no verb-first order in that second clause). Optionally a short initial subordinate "
            "clause only if the matrix still has the subject before the finite verb without inversion. Prefix = everything "
            "through the end of the matrix subject DP; continuation begins with the agreeing finite form."
        ),
        suggested_distance=6,
        suggested_has_attractor=1,
        multiblimp_phenomenon="SV-#",
    ),
    DiscoveryTemplate(
        id="pronoun_subject",
        instruction=(
            "The grammatical subject is a third-person personal pronoun: singular (er, sie, or es) vs. plural (sie). "
            "The prefix does not have to start with the pronoun - optionally add a short sentence-initial adverbial "
            "(e.g. 'Gestern,', 'Jeden Morgen,', 'In diesem Fall,') or a coordinating conjunction clause before the pronoun "
            "in the prefix, and optionally add material after the pronoun (an adverb, a time phrase, a PP) before the "
            "agreeing verb. Vary which singular pronoun is used (er/sie/es) across pairs and use natural, varied verb choices. "
            "Note that plural 'sie' and singular feminine 'sie' are distinguished only by the verb form - make sure the "
            "verb form makes the number contrast unambiguous."
        ),
        suggested_distance=1,
        suggested_has_attractor=0,
        multiblimp_phenomenon="SV-#",
    ),
)

# Per-template diversity hints sent in the user message (always in English regardless of target language).
TEMPLATE_SPECIFIC_DIVERSITY: dict[str, str] = {
    "simple_np": (
        "Vary determiner type across pairs: mix the, a/an (SG where natural), this/these, that/those, "
        "possessives (my/our/their…), bare plurals where natural, and quantificational openers (many, several, some). "
        "Vary subject classes: humans, organizations, animals, artifacts, documents, natural objects, abstract/institutional "
        "nouns where natural-not only professions. Do not reuse the same subject head noun lemma or main verb lemma across pairs."
    ),
    "adj_np": (
        "Use different adjective types: size, age, quality, color, origin, domain-specific, etc. "
        "Do not make every subject human. Do not reuse the same adjective lemma or head noun lemma across pairs."
    ),
    "long_subject_no_attractor": (
        "Every subject must have at least two genuine content modifiers - the article/possessive does NOT count. "
        "Spread the modifier pattern across pairs: do not use the same combination (e.g. numeral+adj) in every pair. "
        "Mix all permitted patterns: two adjectives, numeral+adjective, participial+adjective, possessive+two-adjectives. "
        "No subject should be achievable by simply adding one adjective to adj_np. "
        "Keep the subject internally simple: no post-nominal PPs, relative clauses, or appositive NPs inside it. "
        "Vary determiner types and semantic class of the head noun. Do not reuse head nouns or main verb lemmas."
    ),
    "pp_attractor": (
        "Vary the preposition in the post-nominal PP across pairs (e.g. of, near, beside, behind, under, inside, from, among, "
        "next to-use natural choices for the target language). Vary whether the noun inside the PP is singular or plural. "
        "Avoid always pairing human heads with artifact attractors; mix animacy. Do not reuse the same head noun, attractor noun, "
        "or preposition in every pair."
    ),
    "quantifier_head_attractor": (
        "Vary the quantifier head across pairs - do not use the same quantifier in more than two pairs. "
        "For singular: rotate through one/each/either (EN), uno/una/cualquiera (SPA), einer/eine/jeder/jede (DEU). "
        "For plural: rotate through several/many/most/two/three/few (EN), varios/muchos/la mayoría/dos/tres (SPA), "
        "mehrere/viele/die meisten/zwei/drei (DEU). "
        "Vary the postmodifier noun class across pairs: mix human professions, institutions, animals, and abstract groups. "
        "Vary the main verb and the broader sentence context. "
        "Do not reuse the same postmodifier noun in more than one pair."
    ),
    "perfect_aux": (
        "Vary auxiliary choice and complement (participle, infinitive where appropriate). Vary subject determiners and semantic "
        "class. Do not reuse the same subject head or main lexical verb lemma across pairs."
    ),
    "aux_or_modal": (
        "Wechsle Hilfsverb/Modal und Komplement (Partizip, Infinitiv). Variiere Determinierer und semantische Klasse des Subjekts. "
        "Keine Wiederholung desselben Kopf-Nom-Lemmas oder desselben lexikalischen Hauptverbs in jedem Paar."
    ),
    "irregular_plural": (
        "Use a different irregular noun stem in every pair - do not repeat the same noun. "
        "Vary the subject DP complexity: some pairs should have a bare determiner + noun, others should add an adjective, "
        "a post-nominal PP, or a genitive/possessive. Do not make every subject maximally minimal."
    ),
    "relative_clause_subject": (
        "Mix relative-clause types: subject relatives, object relatives, RCs with a PP inside, and varied animacy pairings "
        "between head and embedded noun. Do not reuse the same relative-pronoun pattern or the same head noun lemma in every pair. "
        "Vary matrix verbs."
    ),
    "pronoun_subject": (
        "Vary which singular pronoun is used across pairs (he/she/it for EN; él/ella for SPA; er/sie/es for DEU). "
        "Vary what precedes the pronoun: some pairs should have a sentence-initial temporal/causal adverbial "
        "('Yesterday,', 'Ayer,', 'Gestern,'), others a brief subordinate clause, and some can start directly with "
        "the pronoun. After the pronoun, only a single short adverb is allowed (e.g. 'also', 'ya', 'auch', 'still', "
        "'aún', 'noch') - do NOT insert a PP, locative phrase, or time NP between the pronoun and the verb. "
        "Do not produce bare 'He/She/They + verb' in more than a third of the pairs. "
        "Vary verb type and tense: mix actions, states, progressives, modals, and auxiliaries."
    ),
    "non_initial_subject": (
        "Mix patterns: initial temporal/causal/conditional subordinates vs. coordinated second conjuncts (and/but/y/pero/und/aber). "
        "When a subordinate clause precedes the matrix, vary embedded verbs and embedded subjects so they can act as distractors "
        "but keep matrix agreement on the target head. For German, favor und/aber second clauses so subject–verb order stays "
        "unambiguous in the target conjunct. Do not reuse the same subordinator, coordinator, or matrix main verb across pairs."
    ),
}


def _default_batch_diversity_text(n_pairs: int) -> str:
    """Cross-pair diversity requirements (quotas scale down for tiny batches)."""
    lines = [
        "Batch diversity (applies to this whole batch):",
        "- Do not reuse the same subject head noun lemma across pairs.",
        "- Do not reuse the same main finite verb lemma (or copula + same predicative head) across pairs.",
        "- Vary determiner type across pairs (not every subject starts with the).",
        "- Vary semantic classes of subjects: people, animals, artifacts, institutions, places, documents, natural objects, "
        "abstracts where natural.",
        "- Avoid making every prefix follow the same narrow lexical or structural pattern.",
    ]
    if n_pairs >= 3:
        lines.append(
            "- At least three pairs should use non-human subjects when still natural for the template."
        )
    else:
        lines.append(
            "- Include non-human subjects in several pairs when natural; avoid an all-human batch unless the template forbids it."
        )
    if n_pairs >= 2:
        lines.append(
            "- At least two pairs should avoid the determiner \"the\" in the subject when natural "
            "(e.g. a/an, this/these, that/those, possessives, quantifiers, bare plurals)."
        )
    else:
        lines.append(
            "- When natural, prefer a determiner other than \"the\" for this single pair."
        )
    lines.append(
        "- Minimality applies within each SG/PL pair only; across pairs, prioritize lexical and structural variety."
    )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

DISCOVERY_SYSTEM_PROMPT_EN = """Generate candidate English subject-verb agreement minimal pairs.

Goal:
- both sentences grammatical
- singular and plural versions semantically as close as possible
- main difference is subject number and the agreement it triggers
- full subject DP stays in the prefix
- continuation begins at the agreeing finite region

Output:
- Return ONLY JSON with one key: "pairs"
- Each pair object must have exactly:
  pair_id, distance, has_attractor, prefix_sg, prefix_pl, continuation_sg, continuation_pl

Rules:
- pair_id: use the numbering requested by the user
- distance: int >= 1, best structural estimate
- has_attractor: 0 or 1
- prefix_sg / prefix_pl:
  - full subject DP and all material before the agreeing finite form
  - no final punctuation
  - minimal difference between SG and PL
  - prefix_sg and prefix_pl must never be identical: subject number must show in the prefix (determiners, noun pluralization, etc.)
  - no inversion, no existential there, no verb-before-subject
- continuation_sg / continuation_pl:
  - begin with the agreeing finite verb or auxiliary
  - keep the rest of the sentence as parallel as possible
  - change only what is required by number/agreement and very local morphology
- Across pairs in one batch, follow the user message's diversity requirements; do not repeat a single lexical prototype.
  Within each pair, SG and PL should still stay maximally parallel.

Do not output explanations, comments, or markdown.
"""

DISCOVERY_SYSTEM_PROMPT_ES = """Generate candidate Spanish subject-verb agreement minimal pairs.

Goal:
- both sentences grammatical
- singular and plural versions semantically as close as possible
- main difference is subject number and the agreement it triggers
- full subject DP stays in the prefix
- continuation begins at the agreeing finite region

Output:
- Return ONLY JSON with one key: "pairs"
- Each pair object must have exactly:
  pair_id, distance, has_attractor, prefix_sg, prefix_pl, continuation_sg, continuation_pl

Rules:
- all natural-language content must be in Spanish
- pair_id: use the numbering requested by the user
- distance: int >= 1, best structural estimate
- has_attractor: 0 or 1
- prefix_sg / prefix_pl:
  - full subject DP and all material before the agreeing finite form
  - no final punctuation
  - minimal difference between SG and PL
  - prefix_sg and prefix_pl must never be identical: subject number must show in the prefix
  - canonical subject-before-verb order
- continuation_sg / continuation_pl:
  - begin with the agreeing finite verb or auxiliary
  - keep the rest of the sentence as parallel as possible
  - change only what is required by number/agreement and very local morphology
- Across pairs in one batch, follow the user message's diversity requirements; do not repeat a single lexical prototype.
  Within each pair, SG and PL should still stay maximally parallel.

Do not output explanations, comments, or markdown.
"""

DISCOVERY_SYSTEM_PROMPT_DE = """Generate candidate German subject-verb agreement minimal pairs.

Goal:
- both sentences grammatical
- singular and plural versions semantically as close as possible
- main difference is subject number and the agreement it triggers
- full subject DP stays in the prefix
- continuation begins at the agreeing finite region

Output:
- Return ONLY JSON with one key: "pairs"
- Each pair object must have exactly:
  pair_id, distance, has_attractor, prefix_sg, prefix_pl, continuation_sg, continuation_pl

Rules:
- all natural-language content must be in standard High German
- pair_id: use the numbering requested by the user
- distance: int >= 1, best structural estimate
- has_attractor: 0 or 1
- prefix_sg / prefix_pl:
  - full subject DP and all material before the agreeing finite form
  - no final punctuation
  - minimal difference between SG and PL
  - prefix_sg and prefix_pl must never be identical: subject number must show in the prefix
  - canonical subject-before-verb order
  - no inversion or verb-first frames with the lexical subject after the verb
- continuation_sg / continuation_pl:
  - begin with the agreeing finite verb or auxiliary
  - keep the rest of the sentence as parallel as possible
  - change only what is required by number/agreement and very local morphology
- Across pairs in one batch, follow the user message’s diversity requirements; do not repeat a single lexical prototype.
  Within each pair, SG and PL should still stay maximally parallel.

Do not output explanations, comments, or markdown.
"""


DISCOVERY_CONFIGS: dict[DiscoveryLangKey, DiscoveryLanguageConfig] = {
    "en": DiscoveryLanguageConfig(
        key="en",
        multiblimp_tsv="eng.tsv",
        # Same code as in data/number_pairs/eng_same_verb.tsv (not ISO "en").
        tsv_language="eng",
        pair_id_prefix="eng",
        system_prompt=DISCOVERY_SYSTEM_PROMPT_EN,
        templates=DISCOVERY_TEMPLATES_EN_CORE,
    ),
    "spa": DiscoveryLanguageConfig(
        key="spa",
        multiblimp_tsv="spa.tsv",
        tsv_language="spa",
        pair_id_prefix="spa",
        system_prompt=DISCOVERY_SYSTEM_PROMPT_ES,
        templates=DISCOVERY_TEMPLATES_SPA_CORE,
    ),
    "deu": DiscoveryLanguageConfig(
        key="deu",
        multiblimp_tsv="deu.tsv",
        tsv_language="deu",
        pair_id_prefix="deu",
        system_prompt=DISCOVERY_SYSTEM_PROMPT_DE,
        templates=DISCOVERY_TEMPLATES_DEU_CORE,
    ),
}


def get_discovery_config(language: DiscoveryLangKey) -> DiscoveryLanguageConfig:
    return DISCOVERY_CONFIGS[language]


def parse_discovery_pairs_response(text: str) -> list[dict[str, Any]]:
    obj = parse_response(text)
    if not obj:
        return []
    if isinstance(obj, list):
        return obj
    pairs = obj.get("pairs")
    if isinstance(pairs, list):
        return pairs
    return []


def assign_source_indices_from_pair_ids(records: list[dict[str, Any]]) -> None:
    """Set source_idx to the numeric suffix of pair_id (eng_0000 -> 0)."""
    for r in records:
        pid = str(r["pair_id"]).strip()
        suffix = pid.split("_")[-1]
        r["source_idx"] = int(suffix)


def trim_to_valid_pairs(
    records: list[dict[str, Any]],
    *,
    target_n: int,
    pair_id_prefix: str,
    pair_id_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen_kept_ids: set[str] = set()

    for r in records:
        ok, reason = validate_pair_record(r)
        pid = str(r.get("pair_id", "")).strip()
        if ok and pid in seen_kept_ids:
            ok, reason = False, "duplicate pair_id"

        if ok and len(kept) < target_n:
            seen_kept_ids.add(pid)
            kept.append(r)
        else:
            msg = reason if not ok else "surplus_over_target_n"
            dropped.append({**r, "_reject_reason": msg})

    for i, r in enumerate(kept):
        r["pair_id"] = f"{pair_id_prefix}_{pair_id_start + i:04d}"
    assign_source_indices_from_pair_ids(kept)

    if len(kept) < target_n:
        logger.warning(
            "Only %d/%d valid pairs after trim (prefix=%s, start=%d); rerun template or raise oversample.",
            len(kept),
            target_n,
            pair_id_prefix,
            pair_id_start,
        )
    return kept, dropped


def validate_pair_record(r: dict[str, Any]) -> tuple[bool, str]:
    for key in ("pair_id", "prefix_sg", "prefix_pl", "continuation_sg", "continuation_pl"):
        if not str(r.get(key, "")).strip():
            return False, f"missing {key}"

    prefix_sg = r["prefix_sg"].strip()
    prefix_pl = r["prefix_pl"].strip()
    cont_sg = r["continuation_sg"].strip()
    cont_pl = r["continuation_pl"].strip()

    if prefix_sg == prefix_pl:
        return False, "prefixes are identical (SG/PL distinction must appear in the prefix)"
    if cont_sg == cont_pl:
        return False, "continuations are identical"
    if prefix_sg.endswith(".") or prefix_pl.endswith("."):
        return False, "prefix ends with a period"

    return True, "ok"


def records_to_long_tsv(
    records: list[dict[str, Any]],
    *,
    language: str = "eng",
    source_idx_offset: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(records):
        ok, reason = validate_pair_record(r)
        if not ok:
            logger.warning("Skipping pair %s: %s", r.get("pair_id", i), reason)
            continue

        pair_id = r["pair_id"].strip()
        source_idx = int(r["source_idx"]) if r.get("source_idx") is not None else source_idx_offset + i
        distance = int(r.get("distance", 1))
        has_attractor = int(bool(r.get("has_attractor", 0)))

        prefix_sg = r["prefix_sg"].strip()
        prefix_pl = r["prefix_pl"].strip()
        continuation_sg = r["continuation_sg"].strip()
        continuation_pl = r["continuation_pl"].strip()

        template_id = r.get("template_id")
        phenomenon_family = r.get("multiblimp_phenomenon")

        rows.append(
            {
                "pair_id": pair_id,
                "language": language,
                "target_number": "SG",
                "good_prefix": prefix_sg,
                "bad_prefix": prefix_pl,
                "continuation": continuation_sg,
                "source_idx": source_idx,
                "distance": distance,
                "has_attractor": has_attractor,
                "template_id": template_id,
                "phenomenon_family": phenomenon_family,
            }
        )
        rows.append(
            {
                "pair_id": pair_id,
                "language": language,
                "target_number": "PL",
                "good_prefix": prefix_pl,
                "bad_prefix": prefix_sg,
                "continuation": continuation_pl,
                "source_idx": source_idx,
                "distance": distance,
                "has_attractor": has_attractor,
                "template_id": template_id,
                "phenomenon_family": phenomenon_family,
            }
        )

    return pd.DataFrame(rows)


def build_user_prompt(
    *,
    n_pairs: int,
    pair_id_start: int,
    diversity_hint: str = "",
    template: DiscoveryTemplate | None = None,
    pair_id_prefix: str = "eng",
    include_default_diversity: bool = True,
) -> str:
    lo = pair_id_start
    hi = pair_id_start + n_pairs - 1

    template_block = ""
    if template is not None:
        lines = [f"TEMPLATE (apply to all {n_pairs} pairs): {template.instruction}"]
        if template.suggested_has_attractor is not None:
            lines.append(f"Use has_attractor={template.suggested_has_attractor} for this batch.")
        if template.multiblimp_phenomenon is not None:
            lines.append(f"Match the phenomenon family {template.multiblimp_phenomenon}.")
        spec = TEMPLATE_SPECIFIC_DIVERSITY.get(template.id)
        if spec:
            lines.append(f"Template-specific diversity:\n{spec}")
        template_block = "\n".join(lines) + "\n\n"

    diversity_sections: list[str] = []
    if include_default_diversity:
        diversity_sections.append(_default_batch_diversity_text(n_pairs))
    if diversity_hint.strip():
        diversity_sections.append(f"Additional instructions:\n{diversity_hint.strip()}")
    diversity_block = ""
    if diversity_sections:
        diversity_block = "\n".join(diversity_sections) + "\n\n"

    return f"""Generate exactly {n_pairs} distinct candidate minimal pairs.

Use pair_id values {pair_id_prefix}_{lo:04d} through {pair_id_prefix}_{hi:04d} in order (one pair per id).

{template_block}{diversity_block}Within each pair, keep SG and PL maximally parallel: change only what is needed for subject number and agreement.
Across pairs, satisfy all diversity requirements above-do not reuse one safe prototype.

Return JSON: {{"pairs": [ ... {n_pairs} objects ... ]}}"""


def call_model_for_pairs(
    *,
    n_pairs: int,
    pair_id_start: int,
    model: str,
    temperature: float = 1.1,
    reasoning_effort: str | None = None,
    diversity_hint: str = "",
    template: DiscoveryTemplate | None = None,
    discovery_language: DiscoveryLangKey = "en",
    include_default_diversity: bool = True,
    max_output_tokens: int | None = 8192,
) -> list[dict[str, Any]]:
    cfg = get_discovery_config(discovery_language)
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {
            "role": "user",
            "content": build_user_prompt(
                n_pairs=n_pairs,
                pair_id_start=pair_id_start,
                diversity_hint=diversity_hint,
                template=template,
                pair_id_prefix=cfg.pair_id_prefix,
                include_default_diversity=include_default_diversity,
            ),
        },
    ]
    text = call_openai(
        messages,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    records = parse_discovery_pairs_response(text)

    for r in records:
        if template is not None:
            r["template_id"] = template.id
            r["multiblimp_phenomenon"] = template.multiblimp_phenomenon
            if r.get("distance") is None and template.suggested_distance is not None:
                r["distance"] = template.suggested_distance
            if r.get("has_attractor") is None and template.suggested_has_attractor is not None:
                r["has_attractor"] = template.suggested_has_attractor

    return records