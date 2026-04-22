# Causal Analysis of Grammatical Number Representations in Multilingual LMs Using Sparse Autoencoders

This repository contains the code, data, and results for my MA thesis at the University of Tübingen (April 2026).

The thesis investigates whether sparse autoencoder (SAE) features encoding subject number are causally involved in subject–verb agreement behavior in multilingual language models and whether such features are reused across languages. The analysis is conducted on Gemma 3 1B PT with pretrained Gemma Scope 2 SAEs and focuses on English, German, and Spanish.

All numerical results reported in the thesis (tables, figures, significance tests) can be reproduced from and are stored in the notebooks listed below.

## Pipeline and notebooks

The pipeline is designed in two stages. Stage 1 isolates candidate number-sensitive SAE features using a synthetic templated minimal-pair dataset. Stage 2 evaluates them causally on MultiBLiMP and SyntaxGym.

| Stage | Notebook | Content |
|---|---|---|
| Competence evaluation | [`notebooks/01_competence_check.ipynb`](notebooks/01_competence_check.ipynb) | Evaluates Gemma 3 PT models (270M, 1B, 4B, 12B) on MultiBLiMP subject–verb number agreement with first-diverging-token scoring. Used for model selection described in Chapter 4. |
| MultiBLiMP number-flipping extension | [`notebooks/02_generate_number_pairs.ipynb`](notebooks/02_generate_number_pairs.ipynb) | Rewrites MultiBLiMP items with the opposite subject number via `gpt-5-mini`, producing SG/PL prefix pairs with matching continuations. Covers the construction and automatic filtering described in Section 3.3. |
| Feature discovery dataset | [`notebooks/03_generate_feature_discovery_dataset.ipynb`](notebooks/03_generate_feature_discovery_dataset.ipynb) | Generates the templated SG/PL minimal pairs used for feature discovery (10 template families × 10 pairs × 3 languages). Corresponds to Section 3.4. |
| Main discovery and causal validation | [`notebooks/04_feature_discovery.ipynb`](notebooks/04_feature_discovery.ipynb) | Full pipeline for candidate detection (FRC + BH-FDR + support filter), single-feature causal screening, in-language ablation on MultiBLiMP, directional ablations, SyntaxGym specificity check, and cross-lingual transfer. Corresponds to Chapters 5 and 6 and Appendix B. |

All tables and figures from the thesis are produced inside these notebooks.

## Datasets

- **MultiBLiMP 1.0** (subject–verb number agreement, SV word order, English, German, Spanish): Used for competence evaluation and all causal ablation experiments. Source: [Jumelet et al. 2025](https://arxiv.org/abs/2504.02768).
- **MultiBLiMP number-flipping extension**: [`data/number_pairs/`](data/number_pairs/). The final, manually filtered set of SG/PL rewrites with matching singular/plural continuations based on `gpt-5-mini` generations. 
- **Feature discovery dataset**: [`data/feature_discovery_dataset/`](data/feature_discovery_dataset/). Templated SG/PL minimal pairs generated with `gpt-5-mini` for English, German, and Spanish (100 pairs per language).
- **SyntaxGym**: Used for the specificity check. Source: [Gauthier et al. 2020](https://aclanthology.org/2020.acl-demos.10/).

## Model and SAEs

Main experiments use [**Gemma 3 1B PT**](https://huggingface.co/google/gemma-3-1b-pt) with [**Gemma Scope 2**](https://huggingface.co/google/gemma-scope-2-1b-pt-res) residual-stream JumpReLU SAEs, width 16k, medium L0, at layers 7, 13, 17, and 22 (25%, 50%, 65%, 85% of model depth). Competence evaluation additionally covers [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m), [4B](https://huggingface.co/google/gemma-3-4b-pt), and [12B PT](https://huggingface.co/google/gemma-3-12b-pt).

## Setup

```bash
uv sync
```

The project uses `uv` and the dependencies pinned in `pyproject.toml` / `uv.lock`. Feature discovery and causal screening were run on a single NVIDIA L4 (Google Colab); the 12B competence evaluation used an A100.

A `.env` file with `HF_TOKEN` is required to download Gemma 3 and Gemma Scope 2 from Hugging Face. Access to Gemma weights must be requested separately on the Hugging Face model page.

## Summary of results

- **Compact causal feature sets.** After statistical and causal filtering, only 8–35 SAE features per language–layer setting (out of 16,384) are left. Ablating them leads to a statistically significant decline in agreement preferences on MultiBLiMP, far beyond matched random baselines.
- **Layer profile.** Effects of ablating the earliest layer (7) is weak or null. The strongest in-language effects are found in the middle and later layers – 17 for English and layer 22 for German and Spanish.
- **Specificity.** Ablation of the discovered feature sets harms number agreement most, but also affects SyntaxGym reflexive binding and partly subordination. Other syntactic categories (filler-gap, center embedding, garden-path, cleft, NPI) remain stable. The discovered features seem to support broad number-related morphosyntactic competence rather than just agreement.
- **Directional structure.** Plural-preferring features show strong, consistent matched effects across all three languages. Singular-preferring features have weaker, more uneven effects. They often reduce log-probability gaps without flipping binary accuracy.
- **Cross-lingual reuse.** Feature overlap and causal transfer across languages can be described as partial and asymmetric. Exact overlap is limited in early layers and larger in later ones, with the largest cross-lingual Jaccard between German and Spanish. Causal effects of ablating transferred features are strongest in layers 17 and 22. Features discovered in German or Spanish transfer to English more strongly than vice versa.
- **Qualitative profile.** Cross-lingually shared features are not purely agreement features. They combine number with broader semantic or discourse regularities (e.g., collective human-group reference, salient singular discourse referents, non-English plural / distributive predication). Insterestingly, their promoted / suppressed tokens (by logits) extend to other languages, beyond the three used in the discovery step.

## Citation

If you use any part of this work, please cite the thesis:

> Smilga, V. (2026). *Causal Analysis of Grammatical Number Representations in Multilingual Language Models Using Sparse Autoencoders*. MA thesis, Eberhard Karls Universität Tübingen.
