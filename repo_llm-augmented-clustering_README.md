# LLM-Augmented Clustering — A Critical Analysis & Enhancement Proposals

Reproduction and **critical extension** of *"Large Language Models Enable Few-Shot Clustering"*
(Viswanathan et al., TACL 2024 — [arXiv:2307.00524](https://arxiv.org/abs/2307.00524)),
carried out as an M.Sc. research project at **Université Paris Cité**.

We reproduce the paper's three-stage LLM-augmented clustering framework, stress-test where it
holds and where it breaks, and propose **two novel methods** that outperform all baselines on
2 of 3 benchmarks — for a total LLM cost under **\$0.50**.

> **Tech:** Python · sentence-transformers · scikit-learn · GPT-4.1-nano (API)
> **Datasets:** Bank77 · CLINC150 · Tweet

---

## TL;DR results

- **Keyphrase expansion** gives consistent, cheap gains across all datasets (feature enrichment > constraint injection).
- Our **LLM Semantic Normalization** reaches the **best accuracy on Bank77 (0.649)**.
- Our **LLM Paraphrase Ensemble** reaches the **best accuracy on CLINC (0.787)** and **best NMI on Bank77 (0.819)**.
- **Post-hoc correction is limited**: the LLM rarely disagrees with K-Means, and when it does it is often wrong — low confidence ≠ misclassified.
- The pairwise-constraint oracle is **high-precision (0.78) but very low-recall (0.23)**, which bottlenecks PCKMeans.

See [`poster.pdf`](./poster.pdf) for the full results and figures.

---

## Methods evaluated

| Stage | Method | Verdict |
|---|---|---|
| Before | Keyphrase Expansion | ✅ best overall |
| Before (ours) | **LLM Semantic Normalization** | ✅ best on Bank77 |
| Before (ours) | **LLM Paraphrase Ensemble** | ✅ best on CLINC |
| During | Pairwise Constraints (PCKMeans) | ≈ limited by oracle recall |
| During | ClusterLLM (k-selection) | ≈ fragile |
| After | Post-hoc Correction | ❌ limited |

---

## Repository structure

```
.
├── src/                 # methods, embedding, clustering, LLM interfaces
├── data/                # dataset loaders / splits
├── experiments/         # scripts to reproduce each table/figure
├── notebooks/           # analysis & visualizations
├── poster.pdf           # A0 conference-style poster
├── requirements.txt
└── README.md
```
*(adapt to your actual layout)*

## Quickstart

```bash
git clone https://github.com/DebianeTassadit/llm-augmented-clustering.git
cd llm-augmented-clustering
pip install -r requirements.txt
export OPENAI_API_KEY=...        # for the LLM-augmented methods
python experiments/run_bank77.py
```

## Team & attribution

Group project by **Aymane Nouhail, Khalil Maadani, Tassadit Debiane, Ala Eddine Choukr-Allah**,
supervised by **Pr. Mohamed Nadif** and **Dr. Imed Keraghel** (Université Paris Cité).
Original team repository: [Aymane-Nouhail/LLM-Augmented-Clustering-Engine](https://github.com/Aymane-Nouhail/LLM-Augmented-Clustering-Engine).
My contributions focused on the novel feature-enrichment methods (Normalization / Paraphrase Ensemble)
and the evaluation/analysis pipeline.

Built on the paper by Viswanathan, Gashteovski, Lawrence, Wu & Neubig (TACL 2024).
