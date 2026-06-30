# Spanish Knowledge Distillation

**Can you shrink a large language model and keep it good in a *specific* language or does the smaller model just quietly trade its multilingual ability for English?**

This repository contains the code, training jobs, and evaluation harness for the paper
*"Exploring Knowledge Distillation with Language-Specific Calibration Data"*

The accompanying paper is included as [`Knowledge_Distillation.pdf`](./Knowledge_Distillation.pdf).

---

## TL;DR

- We took two large, multilingual LLMs (**LLaMA** and **BLOOM**) and distilled them into smaller students using Microsoft's **MiniLLM** framework.
- We calibrated the distillation on **Spanish-only**, **English-only**, and **Spanish+English** instruction data to ask whether language-specific calibration makes a smaller model *better in that language*.
- We **extended MiniLLM to support BLOOM**, and re-engineered the training pipeline (ZeRO-Offload, shorter sequences, fewer steps) so the entire process fits on a **single H100 GPU**, roughly 7× less VRAM than the original setup demanded.
- **Finding:** language-specific calibration did **not** reliably improve same-language performance. A model's multilingual ability appears to come from **pre-training**, not from the calibration data used during distillation, which is exactly the failure mode the study set out to probe.

---

## Why this project exists

This project started with a skeptical reading of how knowledge distillation results are usually reported.

The MiniLLM paper - like most distillation work - distills a multilingual model using **English-only calibration data** and then demonstrates the improvement using **English-only benchmarks**. On paper, the student gets better. But that setup has a blind spot: it only ever *looks* at English.

That led to our core suspicion:

> What if the distilled model isn't actually learning the teacher's knowledge at all? What if the "improvement" is just the model **reallocating its multilingual parameters toward English** and gaining on English benchmarks by cannibalizing its ability in every other language?

If that were true, the English-only evaluation would never catch it. The model would look like a successful distillation while it was really just collapsing toward the one language being measured. The only way to tell the two stories apart is to **distill with a non-English language and then measure that language directly.**

So we introduced **Spanish** as the probe. If distillation genuinely transfers knowledge from the teacher, calibrating on Spanish should improve Spanish performance. If instead the gains come from siphoning multilingual capacity into the calibration language, we'd expect to see the language-specific signal break down and we'd be able to attribute "improvements" to where the multilingual ability actually lives.

For anyone deploying a small model for a non-English use case, the difference between those two explanations matters enormously.

## Research question

> **"How much can we improve a language model's performance on a single language by distilling a larger language model using calibration data from that language?"**

If language-specific calibration worked the way we hoped, a Spanish-calibrated student should beat an English-calibrated student on Spanish tasks. We tested exactly that.

---

## Background: what is Knowledge Distillation?

Knowledge Distillation (KD) trains a small **student** model to imitate a large **teacher** model, so you keep much of the capability at a fraction of the size and inference cost.

Classic KD minimizes the **forward** KL divergence between teacher and student distributions. For open-ended text generation this causes *zero-forcing*: the student spreads probability mass trying to cover every mode of the teacher, including unlikely ones, and ends up bland.

**MiniLLM** (Gu et al., 2023) instead minimizes the **reverse** KL divergence, which is *mode-seeking*: the student focuses its capacity on the teacher's most important modes. To make this trainable in practice, MiniLLM adds:

- **Teacher-mixed sampling** to avoid reward-hacking (degenerate, repetitive outputs that fool the teacher).
- **Single-step regularization + length normalization** to stabilize training and discourage trivially short answers.
- A **language-modelling loss** on a pre-training corpus so the student doesn't forget how to model language while it learns to follow instructions.

---

## How we did it

### Models
| Role | LLaMA | BLOOM |
|------|-------|-------|
| Teacher | 13B | 7B |
| Student | 7B | 3B |

These two families were chosen deliberately: **LLaMA's pre-training is ~0.13% Spanish, while BLOOM's is ~10% Spanish.** That contrast is central to interpreting the results.

### Data
- **OASST1 (OpenAssistant)**: human-written, human-annotated assistant conversations. Used as the **calibration / instruction data**: prompts are pushed through teacher and student, and the student is nudged (via reverse KLD) toward the teacher. We filtered Spanish and English, kept only the start of each conversation tree to avoid duplicates, truncated to the max prompt length, and applied MiniLLM preprocessing.
- **Spanish & English Wikipedia**: plain article text used for the **language-modelling loss**, so the student is penalized for losing general fluency.

Three calibration regimes were trained for each model family: **Spanish-only**, **English-only**, and **Both**.

### Making it fit on one GPU
MiniLLM is very resource-hungry. Three changes brought it down to a single H100:
1. Sequence length **512 → 256** tokens.
2. Training steps **5000 → 1000** (KLD between teacher and student plateaus well before 5000).
3. **ZeRO-Offload** to push optimizer/parameter state to CPU, cutting GPU memory by roughly 7×.

### Evaluation
We evaluated zero-shot with the **Language Model Evaluation Harness** across English, Spanish, and (where available) Mandarin, on five multilingual benchmarks: **HeadQA, PAWS-X, XStoryCloze, XNLI, and LAMBADA**. A later round used **ROUGE-L** on a held-out OASST test set to more directly measure the distilled instruction-following task.

---

## Key findings

- Distillation generally **maintained or slightly reduced** benchmark performance; language-specific calibration did **not** consistently lift same-language scores. The Spanish-calibrated student was not reliably better at Spanish than the English-calibrated one.
- The decisive signal came from the **pre-training mix**, not the calibration data:
  - **LLaMA** (0.13% Spanish in pre-training) kept replying in English and answered Spanish prompts with low adequacy.
  - **BLOOM** (10% Spanish in pre-training) answered Spanish prompts in Spanish.
- **Conclusion:** MiniLLM can perform a domain shift and retains general language-modelling ability, but we found **no empirical support** that distillation with instruction-style calibration data improves a smaller model's ability in a target language. The multilingual capability is inherited from pre-training and consistent with the worry that, absent that pre-training, a student would simply *surrender multilingual parameters to English*.

### An honest methodological note
The zero-shot benchmarks measure *language-modelling likelihood*, not *instruction-following generation*, so they can't fully answer the research question on their own. The paper lays out the better path: evaluate generation directly (e.g. **Super-Natural Instructions** with ROUGE-L) to compare Spanish- vs. English-calibrated students head-to-head. This limitation, and the routes around it, are discussed in detail in the paper.

---

## Repository structure

This repo is a composite of the several repositories used during the project.

| Path | What's inside |
|------|---------------|
| [`minillm/`](./minillm) | The distillation code. Our extension of MiniLLM, including **BLOOM support**. |
| [`bloom_jobs/`](./bloom_jobs) | HPC job scripts to download/process data and run BLOOM distillations (Spanish / English / both). |
| [`llama_jobs/`](./llama_jobs) | HPC job scripts for LLaMA conversion, data processing, and distillations. |
| [`lmeval/`](./lmeval) | Zero-shot evaluation via the LM Evaluation Harness, plus result CSVs (`bloom_results.csv`, `llama_results.csv`) and analysis scripts. |
| [`Tk-Instruct/`](./Tk-Instruct) | Setup for instruction-following (Super-Natural Instructions style) evaluation. |
| `download_*.py` | Scripts to fetch and preprocess the OpenAssistant and Wikipedia datasets (English-only, Spanish-only, and combined variants). |

### Running it
All job scripts are written for the **LSF 10** HPC scheduler and assume a **single H100 GPU with a 32-core CPU and 5 GB RAM per core**. They are straightforward to port to schedulers like Slurm.

- **Distillation:** see `bloom_jobs/` and `llama_jobs/`. The `download_and_process_*` / `process_*` jobs show exactly how to invoke the dataset scripts; the `distill_*` jobs run the distillations themselves.
- **Evaluation:** see the job folders and `README` files inside `lmeval/` and `Tk-Instruct/`, which document how to fetch the per-task data and reproduce the experiments in the paper.
