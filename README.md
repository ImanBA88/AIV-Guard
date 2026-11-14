# AIV-Guard

AIV-Guard is a single-agent workflow for software-requirements ambiguity classification, textual evidence checking, and exploratory Counterfactual Evidence Validation (CEV). It uses Qwen2.5-3B-Instruct for classification and fixed rules to inspect JSON structure and evidence quotations. The CEV extension proposes two candidate refinements and a clarification question for human review. Its structural pass result is **not semantic validation**.

## Repository contents

| Path | Description |
| --- | --- |
| `dataset/orchid_new250_cases.csv` | Packaged 250-case Orchid-HEval evaluation dataset. |
| `dataset/manifest.csv` | Case IDs, source-task IDs, and reference labels for all 250 cases. |
| `dataset/selected_30_cases.csv` | Exact 30 input texts selected for the post-hoc CEV pilot. |
| `notebooks/01_SingleAgent_Test250_Final.ipynb` | Qwen classifier, deterministic evidence gate, checkpoints, and evaluation metrics. |
| `notebooks/02_CEV_Review30_Final.ipynb` | CEV candidate generation, structural checks, and review-form export. |
| `results/predictions_test250.csv` | Original saved classifier predictions and decision traces. |
| `results/cev_predictions_30.csv` | Original saved CEV candidate pairs, questions, and structural-gate outcomes. |
| `results/classification_summary_250.json` | Saved classification summary. |
| `results/cev_structural_summary.json` | Saved CEV structural summary. |
| `results/human_review_30_BLANK.csv` | Reviewer worksheet without gold labels or classifier decisions. No human judgments are recorded. |
| `scripts/rebuild_dataset.py` | Offline verification of the packaged dataset, or optional source reconstruction. |
| `scripts/verify_results.py` | Offline consistency checks for saved classification and CEV results. |
| `protocol.json` | Reconstructed protocol information for this submission; not the original Colab-generated protocol. |

## Recorded results

The reported experiment uses 250 requirement cases from 50 source programming tasks. Each task has one clear case and four ambiguity-modified variants (lexical, semantic, syntactic, vagueness).

| Measurement | Saved result |
| --- | ---: |
| Agent 1 accuracy | 171/250 = 68.4% |
| Macro-F1 | 62.1% |
| Clear recall | 32/50 = 64.0% |
| Ambiguous recall | 139/200 = 69.5% |
| Gate acceptance | 226/250 = 90.4% |
| Gate referrals | 24/250 = 9.6% |
| Accepted-only accuracy | 157/226 = 69.5% |
| CEV structural passes | 27/30 = 90.0% |
| Completed independent semantic reviews | 0/30 |

These are **saved experimental results**. Checking the CSV files does not re-run Qwen. An always-AMBIGUOUS analytical baseline would achieve 80.0% accuracy on this imbalanced test set, but 44.4% macro-F1. The gate's accepted-only accuracy is conditional on its coverage and is not a direct replacement for full-set accuracy. The CEV pilot was selected after classification from accepted AMBIGUOUS decisions.

## Verify the packaged files offline (no GPU or network)

1. Extract the ZIP archive and open a terminal **in the `AIV-Guard` folder** (the folder containing `dataset`, `results`, and `scripts`).
2. Run the following commands:

Windows (Command Prompt):

```bat
py scripts\rebuild_dataset.py --verify-existing
py scripts\verify_results.py
```

Linux, macOS, or an environment where `python` is the command:

```bash
python scripts/rebuild_dataset.py --verify-existing
python scripts/verify_results.py
```

The first command checks dataset structure, 50 source tasks, reference-label distribution, saved manifest, and CEV text matches. It prints a SHA-256 digest. The second command checks saved predictions, deterministic-gate decisions, CEV sample consistency, and the recorded result totals. Both use only the Python standard library. They do not download data, change saved predictions, or load Qwen.

Expected headline checks include 250 cases, 50 tasks, 171 correct classifier predictions, 226 accepted gate decisions with 157 correct, and 27 of 30 CEV structural passes. A successful offline check supports **internal consistency**, not human semantic correctness or an independent rerun of model inference.

## Notebook execution and optional reproduction

To run model inference again, use a CUDA-capable environment, such as a Google Colab GPU runtime, with the dependencies in `requirements.txt` and PyTorch available. Extract the **complete repository** into the runtime, for example at `/content/AIV-Guard`, so the notebooks find the packaged `dataset/` and `results/` files. Uploading the notebook alone does not supply the local project files.

- Open `notebooks/01_SingleAgent_Test250_Final.ipynb` for classification. Its default output location is the project's `outputs/` directory unless Google Drive is mounted/configured. Running its classification cell performs new Qwen inference, potentially 250 calls, except for cases already present in its own checkpoint.
- Open `notebooks/02_CEV_Review30_Final.ipynb` for the selected 30-case extension. It reads the packaged dataset and saved classifier results. When its saved CEV outputs are present and consistent, it inspects those results without loading Qwen. Missing outputs would require fresh inference with a CUDA GPU.
- Google Drive storage is optional. The classification notebook mounts Drive only if `AIV_GUARD_MOUNT_DRIVE=1` is set in the runtime before its setup cell. The environment variable `AIV_GUARD_PROJECT_ROOT` is supported by the CEV notebook when the repository location needs to be specified.
- Do not add credentials, private paths, or access tokens to committed repository files.

For optional reconstruction from the original upstream JSONL, run:

```bash
python -m pip install huggingface_hub
python scripts/rebuild_dataset.py --source path/to/Orchid-HEval/data.jsonl
```

Or supply `--revision` for a remote Hugging Face revision. Reconstruction checks the selected case IDs and all 30 recorded CEV input texts. It refuses to overwrite the packaged CSV when the contents differ. **Reconstruction is not needed to verify this submission.**

## Dataset provenance and limitations

The evaluation data originate from [`SII-YDD/Orchid`](https://huggingface.co/datasets/SII-YDD/Orchid), `Orchid-HEval/data.jsonl`. The complete selected 250-case CSV is included in this repository. The SHA-256 digest of this **packaged copy** is:

```text
73c806e7b6d3587df00326cb43218fbb13a4334fb68ec51db0326adabb450e46
```

The full original upstream JSONL, its exact revision during the original Colab run, and the run-generated original source hash were not supplied. The digest above identifies the bundled CSV; it does **not** independently certify the source revision. See `dataset/README.md` for the deterministic source-task selection procedure.

The evaluation has 50 original tasks with five correlated variants per task, and an 80% AMBIGUOUS class prevalence. Some variants can repeat wording across different reference labels. Literal evidence matching does not prove the evidence is ambiguous. CEV structural checks do not establish two defensible behavioral interpretations, executable divergence, or stakeholder intent. No independent human semantic assessment or downstream code-correctness evaluation has been completed. Numerical comparisons with Orchid, ClarifyGPT, or ReqTestAgent need matched datasets and evaluation metrics.

## Attribution and license

The Orchid dataset is attributed to its authors and is marked Apache-2.0 by its source dataset card. The included copy of the dataset license is at `licenses/Orchid_Apache-2.0_LICENSE.txt`. The repository's own project code is submitted as academic coursework; no additional license is asserted for the student's own code.

Yang, D., et al. (2026). *Clarity Is Not Assumed: Understanding LLM-Based Code Generation under Ambiguous Requirements.* ASE 2026. https://doi.org/10.1145/3832783.3834403
