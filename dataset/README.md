# Dataset and exact selection

This repository uses the publicly available **SII-YDD/Orchid**, subset **Orchid-HEval**.

- Source: https://huggingface.co/datasets/SII-YDD/Orchid
- Upstream raw file: `Orchid-HEval/data.jsonl`
- License on upstream dataset card: **Apache-2.0** (see `../licenses/Orchid_Apache-2.0_LICENSE.txt`). Credit original authors.
- Orchid research: Di Yang et al., *Clarity Is Not Assumed: Understanding LLM-Based Code Generation under Ambiguous Requirements*, ASE 2026, DOI 10.1145/3832783.3834403.

## Available files in this GitHub package

- `manifest.csv`: all 250 case IDs, source-task IDs, and reference labels. **Included.**
- `selected_30_cases.csv`: the exact input text of all 30 CEV pilot cases. **Included.**
- `orchid_new250_cases.csv`: the packaged 250-case evaluation dataset. **Included.**

The bundled evaluation CSV has SHA-256 digest:
`73c806e7b6d3587df00326cb43218fbb13a4334fb68ec51db0326adabb450e46`.
This digest identifies the bundled copy only. The exact upstream source revision from the original Colab run was not preserved.

## Selection rule (important)

Select 50 of the 164 Orchid-HEval source task IDs, excluding the 25 source IDs in the original 125-case development project. Use `random.Random(42).sample()` on the remaining 139 IDs sorted by numeric HumanEval ID. For each selected task, use one `prompt` (Clear) and four fields `Lexical_prompt`, `Semantic_prompt`, `Syntactic_prompt`, `Vagueness_prompt` in the original notebook order. Sort the 250 cases by task numeric ID and then Clear, Lexical, Semantic, Syntactic, Vagueness. Assign `case_id` from 1 to 250.

The rebuild script validates exact case IDs and source tasks against `manifest.csv`. It also checks all 30 audited requirement texts match the exported CEV file. A public repository may have newer data revisions. If checks fail, do **not** silently relabel or substitute requirements. Obtain the original Colab `orchid_new250_cases.csv` and `protocol.json` from the run's Google Drive folder.

The exact upstream data revision and the original Colab-generated SHA-256 were not included in the supplied exports. The checksum reported above was computed from the bundled copy. `protocol.json` states this limitation rather than inventing those values.
