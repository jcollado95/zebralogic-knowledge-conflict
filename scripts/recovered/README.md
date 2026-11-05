# Recovered notebook code

Code for steps that were run interactively in analysis sessions and never saved as standalone scripts. Each file was reconstructed from the provenance lineage of the output it produced and is kept for **provenance and reproducibility notes**. It is not packaged as a CLI. Expect to adjust input paths and set `WORKDIR` before rerunning. Input paths were rewritten relative to the repository root (`src/…`, `data/…`); `raw/…` stands for an excluded raw model output (see [data/README.md](../../data/README.md)).

The E4B pilot eval loop and pilot figure are not included because their lineage copies are truncated. `src/run_eval.py` and `src/analyze_pilot.py` cover the pilot.

| Script | Produced | Location in repo |
|---|---|---|
| `build_pools_v3_gemma4.py` | `pools_v3_gemma4.csv` | data/persons/pools_v3_gemma4.csv |
| `download_zebra_grid.py` | `zebra_grid.parquet` | data/source/zebra_grid.parquet |
| `fetch_candidates_v3_qlever.py` | `candidates_v3.csv` | data/persons/candidates_v3.csv |
| `fetch_facts_extra.py` | `facts_extra.csv` | data/persons/facts_extra.csv |
| `llm_judge_verdicts.py` | `judge_verdicts.csv` | data/persons/judge_verdicts.csv |
| `persons_conflict_cells.py` | `persons_conflict_cells.csv` | results/02_persons_e4b/persons_conflict_cells.csv |
| `plot_candidates_v3_sitelinks.py` | `candidates_v3_sitelinks.png` | results/04_mcqa_probe_v3/candidates_v3_sitelinks.png |
| `plot_mcqa_knowledge_vs_fame.py` | `mcqa_knowledge_vs_fame.png` | results/04_mcqa_probe_v3/mcqa_knowledge_vs_fame.png |
| `plot_persons_e4b_results.py` | `persons_gemma4e4b_results.png` | results/02_persons_e4b/persons_gemma4e4b_results.png |
| `plot_persons_two_models.py` | `persons_two_models.png` | results/03_persons_two_models/persons_two_models.png |
| `plot_pools_v3_by_fame.py` | `pools_v3_by_fame.png` | results/04_mcqa_probe_v3/pools_v3_by_fame.png |
| `plot_v3_results.py` | `persons_v3_results.png` | results/05_persons_v3/persons_v3_results.png |
| `probe_birthdate.py` | `probe_birthdate.csv` | data/persons/probe_birthdate.csv |
| `probe_extra_facts.py` | `probe_extra_scored.jsonl` | data/README.md (excluded raw output) |
| `run_eval_persons_31b.py` | `persons_gemma4_31b.jsonl` | data/README.md (excluded raw output) |
| `run_eval_v3.py` | `persons_v3_gemma-4-31B-it.jsonl` | data/README.md (excluded raw output) |
| `run_mcqa_probe_and_summarize.py` | `mcqa_summary.csv` | results/04_mcqa_probe_v3/mcqa_summary.csv |
| `score_fact_knowledge_by_pool.py` | `fact_knowledge_by_pool.csv` | data/persons/fact_knowledge_by_pool.csv |
| `summarize_inpuzzle_mcqa.py` | `inpuzzle_mcqa_summary.csv` | results/04_mcqa_probe_v3/inpuzzle_mcqa_summary.csv |
| `v3_facilitation_interference.py` | `v3_facilitation_interference.csv` | results/05_persons_v3/v3_facilitation_interference.csv |
| `validate_variants_v3.py` | `variants_v3_validation.csv` | data/puzzles/variants_v3_validation.csv |
