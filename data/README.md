# Data

| Folder | Contents |
|---|---|
| `source/` | ZebraLogic grid-mode file (`zebra_grid.parquet`, WildEval/ZebraLogic), Wikidata relation catalog, frozen KG snapshot, early hand-written seed lists (`kg_seed.json`, superseded by Wikidata) |
| `persons/` | Wikidata person candidates (v1 `candidates_all.csv`, v3 `candidates_v3.csv`), closed-book and MCQA knowledge-probe scores, LLM-judge verdicts, knowledge-stratified pools (`persons_pools.csv`, `pools_v3_gemma4.csv`) and question banks |
| `puzzles/` | Generated variants: templates, bias perturbations, Wikidata variants, real-person variants and pool variants v1–v3, each validated by round-trip solving |

People data comes from Wikidata (CC0). Only public, non-sensitive attributes are used (birthplace, occupation, birth year; earlier versions also used mother, school and height).

## Excluded raw model outputs

Raw generation logs (full reasoning traces) are kept out of git because of their size (69 MB in total) and because they are raw model output. They are stored in the project workspace and can be shared on request, or regenerated with the listed script (greedy decoding, so runs are near-deterministic on the same vLLM build).

| File | MB | Produced by |
|---|---|---|
| `zebralogic_perturbed.jsonl` | 5.3 | generate_variants.py (same content as data/puzzles/zebralogic_perturbed.parquet) |
| `pilot_gemma4e4b.jsonl` | 22.3 | run_eval.py on zebralogic_perturbed + zebralogic_wikidata (E4B pilot) |
| `persons_gemma4e4b.jsonl` | 4.8 | run_eval.py on zebralogic_persons.parquet (E4B) |
| `persons_gemma4_31b.jsonl` | 2.0 | run_eval.py on zebralogic_persons.parquet (31B) |
| `probe_both_scored.jsonl` | 3.2 | probe_pool.py (closed-book probe, both models) |
| `probe_extra_scored.jsonl` | 2.8 | extra-fact probe (scripts/recovered) |
| `probe_mcqa_gemma-4-E4B-it.jsonl` | 5.0 | mcqa_probe.py |
| `probe_mcqa_gemma-4-31B-it.jsonl` | 5.0 | mcqa_probe.py |
| `inpuzzle_mcqa_gemma-4-E4B-it.jsonl` | 1.7 | pool_variants_v3.py / in-puzzle MCQA |
| `inpuzzle_mcqa_gemma-4-31B-it.jsonl` | 1.7 | pool_variants_v3.py / in-puzzle MCQA |
| `persons_v3_gemma-4-E4B-it.jsonl` | 10.2 | run_eval.py on zebralogic_pools_v3.parquet (scripts/recovered) |
| `persons_v3_gemma-4-31B-it.jsonl` | 5.5 | run_eval.py on zebralogic_pools_v3.parquet (scripts/recovered) |

The scored, per-puzzle tables derived from them (`*_scored.csv`, `*_cells.csv`) **are** included under `results/`.
