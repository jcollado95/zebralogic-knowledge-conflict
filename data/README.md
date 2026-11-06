# Data

| File | Contents |
|---|---|
| `zebra_grid.parquet` | ZebraLogic grid-mode puzzles with gold solutions (WildEval/ZebraLogic); the 4×4, 5×4 and 6×4 templates are used |
| `candidates_v3.csv` | 4,515 fame-stratified Wikidata people with displayed and accepted values for birthplace, occupation and birth year ([method](../docs/methods/candidates.md)) |
| `pools_v3_gemma4.csv` | per-person MCQA knowledge flags for both models and the pool assignment (`pool_final`) ([method](../docs/methods/pools.md)) |
| `zebralogic_pools_v3.parquet` | the 720 true/shuffled-fact puzzles with gold solutions, validity flag and per-cell knowledge metadata ([method](../docs/methods/variants.md)) |

The MCQA question bank (13,545 questions) is not stored. `mcqa_probe.build_questions` rebuilds it deterministically from `candidates_v3.csv`.

People data comes from Wikidata (CC0). Only public, non-sensitive attributes are used.

## Raw model outputs

The raw generation logs, which contain full reasoning traces, are not versioned in git. `run_experiment.py` writes them to `runs/`, and `analyze.py` reads them from there. Decoding is at temperature 0, but outputs on the endpoint depend on server state (prefix cache, batching), so reruns are close but not identical. The copies used for the committed results can be shared on request.

| File | MB | Produced by |
|---|---|---|
| `probe_mcqa_gemma-4-E4B-it.jsonl` | 5.0 | `run_experiment.py probe` |
| `probe_mcqa_gemma-4-31B-it.jsonl` | 5.0 | `run_experiment.py probe` |
| `persons_v3_32k_full_gemma-4-E4B-it.jsonl` | 11.2 | `run_experiment.py eval` |
| `persons_v3_32k_full_gemma-4-31B-it.jsonl` | 14.5 | `run_experiment.py eval` |
| `inpuzzle_mcqa_gemma-4-E4B-it.jsonl` | 1.7 | `run_experiment.py inpuzzle` |
| `inpuzzle_mcqa_gemma-4-31B-it.jsonl` | 1.7 | `run_experiment.py inpuzzle` |
