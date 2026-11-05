# Knowledge conflict in closed-world deduction: ZebraLogic with real people

When a logic-grid puzzle is about real, well-known people, does a language model's world knowledge interfere with closed-world deduction? And does the effect depend on model size?

Each ZebraLogic puzzle is rebuilt around real Wikidata people and solved in two conditions:

- **True facts:** the gold solution assigns every person their real birthplace, occupation and birth year.
- **Shuffled facts:** the same people, values and clues, with every attribute reassigned to a different person.

The logic and the clue wording are identical in both conditions. People are grouped into knowledge pools by a per-model multiple-choice probe, so each conflict effect can be tied to what the model actually knows.

Models: `gemma-4-E4B-it` (small) and `gemma-4-31B-it` (large).

## Main result

From [docs/report.md](docs/report.md): 120 ZebraLogic templates (40 each of 4×4, 5×4, 6×4) × 3 knowledge pools × 2 conditions, which is 720 puzzles per model. Greedy decoding, 8,192-token cap.

| Model | Pool | True facts | Shuffled | Gap (pp) | Holm p |
|---|---|---|---|---|---|
| 31B | both know | 85.0% | 65.0% | 20.0 | <0.001 |
| 31B | only 31B knows | 70.8% | 67.5% | 3.3 | 1.0 |
| 31B | neither knows | 70.8% | 73.3% | −2.5 | 1.0 |
| E4B | both know | 48.3% | 36.7% | 11.7 | 0.060 |
| E4B | only 31B knows | 43.3% | 40.8% | 2.5 | 0.86 |
| E4B | neither knows | 44.2% | 39.2% | 5.0 | 0.86 |

- **31B:** the cost is specific to knowledge.
  - Diff-in-diff against the "neither knows" pool: 22.5 pp [10.8, 35.0].
  - Cell-level gap: 13.5 pp [7.1, 19.8] larger on facts the model recognises.
  - Conflicting facts also lengthen its search; the share of answers that hit the token limit rises from 8% to 24%.
- **E4B:** the drop is not knowledge-specific (diff-in-diff 6.7 pp, p = 0.36).

![Main results](results/persons_v3_results.png)

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Fame-stratified Wikidata candidates (4,515 people) | `src/fetch_candidates.py` | `data/candidates_v3.csv` ([method](docs/methods/candidates.md)) |
| 2. MCQA knowledge probe, 3 facts per person and model | `src/run_experiment.py probe` (`src/mcqa_probe.py`) | `runs/probe_mcqa_<model>.jsonl` ([method](docs/methods/probe.md)) |
| 3. Knowledge pools: both / only large / none (strict rule) | `src/build_pools.py` | `data/pools_v3_gemma4.csv` ([method](docs/methods/pools.md)) |
| 4. True vs shuffled puzzles, round-trip validated | `src/puzzle_variants.py` (`src/zebra_core.py`) | `data/zebralogic_pools_v3.parquet` ([method](docs/methods/variants.md)) |
| 5. Puzzle evaluation, both models | `src/run_experiment.py eval` (`src/evaluate.py`) | `runs/persons_v3_<model>.jsonl` |
| 6. In-puzzle MCQA (recognition of each shown fact) | `src/run_experiment.py inpuzzle` | `runs/inpuzzle_mcqa_<model>.jsonl` |
| 7. Statistics | `src/analyze.py` | `results/*.csv` |
| 8. Figures | `src/plots.py` | `results/*.png` |

```bash
conda env create -f environment.yml && conda activate zebralogic     # or: pip install -r requirements.txt
cd src
export VLLM_API_KEY=...                                                # endpoint and models: config.json (VLLM_URL overrides)
python fetch_candidates.py ../data/candidates_v3.csv                   # optional: the snapshot is already in data/
python run_experiment.py probe
python build_pools.py ../data/candidates_v3.csv ../runs gemma-4-E4B-it gemma-4-31B-it gemma4 ../data
python puzzle_variants.py ../data/zebra_grid.parquet ../data/candidates_v3.csv ../data/pools_v3_gemma4.csv ../data
python run_experiment.py eval
python run_experiment.py inpuzzle
python analyze.py
python plots.py
```

All model calls are resumable: outputs are appended as JSONL, and items already done are skipped. Transport errors are retried once.

**Reproducibility.** With the files in `data/` and the raw model outputs, the code regenerates the committed artefacts exactly:

- the 720 puzzles (`zebralogic_pools_v3.parquet`);
- the pools (`pools_v3_gemma4.csv`);
- every table in `results/`.

## Repository layout

```
src/        pipeline code (scripts import each other by module name; run them from src/)
data/       inputs: ZebraLogic grid file, candidate population, knowledge pools, the 720 puzzles
results/    analysis tables and figures
docs/       report.md (full results and discussion) and methods/ (candidates, probe, pools, variants)
runs/       raw model outputs (not versioned; see data/README.md)
```

## Data sources

- **ZebraLogic:** Hugging Face `WildEval/ZebraLogic`, `grid_mode/test-00000-of-00001.parquet`.
- **Wikidata:** via the QLever endpoint (`https://qlever.dev/api/wikidata`), snapshot of 2026-11-04 (CC0). Properties used:
  - P19 place of birth;
  - P106 occupation;
  - P569 date of birth (birth year).
