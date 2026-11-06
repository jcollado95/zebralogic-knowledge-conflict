# Knowledge conflict in closed-world deduction: ZebraLogic with real people

When a logic-grid puzzle is about real, well-known people, does a language model's world knowledge interfere with closed-world deduction? And does the effect depend on model size?

Each ZebraLogic puzzle is rebuilt around real Wikidata people and solved in two conditions:

- **True facts:** the gold solution assigns every person their real birthplace, occupation and birth year.
- **Shuffled facts:** the same people, values and clues, with every attribute reassigned to a different person.

The logic and the clue wording are identical in both conditions. People are grouped into knowledge pools by a per-model multiple-choice probe, so each conflict effect can be tied to what the model actually knows.

Models: `gemma-4-E4B-it` (small) and `gemma-4-31B-it` (large).

## Main result

From [docs/report.md](docs/report.md): 120 ZebraLogic templates (40 each of 4×4, 5×4, 6×4) × 3 knowledge pools × 2 conditions, which is 720 puzzles per model. Temperature 0, a 32,768-token output budget, SSE streaming.

| Model | Pool | True facts | Shuffled | Gap (pp) | Holm p |
|---|---|---|---|---|---|
| 31B | both know | 79.2% | 55.0% | 24.2 | <0.001 |
| 31B | only 31B knows | 69.2% | 57.5% | 11.7 | 0.013 |
| 31B | neither knows | 64.2% | 68.3% | −4.2 | 0.42 |
| E4B | both know | 35.0% | 35.8% | −0.8 | 1.0 |
| E4B | only 31B knows | 36.7% | 33.3% | 3.3 | 1.0 |
| E4B | neither knows | 40.8% | 38.3% | 2.5 | 1.0 |

- **31B:** the cost follows its knowledge.
  - Diff-in-diff against the "neither knows" pool: 28.3 pp [16.7, 40.8] for "both know" and 15.8 pp [5.0, 26.7] for "only 31B knows".
  - The cell-level gap is 14.2 pp [8.1, 20.6] larger on facts the model recognises.
  - Conflicting facts also lengthen its search; in "both know", the share of answers that exhaust the 32k budget rises from 10% to 29%.
- **E4B:** no knowledge-specific cost (diff-in-diff −3.3 and 0.8 pp). It never reaches the budget.

![Main results](results/persons_v3_32k_results.png)

**Puzzle size** ([docs/difficulty.md](docs/difficulty.md)):

- **Small-vs-large gap:** without knowledge, E4B collapses between 4 and 5 houses and 31B between 5 and 6. The gap between the models peaks at 5×4: 78.8% vs 32.5% puzzles solved.
- **31B's conflict cost:** roughly constant across sizes, about 28–30 pp diff-in-diff for well-known people.
- **What the cost consists of:**
  - 4×4 and 5×4: interference, with accuracy falling below the no-knowledge baseline;
  - 6×4: loss of a knowledge shortcut.

![Difficulty effects](results/difficulty/difficulty_effects.png)

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Fame-stratified Wikidata candidates (4,515 people) | `src/fetch_candidates.py` | `data/candidates_v3.csv` ([method](docs/methods/candidates.md)) |
| 2. MCQA knowledge probe, 3 facts per person and model | `src/run_experiment.py probe` (`src/mcqa_probe.py`) | `runs/probe_mcqa_<model>.jsonl` ([method](docs/methods/probe.md)) |
| 3. Knowledge pools: both / only large / none (strict rule) | `src/build_pools.py` | `data/pools_v3_gemma4.csv` ([method](docs/methods/pools.md)) |
| 4. True vs shuffled puzzles, round-trip validated | `src/puzzle_variants.py` (`src/zebra_core.py`) | `data/zebralogic_pools_v3.parquet` ([method](docs/methods/variants.md)) |
| 5. Puzzle evaluation, both models (32,768-token budget, streaming) | `src/run_experiment.py eval` (`src/evaluate.py`) | `runs/persons_v3_32k_full_<model>.jsonl` |
| 6. In-puzzle MCQA (recognition of each shown fact) | `src/run_experiment.py inpuzzle` | `runs/inpuzzle_mcqa_<model>.jsonl` |
| 7. Statistics | `src/analyze.py` | `results/*.csv` |
| 8. Effect of puzzle size | `src/analyze_difficulty.py` | `results/difficulty/*.csv` ([report](docs/difficulty.md)) |
| 9. Figures | `src/plots.py` | `results/*.png`, `results/difficulty/*.png` |

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
python analyze_difficulty.py
python plots.py
```

All model calls are resumable: outputs are appended as JSONL, and items already done are skipped. Transport errors are retried once.

**Rerun the full set.** At temperature 0, outputs on the vLLM endpoint depend on server state (prefix cache, batching). Rerun the full set rather than only failed or truncated items, so that both conditions of a comparison come from the same run.

**Reproducibility.** With the files in `data/` and the raw model outputs, the code regenerates the committed artefacts exactly:

- the 720 puzzles (`zebralogic_pools_v3.parquet`);
- the pools (`pools_v3_gemma4.csv`);
- every table in `results/` and `results/difficulty/`.

## Repository layout

```
src/        pipeline code (scripts import each other by module name; run them from src/)
data/       inputs: ZebraLogic grid file, candidate population, knowledge pools, the 720 puzzles
results/    analysis tables and figures (difficulty/: effect of puzzle size)
docs/       report.md (main results), difficulty.md (puzzle size), methods/ (candidates, probe, pools, variants)
runs/       raw model outputs (not versioned; see data/README.md)
```

## Data sources

- **ZebraLogic:** Hugging Face `WildEval/ZebraLogic`, `grid_mode/test-00000-of-00001.parquet`.
- **Wikidata:** via the QLever endpoint (`https://qlever.dev/api/wikidata`), snapshot of 2026-11-04 (CC0). Properties used:
  - P19 place of birth;
  - P106 occupation;
  - P569 date of birth (birth year).
