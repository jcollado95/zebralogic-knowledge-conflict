# ZebraLogic variants v3: true vs shuffled facts with MCQA-defined pools

**Puzzles:** `zebralogic_pools_v3.parquet`, 720 = 120 ZebraLogic templates (40 each of 4×4, 5×4, 6×4) × 3 pools (`only_large` = only gemma-4 31B knows, `both`, `none`; strict MCQA rule from `pools_v3_gemma4.csv`) × 2 conditions (`aligned` = true facts, `conflict` = shuffled facts). 1 draw of people per template and pool; `pool_variants_v3.py build_all(..., draws=k)` gives more. All 720 pass the round-trip check (re-parsed and solved, exactly one solution, equal to the gold).

**Checks:** true-fact golds are 100% true facts and shuffled-fact golds 0%. The header value lists are identical in both conditions (360/360). The v2 constraints apply: distinct labels, no substrings, a shown value is never true of another person in the puzzle (birthplace via admin parents and aliases, occupations not word-related), and birth years at least 3 apart.

**People used:** only_large 48 of 48 (max 31 uses), both 344 of 560 (max 5), none 203 of 222 (max 8). Median sitelinks of drawn people: 20 / 112 / 5.

## In-puzzle MCQA (`inpuzzle_mcqa_<model>.jsonl`)
For every person × fact in each puzzle, both models answered one brief-recall question whose options were that puzzle's own values for the column, in header order (4–6 options; chance 17–25%). 5,400 questions per model, 0 errors.

| Pool | 31B correct | E4B correct |
|---|---|---|
| both | 98.1% | 91.7% |
| only_large | 83.8% | 39.9% |
| none | 39.7% | 30.8% |

By fact, birthplace is recognised well above chance even in `none` (31B 67%, E4B 55%), against 23–29% for occupation and birth year. A puzzle's birthplaces usually come from different countries, so names give them away: a Japanese name picks the Japanese city. The pool-defining probe avoided this by using same-country distractors. In-puzzle flags therefore measure *recognition within the puzzle*, the knowledge the model can actually use while solving, and they include this name-based plausibility.

Files: `zebralogic_pools_v3.parquet`, `inpuzzle_mcqa_gemma-4-E4B-it.jsonl`, `inpuzzle_mcqa_gemma-4-31B-it.jsonl`, `inpuzzle_mcqa_summary.csv`, `variants_v3_validation.csv`, `pool_variants_v3.py` (with `pool_variants_v2.py`, `pool_variants.py`, `zebra_perturb.py`, `mcqa_probe.py`).
