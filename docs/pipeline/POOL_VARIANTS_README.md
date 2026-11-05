# True-fact vs shuffled-fact ZebraLogic variants from the knowledge pools

**Puzzles:** 720 = 120 ZebraLogic templates (40 each of 4×4, 4×5, 5×4) × 3 pools (`only_31b`, `both`, `none`) × 2 conditions. All 720 pass the round-trip check: the rendered text is re-parsed and the CSP solver finds exactly one solution, equal to the gold.

| Condition | Gold grid |
|---|---|
| `aligned` | every real person holds their own Wikidata facts (100% of gold cells are true facts) |
| `conflict` | every attribute is deranged, so no person holds a value that is true of them (0% true) |

The fictional-name control is dropped. The `none` pool is now the knowledge-free baseline: for those people, `aligned` and `conflict` should look identical to both models.

## Design
- **Relations:** BirthPlace, Occupation and Height for 3-attribute templates, plus EducatedAt for 4×5. These are the facts the models know best; mother is rarely known even in `both`. The template attribute that receives each relation is chosen per template (seeded) and is the same for all pools.
- **People:** *n* people per template and pool are drawn from `persons_pools.csv`, seeded by template id and pool. Display values are the `(salient)` values (named by the model(s) where known). 2 people whose displayed birthplace is a memorial or hospital with no usable municipality were excluded (B. R. Ambedkar, Valentýna Vojtková); 485 of 487 remain.
- **Known pools:** every drawn person is known by the pool's model(s) on at least 2 of the puzzle's relations.
- **Within-puzzle constraints:**
  - all labels distinct; no label a substring of another (names included); no ' and ' / ' is ' / trailing '.';
  - **cross-truth:** no shown value may be true of another drawn person. This makes the shuffled facts genuine contradictions and the true facts unambiguous:
    - birthplace is compared against accepted names, including administrative parents;
    - occupations may not be substring-related (actor / film actor);
    - heights must be more than 3 cm apart from every other person's true heights.
- **Shuffling:** each attribute gets its own derangement, chosen to minimise *cross-attribute coincidences*, i.e. two values in one house belonging to the same other person. There are 0 in 240 conflict puzzles. In 120 conflict puzzles (4×5 templates: 4 houses, 4 attributes) there are 4, which cannot be avoided.
- **Identical surface form:** both conditions use the same people, the same header value lists in the same order, and the same clue wording and order. Only the house assignment of values differs.

## Knowledge coverage of the shown facts (true-fact puzzles)

| Pool | E4B knows the person's fact | 31B knows it | E4B named the shown value | 31B named the shown value | Distinct people | Max uses | Median sitelinks |
|---|---|---|---|---|---|---|---|
| `only_31b` | 0.0% | 70.3% | 0.0% | 67.7% | 68 | 27 | 20 |
| `both` | 71.6% | 78.2% | 61.5% | 70.1% | 134 | 18 | 104 |
| `none` | 0.0% | 0.0% | 0.0% | 0.0% | 190 | 9 | 6 |

By relation (share of cells whose fact the model knows):

| Relation | `only_31b` E4B | `only_31b` 31B | `both` E4B | `both` 31B |
|---|---|---|---|---|
| BirthPlace | 0.0% | 86.7% | 94.4% | 98.3% |
| Occupation | 0.0% | 80.2% | 100.0% | 99.2% |
| Height | 0.0% | 53.8% | 37.9% | 52.1% |
| EducatedAt | 0.0% | 38.8% | 14.4% | 29.4% |

## Files
- `zebralogic_pools.parquet`: one row per puzzle. Columns: `id` (`<template>|<pool>|<condition>`), `base_id`, `size`, `pool`, `condition`, `puzzle`, `solution` (`{header, rows}`), `valid`, `meta` (relations, people, QIDs, sitelinks, per-cell knowledge, cross-attribute coincidences).
- `pool_variants_cells.csv`: per person × relation in each puzzle, whether each model knows that person's fact (`knows_*`) and whether the shown value is the one it named (`shown_known_*`). Use this for knowledge-conditioned analyses.
- `pool_variants.py`: the generator (`python pool_variants.py zebralogic_perturbed.parquet persons_pools.csv out`). It needs `zebra_perturb.py` alongside.

## Running the evaluation
Prompts are built as in earlier runs:

    run_eval.build_prompt(row.puzzle, sol['header'], len(sol['rows']))

followed by `run_eval.run(items, model, key, out_path, workers=24)`. That is 720 prompts per model, roughly 2 h each at the throughput seen on nodo08.

## Caveats
- **Fame differs by pool** (median sitelinks above). Compare aligned vs conflict *within* a pool. Read the pool × condition interaction, not raw accuracy across pools.
- **Uneven reuse in `only_31b`:** with only 68 usable people, some appear in up to 27 of 120 templates. Consider a random effect for person in the analysis.
- **Knowledge flags measure free recall,** and occupation and birthplace partly rest on a reviewed LLM judge (see `POOLS_README.md`).
