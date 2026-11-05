# True-fact vs shuffled-fact ZebraLogic variants, v2 (birthplace, occupation, birth year)

**720 puzzles** = 120 templates (40 each of **4×4, 5×4, 6×4**) × 3 pools (`only_31b`, `both`, `none`) × 2 conditions (`aligned` = true facts, `conflict` = shuffled facts). All 720 pass the round-trip check: the rendered text is re-parsed and the CSP solver finds exactly one solution, equal to the gold.

## Changes from v1 (`zebralogic_pools.parquet`)
- **Facts:** every puzzle uses only **BirthPlace, Occupation and BirthYear**. Every pooled person has all three in Wikidata, and the values can be made distinct within a puzzle. Height, school and mother, which were weakly known, are gone, and spouse and death date were rejected because many people don't have them.
- **Templates:** the 4×5 templates (4 attributes) are replaced by the 40 **6×4** templates from the full WildEval/ZebraLogic `grid_mode` file (`zebra_grid.parquet`), so every template has exactly 3 non-Name attributes.
- **Eligibility is now defined on the three puzzle facts** (per model; birth year from the exact-year probe in `probe_extra_scored.jsonl`):
  - `only_31b`: 31B knows ≥ 2 of the 3 facts, E4B knows none → 50 eligible people;
  - `both`: both models know ≥ 2 of the 3 → 180 eligible;
  - `none`: neither model knows any of the 3 (on top of the pool's 0/5 on the original facts) → 215 eligible.
  - People with no birth year or conflicting Wikidata birth years are excluded, as are the 2 people with non-settlement birthplaces.
- **Clue phrase for the year:** "the person who was born in the year {v}", distinct from the birthplace phrase "the person who was born in {v}".
- **Within a puzzle, birth years are at least 3 years apart**, so a shuffled year is a real contradiction. The other v1 constraints are kept: distinct labels, no substrings, and cross-truth (a shown value is never true of another drawn person; birthplace via admin parents; occupations not substring-related).
- **Shuffles:** with 3 attributes, every conflict puzzle has **0 cross-attribute coincidences**.

## Knowledge coverage of the shown facts (true-fact puzzles)

| Pool | E4B knows the person's fact | 31B knows it | E4B named the shown value | 31B named the shown value | Distinct people | Max uses | Median sitelinks |
|---|---|---|---|---|---|---|---|
| `only_31b` | 0.0% | 74.5% | 0.0% | 71.4% | 48 | 31 | 18 |
| `both` | 85.2% | 90.0% | 74.9% | 81.7% | 141 | 18 | 100 |
| `none` | 0.0% | 0.0% | 0.0% | 0.0% | 196 | 12 | 8 |

By relation:

| Relation | `only_31b` E4B | `only_31b` 31B | `both` E4B | `both` 31B |
|---|---|---|---|---|
| BirthPlace | 0.0% | 88.3% | 89.3% | 95.0% |
| Occupation | 0.0% | 94.8% | 99.5% | 98.0% |
| BirthYear | 0.0% | 40.3% | 66.8% | 77.0% |

In v1, the `both` pool's shown facts were known 71.6% (E4B) and 78.2% (31B), and `only_31b` 70.3% (31B).

## Files
- `zebralogic_pools_v2.parquet`: one row per puzzle. Columns: `id` (`<template>|<pool>|<condition>`), `base_id`, `size`, `pool`, `condition`, `puzzle`, `solution`, `valid`, `meta` (relations, people, QIDs, sitelinks, per-cell knowledge).
- `pool_variants_v2_cells.csv`: per person × relation in each puzzle, `knows_*` and `shown_known_*` for both models.
- `pool_variants_v2.py` (with `pool_variants.py` and `zebra_perturb.py`): the generator. Run as `python pool_variants_v2.py zebra_grid.parquet persons_pools.csv facts_extra.csv probe_extra_scored.jsonl out`.
- `zebra_grid.parquet`: the full WildEval/ZebraLogic `grid_mode` test file (1,000 puzzles), saved so the templates can be rebuilt offline.

## Caveats
- **Reuse in `only_31b`:** with only 48 usable people, one person appears in up to 31 of 120 templates. Use a person random effect in the analysis.
- **The 6×4 templates are harder** (6 houses) than the 4×5 they replace, so overall accuracy isn't comparable with v1.
- **Fame differs by pool.** Compare aligned vs conflict *within* a pool, i.e. read the pool × condition interaction.
- **Occupation is often guessable.** It stays in the puzzle as a well-known fact, but it isn't counted as evidence of knowing a person when the pools are defined.
- **Birth-year knowledge in `only_31b` is lower** (31B 40%), so that pool's contrast rests mostly on birthplace and occupation.
