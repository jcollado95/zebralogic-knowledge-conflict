# Wikidata-backed entity sampling for ZebraLogic perturbations

The hand-written lists in `kg_seed.json` are replaced by a **relation catalog** (`wikidata_relations.json`).
Each entry is turned into a SPARQL query against `query.wikidata.org`, and the results are cached and frozen into a snapshot (`wikidata_kg_snapshot.json`).
More samples come from more seeds (`generate_wd_variants.py --samples K`). The candidate pools themselves change by editing the catalog (thresholds, limits, classes) and rebuilding.

```
python -c "import wikidata_kg as wk; wk.build_kg('wikidata_relations.json','kg_cache').save('wikidata_kg_snapshot.json')"
python generate_wd_variants.py data/zebra_grid.parquet wikidata_kg_snapshot.json out_wd --samples 3
```

## 1. Value pools (one per ZebraLogic attribute; relabelling keeps the puzzle's own phrasing)
| attribute | query | Wikidata pattern | size |
|---|---|---|---|
| Color | class | P31 Q1075 (colour), sitelinks ≥ 25 | 25 |
| Food | class | P31 Q746549 (dish), ≥ 40 | 47 |
| Drink *(low)* | class | P31/P279 Q40050 (drink), ≥ 30 | 29 |
| Smoothie | class | P31/P279 Q3314483 (fruit), English common name P1843 | 30 |
| BookGenre | class | P31 Q223393 (literary genre), ≥ 30 | 102 |
| MusicGenre | class | P31 Q188451 (music genre), ≥ 50 | 101 |
| FavoriteSport | usage | values of P641 (sport) ranked by #statements, filtered to subclasses of Q349 | 24 |
| Hobby | class | P31 Q47728 (hobby), ≥ 30 | 83 |
| Occupation | usage | values of P106 ranked by #statements | 193 |
| Nationality | demonym | P27 values that are current sovereign states (Q3624078, no P576) → English demonym P1549 | 142 |
| CarModel | class | P31 Q3231690 (automobile model), ≥ 30 | 186 |
| PhoneModel | class | P31 Q19723451 (smartphone model), ≥ 15 | 132 |
| HouseStyle | usage | values of P149 (architectural style) | 108 |
| Education *(low)* | usage | values of P512 (academic degree) that are subclasses of Q189533 | 45 |
| Pet / Animal *(low)* | class | has use P366 = pet (Q39201) ∪ domesticated animal Q622852 | 10 |
| Flower *(low)* | class | has use P366 = ornamental plant (Q199182) | 17 |
| HairColor | usage | values of P1884 (hair colour) | 38 |
| Vacation | class | subclasses of tourism Q49389 | 35 |
| Cigar | class | cigarette brand Q19509097 ∪ cigar brand Q110684031 | 87 |
| Birthday | class | Gregorian month Q47018901 (top 12) | 12 |
| Height | — | ordinal; original values kept | — |

*(low)* = the pool contains category noise (e.g. `livestock`, `pet`, `watercress`, `full professor`). These pools are skipped by default (`include_low=True` turns them on), and the puzzle keeps its original values for them.

## 2. Given-name pools
Female given name Q11879590 / male given name Q12308941 (≥ 12 sitelinks: 319 / 336), plus
language-specific pools via P407 "language of work or name". Usable pools (≥ 26 names per gender): English, Spanish, German,
French, Italian, Swedish, Polish, Russian, Turkish, Arabic, Japanese. **Wikidata coverage gap:** Chinese, Hindi, Yoruba, Igbo,
Swahili, Persian, Hebrew, Greek and Korean name items rarely carry P407 (0–36 names), so those groups are not generated yet.

## 3. Person relations (world-grounded triples: person —P→ value)
| relation | property | anchors | people | rendered as |
|---|---|---|---|---|
| occupation | P106 | top 40 occupations | 967 | "the person who works as a {v}" |
| citizenship | P27 → P1549 demonym | top 40 countries | 1115 | "the {v} person" |
| sport | P641 | 24 sports | 140 | "the person who plays {v}" |
| instrument | P1303 | top 25 instruments (excl. *voice*) | 256 | "the person who plays the {v}" |
| genre | P136 | top 30 music genres | 349 | "the person known for {v}" |
| birthplace | P19 | top 40 birthplaces | 676 | "the person born in {v}" |

People are humans (Q5) with ≥ 60 sitelinks who hold an anchor value. Their **full** value sets for the property are fetched
afterwards, which matters because the property is multi-valued (George Washington: British + American).

**Uniqueness rules** (`WikidataKG.people_assignment`, `_draw`):
1. Within an attribute: distinct QIDs and distinct labels.
2. Across the puzzle: no label equals, or occurs as a whole word inside, any other label. The plural endings the clue parser accepts count as matches (`blue` vs `blues`), which prevents ambiguous clue matching.
3. Aligned: v_i ∈ R(p_i) and v_i ∉ R(p_j) for all j ≠ i, so world knowledge identifies exactly one person per value (two people never share a nationality, etc.). R(p) is compared as a set of *labels*, because distinct QIDs can share a label after normalisation (`pop` / `pop music`).
4. Conflict: a derangement of an aligned draw. Each v_i belongs to exactly one *other* person and never to p_i.
5. Person labels containing a period are skipped ("Jr." would end a clue with "..").

Draws are seeded per (puzzle id, sample index, operator). Aligned and conflict variants of the same puzzle and sample therefore share their draws, while different puzzles get different entities.

**Audit of the generated set** (120 puzzles × 29 operators × 3 samples = 10,440 variants): 10,440 / 10,440 re-parse and solve to the
expected unique solution. 0 violations of rules 3–4 across 4,320 grounded variants. All 2,160 aligned/conflict pairs share the same people.
The grounded variants use 2,146 distinct people and 184 distinct relation values, and 6,289 distinct Name entities appear across all variants.

If the puzzle has the matching attribute (Occupation, Nationality, FavoriteSport, MusicGenre), it is relabelled in place.
Otherwise a non-name attribute is **retyped**: its header line, clue phrasing and solution key are replaced and the logic is untouched.

## 4. Data-driven gender skew (replaces the hand-written stereotype lexicon)
For each of the 193 occupations: female share of P21 among the first 3,000 holders returned by the endpoint
(not a random sample). The extremes range from 0.00 (Catholic bishop, American football player, military personnel) to
0.82 (model) and 0.90 (nurse). `wd_gender_occupation` draws ⌊n/2⌋ or ⌈n/2⌉ occupations from the 8 most female-skewed
and the rest from the 8 most male-skewed (n ≥ 200 sampled holders). It then assigns given names whose gender matches
(aligned) or opposes (conflict) the skew. The aligned and conflict variants use the same draws.

## Operators (zebra_perturb.py)
`wd_relabel` (all attributes; `top=15` restricts to the most-linked, typical values) · `wd_names(gender= | lang=)` ·
`wd_gender_occupation(aligned|conflict)` · `wd_grounded(rel, aligned|conflict)`. All are seeded and composable with the
earlier operators (shuffle, symbolic, underdetermine, contradict).

## Caveats
- Labels are English Wikidata labels after automatic filtering (≤ 3 words, ≤ 28 chars, no digits-as-years or punctuation); some remain unusual (`kumis`, `seiyū`).
- Occupations, birthplaces and citizenships of famous people reflect Wikidata's skew towards Western, male, historical figures.
- The endpoint load varies: one occupation anchor (`politician`) timed out and is missing from the occupation person relation.
- A few exclusions are set in the catalog (occupation: pornographic actor, pensioner, …; vacation: sex/drug/suicide/war/dark tourism).
