# Knowledge-stratified person pools (gemma-4-E4B-it vs gemma-4-31B-it)

Three pools of real people for the ZebraLogic knowledge-conflict experiments, defined by what each model *knows* about them:

| Pool | Rule | People | Unique-value subset | Median sitelinks | Mean facts known (E4B / 31B, of 5) |
|---|---|---|---|---|---|
| `only_31b` | 31B knows the person, E4B knows nothing | 70 | 25 | 18.5 | 0.00 / 2.49 |
| `both` | both models know the person | 195 | 37 | 96 | 2.54 / 3.04 |
| `none` | neither model knows anything | 222 | 46 | 6 | 0.00 / 0.00 |

Share of people per pool for whom each model knows each fact:

| Fact | only_31b: E4B | only_31b: 31B | both: E4B | both: 31B | none: E4B | none: 31B |
|---|---|---|---|---|---|---|
| place of birth | 0% | 81% | 88% | 91% | 0% | 0% |
| occupation | 0% | 83% | 99% | 99% | 0% | 0% |
| height (±3 cm) | 0% | 46% | 40% | 59% | 0% | 0% |
| educated at | 0% | 29% | 17% | 28% | 0% | 0% |
| mother | 0% | 10% | 10% | 27% | 0% | 0% |

## How it was built

1. **Candidates.** The Wikidata query from `unique_persons.py` was run on QLever: humans with place of birth (P19), occupation (P106), mother (P25), educated at (P69) and height (P2048), using the script's height normalisation and labelling rules (en → mul → majority label).
   - The single-value rule was relaxed (the script's `SINGLE_VALUES_ONLY = False` behaviour): only 176 people have one value per property, while 1,189 have all five properties. Every value is kept as a JSON list.
   - 6 people with unlabelled values or duplicate names were dropped, leaving **1,183 candidates** (`candidates_all.csv`).
   - Birthplaces that are a hospital, building, palace, historic site or estate (50 cases, detected from their P31 type) are displayed as their administrative parent (P131), e.g. Kapiolani Medical Center → Honolulu. `place of birth (accepted)` lists the label, its P131 parent and, for those non-place cases only, the grandparent.
2. **Closed-book probe** (`probe_pool.py`, which reuses the questions and rules of `knowledge_probe.py`). Each model answered 5 direct questions per person (5,915 per model; greedy decoding; 0 request errors). An answer counts as correct if it matches **any** Wikidata value of that relation:
   - occupation by string match or synonym;
   - birthplace by the label or its admin parents;
   - mother and school must contain every content word of the label; for mothers, a partial name of two or more words that includes the surname is also accepted ("Ann Dunham" for "Stanley Ann Dunham");
   - height within ±3 cm.

   Refusals and clarification requests never count as correct.
3. **LLM judge for occupation and birthplace.** Every non-refusal occupation or birthplace answer that the rules marked wrong (1,703) was judged YES/NO by gemma-4-31B-it, given only the answer and the Wikidata values. This catches synonyms and broader terms (cosmonaut ↔ astronaut, king ↔ monarch, cyclist ↔ sport cyclist) and city-level birthplaces nested deeper than two levels (Tokyo for Akasaka Estate).
   - **Occupation:** 220 YES; 219 kept (one "None (She is…)" refusal removed). Spot-check precision is about 85–90%, with errors mostly over-general terms ("Artist", "Entertainer").
   - **Birthplace:** 77 YES, **all reviewed by hand**. 41 were kept (containing city or historical name: London for Westminster, Leningrad = Saint Petersburg, Allahabad = Prayagraj). 36 were rejected (a different town, a neighbouring municipality, or only a state or country, e.g. Boston for Kent, CT; Athens for Marousi).

   Verdicts are in `judge_verdicts.csv`. Note that the judge is also one of the probed models.
4. **Pool assignment** (`build_pools.py`). For each model, *k* = facts correct out of 5, and *specific* = birthplace, mother or school.
   - **Known:** *k* ≥ 2 with at least one specific fact. Occupation and height alone are not used as evidence, because they are often guessed right: among people with no specific fact correct, occupation is still "correct" for 38% (E4B) and 45% (31B), and height for 12% and 24%.
   - **Unknown:** *k* = 0.
   - `none` also excludes 52 people for whom either model asked for clarification ("which Victoria?"). Their non-recognition comes from an ambiguous name, not from lack of knowledge.
   - 696 candidates fit no pool (partial knowledge) and are left out.
5. **Display values.**
   - **`... (salient)`** in `persons_pools.csv`: one value per column. Priority: a value both models named > a value 31B named > a value E4B named > the most frequent label among all candidates. For example, Modi → *politician* (not *bibliographer*).
   - **`unique_<pool>.csv`:** the same exact MILP as `unique_persons.py`, run per pool: the largest set of people whose shown values never repeat within a column (case-insensitive). Two changes:
     - Where the pool's model(s) named a value, only named values may be shown. That is `31b` for `only_31b`, both models for `both`.
     - Labels containing " and " or " is ", or ending in ".", are excluded because they break the clue grammar.

     Height is the binding column (35–54 distinct heights per pool), which is why these subsets are much smaller than the pools. `shown_known_<col>_<model>` marks whether each shown value is one that model named.

## Files

- `persons_pools.csv`: all 487 pooled people. Columns:
  - `pool`;
  - all Wikidata values as JSON lists, including `place of birth (wikidata)` and `place of birth (accepted)`;
  - `<col> (salient)`;
  - per model (`e4b`, `31b`): `k_*`, `k_specific_*`, `knows_<col>_*`, `known_values_<col>_*` and `answer_<col>_*`.
- `unique_only_31b.csv`, `unique_both.csv`, `unique_none.csv`: the per-pool unique-value subsets (`<col> (shown)` columns).
- `candidates_all.csv`, `probe_both_scored.jsonl`, `judge_verdicts.csv`, `pools_summary.csv`: the inputs and intermediate results.
- `probe_pool.py`, `build_pools.py`: code (`knowledge_probe.py` is needed alongside).

## Caveats and recommendations for puzzle generation

- **Use birthplace + occupation (+ height) as the puzzle relations** when the goal is to manipulate *known* facts; mother and school are rarely known even in `both`.
- **Uniqueness per puzzle instead of globally:** drawing 4–5 people per puzzle from the full pool and requiring distinct salient values within the puzzle keeps the pools large. For conflict variants, also require that a shown value is in no other drawn person's full value set (the `wikidata_kg.py` rule), because multi-valued occupations overlap (many politicians).
- **Fame is confounded with pool** (median sitelinks 6 / 18.5 / 96). Name frequency and culture may differ as well. Compare conditions *within* a pool (aligned vs conflict vs control) rather than raw accuracy across pools.
- "Unknown" means the model gave none of these five facts correctly; it does not mean the model has never heard the name. In `none`, 31B often guesses from the name's apparent nationality (Kraków and University of Warsaw for a Polish name), a name-based prior that could itself be a perturbation to study.
- The probe measures free recall. A person the model recognises in a multiple-choice format may still fall in `none`.
- `both` includes well-known historical figures of every kind (e.g. Adolf Hitler, Vladimir Lenin); filter by QID if needed.
