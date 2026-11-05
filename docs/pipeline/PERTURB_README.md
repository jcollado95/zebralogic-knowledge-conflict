# ZebraLogic template extraction + perturbation generator (prototype)

Source: `WildEval/ZebraLogic`, `grid_mode/test-00000-of-00001.parquet` (Hugging Face), filtered to sizes
`4*4`, `4*5`, `5*4` (houses*attributes) → 120 base puzzles, 1,459 clues.

## Files
- `zebra_perturb.py` – parser → `Template`, CSP solver, renderer (`Variant`), perturbation operators, `validate_text`.
- `generate_variants.py` – `python generate_variants.py data/zebra_grid.parquet out` (≈40 s on CPU).
- `kg_seed.json` – seed triple store used as entity priors (gender stereotype, name→nationality,
  celebrity→sport, cultural name pools, implausible nouns). Swap for a Wikidata- or LLM-logprob-derived KG.
- `templates.jsonl` – one variable template per puzzle (values → slots `{Attr.i}`; clues as typed relations + phrase templates; gold in slot form).
- `zebralogic_perturbed.parquet` / `.jsonl` – all variants. Columns: `id, base_id, size, perturbation, family,
  expected (unique|underdetermined|contradictory), puzzle, solution (JSON {header, rows} – ZebraLogic grid format, null if no unique solution), meta (JSON), valid`.

## Extraction checks
- Every clue parses to one of 9 relation types (same, found_at, not_at, left_adj, left_of, right_of, next_to, one_between, two_between).
- Re-rendering every template reproduces all 120 original puzzle texts byte-for-byte.
- The CSP over the parsed logical form has exactly one solution for all 120 puzzles, and it equals the gold table.
- Each emitted variant is re-parsed **from its rendered text** and solved; `valid` = outcome matches `expected`.

## Perturbation families (logic preserved unless noted)
| family | operator | notes |
|---|---|---|
| gender | `gender_all(female/male)`, `gender_congruence(aligned/conflict)` | congruence: names chosen so that, in the gold solution, gender matches / contradicts the stereotype of the first of Occupation > Hobby > FavoriteSport. Neutral values share one seeded gender in both conditions (matched pair). |
| nationality / favoritesport | `prior_congruence(attr, aligned/conflict)` | world-prior names (Astrid→swede, Messi→soccer). conflict = derangement: nobody holds their prior value. |
| cultural | `cultural_names(group)` | name origin only, no co-occurrence manipulation. |
| symbolic | `symbolic(full=False/True)` | values → `P1`, `A1…`; `full` also replaces phrases ("the person with A2") and header lines. |
| shuffle / order | `shuffle_clues(seed)`, `reorder_clues(reverse/unary_first/unary_last)` | unary = found_at / not_at clues. |
| implausible | `implausible('nouns')` ("loves gravel books"), `implausible('donor')` (values from an absent attribute: "is a tennis") | |
| madeup | `madeup()` | seeded pseudowords, substring-safe. |
| underdetermined | `underdetermine()` | one clue dropped → ≥2 solutions (count in meta). Target answer: abstain. |
| contradictory | `contradict()` | one clue altered → 0 solutions (original/altered clue in meta). |

Operators return a new `Variant`, so they compose: e.g. `shuffle_clues(gender_congruence(Variant(t), kg, 'conflict'), 1)`.

## Known limitations
- Some phrase templates depend on the value ("the person with an aquarium of {v}", "the {v} lover"), so in surface mode the original value's category can show through. `symbolic_full` does not have this problem.
- In the original data, Children clues say "the person who is the mother of X" regardless of the name's gender (e.g. Peter, Bob).
- The gender stereotype lexicon and name pools are small placeholders. Results should be reported as a function of the lexicon.
- In `*_aligned` variants the name gives away the value, so a model can shortcut. Compare aligned vs conflict against the original rather than reading either alone.
- Only one seed per operator was emitted. Raise SEED, or loop over seeds, for more samples per puzzle.
