# Candidate population v3 (persons for knowledge probes and puzzles)

**Snapshot:** QLever Wikidata endpoint (https://qlever.dev/api/wikidata), queried 2026-11-04. **4,515 people** in `candidates_v3.csv`.

## Population and sampling
- **Population:** humans (P31 = Q5) with a place of birth (P19), an occupation (P106) and a date of birth (P569). Counts per sitelinks band at query time: 0–4: 3,052,069; 5–9: 461,028; 10–19: 206,651; 20–49: 76,155; 50–99: 9,719; 100+: 1,455.
- **Fame-stratified sample, weighted toward fame:** 1,000 each from 100+, 50–99 and 20–49; 800 from 10–19; 600 each from 5–9 and 0–4. Within a band, people are the first *n* when ordered by `MD5(STR(?person))`, so the sample is deterministic and reproducible.

Kept after filtering, by band: a_0-4: 548, b_5-9: 566, c_10-19: 758, d_20-49: 935, e_50-99: 909, f_100+: 863 (4,995 sampled, 416 excluded by the rules below).

## Facts per person
- **Birthplace:** the P19 label. If P19 is a hospital, building, palace or similar (detected by its P31 type or label), the first P131 administrative parent is shown instead, e.g. Kapiolani Medical Center → Honolulu. `birthplace_accepted` lists the label, English aliases and the P131 parent (plus the grandparent for those mapped cases).
- **Occupation:** one display value per person. It is the P106 occupation mentioned first in the person's English Wikidata description (2,906 people). Otherwise it comes from office or title words in the description ("President…" → politician; 302), then from description synonyms ("footballer" → association football player; 615), then from the person's most common occupation label across all candidates (756). `occupations_all` keeps every P106 label, so distractors and shuffled values never use a true occupation.
- **Birth year:** the year of P569 (any precision). People with several different years were excluded.

## Exclusions
Applied in this order: missing fact, unlabelled name, conflicting birth year, several birthplaces, one-word name (ambiguous when shown bare in a puzzle, e.g. "Victoria"), label containing ' and ' / ' is ' / a trailing '.' (these break the clue grammar), unlabelled value, non-settlement birthplace after mapping, duplicate name, and birthplace that is a whole country (64 people, e.g. P19 = Japan; city-states such as Hong Kong and Singapore are caught by this rule too).

## Columns
`qid, person, sitelinks, band, birth_country (P17 QID of the birthplace), birthplace, birthplace_accepted (JSON), occupation, occupations_all (JSON), occupation_aliases (JSON), birth_year, occupation_source, description`.
