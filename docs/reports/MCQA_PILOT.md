# MCQA probe: pilot validation

**Pilot:** 100 random candidates × 3 facts × 2 models (gemma-4-E4B-it, gemma-4-31B-it), letter answers, greedy decoding. 0 request errors.

**Accuracy (chance = 25%):**

| Model | Birth year | Birthplace | Occupation |
|---|---|---|---|
| gemma-4-31B-it | 67% | 65% | 78% |
| gemma-4-E4B-it | 27% | 41% | 57% |

**Non-letter answers** (refusals such as "I do not have the information…"): E4B 17.7%, 31B 2.0%. They are scored as not known.

**Position bias:** the correct answer is spread evenly over A–D, but the models' choices are not.

| Model | Chose A | Chose B | Chose C | Chose D | Accuracy when answer is A / B / C / D |
|---|---|---|---|---|---|
| gemma-4-31B-it | 21% | 32% | 24% | 22% | 63% / 80% / 72% / 65% |
| gemma-4-E4B-it | 26% | 51% | 17% | 6% | 44% / 69% / 37% / 16% |

Asking for the option *text* instead of the letter did not remove it: E4B still chose the second option 46% of the time and the last 7%. Its correct/incorrect flags agreed with letter mode on 95.7% of items (31B: 97.3%). The bias is about position, not about letter labels. Because the correct position is balanced, any guessing strategy is right on 25% of unknown items, biased or not. The bias changes which unknown items look known, not how many. The probe therefore keeps standard letter answers, and the chance rate is handled when the pools are defined (at least 2 of 3 facts, one of them birthplace or birth year).

**Distractor review** (30 random questions read by hand): distractors were plausible and never true. Two problems were found and fixed:
- some candidates' Wikidata birthplace is a whole country (e.g. "Japan" appeared as an option), so 64 people with a country-level birthplace were removed;
- distractor years could fall after 2010 (e.g. 2012 for an adult public figure), so they are now capped at 2010 unless the true year is later.

In the pilot, same-country birthplace distractors were drawn only from the 100 pilot people, so a few were from other countries. The full run draws from all candidates.

**Fame gradient (pilot, all facts):** accuracy rises with sitelinks for both models. In the 100+ band it is 31B 97% vs E4B 55%; in the 0–4 band, 40% vs 24%.
