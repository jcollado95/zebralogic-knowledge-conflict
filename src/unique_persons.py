#!/usr/bin/env python3
"""
Wikidata humans with place of birth, occupation, mother, educated at and height,
and the largest possible list in which no two people share any of those values.

Data comes from QLever's live Wikidata endpoint. Needs Python 3.9+ and scipy
(pip install scipy). Writes two CSV files (labels only) to the current folder:
  unique_persons.csv   the largest list with no shared values in any column
  all_candidates.csv   everyone who meets the per-person rules (values may repeat)
"""
import csv
import json
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ENDPOINT = "https://qlever.dev/api/wikidata"

# True: only people with exactly one value per property (the "single" rule).
# False: also people with several values; the solver picks which one to show.
SINGLE_VALUES_ONLY = True

COLUMNS = ["birthPlace", "occupation", "mother", "educatedAt", "height"]
HEADER = ["person", "place of birth", "occupation", "mother", "educated at", "height (cm)"]

PREFIXES = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

DATA_QUERY = PREFIXES + """
SELECT ?person ?birthPlace ?occupation ?mother ?educatedAt ?rank ?metres ?sitelinks WHERE {
  ?person wdt:P31 wd:Q5 ; wdt:P19 ?birthPlace ; wdt:P106 ?occupation ;
          wdt:P25 ?mother ; wdt:P69 ?educatedAt ; p:P2048 ?heightStatement .
  # normalised value = metres, whatever unit was entered (cm, m, ft, ...)
  ?heightStatement wikibase:rank ?rank ; psn:P2048/wikibase:quantityAmount ?metres .
  OPTIONAL { ?person wikibase:sitelinks ?sitelinks }
}
"""

LABEL_QUERY = PREFIXES + """
SELECT ?item ?en ?mul WHERE {
  VALUES ?item { %s }
  OPTIONAL { ?item rdfs:label ?en FILTER(LANG(?en) = "en") }
  OPTIONAL { ?item rdfs:label ?mul FILTER(LANG(?mul) = "mul") }
}
"""


def sparql(query):
    body = urllib.parse.urlencode({"query": query}).encode()
    request = urllib.request.Request(ENDPOINT, data=body, headers={
        "Accept": "application/sparql-results+json",
        "User-Agent": "unique-persons-script/1.0 (Wikidata research script)"})
    with urllib.request.urlopen(request, timeout=300) as response:
        rows = json.load(response)["results"]["bindings"]
    return [{key: cell["value"] for key, cell in row.items()} for row in rows]


def item_id(iri):
    """Q-id of an item, or None for 'unknown value' placeholders."""
    prefix = "http://www.wikidata.org/entity/Q"
    return iri.rsplit("/", 1)[-1] if iri.startswith(prefix) else None


# 1. Collect each person's values. wdt: values are already best-rank; height comes
#    from full statements (to get normalised units), so apply the ranks here.
values = defaultdict(lambda: defaultdict(set))
heights = defaultdict(lambda: defaultdict(set))
sitelinks = defaultdict(int)
for row in sparql(DATA_QUERY):
    person = item_id(row["person"])
    for column in COLUMNS[:4]:
        value = item_id(row[column])
        if value:
            values[person][column].add(value)
    rank = row["rank"].rsplit("#", 1)[-1]
    heights[person][rank].add(str(round(float(row["metres"]) * 100)))
    sitelinks[person] = int(row.get("sitelinks", 0))
for person, by_rank in heights.items():
    values[person]["height"] = by_rank.get("PreferredRank") or by_rank.get("NormalRank") or set()

complete = [p for p in values if all(values[p][c] for c in COLUMNS)]
if SINGLE_VALUES_ONLY:
    candidates = [p for p in complete if all(len(values[p][c]) == 1 for c in COLUMNS)]
else:
    candidates = complete

# 2. Labels: English, then "mul" (many names now only have a mul label), then the
#    label most other languages agree on, and only as a last resort the Q-id.
labels = {}
needed = sorted(set(candidates) | {v for p in candidates for c in COLUMNS[:4] for v in values[p][c]})
for start in range(0, len(needed), 2000):
    chunk = " ".join("wd:" + q for q in needed[start:start + 2000])
    for row in sparql(LABEL_QUERY % chunk):
        labels[item_id(row["item"])] = row.get("en") or row.get("mul")
missing = [q for q in needed if not labels.get(q)]
if missing:
    votes = defaultdict(Counter)
    query = PREFIXES + "SELECT ?item ?label WHERE { VALUES ?item { %s } ?item rdfs:label ?label }"
    for row in sparql(query % " ".join("wd:" + q for q in missing)):
        votes[item_id(row["item"])][row["label"]] += 1
    for q, counter in votes.items():
        labels[q] = counter.most_common(1)[0][0]


def shown(column, value):
    return value if column == "height" else (labels.get(value) or value)


# 3. Exact optimisation: pick the most people such that every column shows each
#    value at most once (compared by label, as it appears in the CSV). Among
#    equally long lists, prefer better-known people (more Wikipedia sitelinks).
options = [(i, c, v) for i, p in enumerate(candidates) for c in COLUMNS
           for v in sorted({shown(c, x) for x in values[p][c]})]
n, m = len(candidates), len(options)
rows, cols, coefs = [], [], []
for i in range(n):                            # a chosen person shows exactly one
    for k in range(len(COLUMNS)):             # value per column: sum(z) - y = 0
        rows.append(5 * i + k); cols.append(i); coefs.append(-1)
value_row = {}
for j, (i, c, v) in enumerate(options):
    rows.append(5 * i + COLUMNS.index(c)); cols.append(n + j); coefs.append(1)
    rows.append(value_row.setdefault((c, v), 5 * n + len(value_row)))  # each value
    cols.append(n + j); coefs.append(1)                                # used <= once
A = coo_matrix((coefs, (rows, cols)), shape=(5 * n + len(value_row), n + m))
upper = np.r_[np.zeros(5 * n), np.ones(len(value_row))]
total = sum(sitelinks[p] for p in candidates) + 1
objective = np.r_[[-(1 + sitelinks[p] / total) for p in candidates], np.zeros(m)]
result = milp(objective, constraints=LinearConstraint(A, 0, upper), integrality=np.ones(n + m),
              bounds=Bounds(0, 1), options={"mip_rel_gap": 0})
x = np.round(result.x).astype(int)
chosen = defaultdict(dict)
for j, (i, c, v) in enumerate(options):
    if x[n + j]:
        chosen[i][c] = v


# 4. Write the CSVs (UTF-8 with BOM so Excel shows accented names correctly).
def write(path, table):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(sorted(table, key=lambda r: r[0].casefold()))


write("unique_persons.csv",
      [[shown("person", candidates[i])] + [chosen[i][c] for c in COLUMNS] for i in range(n) if x[i]])
write("all_candidates.csv",
      [[shown("person", p)] + ["; ".join(sorted({shown(c, v) for v in values[p][c]})) for c in COLUMNS]
       for p in candidates])
print(f"{len(complete)} people have all five properties; {len(candidates)} meet the per-person rules; "
      f"the largest list without shared values has {int(x[:n].sum())} people.")
