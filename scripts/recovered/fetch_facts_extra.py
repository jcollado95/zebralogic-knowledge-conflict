"""Recovered notebook code that produced `facts_extra.csv`.

Reconstructed from session lineage, not written as a standalone script.
Expect adjustments (input paths, working directory via WORKDIR) before rerunning.
Steps that restored files from the artifact store are replaced by `pass`;
those files are under src/ and data/ in this repository.
"""
import os, json, time, urllib.request, urllib.parse, re
import pandas as pd
from collections import defaultdict, Counter

cand = pd.read_csv('data/persons/candidates_all.csv')
qids = cand.qid.tolist()

EP = "https://qlever.dev/api/wikidata"

def sparql(q, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(EP, data=urllib.parse.urlencode({'query': q}).encode(),
                headers={"Accept": "application/sparql-results+json", "User-Agent": "unique-persons-script/1.0 (Wikidata research script)"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return [{kk: c['value'] for kk, c in row.items()} for row in json.load(r)['results']['bindings']]
        except urllib.error.HTTPError as e:
            if e.code == 429 and k < tries - 1:
                time.sleep(20 * (k + 1))
                continue
            raise

PFX = """PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#>"""

PROPS = {'birth_year': 'P569', 'death_year': 'P570', 'citizenship': 'P27', 'spouse': 'P26', 'father': 'P22', 'death_place': 'P20'}

qid_of = lambda iri: iri.rsplit('/', 1)[-1]

P2N = {'http://www.wikidata.org/prop/direct/' + p: n for n, p in PROPS.items()}
vals = defaultdict(lambda: defaultdict(set))
t = time.time()
for s in range(0, len(qids), 600):
    rows = sparql(PFX + f"""SELECT ?p ?prop ?v WHERE {{ VALUES ?p {{ {' '.join('wd:' + q for q in qids[s:s+600])} }}
      VALUES ?prop {{ {' '.join('wdt:' + p for p in PROPS.values())} }} ?p ?prop ?v }}""")
    for r in rows:
        n = P2N[r['prop']]; v = r['v']
        if n.endswith('_year'):
            m = re.match(r'^(-?\d{1,4})-', v)
            if m: vals[qid_of(r['p'])][n].add(str(int(m.group(1))))
        elif v.startswith('http://www.wikidata.org/entity/Q'):
            vals[qid_of(r['p'])][n].add(qid_of(v))
    time.sleep(3)
print('fetched in', round(time.time() - t, 1), 's')
cov = {n: sum(1 for q in qids if vals[q][n]) for n in PROPS}
print('coverage of 1183:', cov)

ents = sorted({v for q in qids for n in ['citizenship', 'spouse', 'father', 'death_place'] for v in vals[q][n]})
lab = {}
alias = defaultdict(set)
par = defaultdict(set)

for s in range(0, len(ents), 1500):
    ch = ' '.join('wd:' + q for q in ents[s:s+1500])
    for r in sparql(PFX + f"""SELECT ?i ?en ?mul WHERE {{ VALUES ?i {{ {ch} }} OPTIONAL {{ ?i rdfs:label ?en FILTER(LANG(?en)="en") }}
                              OPTIONAL {{ ?i rdfs:label ?mul FILTER(LANG(?mul)="mul") }} }}"""):
        lab[qid_of(r['i'])] = r.get('en') or r.get('mul')
    time.sleep(2)
    for r in sparql(PFX + f"""SELECT ?i ?a WHERE {{ VALUES ?i {{ {ch} }} ?i skos:altLabel ?a FILTER(LANG(?a)="en") }}"""):
        alias[qid_of(r['i'])].add(r['a'])
    time.sleep(2)

dps = sorted({v for q in qids for v in vals[q]['death_place']})
for s in range(0, len(dps), 800):
    for r in sparql(PFX + f"""SELECT ?i ?a ?al WHERE {{ VALUES ?i {{ {' '.join('wd:' + q for q in dps[s:s+800])} }} ?i wdt:P131 ?a .
                                 OPTIONAL {{ ?a rdfs:label ?al FILTER(LANG(?al)="en") }} }}"""):
        if r.get('al'): par[qid_of(r['i'])].add(r['al'])
    time.sleep(2)

print(len(ents), 'entities;', sum(1 for e in ents if not lab.get(e)), 'without label;', sum(len(a) for a in alias.values()), 'aliases')

EXTRA = ['birth_year', 'death_year', 'citizenship', 'spouse', 'father', 'death_place']
rows = []
for q in qids:
    d = dict(qid=q)
    for n in EXTRA:
        v = vals[q][n]
        if n.endswith('_year'):
            disp = sorted(v); acc = sorted(v)
        else:
            disp = sorted({lab.get(x) for x in v if lab.get(x)})
            acc = set(disp) | {a for x in v for a in alias[x]}
            if n == 'death_place': acc |= {a for x in v for a in par[x]}
            acc = sorted(acc)
        d[n] = json.dumps(disp, ensure_ascii=False)
        d[n + ' (accepted)'] = json.dumps(acc, ensure_ascii=False)
    rows.append(d)

X = cand[['qid', 'person', 'sitelinks']].merge(pd.DataFrame(rows), on='qid')
X.to_csv('facts_extra.csv', index=False, encoding='utf-8')
