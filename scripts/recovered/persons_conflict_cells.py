"""Recovered notebook code that produced `persons_conflict_cells.csv`.

Reconstructed from session lineage, not written as a standalone script.
Expect adjustments (input paths, working directory via WORKDIR) before rerunning.
Steps that restored files from the artifact store are replaced by `pass`;
those files are under src/ and data/ in this repository.
"""
import os, sys, json, shutil, numpy as np, pandas as pd, csv

os.makedirs('rep', exist_ok=True)
for fn, v in {'run_eval.py': 'src/run_eval.py',
              'zebralogic_persons.parquet': 'data/puzzles/zebralogic_persons.parquet',
              'persons_gemma4e4b.jsonl': 'raw/persons_gemma4e4b.jsonl',
              'knowledge_probe.csv': 'data/persons/knowledge_probe.csv',
              'unique_persons.csv': 'data/persons/unique_persons.csv'}.items():
    shutil.copy(v, f'rep/{fn}')
sys.path.insert(0, 'rep')
import run_eval as E

REL = {'place of birth': 'BirthPlace', 'occupation': 'Occupation', 'mother': 'Mother',
       'educated at': 'EducatedAt', 'height (cm)': 'Height'}
truth = {r['person']: {REL[c]: r[c] for c in REL}
         for r in csv.DictReader(open('rep/unique_persons.csv', encoding='utf-8-sig'))}

kp = pd.read_csv('rep/knowledge_probe.csv')
fp = {('José Luis Rodríguez Zapatero', 'Mother'), ('Nancy Swider-Peltz, Jr', 'Mother')}
kp['correct'] = [c and (p, r) not in fp for c, p, r in zip(kp.correct, kp.person, kp.relation)]
known = set(zip(kp.person[kp.correct], kp.relation[kp.correct]))

var = pd.read_parquet('rep/zebralogic_persons.parquet').set_index('id')
recs = {}
for l in open('rep/persons_gemma4e4b.jsonl', encoding='utf-8'):
    r = json.loads(l)
    recs[r['uid']] = r

rows = []
n_ = E.norm
for uid, r in recs.items():
    v = var.loc[uid]
    if v.condition != 'conflict':
        continue
    sol = json.loads(v.solution)
    keys = sol['header'][1:]
    n = len(sol['rows'])
    dom = {k: {n_(row[j + 1]) for row in sol['rows']} for j, k in enumerate(keys)}
    _, tab = E.parse(r['content'])
    if tab is None:
        continue
    gold = {row[1]: dict(zip(keys, row[1:])) for row in sol['rows']}
    for h, pr in tab.items():
        pr = {n_(k): x for k, x in pr.items()}
        m = next((g for g in gold if n_(g) == n_(pr.get('name', ''))), None)
        if m is None:
            continue
        for k in keys[1:]:
            p = n_(pr.get(n_(k), ''))
            t = n_(truth[m][k])
            rows.append(dict(uid=uid, n=n, person=m, attr=k,
                             correct=p == n_(gold[m][k]),
                             in_domain=p in dom[k],
                             intrusion=p == t and t != n_(gold[m][k]),
                             known=(m, k) in known))

C = pd.DataFrame(rows)
C.to_csv('persons_conflict_cells.csv', index=False)
