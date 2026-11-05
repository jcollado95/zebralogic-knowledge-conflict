"""Recovered notebook code that produced `variants_v3_validation.csv`.

Reconstructed from session lineage, not written as a standalone script.
Expect adjustments (input paths, working directory via WORKDIR) before rerunning.
Steps that restored files from the artifact store are replaced by `pass`;
those files are under src/ and data/ in this repository.
"""
import shutil, os, sys, json, importlib
import pandas as pd

os.makedirs('out', exist_ok=True)

# Copy required files
shutil.copy('src/zebra_perturb.py', 'zebra_perturb.py')
shutil.copy('src/pool_variants.py', 'pool_variants.py')
shutil.copy('src/pool_variants_v2.py', 'pool_variants_v2.py')
shutil.copy('data/source/zebra_grid.parquet', 'zebra_grid.parquet')

sys.path.insert(0, '.')

import pool_variants as PV0
if hasattr(PV0, '_orig_compatible'):
    del PV0._orig_compatible
importlib.reload(PV0)

import pool_variants_v2 as V2m
importlib.reload(V2m)

# candidates_v3.csv and pools_v3_gemma4.csv must be present (built by earlier steps in the trace)
# For the pool_variants_v3 module inline here:

import zebra_perturb as zp
import pool_variants_v2 as V2
PV = V2.PV

import re

POOLS = ['only_large', 'both', 'none']
PV.POOL_MODELS = {p: [] for p in POOLS}
LETTERS = 'ABCDEF'
FACT_OF = {'place of birth': 'birthplace', 'occupation': 'occupation', 'birth year': 'birth_year'}
SIZES = ['4*4', '5*4', '6*4']


def load_people(cand_csv, pools_csv):
    C = pd.read_csv(cand_csv, encoding='utf-8-sig').set_index('qid')
    P = pd.read_csv(pools_csv, encoding='utf-8-sig').set_index('qid')
    out = []
    for q, r in P[P.pool_final.notna()].iterrows():
        c = C.loc[q]
        d = dict(qid=q, person=c.person, pool=r.pool_final, sitelinks=int(c.sitelinks), shown={}, true={}, knows={}, shown_known={})
        d['shown']['place of birth'] = c.birthplace
        d['true']['place of birth'] = set(json.loads(c.birthplace_accepted)) | {c.birthplace}
        d['shown']['occupation'] = c.occupation
        d['true']['occupation'] = set(json.loads(c.occupations_all)) | set(json.loads(c.occupation_aliases)) | {c.occupation}
        d['shown']['birth year'] = str(int(c.birth_year))
        d['true']['birth year'] = {str(int(c.birth_year))}
        for rel, f in FACT_OF.items():
            for t, suf in (('e4b', 'small'), ('31b', 'large')):
                d['knows'][(rel, t)] = d['shown_known'][(rel, t)] = bool(r[f'knows_{f}_{suf}'])
        out.append(d)
    return out


def eligible(p, pool, rels):
    return True


PV.eligible = eligible


def build_all(grid, people, draws=1):
    g = pd.read_parquet(grid)
    g = g[g['size'].isin(SIZES)]
    recs = [(r.id, r.puzzle, dict(header=list(r.solution['header']), rows=[list(map(str, x)) for x in r.solution['rows']]))
            for r in g.itertuples()]
    tpls = zp.build_corpus(recs)
    rows = []
    for tpl, (pid, _, _) in zip(tpls, recs):
        size = '*'.join(pid.split('-')[2].split('x'))
        for pool in POOLS:
            for d in range(draws):
                t2 = tpl if d == 0 else type(tpl)(**{**tpl.__dict__, 'id': f'{tpl.id}#d{d}'})
                for cond, v in PV.build(t2, people, pool).items():
                    text, sol = v.render()
                    try:
                        ok = zp.validate_text(text, sol['header'][1:], sol['rows'], 'unique')
                    except (ValueError, AssertionError):
                        ok = False
                    rows.append(dict(id=f'{pid}|{pool}|d{d}|{cond}', base_id=pid, size=size, pool=pool, draw=d, condition=cond,
                                     puzzle=text, solution=json.dumps(sol, ensure_ascii=False), valid=bool(ok),
                                     meta=json.dumps(v.meta, ensure_ascii=False)))
    return pd.DataFrame(rows)


people = load_people('candidates_v3.csv', 'pools/pools_v3_gemma4.csv')

df = build_all('zebra_grid.parquet', people, 1)

df.drop(columns=['true_rate'], errors='ignore').to_parquet('out/zebralogic_pools_v3.parquet', index=False)

V = df.groupby(['pool', 'condition', 'size']).valid.agg(n='size', valid='sum').reset_index()
V.to_csv('variants_v3_validation.csv', index=False)
