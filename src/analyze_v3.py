"""Analysis of the v3 real-person ZebraLogic runs (true vs shuffled facts, MCQA-defined pools).

Inputs (out/): zebralogic_pools_v3.parquet, persons_v3_<model>.jsonl, inpuzzle_mcqa_<model>.jsonl; candidates_v3.csv
Outputs (out/analysis/):
  v3_scored.csv          one row per model x puzzle
  v3_cells.csv           one row per model x puzzle x person x fact (correct, recognised in the in-puzzle MCQA)
  v3_summary.csv         accuracy per model x pool x condition (+ bootstrap CIs)
  v3_tests.csv           true - shuffled per model x pool: McNemar (Holm within model) + paired bootstrap CI on cells
  v3_knowledge_gap.csv   cell-level true - shuffled gap for recognised vs not recognised facts (cluster bootstrap)
"""
import json, os, sys
from math import comb
import numpy as np, pandas as pd
import run_eval as E

OUT = 'out'; A = os.path.join(OUT, 'analysis'); os.makedirs(A, exist_ok=True)
MODELS = [m['name'] for m in json.load(open('config.json'))['models']]
POOLS = ['both', 'only_large', 'none']
KEY2FACT = {'BirthPlace': 'birthplace', 'Occupation': 'occupation', 'BirthYear': 'birth_year'}
rng = np.random.default_rng(0); norm = E.norm


def binom2(k, n):
    if n == 0: return 1.0
    pk = comb(n, k) / 2 ** n
    return min(1.0, sum(comb(n, i) / 2 ** n for i in range(n + 1) if comb(n, i) / 2 ** n <= pk * (1 + 1e-9)))


def boot(x, B=5000):
    x = np.asarray(x, float); return np.percentile(x[rng.integers(0, len(x), (B, len(x)))].mean(1), [2.5, 97.5])


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(p) - r) * p[i])); adj[i] = run
    return adj


var = pd.read_parquet(os.path.join(OUT, 'zebralogic_pools_v3.parquet')).set_index('id')
S, C = [], []
for m in MODELS:
    ip = pd.read_json(os.path.join(OUT, f'inpuzzle_mcqa_{m}.jsonl'), lines=True)
    rec_flag = {(r.puzzle_key, r.person, r.fact): bool(r.correct) for r in ip.itertuples()}
    recs = {}
    for l in open(os.path.join(OUT, f'persons_v3_{m}.jsonl'), encoding='utf-8'):
        r = json.loads(l)
        if r.get('finish') != 'error': recs[r['uid']] = r
    for uid, r in recs.items():
        v = var.loc[uid]; sol = json.loads(v.solution); sc = E.score(r['content'], sol); u = r.get('usage') or {}
        S.append(dict(model=m, uid=uid, base_id=v.base_id, size=v['size'], pool=v.pool, condition=v.condition, finish=r['finish'],
                      completion_tokens=u.get('completion_tokens'), **sc))
        _, tab = E.parse(r['content']); keys = sol['header'][1:]; pk = f'{v.base_id}|{v.pool}|d{v.draw}'
        pred = {}
        for h, row in (tab or {}).items():
            row = {norm(k): x for k, x in row.items()}; pred[norm(row.get('name', ''))] = row
        for row in sol['rows']:
            g = dict(zip(keys, row[1:])); pr = pred.get(norm(g['Name']), {})
            for k in keys[1:]:
                f = KEY2FACT[k]
                C.append(dict(model=m, uid=uid, base_id=v.base_id, size=v['size'], pool=v.pool, condition=v.condition, person=g['Name'],
                              fact=f, correct=norm(pr.get(norm(k), '')) == norm(g[k]), recognised=rec_flag.get((pk, g['Name'], f))))
S = pd.DataFrame(S); C = pd.DataFrame(C)
S.to_csv(os.path.join(A, 'v3_scored.csv'), index=False); C.to_csv(os.path.join(A, 'v3_cells.csv'), index=False)

# 1. summary per model x pool x condition
summ = []
for (m, p, c), g in S.groupby(['model', 'pool', 'condition']):
    lo, hi = boot(g.puzzle_correct); clo, chi = boot(g.cell_acc)
    summ.append(dict(model=m, pool=p, condition=c, n=len(g), puzzle_acc=g.puzzle_correct.mean(), puzzle_lo=lo, puzzle_hi=hi,
                     cell_acc=g.cell_acc.mean(), cell_lo=clo, cell_hi=chi, truncated=int((g.finish == 'length').sum()),
                     parse_fail=int((g.parse == 'fail').sum()), median_tokens=g.completion_tokens.median()))
pd.DataFrame(summ).to_csv(os.path.join(A, 'v3_summary.csv'), index=False)

# 2. true - shuffled within each model x pool (paired over templates); Holm across the 3 pools within a model
T = []
for m in MODELS:
    rows = []
    for p in POOLS:
        g = S[(S.model == m) & (S.pool == p)]
        W = g.pivot_table(index='base_id', columns='condition', values=['puzzle_correct', 'cell_acc'])
        a, c = W['puzzle_correct']['aligned'].astype(bool), W['puzzle_correct']['conflict'].astype(bool)
        n10, n01 = int((a & ~c).sum()), int((~a & c).sum()); d = W['cell_acc']['aligned'] - W['cell_acc']['conflict']; lo, hi = boot(d)
        rows.append(dict(model=m, pool=p, n_pairs=len(W), puzzle_true=a.mean(), puzzle_shuffled=c.mean(), puzzle_gap_pp=100 * (a.mean() - c.mean()),
                         true_only=n10, shuffled_only=n01, mcnemar_p=binom2(n10, n10 + n01),
                         cell_gap_pp=100 * d.mean(), cell_gap_lo=100 * lo, cell_gap_hi=100 * hi))
    for r, q in zip(rows, holm([r['mcnemar_p'] for r in rows])): r['holm_p'] = q
    T += rows
pd.DataFrame(T).to_csv(os.path.join(A, 'v3_tests.csv'), index=False)

# 3. cell-level gap by in-puzzle recognition, cluster bootstrap over templates
K = []
for m in MODELS:
    c = C[(C.model == m) & C.recognised.notna()].copy(); c['recognised'] = c.recognised.astype(bool)
    agg = c.groupby(['base_id', 'recognised', 'condition']).correct.agg(['sum', 'size']).unstack(['recognised', 'condition'], fill_value=0)
    def gaps(a):
        out = {}
        for kn in (True, False):
            sa, na = a[('sum', kn, 'aligned')].sum(), a[('size', kn, 'aligned')].sum()
            sc, nc = a[('sum', kn, 'conflict')].sum(), a[('size', kn, 'conflict')].sum()
            out[kn] = (sa / na - sc / nc) if na and nc else np.nan
        return out
    est = gaps(agg); bs = [gaps(agg.iloc[rng.integers(0, len(agg), len(agg))]) for _ in range(3000)]
    for kn in (True, False):
        v = np.array([b[kn] for b in bs]); lo, hi = np.nanpercentile(v, [2.5, 97.5])
        K.append(dict(model=m, recognised=kn, cells_per_condition=int(agg[('size', kn, 'aligned')].sum()),
                      gap_pp=100 * est[kn], gap_lo=100 * lo, gap_hi=100 * hi))
    did = np.array([b[True] - b[False] for b in bs]); lo, hi = np.nanpercentile(did, [2.5, 97.5])
    K.append(dict(model=m, recognised='difference', gap_pp=100 * (est[True] - est[False]), gap_lo=100 * lo, gap_hi=100 * hi,
                  p_boot=2 * min(np.mean(did <= 0), np.mean(did >= 0))))
pd.DataFrame(K).to_csv(os.path.join(A, 'v3_knowledge_gap.csv'), index=False)
print(pd.DataFrame(T).round(3).to_string()); print(pd.DataFrame(K).round(3).to_string())
