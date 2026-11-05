"""Compare models on the real-person ZebraLogic variants (aligned / conflict / control).
Inputs (out_persons/): zebralogic_persons.parquet, persons_truth.json, knowledge_probe_both.csv, persons_<tag>.jsonl
Outputs: models_summary.csv, models_paired_tests.csv, models_between.csv, models_cells.csv, models_knowledge_did.csv,
         models_intrusion.csv"""
import json, numpy as np, pandas as pd
from math import comb
import run_eval as E

D = 'out_persons'
MODELS = {'gemma-4-E4B-it': 'persons_gemma4e4b.jsonl', 'gemma-4-31B-it': 'persons_gemma4_31b.jsonl'}
var = pd.read_parquet(f'{D}/zebralogic_persons.parquet').set_index('id')
truth = json.load(open(f'{D}/persons_truth.json', encoding='utf-8'))
KP = pd.read_csv(f'{D}/knowledge_probe_both.csv')
known = {m: set(zip(g.person[g.correct], g.relation[g.correct])) for m, g in KP.groupby('model')}
rng = np.random.default_rng(0); n_ = E.norm


def binom2(k, n):
    if n == 0: return 1.0
    pk = comb(n, k) / 2 ** n
    return min(1.0, sum(comb(n, i) / 2 ** n for i in range(n + 1) if comb(n, i) / 2 ** n <= pk * (1 + 1e-9)))


def boot(x, B=5000):
    x = np.asarray(x, float); return np.percentile(x[rng.integers(0, len(x), (B, len(x)))].mean(1), [2.5, 97.5])


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(p) - r) * p[i])); adj[i] = run
    return adj


S, C = [], []
for m, fn in MODELS.items():
    recs = {}
    for l in open(f'{D}/{fn}', encoding='utf-8'):
        r = json.loads(l)
        if r.get('finish') != 'error': recs[r['uid']] = r
    for uid, r in recs.items():
        v = var.loc[uid]; sol = json.loads(v.solution); sc = E.score(r['content'], sol); u = r.get('usage') or {}
        S.append(dict(model=m, uid=uid, base_id=v.base_id, size=v['size'], condition=v.condition, finish=r['finish'],
                      completion_tokens=u.get('completion_tokens'), **sc))
        # cell level: every gold cell, with the person's true value and whether the model knows it
        _, tab = E.parse(r['content']); keys = sol['header'][1:]; n = len(sol['rows'])
        pred = {}
        for h, pr in (tab or {}).items():
            pr = {n_(k): x for k, x in pr.items()}; pred[n_(pr.get('name', ''))] = pr
        dom = {k: {n_(row[j + 1]) for row in sol['rows']} for j, k in enumerate(keys)}
        for row in sol['rows']:
            g = dict(zip(keys, row[1:])); name = g['Name']
            real = name in truth          # control rows carry fictional names
            pr = pred.get(n_(name), {})
            for k in keys[1:]:
                pv = n_(pr.get(n_(k), ''))
                # in control, the 'owner' of the gold value is the real person whose fact it is
                C.append(dict(model=m, uid=uid, condition=v.condition, n=n, attr=k, person=name if real else None,
                              correct=pv == n_(g[k]), in_domain=pv in dom[k],
                              intrusion=bool(real and pv == n_(truth[name][k]) and n_(truth[name][k]) != n_(g[k])),
                              known=bool(real and (name, k) in known[m])))
S = pd.DataFrame(S); C = pd.DataFrame(C)
S.to_csv(f'{D}/models_scored.csv', index=False); C.to_csv(f'{D}/models_cells.csv', index=False)

# 1. per model x condition summary
summ = []
for (m, c), g in S.groupby(['model', 'condition']):
    lo, hi = boot(g.puzzle_correct); clo, chi = boot(g.cell_acc)
    summ.append(dict(model=m, condition=c, n=len(g), puzzle_acc=g.puzzle_correct.mean(), puzzle_lo=lo, puzzle_hi=hi,
                     cell_acc=g.cell_acc.mean(), cell_lo=clo, cell_hi=chi, parse_fail=int((g.parse == 'fail').sum()),
                     truncated=int((g.finish == 'length').sum()), median_tokens=g.completion_tokens.median()))
summ = pd.DataFrame(summ); summ.to_csv(f'{D}/models_summary.csv', index=False)

# 2. within-model paired contrasts between conditions (Holm within model)
T = []
for m, gm in S.groupby('model'):
    W = gm.pivot_table(index='base_id', columns='condition', values=['puzzle_correct', 'cell_acc'])
    rows = []
    for a, b in [('aligned', 'control'), ('conflict', 'control'), ('aligned', 'conflict')]:
        pa, pb = W['puzzle_correct'][a].astype(bool), W['puzzle_correct'][b].astype(bool)
        n10, n01 = int((pa & ~pb).sum()), int((~pa & pb).sum()); d = W['cell_acc'][a] - W['cell_acc'][b]; lo, hi = boot(d)
        rows.append(dict(model=m, contrast=f'{a} - {b}', puzzle_diff_pp=100 * (pa.mean() - pb.mean()), a_only=n10, b_only=n01,
                         mcnemar_p=binom2(n10, n10 + n01), cell_diff_pp=100 * d.mean(), cell_ci_lo=100 * lo, cell_ci_hi=100 * hi))
    for r, p in zip(rows, holm([r['mcnemar_p'] for r in rows])): r['holm_p'] = p
    T += rows
pd.DataFrame(T).to_csv(f'{D}/models_paired_tests.csv', index=False)

# 3. between models, same puzzle & condition; plus interaction (aligned-conflict gap differs between models?)
mA, mB = list(MODELS)
B = []
for c in ['control', 'aligned', 'conflict']:
    a = S[(S.model == mA) & (S.condition == c)].set_index('base_id'); b = S[(S.model == mB) & (S.condition == c)].set_index('base_id')
    ix = a.index.intersection(b.index); pa, pb = a.loc[ix].puzzle_correct.astype(bool), b.loc[ix].puzzle_correct.astype(bool)
    n10, n01 = int((pb & ~pa).sum()), int((~pb & pa).sum()); d = b.loc[ix].cell_acc - a.loc[ix].cell_acc; lo, hi = boot(d)
    B.append(dict(contrast=f'{mB} - {mA} | {c}', n=len(ix), puzzle_diff_pp=100 * (pb.mean() - pa.mean()), big_only=n10, small_only=n01,
                  mcnemar_p=binom2(n10, n10 + n01), cell_diff_pp=100 * d.mean(), cell_ci_lo=100 * lo, cell_ci_hi=100 * hi))
for r, p in zip(B, holm([r['mcnemar_p'] for r in B])): r['holm_p'] = p
for a_, b_ in [('aligned', 'conflict'), ('aligned', 'control'), ('conflict', 'control')]:
    gap = {}
    for m in MODELS:
        W = S[S.model == m].pivot_table(index='base_id', columns='condition', values='cell_acc'); gap[m] = W[a_] - W[b_]
    dd = (gap[mB] - gap[mA]).dropna(); lo, hi = boot(dd)
    B.append(dict(contrast=f'interaction: ({a_} - {b_}) gap, {mB} minus {mA}', n=len(dd), cell_diff_pp=100 * dd.mean(),
                  cell_ci_lo=100 * lo, cell_ci_hi=100 * hi))
pd.DataFrame(B).to_csv(f'{D}/models_between.csv', index=False)

# 4. knowledge-conditioned cell accuracy: (aligned - conflict) on known vs unknown facts, cluster bootstrap over puzzles
K = []
for m in MODELS:
    c = C[(C.model == m) & (C.condition.isin(['aligned', 'conflict']))].copy()
    c['base'] = c.uid.str.split('|').str[0]
    tab = c.groupby(['known', 'condition']).correct.agg(['mean', 'size']).unstack('condition')
    bases = c.base.unique(); grp = {b: g for b, g in c.groupby('base')}; dids = []
    for _ in range(2000):
        s = pd.concat([grp[b] for b in rng.choice(bases, len(bases))]); t = s.groupby(['known', 'condition']).correct.mean()
        try: dids.append((t[(True, 'aligned')] - t[(True, 'conflict')]) - (t[(False, 'aligned')] - t[(False, 'conflict')]))
        except KeyError: pass
    gk = tab.loc[True, ('mean', 'aligned')] - tab.loc[True, ('mean', 'conflict')]
    gu = tab.loc[False, ('mean', 'aligned')] - tab.loc[False, ('mean', 'conflict')]
    lo, hi = np.percentile(dids, [2.5, 97.5])
    K.append(dict(model=m, known_cells_aligned=int(tab.loc[True, ('size', 'aligned')]), known_cells_conflict=int(tab.loc[True, ('size', 'conflict')]),
                  acc_known_aligned=tab.loc[True, ('mean', 'aligned')], acc_known_conflict=tab.loc[True, ('mean', 'conflict')],
                  acc_unknown_aligned=tab.loc[False, ('mean', 'aligned')], acc_unknown_conflict=tab.loc[False, ('mean', 'conflict')],
                  gap_known_pp=100 * gk, gap_unknown_pp=100 * gu, did_pp=100 * (gk - gu), did_ci_lo=100 * lo, did_ci_hi=100 * hi,
                  did_p_boot=2 * min(np.mean(np.array(dids) <= 0), np.mean(np.array(dids) >= 0))))
pd.DataFrame(K).to_csv(f'{D}/models_knowledge_did.csv', index=False)

# 5. intrusion in conflict vs chance 1/(n-1), by known/unknown
I = []
for m in MODELS:
    c = C[(C.model == m) & (C.condition == 'conflict')]
    for nm, g in [('all', c), ('unknown', c[~c.known]), ('known', c[c.known])]:
        w = g[(~g.correct) & g.in_domain]; p = (1 / (w.n - 1)).values; obs = int(w.intrusion.sum()); exp = p.sum()
        sim = (rng.random((50000, len(p))) < p).sum(1)
        I.append(dict(model=m, subset=nm, cells=len(g), cell_acc=g.correct.mean(), wrong_in_domain=len(w), intrusions=obs,
                      expected=exp, obs_rate=obs / max(len(w), 1), chance_rate=exp / max(len(w), 1),
                      p_sim=(np.abs(sim - exp) >= abs(obs - exp)).mean() if len(w) else np.nan))
pd.DataFrame(I).to_csv(f'{D}/models_intrusion.csv', index=False)
