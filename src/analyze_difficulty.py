"""Effect of puzzle difficulty (4x4 / 5x4 / 6x4 = 4 / 5 / 6 houses, 4 attributes each).

Run after analyze.py. Reads results/v3_scored.csv and results/v3_cells.csv.
Writes results/difficulty/:
  diff_accuracy.csv                  accuracy per model x size x pool x condition (+ bootstrap CI, truncation, tokens)
  diff_knowledge_free_curve.csv      'none' pool accuracy per model x size, averaged over conditions
  diff_gaps.csv                      per model x size x pool: true - shuffled puzzle gap (paired bootstrap, McNemar),
                                     cell gap, diff-in-diff vs 'none', facilitation, interference
  diff_trend.csv                     6x4 - 4x4 contrast for each of these metrics (bootstrap)
  diff_recognition.csv               recognition-conditioned cell gap by size (pools pooled)
  diff_model_gap.csv                 31B - E4B accuracy per size x pool x condition (paired by puzzle)
  diff_tokens.csv                    truncation and completion tokens per size x condition
  diff_sensitivity_no_truncation.csv puzzle gap after dropping templates truncated in either condition
Usage: python analyze_difficulty.py [--results ../results]
"""
import argparse, os
from math import comb
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument('--results', default='../results'); args = ap.parse_args()
A = args.results; OUT = os.path.join(A, 'difficulty'); os.makedirs(OUT, exist_ok=True)
SIZES = ['4*4', '5*4', '6*4']; POOLS = ['both', 'only_large', 'none']; B = 10000
rng = np.random.default_rng(2026)


def binom2(k, n):
    if n == 0: return 1.0
    pk = comb(n, k) / 2 ** n
    return min(1.0, sum(comb(n, i) / 2 ** n for i in range(n + 1) if comb(n, i) / 2 ** n <= pk * (1 + 1e-9)))


def bmean(x):
    """bootstrap distribution of the mean of per-template values"""
    x = np.asarray(x, float); return x[rng.integers(0, len(x), (B, len(x)))].mean(1)


def ci(bs): return 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def p2(bs): return min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))


S = pd.read_csv(os.path.join(A, 'v3_scored.csv')); C = pd.read_csv(os.path.join(A, 'v3_cells.csv'))
S['puzzle_correct'] = S.puzzle_correct.astype(float)
cellp = C.groupby(['model', 'uid', 'base_id', 'size', 'pool', 'condition']).correct.mean().reset_index()

# accuracy table
acc = []
for (m, sz, p, c), g in S.groupby(['model', 'size', 'pool', 'condition']):
    lo, hi = ci(bmean(g.puzzle_correct))
    acc.append(dict(model=m, size=sz, pool=p, condition=c, n=len(g), puzzle_acc=100 * g.puzzle_correct.mean(), lo=lo, hi=hi,
                    truncated_pct=100 * (g.finish == 'length').mean(), median_tokens=g.completion_tokens.median()))

# knowledge-free curve: 'none' pool, per template mean over the two conditions
curve = []
for (m, sz), g in S[S.pool == 'none'].groupby(['model', 'size']):
    v = g.groupby('base_id').puzzle_correct.mean(); lo, hi = ci(bmean(v))
    curve.append(dict(model=m, size=sz, acc=100 * v.mean(), lo=lo, hi=hi))

# gaps, diff-in-diff, facilitation / interference, trend
gaps, trend = [], []
for m, gm in S.groupby('model'):
    P = gm.pivot_table(index=['size', 'base_id'], columns=['pool', 'condition'], values='puzzle_correct')
    Q = cellp[cellp.model == m].pivot_table(index=['size', 'base_id'], columns=['pool', 'condition'], values='correct')
    per = {}
    for sz in SIZES:
        Ps, Qs = P.loc[sz], Q.loc[sz]
        gap_none = (Ps[('none', 'aligned')] - Ps[('none', 'conflict')]).values
        for p in POOLS:
            d_p = (Ps[(p, 'aligned')] - Ps[(p, 'conflict')]).values
            d_c = (Qs[(p, 'aligned')] - Qs[(p, 'conflict')]).values
            a, s = Ps[(p, 'aligned')].astype(bool), Ps[(p, 'conflict')].astype(bool)
            n10, n01 = int((a & ~s).sum()), int((~a & s).sum())
            bs_p, bs_c = bmean(d_p), bmean(d_c)
            row = dict(model=m, size=sz, pool=p, n=len(d_p), true=100 * a.mean(), shuffled=100 * s.mean(),
                       gap_pp=100 * d_p.mean(), gap_lo=ci(bs_p)[0], gap_hi=ci(bs_p)[1],
                       true_only=n10, shuffled_only=n01, mcnemar_p=binom2(n10, n10 + n01),
                       cell_gap_pp=100 * d_c.mean(), cell_lo=ci(bs_c)[0], cell_hi=ci(bs_c)[1])
            per[(sz, p, 'gap')] = d_p; per[(sz, p, 'cell')] = d_c
            if p != 'none':
                dd = d_p - gap_none; bs = bmean(dd)            # same templates in both pools -> paired
                row.update(did_pp=100 * dd.mean(), did_lo=ci(bs)[0], did_hi=ci(bs)[1], did_p=p2(bs)); per[(sz, p, 'did')] = dd
                for eff, cond in [('facil', 'aligned'), ('interf', 'conflict')]:
                    x = (Ps[(p, cond)] - Ps[('none', cond)]).values; bs = bmean(x)
                    row.update({f'{eff}_pp': 100 * x.mean(), f'{eff}_lo': ci(bs)[0], f'{eff}_hi': ci(bs)[1], f'{eff}_p': p2(bs)})
                    per[(sz, p, eff)] = x
            gaps.append(row)
    for p in POOLS:
        for k in ['gap', 'cell'] + (['did', 'facil', 'interf'] if p != 'none' else []):
            hi_, lo_ = per[('6*4', p, k)], per[('4*4', p, k)]
            bs = bmean(hi_) - bmean(lo_)                        # different template sets -> independent bootstraps
            trend.append(dict(model=m, pool=p, metric=k, d4=100 * lo_.mean(), d5=100 * per[('5*4', p, k)].mean(), d6=100 * hi_.mean(),
                              diff_6_minus_4=100 * (hi_.mean() - lo_.mean()), lo=ci(bs)[0], hi=ci(bs)[1], p=p2(bs)))

# recognition-conditioned cell gap by size (pools pooled), template-cluster bootstrap
recog = []
Cr = C[C.recognised.notna()].copy(); Cr['recognised'] = Cr.recognised.astype(str).str.lower().eq('true')
for (m, sz), g in Cr.groupby(['model', 'size']):
    t = g.groupby(['base_id', 'recognised', 'condition']).correct.agg(['sum', 'count']).unstack(['recognised', 'condition']).fillna(0)
    arr = {col: t[col].values for col in t.columns}; idx = rng.integers(0, len(t), (2000, len(t))); bsr, obs = {}, {}
    for kn in (True, False):
        sa, ca = arr[('sum', kn, 'aligned')], arr[('count', kn, 'aligned')]
        sc, cc = arr[('sum', kn, 'conflict')], arr[('count', kn, 'conflict')]
        bsr[kn] = sa[idx].sum(1) / ca[idx].sum(1) - sc[idx].sum(1) / cc[idx].sum(1)
        obs[kn] = sa.sum() / ca.sum() - sc.sum() / cc.sum()
    dif = bsr[True] - bsr[False]
    recog.append(dict(model=m, size=sz, recognised_gap=100 * obs[True], rec_lo=ci(bsr[True])[0], rec_hi=ci(bsr[True])[1],
                      unrecognised_gap=100 * obs[False], unrec_lo=ci(bsr[False])[0], unrec_hi=ci(bsr[False])[1],
                      difference=100 * (obs[True] - obs[False]), diff_lo=ci(dif)[0], diff_hi=ci(dif)[1], diff_p=p2(dif),
                      cells_recognised=int(arr[('count', True, 'aligned')].sum()), cells_unrecognised=int(arr[('count', False, 'aligned')].sum())))

# 31B - E4B accuracy, paired by puzzle uid
mgap = []
W = S.pivot_table(index=['size', 'pool', 'condition', 'uid'], columns='model', values='puzzle_correct')
for (sz, p, c), g in W.groupby(level=[0, 1, 2]):
    x = (g['gemma-4-31B-it'] - g['gemma-4-E4B-it']).values; bs = bmean(x)
    mgap.append(dict(size=sz, pool=p, condition=c, acc_31b=100 * g['gemma-4-31B-it'].mean(), acc_e4b=100 * g['gemma-4-E4B-it'].mean(),
                     gap_pp=100 * x.mean(), lo=ci(bs)[0], hi=ci(bs)[1]))

# tokens / truncation
toks = []
for (m, sz, c), g in S.groupby(['model', 'size', 'condition']):
    toks.append(dict(model=m, size=sz, condition=c, truncated=int((g.finish == 'length').sum()), n=len(g),
                     median_tokens=g.completion_tokens.median(), p90_tokens=g.completion_tokens.quantile(0.9)))

# sensitivity: drop templates truncated in either condition
sens = []
for (m, sz, p), g in S.groupby(['model', 'size', 'pool']):
    bad = set(g[g.finish == 'length'].base_id)
    Wp = g[~g.base_id.isin(bad)].pivot_table(index='base_id', columns='condition', values='puzzle_correct').astype(bool)
    n10, n01 = int((Wp.aligned & ~Wp.conflict).sum()), int((~Wp.aligned & Wp.conflict).sum())
    sens.append(dict(model=m, size=sz, pool=p, n_kept=len(Wp), n_dropped=len(bad), true=100 * Wp.aligned.mean(),
                     shuffled=100 * Wp.conflict.mean(), gap_pp=100 * (Wp.aligned.mean() - Wp.conflict.mean()),
                     true_only=n10, shuffled_only=n01, mcnemar_p=binom2(n10, n10 + n01)))

for name, rows in [('diff_accuracy', acc), ('diff_knowledge_free_curve', curve), ('diff_gaps', gaps), ('diff_trend', trend),
                   ('diff_recognition', recog), ('diff_model_gap', mgap), ('diff_tokens', toks),
                   ('diff_sensitivity_no_truncation', sens)]:
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f'{name}.csv'), index=False)
print('written to', OUT)
