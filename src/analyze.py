"""Analysis of the knowledge-conflict experiment (true vs shuffled facts, MCQA-defined pools, E4B vs 31B).

Inputs : data/ candidates_v3.csv, zebralogic_pools_v3.parquet
         runs/ probe_mcqa_<model>.jsonl, persons_v3_32k_full_<model>.jsonl, inpuzzle_mcqa_<model>.jsonl (see run_experiment.py)
Outputs (results/):
  mcqa_summary.csv, mcqa_by_band_long.csv   knowledge probe accuracy per model x fact (x fame band)
  variants_v3_validation.csv                round-trip validity of the 720 puzzles
  inpuzzle_mcqa_summary.csv                 in-puzzle recognition per pool x model x fact
  v3_scored.csv          one row per model x puzzle
  v3_cells.csv           one row per model x puzzle x person x fact (correct, recognised in the in-puzzle MCQA)
  v3_summary.csv         accuracy per model x pool x condition (+ bootstrap CIs)
  v3_tests.csv           true - shuffled per model x pool: McNemar (Holm within model) + paired bootstrap CI on cells
  v3_knowledge_gap.csv   cell-level true - shuffled gap for recognised vs not recognised facts (cluster bootstrap)
  v3_knowledge_gap_by_pool.csv       the same gap within each pool
  v3_diff_in_diff.csv                (true - shuffled) in a knowledge pool minus the same gap in 'none'
  v3_facilitation_interference.csv   each condition vs the same condition in 'none', template by template
  v3_sensitivity_no_truncation.csv   true vs shuffled after dropping templates truncated in either condition
  v3_by_size.csv                     puzzle accuracy by model x size x pool x condition
Usage: python analyze.py [--config config.json] [--data ../data] [--runs ../runs] [--out ../results]
"""
import argparse, json, os, sys
from math import comb
import numpy as np, pandas as pd
import evaluate as E

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='config.json'); ap.add_argument('--data', default='../data')
ap.add_argument('--runs', default='../runs'); ap.add_argument('--out', default='../results')
args = ap.parse_args()
DATA, RUNS, A = args.data, args.runs, args.out; os.makedirs(A, exist_ok=True)
MODELS = [m['name'] for m in json.load(open(args.config))['models']]
POOLS = ['both', 'only_large', 'none']
KEY2FACT = {'BirthPlace': 'birthplace', 'Occupation': 'occupation', 'BirthYear': 'birth_year'}
rng = np.random.default_rng(0); norm = E.norm

# 0. knowledge probe, puzzle validity, in-puzzle recognition
cand = pd.read_csv(os.path.join(DATA, 'candidates_v3.csv'), encoding='utf-8-sig').set_index('qid')
probe_files = [os.path.join(RUNS, f'probe_mcqa_{m}.jsonl') for m in MODELS]
if all(os.path.exists(f) for f in probe_files):
    P = pd.concat([pd.read_json(f, lines=True) for f in probe_files])
    P['band'] = P.qid.map(cand.band); P['nonletter'] = P.letter.isna()
    P.groupby(['model', 'fact']).agg(accuracy=('correct', 'mean'), non_letter=('nonletter', 'mean'), n=('correct', 'size')).round(3) \
        .reset_index().to_csv(os.path.join(A, 'mcqa_summary.csv'), index=False)
    P.groupby(['model', 'fact', 'band']).correct.agg(['mean', 'size']).reset_index().to_csv(os.path.join(A, 'mcqa_by_band_long.csv'), index=False)
VAR = pd.read_parquet(os.path.join(DATA, 'zebralogic_pools_v3.parquet'))
VAR.groupby(['pool', 'condition', 'size']).valid.agg(n='size', valid='sum').reset_index().to_csv(os.path.join(A, 'variants_v3_validation.csv'), index=False)
IP = pd.concat([pd.read_json(os.path.join(RUNS, f'inpuzzle_mcqa_{m}.jsonl'), lines=True) for m in MODELS])
IP['pool'] = IP.puzzle_key.str.split('|').str[1]
IP.groupby(['pool', 'model', 'fact']).correct.mean().reset_index().to_csv(os.path.join(A, 'inpuzzle_mcqa_summary.csv'), index=False)


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


var = pd.read_parquet(os.path.join(DATA, 'zebralogic_pools_v3.parquet')).set_index('id')
S, C = [], []
for m in MODELS:
    ip = pd.read_json(os.path.join(RUNS, f'inpuzzle_mcqa_{m}.jsonl'), lines=True)
    rec_flag = {(r.puzzle_key, r.person, r.fact): bool(r.correct) for r in ip.itertuples()}
    recs = {}
    for l in open(os.path.join(RUNS, f'persons_v3_32k_full_{m}.jsonl'), encoding='utf-8'):
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


# 4. extra contrasts (computed from the saved tables; each with its own seed)
Sc = pd.read_csv(os.path.join(A, 'v3_scored.csv')); Ce = pd.read_csv(os.path.join(A, 'v3_cells.csv'))

# 4a. recognition-conditioned cell gap within each pool (cluster bootstrap over templates)
rng1 = np.random.default_rng(1); rows = []
for (m, p), c in Ce[Ce.recognised.notna()].groupby(['model', 'pool']):
    c = c.copy(); c['recognised'] = c.recognised.astype(str).str.lower().eq('true')
    for kn in (True, False):
        g = c[c.recognised == kn]
        if g.empty: continue
        grp = {b: x for b, x in g.groupby('base_id')}; bases = list(grp)
        a = g[g.condition == 'aligned'].correct.mean(); s = g[g.condition == 'conflict'].correct.mean(); bs = []
        for _ in range(1000):
            smp = pd.concat([grp[b] for b in rng1.choice(bases, len(bases))])
            bs.append(smp[smp.condition == 'aligned'].correct.mean() - smp[smp.condition == 'conflict'].correct.mean())
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rows.append(dict(model=m, pool=p, recognised=kn, cells=int((g.condition == 'aligned').sum()), gap_pp=100 * (a - s), lo=100 * lo, hi=100 * hi))
pd.DataFrame(rows).to_csv(os.path.join(A, 'v3_knowledge_gap_by_pool.csv'), index=False)

# 4b. diff-in-diff: (true - shuffled) in a knowledge pool minus (true - shuffled) in 'none'
rng7 = np.random.default_rng(7); out = []
for m, g in Sc.groupby('model'):
    P_ = g.pivot_table(index=['pool', 'base_id'], columns='condition', values='puzzle_correct').astype(float)
    d = {p: (P_.loc[p].aligned - P_.loc[p].conflict).values for p in ['both', 'only_large', 'none']}
    for p in ['both', 'only_large']:
        obs = d[p].mean() - d['none'].mean()
        bs = np.array([rng7.choice(d[p], len(d[p])).mean() - rng7.choice(d['none'], len(d['none'])).mean() for _ in range(20000)])
        out.append(dict(model=m, contrast=f'{p} - none', did_pp=100 * obs, lo=100 * np.percentile(bs, 2.5), hi=100 * np.percentile(bs, 97.5),
                        p=min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))))
pd.DataFrame(out).to_csv(os.path.join(A, 'v3_diff_in_diff.csv'), index=False)

# 4c. facilitation (true facts vs 'none' true) and interference (shuffled vs 'none' shuffled), paired over templates
rng11 = np.random.default_rng(11); out = []
for m, g in Sc.groupby('model'):
    P_ = g.pivot_table(index='base_id', columns=['pool', 'condition'], values='puzzle_correct').astype(float)
    for p in ['both', 'only_large']:
        for name, cond in [('facilitation', 'aligned'), ('interference', 'conflict')]:
            dv = (P_[(p, cond)] - P_[('none', cond)]).values
            bs = np.array([rng11.choice(dv, len(dv)).mean() for _ in range(20000)])
            out.append(dict(model=m, pool=p, effect=name, diff_pp=100 * dv.mean(), lo=100 * np.percentile(bs, 2.5), hi=100 * np.percentile(bs, 97.5),
                            p=min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))))
pd.DataFrame(out).to_csv(os.path.join(A, 'v3_facilitation_interference.csv'), index=False)

# 4d. sensitivity: drop every template truncated (finish == 'length') in either condition
sens = []
for (m, p), g in Sc.groupby(['model', 'pool']):
    bad = set(g[g.finish == 'length'].base_id)
    W = g[~g.base_id.isin(bad)].pivot_table(index='base_id', columns='condition', values='puzzle_correct')
    a, s = W.aligned.astype(bool), W.conflict.astype(bool); n10, n01 = int((a & ~s).sum()), int((~a & s).sum())
    sens.append(dict(model=m, pool=p, n_pairs=len(W), true=a.mean(), shuffled=s.mean(), gap_pp=100 * (a.mean() - s.mean()),
                     true_only=n10, shuffled_only=n01, p=binom2(n10, n10 + n01)))
pd.DataFrame(sens).to_csv(os.path.join(A, 'v3_sensitivity_no_truncation.csv'), index=False)

# 4e. accuracy by puzzle size
(Sc.groupby(['model', 'size', 'pool', 'condition'])
   .agg(n=('puzzle_correct', 'size'), puzzle_acc=('puzzle_correct', 'mean'), truncated=('finish', lambda s: int((s == 'length').sum())),
        median_tokens=('completion_tokens', 'median'))
   .reset_index().to_csv(os.path.join(A, 'v3_by_size.csv'), index=False))
