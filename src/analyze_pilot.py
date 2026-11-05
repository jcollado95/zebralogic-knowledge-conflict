"""Score and summarise pilot results (JSONL from run_eval.run)."""
import json, sys, numpy as np, pandas as pd
from scipy.stats import binomtest
import run_eval as ev


def load(path):
    rows = []
    for l in open(path, encoding='utf-8'):
        r = json.loads(l); s = ev.score(r['content'], r['solution'])
        rows.append(dict(uid=r['uid'], base_id=r['base_id'], size=r['size'], perturbation=r['perturbation'], condition=r['condition'],
                         finish=r['finish'], completion_tokens=(r.get('usage') or {}).get('completion_tokens'), **s))
    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summary(df):
    g = df.groupby(['condition', 'perturbation'])
    out = g.agg(n=('uid', 'size'), puzzle_acc=('puzzle_correct', 'mean'), cell_acc=('cell_acc', 'mean'),
                abstain_rate=('abstained', 'mean'), parse_fail=('parse', lambda s: (s == 'fail').mean()),
                truncated=('finish', lambda s: (s == 'length').mean()), tokens=('completion_tokens', 'median')).reset_index()
    k = g['puzzle_correct'].sum().values
    ci = [wilson(a, b) for a, b in zip(k, out.n)]
    out['puzzle_acc_lo'] = [c[0] for c in ci]; out['puzzle_acc_hi'] = [c[1] for c in ci]
    return out


def paired(df, a, b, cond='standard', B=5000, seed=0):
    """b minus a on the same base puzzles: puzzle-acc delta + exact McNemar p; cell-acc delta + bootstrap CI."""
    d = df[df.condition == cond].pivot_table(index='base_id', columns='perturbation', values=['puzzle_correct', 'cell_acc'], aggfunc='first')
    if a not in d['puzzle_correct'] or b not in d['puzzle_correct']: return None
    pa, pb = d['puzzle_correct'][a], d['puzzle_correct'][b]; m = pa.notna() & pb.notna()
    pa, pb = pa[m].astype(bool), pb[m].astype(bool)
    n10, n01 = int((pa & ~pb).sum()), int((~pa & pb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    ca, cb = d['cell_acc'][a][m].astype(float).values, d['cell_acc'][b][m].astype(float).values
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(ca), (B, len(ca))); bs = (cb[idx] - ca[idx]).mean(1)
    return dict(a=a, b=b, n=int(m.sum()), puzzle_acc_a=pa.mean(), puzzle_acc_b=pb.mean(), d_puzzle=pb.mean() - pa.mean(),
                a_only=n10, b_only=n01, mcnemar_p=p, d_cell=(cb - ca).mean(), d_cell_lo=np.quantile(bs, .025), d_cell_hi=np.quantile(bs, .975))


STD = ['symbolic_full', 'madeup', 'shuffle_s0', 'implausible_nouns', 'wd_gender_aligned', 'wd_gender_conflict',
       'wd_citizenship_aligned', 'wd_citizenship_conflict', 'wd_occupation_aligned', 'wd_occupation_conflict']
PAIRS = [('wd_gender_aligned', 'wd_gender_conflict'), ('wd_citizenship_aligned', 'wd_citizenship_conflict'),
         ('wd_occupation_aligned', 'wd_occupation_conflict')]


def report(df):
    s = summary(df)
    vs_orig = pd.DataFrame([x for x in (paired(df, 'original', p) for p in STD) if x])
    pairs = pd.DataFrame([x for x in (paired(df, a, b) for a, b in PAIRS) if x])
    by_size = df[df.condition == 'standard'].groupby(['size', 'perturbation']).puzzle_correct.mean().unstack(0)
    return s, vs_orig, pairs, by_size


if __name__ == '__main__':
    df = load(sys.argv[1]); s, v, p, bs = report(df)
    pd.set_option('display.width', 200); print(s.round(3).to_string()); print(v.round(3).to_string()); print(p.round(3).to_string())
