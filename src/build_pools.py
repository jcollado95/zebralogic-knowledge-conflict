"""Build the three knowledge pools and their unique-value subsets.

Inputs : candidates_all.csv (Wikidata humans with P19/P106/P25/P69/P2048; all values, JSON lists)
         probe_both_scored.jsonl (closed-book probe of both models; rule-based + reviewed LLM-judge scoring)
Outputs: persons_pools.csv          every candidate in a pool, all values + per-model knowledge flags
         unique_<pool>.csv          largest subset whose displayed values never repeat within a column
                                    (same layout as unique_persons.csv, plus knowledge columns)

Knowledge (per model, per person): k = facts correct out of 5; specific = birthplace/mother/school.
  known   : k >= 2 and at least one specific fact      unknown : k == 0
Pools   : only_31b = 31B known & E4B unknown;  both = both known;
          none = both unknown and no clarification request ("which Victoria?") from either model.
"""
import json, re
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
import probe_pool as PP

MODELS = {'gemma-4-E4B-it': 'e4b', 'gemma-4-31B-it': '31b'}
REL2COL = {'BirthPlace': 'place of birth', 'Occupation': 'occupation', 'Mother': 'mother', 'EducatedAt': 'educated at',
           'Height': 'height (cm)'}
COLS = list(REL2COL.values()); SPEC = ['BirthPlace', 'Mother', 'EducatedAt']
BAD = re.compile(r" and | is |\.$")          # break the ZebraLogic clue grammar


def assign(cand, pr):
    W = pr.pivot_table(index=['qid', 'model'], columns='relation', values='correct', aggfunc='first').astype(int)
    W['k_spec'] = W[SPEC].sum(axis=1); W['k_all'] = W[list(REL2COL)].sum(axis=1)
    clar = pr.groupby('qid').clarification.any()
    e, b = W.xs('gemma-4-E4B-it', level='model'), W.xs('gemma-4-31B-it', level='model')
    kn = lambda w: (w.k_all >= 2) & (w.k_spec >= 1); un = lambda w: w.k_all == 0
    pool = pd.Series(None, index=e.index, dtype=object)
    pool[kn(b) & un(e)] = 'only_31b'; pool[kn(b) & kn(e)] = 'both'
    pool[un(b) & un(e) & ~clar.reindex(e.index, fill_value=False)] = 'none'
    out = cand.set_index('qid').copy(); out['pool'] = pool
    # which displayed values each model named (string match; judged-only answers -> knows=True, values unknown)
    for m, tag in MODELS.items():
        w = W.xs(m, level='model'); out[f'k_{tag}'] = w.k_all; out[f'k_specific_{tag}'] = w.k_spec
        sub = pr[pr.model == m].set_index(['qid', 'relation'])
        for rel, col in REL2COL.items():
            out[f'knows_{col}_{tag}'] = w[rel].astype(bool)
            kv, ans = [], []
            for q in out.index:
                r = sub.loc[(q, rel)]; disp = json.loads(out.at[q, col]); ans.append(r.answer)
                if not r.correct: kv.append([])
                elif rel == 'BirthPlace': kv.append(disp)
                else: kv.append([v for v in disp if PP.score(rel, [v], r.answer)])
            out[f'known_values_{col}_{tag}'] = [json.dumps(x, ensure_ascii=False) for x in kv]
            out[f'answer_{col}_{tag}'] = ans
    return out.reset_index()


def salient(P, cand):
    """One display value per column: named by both models > by 31B > by E4B > most frequent label among all candidates."""
    freq = {c: pd.Series([v for x in cand[c] for v in json.loads(x)]).value_counts() for c in COLS}
    for c in COLS:
        out = []
        for _, r in P.iterrows():
            vals = json.loads(r[c]); ke = set(json.loads(r[f'known_values_{c}_e4b'])); kb = set(json.loads(r[f'known_values_{c}_31b']))
            for pick in (ke & kb, kb, ke):
                pick = [v for v in vals if v in pick]
                if pick: out.append(pick[0]); break
            else: out.append(max(vals, key=lambda v: (freq[c].get(v, 0), v)))
        P[f'{c} (salient)'] = out
    return P


def unique_subset(P, prefer, hard=True):
    """Exact MILP (as in unique_persons.py): max people such that each displayed value is used at most once per
    column (case-insensitive); among equal sizes prefer displayed values the `prefer` models named, then sitelinks."""
    P = P.reset_index(drop=True)
    opts = []                                              # (person i, column, value, bonus)
    for i, r in P.iterrows():
        if BAD.search(r.person): continue
        for c in COLS:
            pref = set.intersection(*[set(json.loads(r[f'known_values_{c}_{t}'])) for t in prefer]) if prefer else set()
            vals = json.loads(r[c])
            if hard and pref: vals = [v for v in vals if v in pref]       # show only values the pool model(s) named
            for v in vals:
                if not BAD.search(v): opts.append((i, c, v, 1.0 if v in pref else 0.0))
    have = defaultdict(set)
    for i, c, v, _ in opts: have[i].add(c)
    ok = [i for i in range(len(P)) if have[i] == set(COLS)]
    opts = [o for o in opts if o[0] in set(ok)]; idx = {i: k for k, i in enumerate(ok)}; n, m = len(ok), len(opts)
    rows, cols, coef = [], [], []
    for i in ok:
        for k in range(len(COLS)): rows.append(5 * idx[i] + k); cols.append(idx[i]); coef.append(-1)
    vrow = {}
    for j, (i, c, v, _) in enumerate(opts):
        rows.append(5 * idx[i] + COLS.index(c)); cols.append(n + j); coef.append(1)
        rows.append(vrow.setdefault((c, v.lower()), 5 * n + len(vrow))); cols.append(n + j); coef.append(1)
    A = coo_matrix((coef, (rows, cols)), shape=(5 * n + len(vrow), n + m))
    ub = np.r_[np.zeros(5 * n), np.ones(len(vrow))]
    sl = P.loc[ok, 'sitelinks'].to_numpy(float); tot = sl.sum() + 1
    # weights: person 1 >> preferred-value bonus (<1 in total per person) >> sitelinks tie-break
    obj = np.r_[-(1 + 0.001 * sl / tot), [-(0.5 / (5 * n)) * b for *_, b in opts]]
    res = milp(obj, constraints=LinearConstraint(A, 0, ub), integrality=np.ones(n + m), bounds=Bounds(0, 1),
               options={'mip_rel_gap': 0, 'time_limit': 600})
    x = np.round(res.x).astype(int); chosen = defaultdict(dict)
    for j, (i, c, v, _) in enumerate(opts):
        if x[n + j]: chosen[i][c] = v
    sel = [i for i in ok if x[idx[i]]]
    U = P.loc[sel].copy()
    for c in COLS: U[f'{c} (shown)'] = [chosen[i][c] for i in sel]
    for t in MODELS.values():
        for c in COLS:
            U[f'shown_known_{c}_{t}'] = [chosen[i][c] in json.loads(P.at[i, f'known_values_{c}_{t}']) for i in sel]
    return U, res


if __name__ == '__main__':
    cand = pd.read_csv('candidates_all.csv'); pr = pd.read_json('probe_both_scored.jsonl', lines=True)
    A_ = assign(cand, pr); P = salient(A_[A_.pool.notna()].copy(), cand); P.to_csv('persons_pools.csv', index=False, encoding='utf-8-sig')
    for pool, prefer in [('only_31b', ['31b']), ('both', ['31b', 'e4b']), ('none', [])]:
        U, res = unique_subset(P[P.pool == pool], prefer)
        U.to_csv(f'unique_{pool}.csv', index=False, encoding='utf-8-sig'); print(pool, len(P[P.pool == pool]), '->', len(U), res.message)
