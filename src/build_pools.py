"""Knowledge pools from the MCQA probe, for any (small, large) model pair.

Per model and person (3 facts: birthplace, occupation, birth year; refusals/unparsed count as wrong):
  known   : >= 2 of 3 correct, at least one of them birthplace or birth year   (strict: 3 of 3)
  unknown : 0 of 3 correct
Pools: only_large = large known & small unknown; both = both known; none = both unknown.
`pool_final` uses the strict rule (all 3 correct), since a single 4-option question is guessed 25% of the time;
`pool` (looser rule) is kept for reference.
Chance: a model that knows nothing about a person, guessing at 25% per question, is labelled
  known with p = 3*(1/4)^2*(3/4) + (1/4)^3 = 0.156 (strict: 0.016) and unknown with p = (3/4)^3 = 0.422.
  So #ignorant ~= #(0/3) / 0.422 and expected chance-known labels ~= 0.156 * #ignorant.
Fame matching: `matched_<pool>` flags a subsample in which only_large, both and none have equal counts per
sitelinks band (bands where all three pools have people).
Outputs: pools_v3_<family>.csv, pools_v3_summary.csv, pools_v3_by_band.csv
Usage: python build_pools.py candidates_v3.csv probe_dir small_model large_model family out_dir
"""
import json, os, sys
import numpy as np, pandas as pd

FACTS = ['birthplace', 'occupation', 'birth_year']; SPECIFIC = ['birthplace', 'birth_year']
P_KNOWN, P_KNOWN_STRICT, P_UNKNOWN = 3 * (1 / 4) ** 2 * (3 / 4) + (1 / 4) ** 3, (1 / 4) ** 3, (3 / 4) ** 3


def flags(probe):
    W = probe.pivot_table(index='qid', columns='fact', values='correct', aggfunc='first').reindex(columns=FACTS).fillna(False).astype(bool)
    k = W.sum(axis=1)
    return pd.DataFrame({**{f'knows_{f}': W[f] for f in FACTS}, 'k': k,
                         'known': (k >= 2) & W[SPECIFIC].any(axis=1), 'known_strict': k == 3, 'unknown': k == 0})


def build(cand, probe_dir, small, large, family, seed=0):
    rd = lambda m: pd.read_json(os.path.join(probe_dir, f"probe_mcqa_{m.replace('/', '_')}.jsonl"), lines=True)
    fs, fl = flags(rd(small)), flags(rd(large))
    C = cand.set_index('qid').copy()
    for tag, f in (('small', fs), ('large', fl)):
        C = C.join(f.add_suffix(f'_{tag}'), how='left')
    for rule, col in (('', 'known'), ('_strict', 'known_strict')):
        p = pd.Series(None, index=C.index, dtype=object)
        p[C[f'{col}_large'] & C['unknown_small']] = 'only_large'
        p[C[f'{col}_large'] & C[f'{col}_small']] = 'both'
        p[C['unknown_large'] & C['unknown_small']] = 'none'
        C['pool' + rule] = p
    # fame-matched subsample (default rule): equal counts per band across the three pools
    rng = np.random.default_rng(seed); C['matched'] = False
    for band, g in C[C.pool.notna()].groupby('band'):
        n = min((g.pool == p_).sum() for p_ in ('only_large', 'both', 'none'))
        for p_ in ('only_large', 'both', 'none'):
            ids = g.index[g.pool == p_]
            if n: C.loc[rng.choice(ids, n, replace=False), 'matched'] = True
    # final pools: strict rule; fame-matched subsample on the final pools
    C['pool_final'] = C['pool_strict']; rng = np.random.default_rng(seed); C['matched_final'] = False
    for band, g in C[C.pool_final.notna()].groupby('band'):
        n = min((g.pool_final == p_).sum() for p_ in ('only_large', 'both', 'none'))
        for p_ in ('only_large', 'both', 'none'):
            if n: C.loc[rng.choice(g.index[g.pool_final == p_], n, replace=False), 'matched_final'] = True
    C['family'] = family; C['small_model'] = small; C['large_model'] = large
    chance = {}
    for tag, f in (('small', fs), ('large', fl)):
        ign = f.unknown.sum() / P_UNKNOWN
        chance[tag] = dict(zero_of_3=int(f.unknown.sum()), est_ignorant=round(float(ign)), est_chance_known=round(float(P_KNOWN * ign)),
                           est_chance_known_strict=round(float(P_KNOWN_STRICT * ign)), labelled_known=int(f.known.sum()), labelled_known_strict=int(f.known_strict.sum()))
    return C.reset_index(), chance


def summaries(C):
    F = C[C.pool_final.notna()]
    summ = F.groupby('pool_final').agg(people=('qid', 'size'), median_sitelinks=('sitelinks', 'median'), matched=('matched_final', 'sum'),
                                       e4b_facts=('k_small', 'mean'), b31_facts=('k_large', 'mean')).round(2).reset_index()
    by_band = pd.crosstab(F.band, F.pool_final).reset_index().rename_axis(None, axis=1)
    return summ, by_band


if __name__ == '__main__':
    cand = pd.read_csv(sys.argv[1], encoding='utf-8-sig'); out = sys.argv[6]; os.makedirs(out, exist_ok=True)
    C, chance = build(cand, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    C.to_csv(os.path.join(out, f'pools_v3_{sys.argv[5]}.csv'), index=False, encoding='utf-8-sig')
    summ, by_band = summaries(C)
    summ.to_csv(os.path.join(out, 'pools_v3_summary.csv'), index=False); by_band.to_csv(os.path.join(out, 'pools_v3_by_band.csv'), index=False)
    print('final (strict):', C.pool_final.value_counts().to_dict(), '| looser rule:', C.pool.value_counts().to_dict(), json.dumps(chance))
