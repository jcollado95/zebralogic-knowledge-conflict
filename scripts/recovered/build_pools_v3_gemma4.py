"""Recovered notebook code that produced `pools_v3_gemma4.csv`.

Reconstructed from session lineage, not written as a standalone script.
Expect adjustments (input paths, working directory via WORKDIR) before rerunning.
Steps that restored files from the artifact store are replaced by `pass`;
those files are under src/ and data/ in this repository.
"""
import json, os, sys
import numpy as np, pandas as pd

# build_pools_v3.py content (inlined)
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
    # final pools (user decision): strict rule; fame-matched subsample on the final pools
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


import shutil

os.makedirs('probe', exist_ok=True)
os.makedirs('pools', exist_ok=True)

# Copy probe JSONL files - these were produced by mcqa_probe.py run against the endpoint
# They are stored as artifacts; copy them to the expected probe/ directory
# NOTE: probe_mcqa_*.jsonl files are not listed in dependency mappings,
# so we must rebuild from candidates_v3.csv which itself requires a live SPARQL query.
# The candidates_v3.csv was built from a live QLever query - not available as artifact.
# The closest available artifact is candidates_all.csv; candidates_v3.csv is the stratified sample.

# Since candidates_v3.csv and probe files are not in dependency mappings,
# we reconstruct using available artifacts where possible.
# The probe JSONL files were saved to 'probe/' during the run - no artifact reference is given.
# We load candidates_v3.csv that was saved to disk during the session.

cand = pd.read_csv('candidates_v3.csv', encoding='utf-8-sig')

C2, _ = build(cand, 'probe', 'gemma-4-E4B-it', 'gemma-4-31B-it', 'gemma4')
C2.to_csv('pools/pools_v3_gemma4.csv', index=False, encoding='utf-8-sig')
print('final (strict):', C2.pool_final.value_counts().to_dict(), '| looser rule:', C2.pool.value_counts().to_dict())
