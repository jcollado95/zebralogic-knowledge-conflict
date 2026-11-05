"""True-fact vs shuffled-fact ZebraLogic variants, v2: three universal, well-known facts per person.

Templates: ZebraLogic grid_mode sizes 4*4, 5*4, 6*4 (40 each; all have 3 non-Name attributes).
Relations: BirthPlace, Occupation, BirthYear  (every pooled person has all three).
Pools (from persons_pools.csv) with per-puzzle eligibility on these three facts:
  only_31b : 31B knows >= 2 of the 3 facts; E4B knows none of them (incl. birth year)
  both     : both models know >= 2 of the 3 facts
  none     : neither model knows any of the 3 facts (pool already requires 0/5 on the original facts)
People with conflicting Wikidata birth years are excluded.
Per-puzzle constraints: as pool_variants.py (distinct labels, no substrings, cross-truth: birthplace via admin parents,
occupations not substring-related) plus birth years >= 3 years apart.
Usage: python pool_variants_v2.py zebra_grid.parquet persons_pools.csv facts_extra.csv probe_extra_scored.jsonl out
"""
import sys, json, os, re
import pandas as pd
import zebra_perturb as zp
import pool_variants as PV

PV.REL = {'place of birth': ('BirthPlace', 'Each person was born in a different place', 'the person who was born in {v}'),
          'occupation': ('Occupation', 'Each person has a different occupation', 'the person who works as a {v}'),
          'birth year': ('BirthYear', 'Each person was born in a different year', 'the person who was born in the year {v}')}
PV.BASE_RELS = ['place of birth', 'occupation', 'birth year']
SIZES = ['4*4', '5*4', '6*4']
MIN_YEAR_GAP = 3


def load_people(pools_csv, facts_csv, probe_jsonl):
    P = pd.read_csv(pools_csv, encoding='utf-8-sig'); F = pd.read_csv(facts_csv).set_index('qid')
    pr = pd.read_json(probe_jsonl, lines=True); pr = pr[pr.relation == 'birth_year']
    by_known = {(q, {'gemma-4-E4B-it': 'e4b', 'gemma-4-31B-it': '31b'}[m]): bool(c) for q, m, c in zip(pr.qid, pr.model, pr.correct)}
    out = []
    for r in P.to_dict('records'):
        years = json.loads(F.at[r['qid'], 'birth_year']) if r['qid'] in F.index else []
        if len(years) != 1: continue                                   # missing or conflicting birth year
        d = dict(qid=r['qid'], person=r['person'], pool=r['pool'], sitelinks=r['sitelinks'], shown={}, true={}, knows={}, shown_known={})
        d['shown']['place of birth'] = r['place of birth (salient)']
        d['true']['place of birth'] = set(json.loads(r['place of birth (accepted)']) + json.loads(r['place of birth']))
        d['shown']['occupation'] = r['occupation (salient)']; d['true']['occupation'] = set(json.loads(r['occupation']))
        d['shown']['birth year'] = years[0]; d['true']['birth year'] = set(years)
        for t in ('e4b', '31b'):
            for c in ('place of birth', 'occupation'):
                d['knows'][(c, t)] = bool(r[f'knows_{c}_{t}'])
                d['shown_known'][(c, t)] = d['shown'][c] in set(json.loads(r[f'known_values_{c}_{t}']))
            d['knows'][('birth year', t)] = d['shown_known'][('birth year', t)] = by_known.get((r['qid'], t), False)
        if PV.NONPLACE.search(d['shown']['place of birth']): continue
        out.append(d)
    return out


def eligible(p, pool, rels):
    k = {t: sum(p['knows'][(c, t)] for c in rels) for t in ('e4b', '31b')}
    if pool == 'only_31b': return k['31b'] >= 2 and k['e4b'] == 0
    if pool == 'both': return k['31b'] >= 2 and k['e4b'] >= 2
    return k['31b'] == 0 and k['e4b'] == 0


_compat = getattr(PV, '_orig_compatible', PV.compatible); PV._orig_compatible = _compat   # idempotent on reload
def compatible(group, rels):
    if not _compat(group, rels): return False
    ys = [int(p['shown']['birth year']) for p in group]
    return all(abs(a - b) >= MIN_YEAR_GAP for i, a in enumerate(ys) for b in ys[i + 1:])


PV.eligible = eligible; PV.compatible = compatible


def main(grid, pools_csv, facts_csv, probe_jsonl, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    g = pd.read_parquet(grid); g = g[g['size'].isin(SIZES)]
    recs = [(r.id, r.puzzle, dict(header=list(r.solution['header']), rows=[list(map(str, x)) for x in r.solution['rows']]))
            for r in g.itertuples()]
    tpls = zp.build_corpus(recs); people = load_people(pools_csv, facts_csv, probe_jsonl); rows = []
    for tpl, (pid, _, _) in zip(tpls, recs):
        size = '*'.join(pid.split('-')[2].split('x'))
        for pool in PV.POOL_MODELS:
            for cond, v in PV.build(tpl, people, pool).items():
                text, sol = v.render()
                try: ok = zp.validate_text(text, sol['header'][1:], sol['rows'], 'unique')
                except (ValueError, AssertionError): ok = False
                rows.append(dict(id=f'{pid}|{pool}|{cond}', base_id=pid, size=size, pool=pool, condition=cond, puzzle=text,
                                 solution=json.dumps(sol, ensure_ascii=False), valid=bool(ok), meta=json.dumps(v.meta, ensure_ascii=False)))
    out = pd.DataFrame(rows); out.to_parquet(f'{out_dir}/zebralogic_pools_v2.parquet', index=False)
    print(out.groupby(['pool', 'condition']).valid.agg(['size', 'sum']).to_string())
    return out, people


if __name__ == '__main__':
    main(*sys.argv[1:6])
