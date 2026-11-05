"""True-fact vs shuffled-fact ZebraLogic variants from the knowledge-stratified person pools.

For every template (120 originals: 4x4, 4x5, 5x4) and every pool (only_31b, both, none):
  aligned  : real people with their own (salient) Wikidata values
  conflict : same people, same header value lists; every attribute deranged so nobody holds a value that is true of them
Relations: BirthPlace, Occupation, Height (+ EducatedAt for 4-attribute templates) - the facts the models know best.
Per-puzzle draw constraints (n people):
  * shown labels distinct and no label a substring of another (incl. names); no ' and ' / ' is ' / trailing '.'
  * cross-truth: a shown value of person i is not true of any other drawn person j (birthplace via accepted
    names incl. admin parents; occupation also not substring-related, e.g. actor / film actor; heights > 3 cm apart)
  * known pools: every drawn person is known by the pool's model(s) on >= 2 of the puzzle's relations
Usage: python pool_variants.py zebralogic_perturbed.parquet persons_pools.csv out_dir
"""
import sys, json, itertools, hashlib, random, copy, re, os
import pandas as pd
import zebra_perturb as zp

REL = {'place of birth': ('BirthPlace', 'Each person was born in a different place', 'the person who was born in {v}'),
       'occupation': ('Occupation', 'Each person has a different occupation', 'the person who works as a {v}'),
       'educated at': ('EducatedAt', 'Each person studied at a different institution', 'the person who studied at {v}'),
       'height (cm)': ('Height', 'Each person has a different height in cm', 'the person who is {v} cm tall')}
BASE_RELS = ['place of birth', 'occupation', 'height (cm)']; EXTRA_REL = 'educated at'
POOL_MODELS = {'only_31b': ['31b'], 'both': ['31b', 'e4b'], 'none': []}
BAD = re.compile(r" and | is |\.$")
# shown birthplaces that are memorials/buildings without a usable admin parent (e.g. 'Bhim Janmabhoomi') -> person excluded
NONPLACE = re.compile(r"janmabhoomi|memorial|\bsite\b|hospital|\bhouse\b|palace|\bestate\b|cent(er|re)\b|clinic|castle|manor|\bhall\b|residence|museum|birthplace", re.I)


def rng_for(*parts):
    return random.Random(int(hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()[:12], 16))


def load_people(path):
    P = pd.read_csv(path, encoding='utf-8-sig'); out = []
    for r in P.to_dict('records'):
        d = dict(qid=r['qid'], person=r['person'], pool=r['pool'], sitelinks=r['sitelinks'], shown={}, true={}, knows={}, shown_known={})
        for c in REL:
            d['shown'][c] = r[f'{c} (salient)'] if c != 'height (cm)' else str(r[f'{c} (salient)'])
            d['true'][c] = set(json.loads(r['place of birth (accepted)']) + json.loads(r['place of birth'])) if c == 'place of birth' \
                else set(map(str, json.loads(r[c])))
            for t in ('e4b', '31b'):
                d['knows'][(c, t)] = bool(r[f'knows_{c}_{t}'])
                d['shown_known'][(c, t)] = d['shown'][c] in set(map(str, json.loads(r[f'known_values_{c}_{t}'])))
        if not NONPLACE.search(d['shown']['place of birth']): out.append(d)
    return out


def _sub(a, b):
    a, b = a.lower(), b.lower(); return a in b or b in a


def compatible(group, rels):
    labels = [p['person'] for p in group] + [p['shown'][c] for p in group for c in rels]
    if any(BAD.search(x) for x in labels): return False
    low = [x.lower() for x in labels]
    if len(set(low)) < len(low) or any(i != j and low[i] in low[j] for i in range(len(low)) for j in range(len(low))): return False
    for p, q in itertools.permutations(group, 2):
        for c in rels:
            v = p['shown'][c]
            if c == 'height (cm)':
                if any(abs(int(v) - int(h)) <= 3 for h in q['true'][c]): return False
            elif c == 'occupation':
                if any(_sub(v, t) for t in q['true'][c]): return False
            elif v.lower() in {t.lower() for t in q['true'][c]}: return False
    return True


def eligible(p, pool, rels):
    ms = POOL_MODELS[pool]
    return not ms or sum(all(p['knows'][(c, t)] for t in ms) for c in rels) >= 2


def draw(tid, pool, people, n, rels, tries=20000):
    rng = rng_for(tid, pool, 'people'); cands = [p for p in people if p['pool'] == pool and eligible(p, pool, rels)]
    for _ in range(tries):
        G = rng.sample(cands, n)
        if compatible(G, rels): return G
    raise ValueError(f'no compatible draw for {tid} {pool}')


def conflict_perms(tid, n, k):
    rng = rng_for(tid, 'conflict'); der = [p for p in itertools.permutations(range(n)) if all(p[i] != i for i in range(n))]
    best = None
    for _ in range(3000):
        S = [rng.choice(der) for _ in range(k)]
        cost = sum(a[i] == b[i] for a, b in itertools.combinations(S, 2) for i in range(n))
        if best is None or cost < best[0]: best = (cost, S)
        if cost == 0: break
    return best


def build(t, people, pool):
    nonname = [a['key'] for a in t.attrs if a['key'] != 'Name']; n = t.n
    rels = BASE_RELS + ([EXTRA_REL] if len(nonname) == 4 else [])
    assert len(rels) == len(nonname), (t.id, nonname)
    rng_for(t.id, 'relorder').shuffle(rels)                    # which template attribute gets which relation
    G = draw(t.id, pool, people, n, rels); g = t.gold_pos()
    house_of = {k: {x: g[(k, x)] for x in t.attrs[t.keys.index(k)]['values']} for k in t.keys}
    base = zp.Variant(t)
    for k, c in zip(nonname, rels):
        key, line, phrase = REL[c]; base.retype[k] = dict(key=key, line=line, phrase=phrase)
    names = {x: G[h - 1]['person'] for x, h in house_of['Name'].items()}
    al = base.fork('pool_aligned'); al.relabel['Name'] = dict(names)
    for k, c in zip(nonname, rels): al.relabel[k] = {x: G[h - 1]['shown'][c] for x, h in house_of[k].items()}
    cost, S = conflict_perms(t.id, n, len(nonname))
    cf = base.fork('pool_conflict'); cf.relabel['Name'] = dict(names)
    for (k, c), perm in zip(zip(nonname, rels), S):
        cf.relabel[k] = {x: G[perm[h - 1]]['shown'][c] for x, h in house_of[k].items()}
    t2 = copy.deepcopy(t)                                      # identical header value order in both conditions
    for a, a2 in zip(t.attrs, t2.attrs):
        if a['key'] == 'Name': continue
        shown = [al.val(a['key'], x) for x in a['values']]; inv = {v: o for o, v in cf.relabel[a['key']].items()}
        a2['values'] = [inv[s] for s in shown]
    cf.tpl = t2
    # per-cell knowledge annotation: does the model know person p's true fact for relation c?
    cells = [dict(person=p['person'], qid=p['qid'], relation=REL[c][0],
                  knows_e4b=p['knows'][(c, 'e4b')], knows_31b=p['knows'][(c, '31b')],
                  shown_known_e4b=p['shown_known'][(c, 'e4b')], shown_known_31b=p['shown_known'][(c, '31b')])
             for p in G for c in rels]
    meta = dict(pool=pool, relations=[REL[c][0] for c in rels], people=[p['person'] for p in G], qids=[p['qid'] for p in G],
                sitelinks=[int(p['sitelinks']) for p in G], cells=cells)
    al.meta.update(meta); cf.meta.update(meta, cross_attr_coincidences=cost)
    return {'aligned': al, 'conflict': cf}


def main(src, pools_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(src); o = df[df.perturbation == 'original']
    recs = [(r.base_id, r.puzzle, json.loads(r.solution)) for r in o.itertuples()]
    tpls = zp.build_corpus(recs); people = load_people(pools_csv); rows = []
    for tpl, (pid, _, _) in zip(tpls, recs):
        size = '*'.join(pid.split('-')[2].split('x'))
        for pool in POOL_MODELS:
            for cond, v in build(tpl, people, pool).items():
                text, sol = v.render()
                try: ok = zp.validate_text(text, sol['header'][1:], sol['rows'], 'unique')
                except (ValueError, AssertionError): ok = False
                rows.append(dict(id=f'{pid}|{pool}|{cond}', base_id=pid, size=size, pool=pool, condition=cond, puzzle=text,
                                 solution=json.dumps(sol, ensure_ascii=False), valid=bool(ok), meta=json.dumps(v.meta, ensure_ascii=False)))
    out = pd.DataFrame(rows); out.to_parquet(f'{out_dir}/zebralogic_pools.parquet', index=False)
    print(out.groupby(['pool', 'condition']).valid.agg(['size', 'sum']).to_string())
    return out


if __name__ == '__main__':
    main(*sys.argv[1:4])
