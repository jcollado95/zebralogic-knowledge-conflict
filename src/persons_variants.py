"""ZebraLogic variants grounded in real people (unique_persons.csv: people whose 5 Wikidata facts are
pairwise distinct across the list).

Every non-Name attribute of a ZebraLogic template is retyped to one of the five person relations, so all
conditions share identical clue wording and logic; only the entity labels differ.
  aligned  : real names, every person holds their own true values (world knowledge agrees with the gold)
  conflict : real names, every attribute is a derangement w.r.t. the names (no person holds their own
             value) and same-person co-occurrences between attribute columns are minimised
  control  : same attribute values as aligned, original fictional ZebraLogic names (no knowledge link)
Usage: python persons_variants.py zebralogic_perturbed.parquet unique_persons.csv out_dir
"""
import sys, json, csv, itertools, hashlib, random
import pandas as pd
import zebra_perturb as zp

REL = {  # column in CSV -> (solution key, header line, clue phrase)
    'place of birth': ('BirthPlace', 'Each person was born in a different place', 'the person who was born in {v}'),
    'occupation': ('Occupation', 'Each person has a different occupation', 'the person who works as a {v}'),
    'mother': ('Mother', 'Each person has a different mother', 'the person whose mother is {v}'),
    'educated at': ('EducatedAt', 'Each person studied at a different institution', 'the person who studied at {v}'),
    'height (cm)': ('Height', 'Each person has a different height in cm', 'the person who is {v} cm tall'),
}


def rng_for(pid, salt):
    return random.Random(int(hashlib.sha1(f'{pid}|{salt}'.encode()).hexdigest()[:12], 16))


def load_people(path):
    with open(path, encoding='utf-8-sig') as f:
        return [dict(r) for r in csv.DictReader(f)]


def clash(labels):
    """substring overlap between labels, or a label containing ' and ' / ' is ' (breaks clue grammar)"""
    low = [x.lower() for x in labels]
    return any(' and ' in a or ' is ' in a or a.endswith('.') for a in low) or any(a != b and a in b for a in low for b in low)


def draw(pid, people, n, rels, orig_names, tries=5000):
    """n people + chosen relations with no substring clashes among all labels shown in the puzzle
    (also checked against the fictional control names)."""
    rng = rng_for(pid, 'people')
    for _ in range(tries):
        P = rng.sample(people, n)
        labels = [p['person'] for p in P] + [p[c] for p in P for c in rels]
        if len({x.lower() for x in labels}) == len(labels) and not clash(labels) \
                and not clash(orig_names + [p[c] for p in P for c in rels]):
            return P
    raise ValueError('no clash-free draw')  # pragma: no cover


def conflict_perms(pid, n, k):
    """k derangements of range(n) (attribute column j at house-slot i gets person perm[j][i]);
    minimise pairs of columns that place the same person's values together."""
    rng = rng_for(pid, 'conflict'); der = [p for p in itertools.permutations(range(n)) if all(p[i] != i for i in range(n))]
    best = None
    for _ in range(3000):
        S = [rng.choice(der) for _ in range(k)]
        cost = sum(a[i] == b[i] for a, b in itertools.combinations(S, 2) for i in range(n))
        if best is None or cost < best[0]: best = (cost, S)
        if cost == 0: break
    return best


def build(tpl, people, seed=0):
    t = tpl; n = t.n; nonname = [a['key'] for a in t.attrs if a['key'] != 'Name']
    rels = rng_for(t.id, 'rels').sample(list(REL), len(nonname))
    P = draw(t.id, people, n, rels, t.attrs[0]['values'])
    g = t.gold_pos()
    # house h (1..n) <- person P[h-1] in the aligned world
    house_of = {k: {x: g[(k, x)] for x in t.attrs[t.keys.index(k)]['values']} for k in t.keys}
    base = zp.Variant(t)
    for k, c in zip(nonname, rels):
        key, line, phrase = REL[c]; base.retype[k] = dict(key=key, line=line, phrase=phrase)
    out = {}
    names_real = {x: P[h - 1]['person'] for x, h in house_of['Name'].items()}
    # aligned
    v = base.fork('persons_aligned'); v.relabel['Name'] = dict(names_real)
    for k, c in zip(nonname, rels): v.relabel[k] = {x: P[h - 1][c] for x, h in house_of[k].items()}
    out['aligned'] = v
    # control: fictional names, same attribute values
    v = base.fork('persons_control')
    for k, c in zip(nonname, rels): v.relabel[k] = {x: P[h - 1][c] for x, h in house_of[k].items()}
    out['control'] = v
    # conflict
    cost, S = conflict_perms(t.id, n, len(nonname))
    v = base.fork('persons_conflict'); v.relabel['Name'] = dict(names_real)
    for (k, c), perm in zip(zip(nonname, rels), S):
        v.relabel[k] = {x: P[perm[h - 1]][c] for x, h in house_of[k].items()}
    v.meta['cross_attr_coincidences'] = cost
    # show header values in the same order as the aligned variant (only the house assignment differs)
    import copy
    t2 = copy.deepcopy(t); al = out['aligned']
    for a, a2 in zip(t.attrs, t2.attrs):
        if a['key'] == 'Name': continue
        shown = [al.val(a['key'], x) for x in a['values']]
        inv = {new: old for old, new in v.relabel[a['key']].items()}
        a2['values'] = [inv[lbl] for lbl in shown]
    v.tpl = t2
    out['conflict'] = v
    for cond, v in out.items():
        v.meta.update(relations=rels, people=[p['person'] for p in P])
    return out


def main(src, persons_csv, out_dir):
    import os; os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(src); o = df[df.perturbation == 'original']
    recs = [(r.base_id, r.puzzle, json.loads(r.solution)) for r in o.itertuples()]
    tpls = zp.build_corpus(recs); people = load_people(persons_csv)
    truth = {p['person']: {REL[c][0]: p[c] for c in REL} for p in people}
    rows = []
    for tpl, (pid, _, _) in zip(tpls, recs):
        size = '*'.join(pid.split('-')[2].split('x'))
        for cond, v in build(tpl, people).items():
            text, sol = v.render()
            try: ok = zp.validate_text(text, sol['header'][1:], sol['rows'], 'unique')
            except (ValueError, AssertionError): ok = False
            rows.append(dict(id=f'{pid}|persons_{cond}', base_id=pid, size=size, condition=cond, puzzle=text,
                             solution=json.dumps(sol, ensure_ascii=False), valid=bool(ok),
                             meta=json.dumps(v.meta, ensure_ascii=False)))
    out = pd.DataFrame(rows)
    out.to_parquet(f'{out_dir}/zebralogic_persons.parquet', index=False)
    json.dump(truth, open(f'{out_dir}/persons_truth.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(out.groupby('condition').valid.agg(['size', 'sum']))
    return out


if __name__ == '__main__':
    main(*sys.argv[1:4])
