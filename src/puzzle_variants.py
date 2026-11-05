"""True-fact vs shuffled-fact ZebraLogic variants with real people from knowledge-defined pools.

Templates : ZebraLogic grid_mode 4*4, 5*4, 6*4 (40 each; 3 non-Name attributes)
Relations : BirthPlace, Occupation, BirthYear; every template attribute is retyped to one of them
            (assignment seeded per template)
People    : pools_<family>.csv (from build_pools.py), column pool_final (strict MCQA rule):
            only_large (only the large model knows), both, none
Conditions: aligned  = real people with their own facts (gold solution = real-world truth)
            conflict = same people, same header value order; every attribute deranged so nobody holds a value
                       that is true of them (cross-attribute coincidences minimised)
Per-puzzle draw constraints (n people):
  * shown labels distinct, no label a substring of another (names included); no ' and ' / ' is ' / trailing '.'
  * cross-truth: a shown value of person i is not true of any other drawn person j (birthplace via accepted names
    incl. admin parents; occupations not substring-related, e.g. actor / film actor)
  * birth years at least MIN_YEAR_GAP apart
Draws     : `draws` people sets per template and pool (default 1)
Every variant is re-parsed from its rendered text and solved (zebra_core.validate_text): exactly one solution,
equal to the gold.
In-puzzle MCQA (run_inpuzzle_mcqa): for every person x fact in each (template, pool, draw), one brief-recall
question whose options are that puzzle's own values for the column, in header order.
Usage     : python puzzle_variants.py zebra_grid.parquet candidates_v3.csv pools_v3_gemma4.csv out_dir [draws]
"""
import sys, os, json, re, time, threading, itertools, hashlib, random, copy
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import zebra_core as zp
import mcqa_probe as M

REL = {'place of birth': ('BirthPlace', 'Each person was born in a different place', 'the person who was born in {v}'),
       'occupation': ('Occupation', 'Each person has a different occupation', 'the person who works as a {v}'),
       'birth year': ('BirthYear', 'Each person was born in a different year', 'the person who was born in the year {v}')}
BASE_RELS = ['place of birth', 'occupation', 'birth year']
SIZES = ['4*4', '5*4', '6*4']
POOLS = ['only_large', 'both', 'none']
MIN_YEAR_GAP = 3
BAD = re.compile(r" and | is |\.$")
LETTERS = 'ABCDEF'
FACT_OF = {'place of birth': 'birthplace', 'occupation': 'occupation', 'birth year': 'birth_year'}


def rng_for(*parts):
    return random.Random(int(hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()[:12], 16))


def load_people(cand_csv, pools_csv):
    C = pd.read_csv(cand_csv, encoding='utf-8-sig').set_index('qid'); P = pd.read_csv(pools_csv, encoding='utf-8-sig').set_index('qid')
    out = []
    for q, r in P[P.pool_final.notna()].iterrows():
        c = C.loc[q]
        d = dict(qid=q, person=c.person, pool=r.pool_final, sitelinks=int(c.sitelinks), shown={}, true={}, knows={}, shown_known={})
        d['shown']['place of birth'] = c.birthplace; d['true']['place of birth'] = set(json.loads(c.birthplace_accepted)) | {c.birthplace}
        d['shown']['occupation'] = c.occupation; d['true']['occupation'] = set(json.loads(c.occupations_all)) | set(json.loads(c.occupation_aliases)) | {c.occupation}
        d['shown']['birth year'] = str(int(c.birth_year)); d['true']['birth year'] = {str(int(c.birth_year))}
        for rel, f in FACT_OF.items():
            for t, suf in (('e4b', 'small'), ('31b', 'large')):   # cell-annotation tags: e4b = small, 31b = large model
                d['knows'][(rel, t)] = d['shown_known'][(rel, t)] = bool(r[f'knows_{f}_{suf}'])
        out.append(d)
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
            if c == 'occupation':
                if any(_sub(v, t) for t in q['true'][c]): return False
            elif v.lower() in {t.lower() for t in q['true'][c]}: return False
    ys = [int(p['shown']['birth year']) for p in group]
    return all(abs(a - b) >= MIN_YEAR_GAP for i, a in enumerate(ys) for b in ys[i + 1:])


def draw(tid, pool, people, n, rels, tries=20000):
    rng = rng_for(tid, pool, 'people'); cands = [p for p in people if p['pool'] == pool]
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
    rels = list(BASE_RELS)
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
    # per-cell knowledge annotation: does each model know person p's fact for relation c?
    cells = [dict(person=p['person'], qid=p['qid'], relation=REL[c][0],
                  knows_e4b=p['knows'][(c, 'e4b')], knows_31b=p['knows'][(c, '31b')],
                  shown_known_e4b=p['shown_known'][(c, 'e4b')], shown_known_31b=p['shown_known'][(c, '31b')])
             for p in G for c in rels]
    meta = dict(pool=pool, relations=[REL[c][0] for c in rels], people=[p['person'] for p in G], qids=[p['qid'] for p in G],
                sitelinks=[int(p['sitelinks']) for p in G], cells=cells)
    al.meta.update(meta); cf.meta.update(meta, cross_attr_coincidences=cost)
    return {'aligned': al, 'conflict': cf}


def build_all(grid, people, draws=1):
    g = pd.read_parquet(grid); g = g[g['size'].isin(SIZES)]
    recs = [(r.id, r.puzzle, dict(header=list(r.solution['header']), rows=[list(map(str, x)) for x in r.solution['rows']])) for r in g.itertuples()]
    tpls = zp.build_corpus(recs); rows = []
    for tpl, (pid, _, _) in zip(tpls, recs):
        size = '*'.join(pid.split('-')[2].split('x'))
        for pool in POOLS:
            for d in range(draws):
                t2 = tpl if d == 0 else type(tpl)(**{**tpl.__dict__, 'id': f'{tpl.id}#d{d}'})   # new seed per draw
                for cond, v in build(t2, people, pool).items():
                    text, sol = v.render()
                    try: ok = zp.validate_text(text, sol['header'][1:], sol['rows'])
                    except (ValueError, AssertionError): ok = False
                    rows.append(dict(id=f'{pid}|{pool}|d{d}|{cond}', base_id=pid, size=size, pool=pool, draw=d, condition=cond,
                                     puzzle=text, solution=json.dumps(sol, ensure_ascii=False), valid=bool(ok), meta=json.dumps(v.meta, ensure_ascii=False)))
    return pd.DataFrame(rows)


def inpuzzle_questions(df):
    """one question per (template, pool, draw, person, fact), options = the aligned puzzle's header values"""
    Q = []
    for r in df[df.condition == 'aligned'].itertuples():
        sol = json.loads(r.solution); keys = sol['header'][1:]
        hdr = [l for l in r.puzzle.split('\n') if l.startswith(' - ')]
        vals = {k: re.findall(r'`([^`]*)`', l) for k, l in zip(keys, hdr)}
        for row in sol['rows']:
            d = dict(zip(keys, row[1:]))
            for k in keys[1:]:
                rel = {v[0]: c for c, v in REL.items()}[k]; opts = vals[k]
                Q.append(dict(puzzle_key=f'{r.base_id}|{r.pool}|d{r.draw}', person=d['Name'], fact=FACT_OF[rel], options=opts,
                              answer=LETTERS[opts.index(d[k])]))
    return Q


def _prompt(q):
    return M.FACTS[q['fact']].format(p=q['person']) + '\n' + '\n'.join(f'{L}. {o}' for L, o in zip(LETTERS, q['options'])) + '\n' + M.INSTRS['brief']


def _parse(text, n):
    t = (text or '').strip(); ok = LETTERS[:n]
    m = re.findall(rf'answer\s*[:\-]?\s*\**\(?([{ok}])\b', t, re.I)
    if m: return m[-1].upper()
    last = [l.strip(' *.()') for l in t.splitlines() if l.strip()][-1:] or ['']
    return last[0] if last[0] in ok else None


def run_inpuzzle_mcqa(cfg, Q, out_path, models=None, log=print):
    key = os.environ[cfg.get('api_key_env', 'VLLM_API_KEY')]
    for m in cfg['models']:
        if models and m['name'] not in models: continue
        path = out_path.format(model=m['name'].replace('/', '_')); done = set()
        if os.path.exists(path):
            for l in open(path, encoding='utf-8'):
                r = json.loads(l)
                if not r['raw'].startswith('ERROR'): done.add((r['puzzle_key'], r['person'], r['fact']))
        todo = [q for q in Q if (q['puzzle_key'], q['person'], q['fact']) not in done]; lock = threading.Lock(); t0 = time.time()
        log(f"{m['name']}: {len(done)} done, {len(todo)} to run")

        def job(q):
            raw = M.ask(cfg['endpoint'], key, m['name'], _prompt(q), max_tokens=80); L = _parse(raw, len(q['options']))
            rec = dict(model=m['name'], **q, raw=raw.strip()[:300], letter=L, correct=L == q['answer'])
            with lock:
                with open(path, 'a', encoding='utf-8') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        with ThreadPoolExecutor(cfg.get('workers', 24)) as ex: list(ex.map(job, todo))
        log(f"{m['name']}: finished {len(todo)} in {time.time() - t0:.0f}s")


if __name__ == '__main__':
    grid, cand, pools, out = sys.argv[1:5]; draws = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    os.makedirs(out, exist_ok=True); people = load_people(cand, pools)
    df = build_all(grid, people, draws); df.to_parquet(f'{out}/zebralogic_pools_v3.parquet', index=False)
    print(df.groupby(['pool', 'condition']).valid.agg(['size', 'sum']).to_string())
