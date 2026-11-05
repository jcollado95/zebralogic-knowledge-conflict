"""True-fact vs shuffled-fact ZebraLogic variants, v3 (MCQA-defined pools; config-driven).

Templates : ZebraLogic grid_mode 4*4, 5*4, 6*4 (40 each; 3 non-Name attributes)
Relations : BirthPlace, Occupation, BirthYear (pool_variants_v2 phrasing and constraints: distinct labels, no substrings,
            cross-truth rules, birth years >= 3 apart, identical header order in both conditions)
People    : pools_v3_<family>.csv, column pool_final (strict MCQA rule); pools only_large / both / none
Draws     : `draws` people sets per template and pool (default 1)
In-puzzle MCQA (optional, run_inpuzzle_mcqa): for every person x fact in each (template, pool, draw), one brief-recall
            question whose options are that puzzle's own values for the column, in header order (n = number of houses).
Usage     : python pool_variants_v3.py zebra_grid.parquet candidates_v3.csv pools/pools_v3_gemma4.csv out [draws]
"""
import sys, os, json, re, time, threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import zebra_perturb as zp
import pool_variants_v2 as V2            # patches pool_variants: relations, year gap, cross-truth
PV = V2.PV
import mcqa_probe as M

POOLS = ['only_large', 'both', 'none']
PV.POOL_MODELS = {p: [] for p in POOLS}   # pools are already knowledge-defined; no extra per-person eligibility
LETTERS = 'ABCDEF'
FACT_OF = {'place of birth': 'birthplace', 'occupation': 'occupation', 'birth year': 'birth_year'}


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
            for t, suf in (('e4b', 'small'), ('31b', 'large')):   # tags used by pool_variants.build's cell annotation
                d['knows'][(rel, t)] = d['shown_known'][(rel, t)] = bool(r[f'knows_{f}_{suf}'])
        out.append(d)
    return out


def eligible(p, pool, rels): return True
PV.eligible = eligible


def build_all(grid, people, draws=1):
    g = pd.read_parquet(grid); g = g[g['size'].isin(V2.SIZES)]
    recs = [(r.id, r.puzzle, dict(header=list(r.solution['header']), rows=[list(map(str, x)) for x in r.solution['rows']])) for r in g.itertuples()]
    tpls = zp.build_corpus(recs); rows = []
    for tpl, (pid, _, _) in zip(tpls, recs):
        size = '*'.join(pid.split('-')[2].split('x'))
        for pool in POOLS:
            for d in range(draws):
                t2 = tpl if d == 0 else type(tpl)(**{**tpl.__dict__, 'id': f'{tpl.id}#d{d}'})   # new seed per draw
                for cond, v in PV.build(t2, people, pool).items():
                    text, sol = v.render()
                    try: ok = zp.validate_text(text, sol['header'][1:], sol['rows'], 'unique')
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
                rel = {v[0]: c for c, v in PV.REL.items()}[k]; opts = vals[k]
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
