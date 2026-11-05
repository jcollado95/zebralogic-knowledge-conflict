"""Generate perturbed ZebraLogic grid_mode variants for the 4*4, 4*5 and 5*4 subset.
Usage: python generate_variants.py data/zebra_grid.parquet out_dir
Every emitted puzzle is re-parsed from its rendered text and solved; `valid` records the outcome."""
import sys, os, json, collections
import pandas as pd
import zebra_perturb as zp

SIZES = ['4*4', '4*5', '5*4']
SEED = 0
CULTURES = ['east_asian', 'south_asian', 'arabic', 'west_african', 'hispanic']


def variants(t, kg, corpus_values):
    o = zp.Variant(t, tags=['original'])
    out = [o]
    # 1. gender / cultural / world-prior name relabelling (logic untouched, isomorphic puzzle)
    out += [zp.gender_all(o, kg, g, SEED) for g in ('female', 'male')]
    out += [zp.gender_congruence(o, kg, m, SEED) for m in ('aligned', 'conflict')]
    out += [zp.prior_congruence(o, kg, a, m, SEED) for a in ('Nationality', 'FavoriteSport') for m in ('aligned', 'conflict')]
    out += [zp.cultural_names(o, kg, c, SEED) for c in CULTURES]
    # 2. symbolic
    out += [zp.symbolic(o, full=False), zp.symbolic(o, full=True)]
    # 3. clue order
    out += [zp.shuffle_clues(o, s) for s in range(3)] + [zp.reorder_clues(o, h) for h in ('reverse', 'unary_first', 'unary_last')]
    # 4. world-implausible
    out += [zp.implausible(o, kg, SEED, 'nouns'), zp.implausible(o, kg, SEED, 'donor', corpus_values)]
    # 5. made-up entities
    out += [zp.madeup(o, SEED)]
    # epistemic humility: no unique solution
    out += [zp.underdetermine(o, SEED), zp.contradict(o, SEED)]
    return [v for v in out if v is not None]


def main(src, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_parquet(src)
    df = df[df['size'].isin(SIZES)]
    recs = [(r.id, r.puzzle, {'header': list(r.solution['header']), 'rows': [list(x) for x in r.solution['rows']]})
            for r in df.itertuples()]
    size = dict(zip(df.id, df['size']))
    T = zp.build_corpus(recs)
    kg = zp.KG(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kg_seed.json'))
    cv = collections.defaultdict(set)
    for t in T:
        for a in t.attrs: cv[a['key']].update(a['values'])
    rows = []
    with open(os.path.join(outdir, 'templates.jsonl'), 'w', encoding='utf-8') as f:
        for t in T: f.write(json.dumps(t.to_json(), ensure_ascii=False) + '\n')
    for t in T:
        for v in variants(t, kg, cv):
            text, sol = v.render()
            ok = zp.validate_text(text, t.keys, sol['rows'] if sol else None, v.expected)
            tag = '+'.join(v.tags)
            rows.append(dict(id=f'{t.id}__{tag}', base_id=t.id, size=size[t.id], perturbation=tag,
                             family=v.tags[-1].split('_')[0] if len(v.tags) > 1 else 'original',
                             expected=v.expected, puzzle=text,
                             solution=json.dumps(sol, ensure_ascii=False) if sol else None,
                             meta=json.dumps(v.meta, ensure_ascii=False), valid=bool(ok)))
    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(outdir, 'zebralogic_perturbed.parquet'), index=False)
    out.to_json(os.path.join(outdir, 'zebralogic_perturbed.jsonl'), orient='records', lines=True, force_ascii=False)
    return out


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
