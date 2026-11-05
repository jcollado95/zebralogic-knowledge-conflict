"""Generate Wikidata-backed ZebraLogic variants (4*4, 4*5, 5*4).
Usage: python generate_wd_variants.py data/zebra_grid.parquet wikidata_kg_snapshot.json out_wd --samples 3
`--samples K` draws K independent samples (seeds 0..K-1) per operator per puzzle; raise it for more data.
Each emitted puzzle is re-parsed from its rendered text and solved (`valid`)."""
import sys, os, json, argparse, collections
import pandas as pd
import zebra_perturb as zp
import wikidata_kg as wk

SIZES = ['4*4', '4*5', '5*4']
LANGS = ['english', 'spanish', 'german', 'french', 'italian', 'swedish', 'polish', 'russian', 'turkish', 'arabic', 'japanese']


def operators(o, kg, s):
    yield zp.wd_relabel(o, kg, s)
    yield zp.wd_relabel(o, kg, s, top=15)                      # only the most-linked (typical) values
    for g in ('female', 'male'): yield zp.wd_names(o, kg, s, gender=g)
    for lang in LANGS: yield zp.wd_names(o, kg, s, lang=lang)
    for m in ('aligned', 'conflict'): yield zp.wd_gender_occupation(o, kg, m, s)
    for rel in kg.people:
        if kg.people[rel]['persons']:
            for m in ('aligned', 'conflict'): yield zp.wd_grounded(o, kg, rel, m, s)


def main(src, kg_path, outdir, samples):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_parquet(src); df = df[df['size'].isin(SIZES)]
    recs = [(r.id, r.puzzle, {'header': list(r.solution['header']), 'rows': [list(x) for x in r.solution['rows']]}) for r in df.itertuples()]
    size = dict(zip(df.id, df['size'])); T = zp.build_corpus(recs); kg = wk.WikidataKG.load(kg_path)
    rows, failed = [], collections.Counter()
    for t in T:
        o = zp.Variant(t, tags=['original'])
        for s in range(samples):
            for v in operators(o, kg, s):
                if v is None: failed['none'] += 1; continue
                text, sol = v.render()
                try: ok = zp.validate_text(text, v.keys(), sol['rows'] if sol else None, v.expected)
                except ValueError: ok = False
                tag = '+'.join(v.tags[1:])
                rows.append(dict(id=f'{t.id}__{tag}', base_id=t.id, size=size[t.id], perturbation=tag.rsplit('_s', 1)[0],
                                 seed=s, expected=v.expected, puzzle=text, solution=json.dumps(sol, ensure_ascii=False),
                                 meta=json.dumps(v.meta, ensure_ascii=False), valid=bool(ok), kg_retrieved=kg.retrieved))
    out = pd.DataFrame(rows).drop_duplicates('puzzle')
    out.to_parquet(os.path.join(outdir, 'zebralogic_wikidata.parquet'), index=False)
    return out, failed


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('kg'); ap.add_argument('out'); ap.add_argument('--samples', type=int, default=3)
    a = ap.parse_args(); out, failed = main(a.src, a.kg, a.out, a.samples)
    print(out.groupby('perturbation').agg(n=('valid', 'size'), valid=('valid', 'sum')).to_string()); print('not generated:', dict(failed))
