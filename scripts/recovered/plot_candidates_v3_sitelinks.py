"""Recovered notebook code that produced `candidates_v3_sitelinks.png`.

Reconstructed from session lineage, not written as a standalone script.
Expect adjustments (input paths, working directory via WORKDIR) before rerunning.
Steps that restored files from the artifact store are replaced by `pass`;
those files are under src/ and data/ in this repository.
"""
import os, re, json, time, urllib.request, urllib.parse, urllib.error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# skill:figure-style kernel.py (auto-injected on skill load)
META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")
    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
    import matplotlib.pyplot as plt
    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 8) + 1
    s = letter.lower() if case == "lower" else letter.upper()
    ax.text(dx, dy, s, transform=ax.transAxes,
            fontweight="bold", fontsize=fontsize, va="bottom", ha="left")


EP = "https://qlever.dev/api/wikidata"
UA = "unique-persons-script/1.0 (Wikidata research script)"


def sparql(q, tries=6, timeout=600):
    for k in range(tries):
        try:
            req = urllib.request.Request(EP, data=urllib.parse.urlencode({'query': q}).encode(),
                                         headers={"Accept": "application/sparql-results+json", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return [{kk: c['value'] for kk, c in row.items()} for row in json.load(r)['results']['bindings']]
        except urllib.error.HTTPError as e:
            if e.code == 429 and k < tries - 1:
                time.sleep(20 * (k + 1))
                continue
            raise


PFX = """PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#>"""

BANDS = [('f_100+', 100, 100000, 1000), ('e_50-99', 50, 100, 1000), ('d_20-49', 20, 50, 1000),
         ('c_10-19', 10, 20, 800), ('b_5-9', 5, 10, 600), ('a_0-4', 0, 5, 600)]

samp = []
t = time.time()
for name, lo, hi, n in BANDS:
    q = PFX + f"""SELECT DISTINCT ?p ?sl WHERE {{ ?p wdt:P31 wd:Q5 ; wdt:P19 ?b ; wdt:P106 ?o ; wdt:P569 ?d ; wikibase:sitelinks ?sl .
      FILTER(?sl >= {lo} && ?sl < {hi}) BIND(MD5(STR(?p)) AS ?h) }} ORDER BY ?h LIMIT {n}"""
    rows = sparql(q)
    samp += [dict(qid=r['p'].rsplit('/', 1)[-1], sitelinks=int(r['sl']), band=name) for r in rows]
    print(name, len(rows), round(time.time() - t, 1), 's')
    time.sleep(3)

S = pd.DataFrame(samp).drop_duplicates('qid')

def band_of(sl):
    for name, lo, hi, _ in BANDS:
        if lo <= sl < hi:
            return name


qid_of = lambda iri: iri.rsplit('/', 1)[-1]
PROPS = {'P19': 'bp', 'P106': 'occ', 'P569': 'dob'}
vals = defaultdict(lambda: defaultdict(set))
C = S.copy()
qids = C.qid.tolist()

t = time.time()
for s in range(0, len(qids), 500):
    rows = sparql(PFX + f"""PREFIX p: <http://www.wikidata.org/prop/> PREFIX ps: <http://www.wikidata.org/prop/statement/>
      SELECT ?p ?prop ?v WHERE {{ VALUES ?p {{ {' '.join('wd:' + q for q in qids[s:s + 500])} }}
      VALUES ?prop {{ wdt:P19 wdt:P106 wdt:P569 }} ?p ?prop ?v }}""")
    for r in rows:
        n = PROPS[r['prop'].rsplit('/', 1)[-1]]
        v = r['v']
        if n == 'dob':
            m = re.match(r'^(-?\d{1,4})-', v)
            if m:
                vals[qid_of(r['p'])]['year'].add(int(m.group(1)))
        elif v.startswith('http://www.wikidata.org/entity/Q'):
            vals[qid_of(r['p'])][n].add(qid_of(v))
    time.sleep(2)

ents = sorted(set(qids) | {v for q in qids for k in ['bp', 'occ'] for v in vals[q][k]})
lab = {}
alias = defaultdict(set)


def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


t = time.time()
for ch in chunked(ents, 1500):
    vs = ' '.join('wd:' + q for q in ch)
    for r in sparql(PFX + f"""SELECT ?i ?en ?mul WHERE {{ VALUES ?i {{ {vs} }} OPTIONAL {{ ?i rdfs:label ?en FILTER(LANG(?en)="en") }} OPTIONAL {{ ?i rdfs:label ?mul FILTER(LANG(?mul)="mul") }} }}"""):
        if r.get('en') or r.get('mul'):
            lab[qid_of(r['i'])] = r.get('en') or r.get('mul')
    time.sleep(2)
    for r in sparql(PFX + f"""SELECT ?i ?a WHERE {{ VALUES ?i {{ {vs} }} ?i skos:altLabel ?a FILTER(LANG(?a)="en") }}"""):
        alias[qid_of(r['i'])].add(r['a'])
    time.sleep(2)

bps = sorted({v for q in qids for v in vals[q]['bp']})
types = defaultdict(set)
par1 = defaultdict(set)
par2 = defaultdict(set)
for ch in chunked(bps, 800):
    for r in sparql(PFX + f"""SELECT ?b ?t ?a1 ?a2 WHERE {{ VALUES ?b {{ {' '.join('wd:' + q for q in ch)} }} OPTIONAL {{ ?b wdt:P31 ?t }} OPTIONAL {{ ?b wdt:P131 ?a1 . OPTIONAL {{ ?a1 wdt:P131 ?a2 }} }} }}"""):
        b = qid_of(r['b'])
        for k, d in [('t', types), ('a1', par1), ('a2', par2)]:
            if r.get(k):
                d[b].add(qid_of(r[k]))
    time.sleep(2)

need = sorted(({x for d in (types, par1, par2) for s_ in d.values() for x in s_}) - set(lab))
for ch in chunked(need, 1500):
    for r in sparql(PFX + f"""SELECT ?i ?en WHERE {{ VALUES ?i {{ {' '.join('wd:' + q for q in ch)} }} ?i rdfs:label ?en FILTER(LANG(?en)="en") }}"""):
        lab[qid_of(r['i'])] = r['en']
    time.sleep(2)

NONPLACE_T = re.compile(r'hospital|building|\bhouse\b|clinic|maternity|museum|palace|castle|residence|structure|birthplace|birth house|medical|manor|\bestate\b|\bvilla\b|apartment|historic site|hotel|prison', re.I)
NONPLACE_L = re.compile(r"janmabhoomi|memorial|\bsite\b|hospital|\bhouse\b|palace|\bestate\b|cent(er|re)\b|clinic|castle|manor|\bhall\b|residence|museum|birthplace", re.I)
BAD = re.compile(r" and | is |\.$")


def is_nonplace(b):
    return any(NONPLACE_T.search(lab.get(t) or '') for t in types[b]) or bool(NONPLACE_L.search(lab.get(b) or ''))


def bp_disp(b):
    if is_nonplace(b) and par1[b]:
        c = sorted(lab[a] for a in par1[b] if a in lab)
        return c[0] if c else lab.get(b)
    return lab.get(b)


def bp_acc(b):
    acc = {lab.get(b)} | alias[b] | {lab.get(a) for a in par1[b]}
    if is_nonplace(b):
        acc |= {lab.get(a) for a in par2[b]}
    return {x for x in acc if x}


occ_freq = Counter(lab[o] for q in qids for o in vals[q]['occ'] if o in lab)
rows = []
drop = Counter()
for _, r in C.iterrows():
    q = r.qid
    v = vals[q]
    name = lab.get(q)
    if not (v['bp'] and v['occ'] and v['year']):
        drop['missing fact'] += 1
        continue
    if not name or re.match(r'^Q\d+$', name):
        drop['unlabelled name'] += 1
        continue
    if len(v['year']) > 1:
        drop['conflicting birth year'] += 1
        continue
    if len(v['bp']) > 1:
        drop['several birthplaces'] += 1
        continue
    if ' ' not in name.strip():
        drop['one-word name'] += 1
        continue
    b = next(iter(v['bp']))
    bdisp = bp_disp(b)
    occs = sorted({lab[o] for o in v['occ'] if o in lab})
    occs_ok = [o for o in occs if not BAD.search(o)]
    if not bdisp or not occs_ok:
        drop['unlabelled value'] += 1
        continue
    if BAD.search(name) or BAD.search(bdisp):
        drop['clue-grammar label'] += 1
        continue
    if NONPLACE_L.search(bdisp):
        drop['non-settlement birthplace'] += 1
        continue
    occ_disp = max(occs_ok, key=lambda o: (occ_freq[o], o))
    rows.append(dict(qid=q, person=name, sitelinks=r.sitelinks, band=r.band,
                     birthplace=bdisp, birthplace_accepted=json.dumps(sorted(bp_acc(b) | {bdisp}), ensure_ascii=False),
                     occupation=occ_disp, occupations_all=json.dumps(occs, ensure_ascii=False),
                     occupation_aliases=json.dumps(sorted({a for o in v['occ'] for a in alias[o]}), ensure_ascii=False),
                     birth_year=next(iter(v['year']))))

V3 = pd.DataFrame(rows)
dup = V3.person.str.lower().duplicated(keep=False)
drop['duplicate name'] += int(dup.sum())
V3 = V3[~dup].reset_index(drop=True)

desc = {}
t = time.time()
for ch in chunked(V3.qid.tolist(), 1500):
    for r in sparql(PFX + f"""PREFIX schema: <http://schema.org/> SELECT ?i ?d WHERE {{ VALUES ?i {{ {' '.join('wd:' + q for q in ch)} }} ?i schema:description ?d FILTER(LANG(?d)="en") }}"""):
        desc[qid_of(r['i'])] = r['d']
    time.sleep(2)

OFFICE = re.compile(r'\b(president|prime minister|minister|senator|governor|chancellor|dictator|mayor|congress(wo)?man|member of parliament|leader|premier|head of state|statesman)\b', re.I)
ROYAL = re.compile(r'\b(king|queen|emperor|empress|monarch|tsar|sultan|shah)\b', re.I)
DSYN = {'footballer': ['association football player', 'American football player'],
        'soccer player': ['association football player'],
        'actress': ['actor', 'film actor', 'television actor', 'stage actor'],
        'actor': ['actor', 'film actor', 'television actor', 'stage actor'],
        'businessman': ['businessperson', 'entrepreneur'], 'businesswoman': ['businessperson', 'entrepreneur'],
        'entrepreneur': ['entrepreneur', 'businessperson'], 'author': ['writer', 'novelist'],
        'novelist': ['novelist', 'writer'], 'cyclist': ['sport cyclist', 'track cyclist'],
        'musician': ['musician', 'singer', 'composer', 'guitarist'], 'painter': ['painter', 'artist'],
        'singer': ['singer', 'singer-songwriter'], 'rapper': ['rapper'], 'swimmer': ['swimmer'],
        'boxer': ['boxer'], 'tennis player': ['tennis player'], 'basketball player': ['basketball player'],
        'golfer': ['golfer', 'professional golfer'], 'model': ['model', 'fashion model'],
        'journalist': ['journalist'], 'composer': ['composer'], 'poet': ['poet'],
        'philosopher': ['philosopher'], 'scientist': ['scientist', 'physicist', 'chemist', 'biologist'],
        'wrestler': ['professional wrestler', 'wrestler'], 'skater': ['figure skater', 'speed skater'],
        'athlete': ['athletics competitor', 'sprinter'], 'runner': ['athletics competitor', 'long-distance runner'],
        'rugby': ['rugby union player', 'rugby league player'], 'cricketer': ['cricketer'],
        'film director': ['film director'], 'director': ['film director', 'theatre director']}


def occ_pick2(r):
    occs = [o for o in json.loads(r.occupations_all) if not BAD.search(o)]
    d = (desc.get(r.qid) or '').lower()
    pos = [(m.start(), o) for o in occs for m in [re.search(r'(^|[^a-z])' + re.escape(o.lower()) + r'([^a-z]|$)', d)] if m]
    if pos:
        return min(pos)[1], 'description'
    for rx, cands in [(OFFICE, ['politician', 'statesperson']), (ROYAL, ['monarch', 'sovereign', 'ruler', 'aristocrat'])]:
        if rx.search(d):
            for c_ in cands:
                if c_ in occs:
                    return c_, 'description (office/title)'
    best = None
    for w, cands in DSYN.items():
        m = re.search(r'(^|[^a-z])' + re.escape(w) + r'([^a-z]|$)', d)
        if m:
            for c_ in cands:
                if c_ in occs and (best is None or m.start() < best[0]):
                    best = (m.start(), c_)
                    break
    if best:
        return best[1], 'description (synonym)'
    return max(occs, key=lambda o: (occ_freq[o], o)), 'most frequent'


pk = V3.apply(occ_pick2, axis=1)
V3['occupation'] = [p[0] for p in pk]
V3['occupation_source'] = [p[1] for p in pk]
V3['description'] = V3.qid.map(desc)

apply_figure_style()
fig, ax = plt.subplots(figsize=(3.6, 2.4))
x = np.log10(V3.sitelinks.clip(lower=1))
ax.hist(x, bins=np.linspace(0, 2.6, 40), color='#2a6fb0')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['1', '10', '100'])
ax.set_xlabel('Wikipedia sitelinks (log scale; 0 shown as 1)')
ax.set_ylabel('People')
ax.set_title(f'Fame-stratified candidates (n = {len(V3):,})', loc='left')
fig.tight_layout()
fig.savefig('candidates_v3_sitelinks.png', dpi=300)
