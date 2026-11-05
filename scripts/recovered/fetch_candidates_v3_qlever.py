"""Recovered notebook code that produced `candidates_v3.csv`.

Reconstructed from session lineage, not written as a standalone script.
Expect adjustments (input paths, working directory via WORKDIR) before rerunning.
Steps that restored files from the artifact store are replaced by `pass`;
those files are under src/ and data/ in this repository.
"""
import os, re, json, time, urllib.request, urllib.parse, urllib.error
import pandas as pd
from collections import defaultdict, Counter

EP = "https://qlever.dev/api/wikidata"
UA = "unique-persons-script/1.0 (Wikidata research script)"

def sparql(q, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(EP, data=urllib.parse.urlencode({'query': q}).encode(),
                headers={"Accept": "application/sparql-results+json", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as r:
                return [{kk: c['value'] for kk, c in row.items()} for row in json.load(r)['results']['bindings']]
        except urllib.error.HTTPError as e:
            if e.code == 429 and k < tries - 1:
                time.sleep(20 * (k + 1))
                continue
            raise

PFX = """PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#>"""

def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

qid_of = lambda iri: iri.rsplit('/', 1)[-1]

BANDS = [('f_100+', 100, 100000, 1000), ('e_50-99', 50, 100, 1000), ('d_20-49', 20, 50, 1000),
         ('c_10-19', 10, 20, 800), ('b_5-9', 5, 10, 600), ('a_0-4', 0, 5, 600)]

samp = []
t = time.time()
for name, lo, hi, n in BANDS:
    q = PFX + f"""SELECT DISTINCT ?p ?sl WHERE {{ ?p wdt:P31 wd:Q5 ; wdt:P19 ?b ; wdt:P106 ?o ; wdt:P569 ?d ; wikibase:sitelinks ?sl .
      FILTER(?sl >= {lo} && ?sl < {hi}) BIND(MD5(STR(?p)) AS ?h) }} ORDER BY ?h LIMIT {n}"""
    rows = sparql(q)
    samp += [dict(qid=r['p'].rsplit('/', 1)[-1], sitelinks=int(r['sl']), band=name) for r in rows]
    print(name, len(rows), round(time.time()-t, 1), 's')
    time.sleep(3)

S = pd.DataFrame(samp).drop_duplicates('qid')
print('sample', len(S))

def band_of(sl):
    for name, lo, hi, _ in BANDS:
        if lo <= sl < hi:
            return name

C = S.copy()
C['source'] = 'stratified'
qids = C.qid.tolist()

PROPS = {'P19': 'bp', 'P106': 'occ', 'P569': 'dob'}
vals = defaultdict(lambda: defaultdict(set))
t = time.time()
for s in range(0, len(qids), 500):
    rows = sparql(PFX + f"""PREFIX p: <http://www.wikidata.org/prop/> PREFIX ps: <http://www.wikidata.org/prop/statement/>
      SELECT ?p ?prop ?v WHERE {{ VALUES ?p {{ {' '.join('wd:'+q for q in qids[s:s+500])} }}
      VALUES ?prop {{ wdt:P19 wdt:P106 wdt:P569 }} ?p ?prop ?v }}""")
    for r in rows:
        n = PROPS[r['prop'].rsplit('/', 1)[-1]]; v = r['v']
        if n == 'dob':
            m = re.match(r'^(-?\d{1,4})-', v)
            if m:
                vals[qid_of(r['p'])]['year'].add(int(m.group(1)))
        elif v.startswith('http://www.wikidata.org/entity/Q'):
            vals[qid_of(r['p'])][n].add(qid_of(v))
    time.sleep(2)
print('fetched', round(time.time()-t, 1), 's')

ents = sorted(set(qids) | {v for q in qids for k in ['bp', 'occ'] for v in vals[q][k]})
lab = {}
alias = defaultdict(set)
t = time.time()
for ch in chunked(ents, 1500):
    vs = ' '.join('wd:'+q for q in ch)
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
    for r in sparql(PFX + f"""SELECT ?b ?t ?a1 ?a2 WHERE {{ VALUES ?b {{ {' '.join('wd:'+q for q in ch)} }} OPTIONAL {{ ?b wdt:P31 ?t }} OPTIONAL {{ ?b wdt:P131 ?a1 . OPTIONAL {{ ?a1 wdt:P131 ?a2 }} }} }}"""):
        b = qid_of(r['b'])
        for k, d in [('t', types), ('a1', par1), ('a2', par2)]:
            if r.get(k):
                d[b].add(qid_of(r[k]))
    time.sleep(2)

need = sorted(({x for d in (types, par1, par2) for s_ in d.values() for x in s_}) - set(lab))
for ch in chunked(need, 1500):
    for r in sparql(PFX + f"""SELECT ?i ?en WHERE {{ VALUES ?i {{ {' '.join('wd:'+q for q in ch)} }} ?i rdfs:label ?en FILTER(LANG(?en)="en") }}"""):
        lab[qid_of(r['i'])] = r['en']
    time.sleep(2)
print(len(ents), 'entities,', sum(1 for e in ents if e not in lab), 'unlabelled; took', round(time.time()-t), 's')

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
    q = r.qid; v = vals[q]; name = lab.get(q)
    if not (v['bp'] and v['occ'] and v['year']):
        drop['missing fact'] += 1; continue
    if not name or re.match(r'^Q\d+$', name):
        drop['unlabelled name'] += 1; continue
    if len(v['year']) > 1:
        drop['conflicting birth year'] += 1; continue
    if len(v['bp']) > 1:
        drop['several birthplaces'] += 1; continue
    if ' ' not in name.strip():
        drop['one-word name'] += 1; continue
    b = next(iter(v['bp'])); bdisp = bp_disp(b)
    occs = sorted({lab[o] for o in v['occ'] if o in lab}); occs_ok = [o for o in occs if not BAD.search(o)]
    if not bdisp or not occs_ok:
        drop['unlabelled value'] += 1; continue
    if BAD.search(name) or BAD.search(bdisp):
        drop['clue-grammar label'] += 1; continue
    if NONPLACE_L.search(bdisp):
        drop['non-settlement birthplace'] += 1; continue
    occ_disp = max(occs_ok, key=lambda o: (occ_freq[o], o))
    rows.append(dict(qid=q, person=name, sitelinks=r.sitelinks, band=r.band, source=r.source,
        birthplace=bdisp, birthplace_accepted=json.dumps(sorted(bp_acc(b) | {bdisp}), ensure_ascii=False),
        occupation=occ_disp, occupations_all=json.dumps(occs, ensure_ascii=False),
        occupation_aliases=json.dumps(sorted({a for o in v['occ'] for a in alias[o]}), ensure_ascii=False),
        birth_year=next(iter(v['year']))))

V3 = pd.DataFrame(rows)
dup = V3.person.str.lower().duplicated(keep=False)
drop['duplicate name'] += int(dup.sum())
V3 = V3[~dup].reset_index(drop=True)
print(len(V3), 'kept; dropped', dict(drop))

desc = {}
t = time.time()
for ch in chunked(V3.qid.tolist(), 1500):
    for r in sparql(PFX + f"""PREFIX schema: <http://schema.org/> SELECT ?i ?d WHERE {{ VALUES ?i {{ {' '.join('wd:'+q for q in ch)} }} ?i schema:description ?d FILTER(LANG(?d)="en") }}"""):
        desc[qid_of(r['i'])] = r['d']
    time.sleep(2)
print(len(desc), 'descriptions in', round(time.time()-t), 's')

OFFICE = re.compile(r'\b(president|prime minister|minister|senator|governor|chancellor|dictator|mayor|congress(wo)?man|member of parliament|leader|premier|head of state|statesman)\b', re.I)
ROYAL = re.compile(r'\b(king|queen|emperor|empress|monarch|tsar|sultan|shah)\b', re.I)
DSYN = {'footballer': ['association football player', 'American football player'], 'soccer player': ['association football player'],
        'actress': ['actor', 'film actor', 'television actor', 'stage actor'],
        'actor': ['actor', 'film actor', 'television actor', 'stage actor'],
        'businessman': ['businessperson', 'entrepreneur'], 'businesswoman': ['businessperson', 'entrepreneur'],
        'entrepreneur': ['entrepreneur', 'businessperson'], 'author': ['writer', 'novelist'], 'novelist': ['novelist', 'writer'],
        'cyclist': ['sport cyclist', 'track cyclist'], 'musician': ['musician', 'singer', 'composer', 'guitarist'],
        'painter': ['painter', 'artist'], 'singer': ['singer', 'singer-songwriter'], 'rapper': ['rapper'],
        'swimmer': ['swimmer'], 'boxer': ['boxer'], 'tennis player': ['tennis player'],
        'basketball player': ['basketball player'], 'golfer': ['golfer', 'professional golfer'],
        'model': ['model', 'fashion model'], 'journalist': ['journalist'], 'composer': ['composer'],
        'poet': ['poet'], 'philosopher': ['philosopher'], 'scientist': ['scientist', 'physicist', 'chemist', 'biologist'],
        'wrestler': ['professional wrestler', 'wrestler'], 'skater': ['figure skater', 'speed skater'],
        'athlete': ['athletics competitor', 'sprinter'], 'runner': ['athletics competitor', 'long-distance runner'],
        'rugby': ['rugby union player', 'rugby league player'], 'cricketer': ['cricketer'],
        'film director': ['film director'], 'director': ['film director', 'theatre director']}

def occ_pick(r):
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
                    best = (m.start(), c_); break
    if best:
        return best[1], 'description (synonym)'
    return max(occs, key=lambda o: (occ_freq[o], o)), 'most frequent'

pk = V3.apply(occ_pick, axis=1)
V3['occupation'] = [p[0] for p in pk]
V3['occupation_source'] = [p[1] for p in pk]
V3['description'] = V3.qid.map(desc)

# Keep only stratified source
V3 = V3[V3.source == 'stratified'].drop(columns=['source']).reset_index(drop=True)

# Fetch birth country
cty = {}
q_ = V3.qid.tolist()
for s in range(0, len(q_), 800):
    for r in sparql(PFX + f"SELECT ?p ?c WHERE {{ VALUES ?p {{ {' '.join('wd:'+x for x in q_[s:s+800])} }} ?p wdt:P19 ?b . ?b wdt:P17 ?c }}"):
        cty.setdefault(r['p'].rsplit('/', 1)[-1], r['c'].rsplit('/', 1)[-1])
    time.sleep(2)
V3['birth_country'] = V3.qid.map(cty)

# Fetch country labels to filter out whole-country birthplaces
cq = sorted(V3.birth_country.dropna().unique())
import unicodedata
def norm(s):
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()).split())

clab = set()
for s in range(0, len(cq), 500):
    for r in sparql(PFX + f"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
      SELECT ?l WHERE {{ VALUES ?c {{ {' '.join('wd:'+x for x in cq[s:s+500])} }} {{ ?c rdfs:label ?l FILTER(LANG(?l)="en") }} UNION {{ ?c skos:altLabel ?l FILTER(LANG(?l)="en") }} }}"""):
        clab.add(norm(r['l']))

isc = V3.birthplace.map(lambda b: norm(b) in clab)
print('birthplace is a country:', int(isc.sum()))
V3 = V3[~isc].reset_index(drop=True)
print(len(V3), 'candidates remain')

V3.to_csv('candidates_v3.csv', index=False, encoding='utf-8-sig')
print('Done. Saved candidates_v3.csv with', len(V3), 'rows.')
