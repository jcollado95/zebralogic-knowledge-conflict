"""Wikidata-backed entity source for ZebraLogic perturbations.

build_kg(catalog_path, cache_dir) -> WikidataKG       (runs SPARQL, caches every query result on disk)
WikidataKG.save(path) / WikidataKG.load(path)         (frozen snapshot for reproducible experiments)

Uniqueness rules enforced by the samplers
  * within an attribute: distinct QIDs and distinct normalised labels
  * across the puzzle: no label equals, or occurs as a whole-word substring of, any other label in the puzzle
  * grounded people (relation R, people p_1..p_n, assigned values v_1..v_n):
      - aligned : v_i in R(p_i)  and  v_i not in R(p_j) for every j != i   (world knowledge identifies exactly one person)
      - conflict: v_i not in R(p_i) and v_i in R(p_j) for exactly one j != i (a derangement of an aligned assignment)
    R(p) is the *full* value set of p in Wikidata (multi-valued: dual citizenship, several occupations).
"""
from __future__ import annotations
import json, os, re, time, hashlib, random, urllib.request, urllib.parse, urllib.error, collections
from concurrent.futures import ThreadPoolExecutor

ENDPOINT = 'https://query.wikidata.org/sparql'
UA = 'zebralogic-perturbation-research/0.1 (academic benchmark generation; python-urllib)'
LABEL_SVC = 'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }'


class Sparql:
    def __init__(self, cache_dir='kg_cache', delay=1.0):
        self.cache_dir = cache_dir; self.delay = delay; os.makedirs(cache_dir, exist_ok=True)

    def __call__(self, q, timeout=120, retries=3):
        h = hashlib.sha1(q.encode()).hexdigest()[:16]; f = os.path.join(self.cache_dir, h + '.json')
        if os.path.exists(f):
            return json.load(open(f, encoding='utf-8'))
        data = urllib.parse.urlencode({'query': q}).encode()
        for k in range(retries):
            try:
                req = urllib.request.Request(ENDPOINT, data=data, headers={'Accept': 'application/sparql-results+json', 'User-Agent': UA})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    rows = json.loads(r.read())['results']['bindings']
                rows = [{k_: v['value'] for k_, v in b.items()} for b in rows]
                json.dump(rows, open(f, 'w', encoding='utf-8'), ensure_ascii=False)
                time.sleep(self.delay)
                return rows
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and k < retries - 1:
                    time.sleep(float(e.headers.get('Retry-After', 10 * (k + 1)))); continue
                raise


qid = lambda uri: uri.rsplit('/', 1)[-1]


def clean_label(lab, rules, spec):
    if not lab: return None
    if spec.get('strip_suffix') and lab.lower().endswith(spec['strip_suffix']):
        lab = lab[: -len(spec['strip_suffix'])]
    lab = lab.strip()
    if spec.get('lower'): lab = lab.lower()
    if re.search(rules['reject_regex'], ' ' + lab + ' '): return None
    if spec.get('exclude_regex') and re.search(spec['exclude_regex'], lab, re.I): return None
    if len(lab) > rules['max_chars'] or len(lab.split()) > rules['max_words'] or not re.search(r'[A-Za-z]', lab): return None
    return lab


# ----------------------------------------------------------------------------- pool queries
def q_class(spec, limit=400):
    cn = 'OPTIONAL { ?x wdt:P1843 ?cn . FILTER(lang(?cn)="en") }' if spec.get('common_name') else ''
    return f'''SELECT ?x ?xLabel ?sl (SAMPLE(?cn) AS ?common) WHERE {{ {spec["pattern"]} ?x wikibase:sitelinks ?sl .
      FILTER(?sl >= {spec["min_sitelinks"]}) {cn} {LABEL_SVC} }} GROUP BY ?x ?xLabel ?sl ORDER BY DESC(?sl) LIMIT {limit}'''


def q_usage(prop, limit, subject='', filt=''):
    return f'''SELECT ?x ?xLabel ?sl WHERE {{ {{ SELECT ?x (COUNT(*) AS ?sl) WHERE {{ ?s wdt:{prop} ?x . {subject} }}
      GROUP BY ?x ORDER BY DESC(?sl) LIMIT {limit} }} {filt} {LABEL_SVC} }} ORDER BY DESC(?sl)'''


def q_demonym(limit):
    return f'''SELECT ?x ?xLabel ?sl (SAMPLE(?d) AS ?dem) WHERE {{ {{ SELECT ?x (COUNT(*) AS ?sl) WHERE {{ ?s wdt:P27 ?x . }}
      GROUP BY ?x ORDER BY DESC(?sl) LIMIT {limit} }} ?x wdt:P31 wd:Q3624078 . FILTER NOT EXISTS {{ ?x wdt:P576 ?end }}
      ?x wdt:P1549 ?d . FILTER(lang(?d)="en") {LABEL_SVC} }} GROUP BY ?x ?xLabel ?sl ORDER BY DESC(?sl)'''


def q_names(cls, minsl, lang=None, limit=600):
    lf = f'?x wdt:P407 wd:{lang} .' if lang else ''
    return f'''SELECT ?x ?xLabel ?sl WHERE {{ ?x wdt:P31 wd:{cls} . {lf} ?x wikibase:sitelinks ?sl . FILTER(?sl >= {minsl})
      {LABEL_SVC} }} ORDER BY DESC(?sl) LIMIT {limit}'''


def q_people(prop, anchor, minsl, limit):
    # no ORDER BY: the engine can stop at LIMIT; the sitelink threshold keeps people well known
    return f'''SELECT ?p ?pLabel ?sl ?g WHERE {{ ?p wdt:{prop} wd:{anchor} ; wdt:P31 wd:Q5 ; wikibase:sitelinks ?sl .
      FILTER(?sl >= {minsl}) OPTIONAL {{ ?p wdt:P21 ?g }} {LABEL_SVC} }} LIMIT {limit}'''


def q_facts(pids, props):
    vp = ' '.join(f'wd:{p}' for p in pids); vr = ' '.join(f'wdt:{p}' for p in props)
    return f'''SELECT ?p ?r ?v ?vLabel WHERE {{ VALUES ?p {{ {vp} }} VALUES ?r {{ {vr} }} ?p ?r ?v . {LABEL_SVC} }}'''


def q_gender_sample(prop, val, limit):
    # streaming form (no aggregation) so the engine can stop at LIMIT; counted client-side
    return f'''SELECT ?g WHERE {{ ?p wdt:{prop} wd:{val} ; wdt:P21 ?g . }} LIMIT {limit}'''


def add_gender_stats(kg, cache_dir='kg_cache', workers=3, log=print):
    spec = kg.catalog['gender_stats']; sp = Sparql(cache_dir); stats = {}
    def job(x):
        try: return x, sp(q_gender_sample('P106', x['qid'], spec['sample_holders']))
        except Exception as e: log(f'gender {x["label"]}: FAILED {e!r}'); return x, None
    with ThreadPoolExecutor(workers) as ex:
        for x, rows in ex.map(job, kg.pools[spec['attr']]):
            if rows is None: continue
            c = collections.Counter(GENDER.get(qid(r['g']), 'other') for r in rows)
            fm = c['female'] + c['male']
            if fm: stats[x['label']] = dict(qid=x['qid'], female_share=c['female'] / fm, n=fm)
    kg.gender_stats = stats; log(f'gender stats: {len(stats)} occupations'); return stats


def q_demonyms_of(cids):
    vc = ' '.join(f'wd:{c}' for c in cids)
    return f'''SELECT ?x (SAMPLE(?d) AS ?dem) WHERE {{ VALUES ?x {{ {vc} }} ?x wdt:P1549 ?d . FILTER(lang(?d)="en") }} GROUP BY ?x'''


# ----------------------------------------------------------------------------- KG
class WikidataKG:
    def __init__(self, catalog, pools, names, people, retrieved=None, gender_stats=None):
        self.gender_stats = gender_stats or {}
        self.catalog = catalog
        self.pools = pools          # attr -> [{qid, label, score}]
        self.names = names          # 'female'/'male'/'female:<lang>'... -> [{qid,label,score}]
        self.people = people        # rel -> {'anchors': {qid: label}, 'persons': {pid: {label, gender, values:[qid...]}}, 'labels': {qid: label}}
        self.retrieved = retrieved or time.strftime('%Y-%m-%d')

    # ------------------------------------------------------------ persistence
    def save(self, path):
        json.dump(dict(catalog=self.catalog, pools=self.pools, names=self.names, people=self.people, retrieved=self.retrieved,
                       gender_stats=self.gender_stats),
                  open(path, 'w', encoding='utf-8'), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        d = json.load(open(path, encoding='utf-8'))
        return cls(d['catalog'], d['pools'], d['names'], d['people'], d['retrieved'], d.get('gender_stats'))

    # ------------------------------------------------------------ uniqueness helpers
    @staticmethod
    def _clash(lab, taken):
        """True if lab equals / contains / is contained in a taken label as whole words, allowing the plural
        endings the clue parser accepts (blue ~ blues, cruise ~ cruises)."""
        l = lab.lower(); pl = r'(?:s|es)?(?!\w)'
        return any(l == t or re.search(r'(?<!\w)' + re.escape(l) + pl, t) or re.search(r'(?<!\w)' + re.escape(t) + pl, l)
                   for t in taken)

    def _draw(self, items, k, rng, taken, key=lambda x: x['label']):
        items = list(items); rng.shuffle(items); out, seen_q = [], set(); taken = set(taken)
        for it in items:
            lab = key(it); q = it.get('qid') or lab
            if q in seen_q or self._clash(lab, taken): continue
            out.append(it); seen_q.add(q); taken.add(lab.lower())
            if len(out) == k: return out
        raise ValueError(f'only {len(out)} unique candidates for k={k}')

    # ------------------------------------------------------------ samplers
    def values(self, attr, k, rng, taken=(), top=None):
        pool = self.pools.get(attr) or []
        if top: pool = pool[:top]
        return [x['label'] for x in self._draw(pool, k, rng, taken)]

    def given_names(self, gender, k, rng, taken=(), lang=None, top=None):
        key = gender if lang is None else f'{gender}:{lang}'
        pool = self.names.get(key, [])
        if top: pool = pool[:top]
        return [x['label'] for x in self._draw(pool, k, rng, taken)]

    def people_assignment(self, rel, k, rng, mode='aligned', taken=(), tries=400):
        """Returns [(person_label, person_gender, value_label, true_value_labels)] satisfying the uniqueness rules."""
        P = self.people[rel]; persons = P['persons']; labels = P['labels']
        # work on *labels*: distinct QIDs can share a label after normalisation (pop / pop music -> 'pop')
        vset = {pid: {labels[v] for v in d['values'] if v in labels} for pid, d in persons.items()}
        anchor_labels = {labels.get(a, al) for a, al in P['anchors'].items()}
        by_val = collections.defaultdict(list)
        for pid, s in vset.items():
            for v in s: by_val[v].append(pid)
        vals = sorted(v for v in by_val if v in anchor_labels)
        for _ in range(tries):
            if len(vals) < k: break
            chosen = rng.sample(vals, k); taken_l = set(t.lower() for t in taken); picks = []
            if any(self._clash(a, taken_l | {b.lower() for b in chosen if b != a}) for a in chosen): continue
            for v in chosen:  # one person per value whose value set meets the chosen values only at v
                cands = [p for p in by_val[v] if not (vset[p] & (set(chosen) - {v})) and '.' not in persons[p]['label']
                         and not self._clash(persons[p]['label'], taken_l | {b.lower() for b in chosen} | {x[0].lower() for x in picks})]
                if not cands: break
                p = rng.choice(sorted(cands)); picks.append((persons[p]['label'], p, v))
            if len(picks) < k: continue
            perm = list(range(k))
            if mode == 'conflict':  # derangement of the values over the people
                while any(i == j for i, j in enumerate(perm)): rng.shuffle(perm)
            return [(picks[i][0], persons[picks[i][1]]['gender'], chosen[perm[i]], sorted(vset[picks[i][1]])) for i in range(k)]
        raise ValueError(f'no valid {mode} assignment for {rel}, k={k}')

    def gender_share(self, rel='occupation', min_people=8):
        """Female share per occupation: gender_stats (sample of holders) if available, else among fetched well-known holders."""
        if self.gender_stats: return {k: v for k, v in self.gender_stats.items() if v['n'] >= min_people}
        P = self.people[rel]; cnt = collections.defaultdict(collections.Counter)
        for d in P['persons'].values():
            for v in d['values']:
                if v in P['anchors']: cnt[v][d['gender']] += 1
        return {P['labels'][v]: dict(female_share=c['female'] / max(1, c['female'] + c['male']), n=c['female'] + c['male'])
                for v, c in cnt.items() if c['female'] + c['male'] >= min_people and v in P['labels']}


# ----------------------------------------------------------------------------- build
GENDER = {'Q6581072': 'female', 'Q6581097': 'male', 'Q1052281': 'female', 'Q2449503': 'male'}


def _pool_from_rows(rows, rules, spec, label_key='xLabel'):
    out, seen = [], set()
    for r in rows:
        raw = r.get('dem') or (r.get('common') if spec.get('common_name') and r.get('common') else None) or r.get(label_key)
        lab = clean_label(raw, rules, spec)
        if lab and lab.lower() not in seen:
            seen.add(lab.lower()); out.append(dict(qid=qid(r['x']), label=lab, score=float(r.get('sl', 0))))
    return out[: spec['top']] if spec.get('top') else out


def build_kg(catalog_path, cache_dir='kg_cache', workers=3, log=print):
    cat = json.load(open(catalog_path, encoding='utf-8')); rules = cat['label_rules']; sp = Sparql(cache_dir)
    pools = {}

    def pool(attr, spec):
        t = spec['type']
        if t == 'class': rows = sp(q_class(spec))
        elif t == 'usage': rows = sp(q_usage(spec['property'], spec['limit'], spec.get('subject', ''), spec.get('filter', '')))
        elif t == 'demonym': rows = sp(q_demonym(spec['limit']))
        else: return attr, None
        return attr, _pool_from_rows(rows, rules, spec)

    def safe_pool(attr, spec):
        try: return pool(attr, spec)
        except Exception as e:  # one failing query must not abort the build
            log(f'pool {attr}: FAILED {e!r}'); return attr, []

    specs = {a: s for a, s in cat['value_pools'].items() if s['type'] not in ('none', 'same_as')}
    with ThreadPoolExecutor(workers) as ex:
        for a, p in ex.map(lambda kv: safe_pool(*kv), specs.items()):
            pools[a] = p; log(f'pool {a}: {len(p)}')
    for a, s in cat['value_pools'].items():
        if s['type'] == 'same_as': pools[a] = pools[s['of']]

    names = {}
    npc = cat['name_pools']
    jobs = [(g, None) for g in ('female', 'male')] + [(g, l) for g in ('female', 'male') for l in npc['languages']]
    def name_job(gl):
        g, l = gl; lq = npc['languages'][l] if l else None
        minsl = npc[g]['min_sitelinks'] if l is None else 3
        rows = sp(q_names(npc[g]['class'], minsl, lq))
        return gl, _pool_from_rows(rows, rules, {})
    with ThreadPoolExecutor(workers) as ex:
        for (g, l), p in ex.map(name_job, jobs):
            names[g if l is None else f'{g}:{l}'] = p
    log('names: ' + ', '.join(f'{k}={len(v)}' for k, v in names.items()))

    people = {}
    pf = cat['person_fetch']; props = sorted({r['property'] for r in cat['person_relations'].values()})
    for rel, spec in cat['person_relations'].items():
        if 'anchors_from' in spec:  # pools already filtered by the catalog rules
            anchors = pools[spec['anchors_from']][: spec['n_anchors']]
        else:
            rows = sp(q_usage(spec['property'], spec['n_anchors'] * 2, spec.get('subject', '')))
            anchors = _pool_from_rows(rows, rules, spec)[: spec['n_anchors']]
        if spec.get('project') == 'demonym':  # anchors already carry demonym labels
            pass
        with ThreadPoolExecutor(workers) as ex:
            def pj(a):
                try: return a, sp(q_people(spec['property'], a['qid'], pf['min_sitelinks'], pf['per_anchor']))
                except Exception as e: log(f'people {rel}/{a["label"]}: FAILED {e!r}'); return a, []
            res = list(ex.map(pj, anchors))
        persons = {}
        for a, rows in res:
            for r in rows:
                lab = clean_label(r.get('pLabel'), dict(rules, reject_regex=r'^Q\d+$|[(),/&:;".]|\d'), {})
                if lab: persons.setdefault(qid(r['p']), dict(label=lab, gender=GENDER.get(qid(r.get('g', '')), 'other')))
        # full value sets for exclusivity checks (all values, not only anchors)
        pids = list(persons); vals = collections.defaultdict(set); vlab = {}
        for i in range(0, len(pids), 150):
            for r in sp(q_facts(pids[i:i + 150], [spec['property']])):
                vals[qid(r['p'])].add(qid(r['v'])); vlab[qid(r['v'])] = r.get('vLabel')
        labels = {}
        if spec.get('project') == 'demonym':
            cids = sorted({v for s in vals.values() for v in s})
            for i in range(0, len(cids), 200):
                for r in sp(q_demonyms_of(cids[i:i + 200])):
                    lab = clean_label(r['dem'], rules, {});
                    if lab: labels[qid(r['x'])] = lab
        else:
            for v, l in vlab.items():
                lab = clean_label(l, rules, spec)
                if lab: labels[v] = lab
        for p in persons: persons[p]['values'] = sorted(vals.get(p, ()))
        people[rel] = dict(anchors={a['qid']: labels.get(a['qid'], a['label']) for a in anchors}, persons=persons, labels=labels,
                           key=spec['key'], line=spec['line'], phrase=spec['phrase'])
        log(f'people {rel}: anchors={len(anchors)} persons={len(persons)}')
    kg = WikidataKG(cat, pools, names, people)
    if cat.get('gender_stats'): add_gender_stats(kg, cache_dir, workers, log)
    return kg
