"""Recovered notebook code that produced `fact_knowledge_by_pool.csv`.

Reconstructed from session lineage, not written as a standalone script.
Expect adjustments (input paths, working directory via WORKDIR) before rerunning.
Steps that restored files from the artifact store are replaced by `pass`;
those files are under src/ and data/ in this repository.
"""
import os, sys, json, re, time, threading, urllib.request, urllib.parse, shutil
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Copy dependency files into working directory
shutil.copy('data/persons/candidates_all.csv', 'candidates_all.csv')
shutil.copy('data/persons/persons_pools.csv', 'persons_pools.csv')
shutil.copy('src/knowledge_probe.py', 'knowledge_probe.py')
shutil.copy('src/probe_pool.py', 'probe_pool.py')

sys.path.insert(0, '.')
import knowledge_probe as K
import probe_pool as PP

cand = pd.read_csv('candidates_all.csv')
qids = cand.qid.tolist()

EP = "https://qlever.dev/api/wikidata"
PFX = """PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#>"""

def sparql(q, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(EP, data=urllib.parse.urlencode({'query': q}).encode(),
                headers={"Accept": "application/sparql-results+json",
                         "User-Agent": "unique-persons-script/1.0 (Wikidata research script)"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return [{kk: c['value'] for kk, c in row.items()} for row in json.load(r)['results']['bindings']]
        except urllib.error.HTTPError as e:
            if e.code == 429 and k < tries - 1:
                time.sleep(20 * (k + 1))
                continue
            raise

qid_of = lambda iri: iri.rsplit('/', 1)[-1]

PROPS = {'birth_year': 'P569', 'death_year': 'P570', 'citizenship': 'P27',
         'spouse': 'P26', 'father': 'P22', 'death_place': 'P20'}
P2N = {'http://www.wikidata.org/prop/direct/' + p: n for n, p in PROPS.items()}

vals = defaultdict(lambda: defaultdict(set))
t = time.time()
for s in range(0, len(qids), 600):
    rows = sparql(PFX + f"""SELECT ?p ?prop ?v WHERE {{ VALUES ?p {{ {' '.join('wd:'+q for q in qids[s:s+600])} }}
      VALUES ?prop {{ {' '.join('wdt:'+p for p in PROPS.values())} }} ?p ?prop ?v }}""")
    for r in rows:
        n = P2N[r['prop']]; v = r['v']
        if n.endswith('_year'):
            m = re.match(r'^(-?\d{1,4})-', v)
            if m: vals[qid_of(r['p'])][n].add(str(int(m.group(1))))
        elif v.startswith('http://www.wikidata.org/entity/Q'):
            vals[qid_of(r['p'])][n].add(qid_of(v))
    time.sleep(3)
print('fetched props in', round(time.time()-t, 1), 's')

ents = sorted({v for q in qids for n in ['citizenship', 'spouse', 'father', 'death_place'] for v in vals[q][n]})
lab = {}
alias = defaultdict(set)
par = defaultdict(set)

for s in range(0, len(ents), 1500):
    ch = ' '.join('wd:' + q for q in ents[s:s+1500])
    for r in sparql(PFX + f"""SELECT ?i ?en ?mul WHERE {{ VALUES ?i {{ {ch} }} OPTIONAL {{ ?i rdfs:label ?en FILTER(LANG(?en)="en") }}
                              OPTIONAL {{ ?i rdfs:label ?mul FILTER(LANG(?mul)="mul") }} }}"""):
        lab[qid_of(r['i'])] = r.get('en') or r.get('mul')
    time.sleep(2)
    for r in sparql(PFX + f"""SELECT ?i ?a WHERE {{ VALUES ?i {{ {ch} }} ?i skos:altLabel ?a FILTER(LANG(?a)="en") }}"""):
        alias[qid_of(r['i'])].add(r['a'])
    time.sleep(2)

dps = sorted({v for q in qids for v in vals[q]['death_place']})
for s in range(0, len(dps), 800):
    for r in sparql(PFX + f"""SELECT ?i ?a ?al WHERE {{ VALUES ?i {{ {' '.join('wd:'+q for q in dps[s:s+800])} }} ?i wdt:P131 ?a .
                                 OPTIONAL {{ ?a rdfs:label ?al FILTER(LANG(?al)="en") }} }}"""):
        if r.get('al'): par[qid_of(r['i'])].add(r['al'])
    time.sleep(2)

EXTRA = ['birth_year', 'death_year', 'citizenship', 'spouse', 'father', 'death_place']
rows = []
for q in qids:
    d = dict(qid=q)
    for n in EXTRA:
        v = vals[q][n]
        if n.endswith('_year'):
            disp = sorted(v); acc = sorted(v)
        else:
            disp = sorted({lab.get(x) for x in v if lab.get(x)})
            acc = set(disp) | {a for x in v for a in alias[x]}
            if n == 'death_place': acc |= {a for x in v for a in par[x]}
            acc = sorted(acc)
        d[n] = json.dumps(disp, ensure_ascii=False)
        d[n + ' (accepted)'] = json.dumps(acc, ensure_ascii=False)
    rows.append(d)
X = cand[['qid', 'person', 'sitelinks']].merge(pd.DataFrame(rows), on='qid')
X.to_csv('facts_extra.csv', index=False, encoding='utf-8')
Xi = X.set_index('qid')

QX = {'birth_year': 'In what year was {p} born? Answer with the year only.',
      'death_year': 'In what year did {p} die? Answer with the year only.',
      'citizenship': 'What is the country of citizenship of {p}? Answer with the country name only.',
      'spouse': "Who is or was {p}'s spouse? Answer with the full name only.",
      'father': "What is the name of {p}'s father? Answer with his full name only.",
      'death_place': 'Where did {p} die? Answer with the place name only.'}

def ask(model, p, n):
    body = json.dumps(dict(model=model, messages=[{'role': 'user', 'content': QX[n].format(p=p)}],
                           max_tokens=60, temperature=0)).encode()
    rq = urllib.request.Request(K.URL, data=body,
                                headers={'Content-Type': 'application/json',
                                         'Authorization': f"Bearer {os.environ['VLLM_API_KEY']}"})
    for k in range(3):
        try:
            return json.loads(urllib.request.urlopen(rq, timeout=600).read())['choices'][0]['message']['content'] or ''
        except Exception as e:
            err = repr(e); time.sleep(10 * (k + 1))
    return 'ERROR ' + err

jobs = [(r.qid, r.person, n) for r in X.itertuples() for n in EXTRA if json.loads(Xi.at[r.qid, n])]
print(len(jobs), 'questions per model')

res = {}
def runm(m):
    with ThreadPoolExecutor(24) as ex:
        res[m] = list(ex.map(lambda j: ask(m, j[1], j[2]), jobs))

th = [threading.Thread(target=runm, args=(m,)) for m in ['gemma-4-E4B-it', 'gemma-4-31B-it']]
t = time.time()
for x in th: x.start()
for x in th: x.join()

recs = [dict(model=m, qid=q, person=p, relation=n, answer=a.strip()[:300])
        for m in res for (q, p, n), a in zip(jobs, res[m])]
pr = pd.DataFrame(recs)
pr.to_json('probe_extra_raw.jsonl', orient='records', lines=True, force_ascii=False)
print('done in', round(time.time()-t), 's; errors', sum(r['answer'].startswith('ERROR') for r in recs))

def norm(s): return ' '.join(K.toks(s))

def score(n, acc, ans):
    if K.refusal(ans): return False
    if n.endswith('_year'):
        m = re.search(r'-?\d{3,4}', ans)
        return bool(m) and str(int(m.group())) in acc
    if n in ('citizenship', 'death_place'):
        a = norm(ans)
        return bool(a) and any(
            (g := norm(x)) and (re.search(r'(^| )' + re.escape(g) + r'( |$)', a) or
                                 (n == 'death_place' and re.search(r'(^| )' + re.escape(a) + r'( |$)', g)))
            for x in acc)
    return any(K.correct('Mother', x, ans) or PP._mother_partial(x, ans) for x in acc)

pr['acc'] = [json.loads(Xi.at[q, n + ' (accepted)']) for q, n in zip(pr.qid, pr.relation)]
pr['refusal'] = pr.answer.map(K.refusal)
pr['correct'] = [score(n, a, s) for n, a, s in zip(pr.relation, pr.acc, pr.answer)]
pr['year_pm1'] = [n.endswith('_year') and bool(m := re.search(r'-?\d{3,4}', s)) and
                  any(abs(int(m.group()) - int(y)) <= 1 for y in a)
                  for n, a, s in zip(pr.relation, pr.acc, pr.answer)]

pools = pd.read_csv('persons_pools.csv', encoding='utf-8-sig').set_index('qid').pool
pr['pool'] = pr.qid.map(pools).fillna('unpooled')

Pp = pd.read_csv('persons_pools.csv', encoding='utf-8-sig').set_index('qid')
old = {'place of birth': 'birthplace', 'occupation': 'occupation', 'height (cm)': 'height',
       'educated at': 'school', 'mother': 'mother'}
rows = []
for pool in ['both', 'only_31b', 'none']:
    g = Pp[Pp.pool == pool]
    for c, nm in old.items():
        rows.append(dict(fact=nm, kind='current', pool=pool, coverage=1.0,
                         e4b=g[f'knows_{c}_e4b'].mean(), b31=g[f'knows_{c}_31b'].mean()))
    for n in ['birth_year', 'spouse', 'father', 'death_year', 'death_place', 'citizenship']:
        s = pr[(pr.pool == pool) & (pr.relation == n)]
        cov = s.qid.nunique() / len(g)
        rows.append(dict(fact=n.replace('_', ' '), kind='new', pool=pool, coverage=cov,
                         e4b=s[s.model == 'gemma-4-E4B-it'].correct.mean(),
                         b31=s[s.model == 'gemma-4-31B-it'].correct.mean()))

CMP = pd.DataFrame(rows)
CMP.to_csv('fact_knowledge_by_pool.csv', index=False)
print(CMP.head())
