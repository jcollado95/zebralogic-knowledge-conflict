"""Recovered notebook code that produced `probe_birthdate.csv`.

Reconstructed from session lineage, not written as a standalone script.
Expect adjustments (input paths, working directory via WORKDIR) before rerunning.
Steps that restored files from the artifact store are replaced by `pass`;
those files are under src/ and data/ in this repository.
"""
import os, sys, json, re, time, threading, shutil
import urllib.request, urllib.parse
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Copy dependency files
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
            if e.code == 429 and k < tries - 1: time.sleep(20 * (k + 1)); continue
            raise

def qid_of(iri):
    return iri.rsplit('/', 1)[-1]

PFX2 = PFX + " PREFIX p: <http://www.wikidata.org/prop/> PREFIX psv: <http://www.wikidata.org/prop/statement/value/> PREFIX wikibase: <http://wikiba.se/ontology#>"
bd = defaultdict(set)
time.sleep(2)
for s in range(0, len(qids), 600):
    for r in sparql(PFX2 + f"""SELECT ?p ?t ?prec WHERE {{ VALUES ?p {{ {' '.join('wd:' + q for q in qids[s:s+600])} }}
        ?p p:P569 ?st . ?st wikibase:rank ?rk . FILTER(?rk != wikibase:DeprecatedRank)
        ?st psv:P569 ?v . ?v wikibase:timeValue ?t ; wikibase:timePrecision ?prec }}"""):
        if int(r['prec']) >= 11:
            m = re.match(r'^(-?\d+)-(\d\d)-(\d\d)', r['t'])
            if m: bd[qid_of(r['p'])].add(f"{int(m.group(2))}-{int(m.group(3))}")
    time.sleep(3)

MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
       'September', 'October', 'November', 'December']

pools_s = pd.read_csv('persons_pools.csv', encoding='utf-8-sig').set_index('qid').pool

QBD = 'On what day and month was {p} born? Answer with the day and month only, for example "4 August".'

def ask_bd(model, p):
    body = json.dumps(dict(model=model, messages=[{'role': 'user', 'content': QBD.format(p=p)}],
                           max_tokens=30, temperature=0)).encode()
    rq = urllib.request.Request(K.URL, data=body,
                                headers={'Content-Type': 'application/json',
                                         'Authorization': f"Bearer {os.environ['VLLM_API_KEY']}"})
    for k in range(3):
        try:
            return json.loads(urllib.request.urlopen(rq, timeout=600).read())['choices'][0]['message']['content'] or ''
        except Exception as e:
            err = repr(e); time.sleep(10 * (k + 1))
    return 'ERROR ' + err

MONRE = {m.lower(): i + 1 for i, m in enumerate(MON)}
MONRE.update({m[:3].lower(): i + 1 for i, m in enumerate(MON)})
MONRE['sept'] = 9

def parse_bd(a):
    a2 = a.lower()
    mm = [(MONRE[w], m.start()) for m in re.finditer(r'[a-z]+', a2) for w in [m.group()] if w in MONRE]
    dd = re.findall(r'\b([12]?\d|3[01])(?:st|nd|rd|th)?\b', a2)
    if not mm or not dd: return None
    return f"{mm[0][0]}-{int(dd[0])}"

X = cand.set_index('qid')
bjobs = [(q, X.at[q, 'person']) for q in qids if bd[q]]

out_bd = {}
t = time.time()

def runm(m):
    with ThreadPoolExecutor(24) as ex:
        out_bd[m] = list(ex.map(lambda j: ask_bd(m, j[1]), bjobs))

th = [threading.Thread(target=runm, args=(m,)) for m in ['gemma-4-E4B-it', 'gemma-4-31B-it']]
for x in th: x.start()
for x in th: x.join()

B = pd.DataFrame([dict(model=m, qid=q, person=p, answer=a.strip()[:120],
                       gold=json.dumps(sorted(bd[q])))
                  for m in out_bd for (q, p), a in zip(bjobs, out_bd[m])])
B['parsed'] = B.answer.map(parse_bd)
B['refusal'] = B.answer.map(K.refusal)
B['correct'] = [p is not None and p in json.loads(g) for p, g in zip(B.parsed, B.gold)]
B['month_correct'] = [p is not None and p.split('-')[0] in {x.split('-')[0] for x in json.loads(g)}
                      for p, g in zip(B.parsed, B.gold)]
B['pool'] = B.qid.map(pools_s).fillna('unpooled')

B.to_csv('probe_birthdate.csv', index=False, encoding='utf-8-sig')
