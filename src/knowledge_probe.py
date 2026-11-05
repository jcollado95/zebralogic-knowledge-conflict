import os
"""Closed-book knowledge probe for the unique_persons pool: 5 direct questions per person, greedy decoding.
Deterministic scoring shared by every model (refusals first, then per-relation rules)."""
import re, json, time, unicodedata, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get('VLLM_URL', 'http://ada01.ujaen.es:8080/v1/chat/completions')
Q = {'BirthPlace': 'Where was {p} born? Answer with the place name only.',
     'Occupation': 'What is the main occupation of {p}? Answer with the occupation only.',
     'Mother': "What is the name of {p}'s mother? Answer with her full name only.",
     'EducatedAt': 'Which school or university did {p} attend? Answer with the institution name only.',
     'Height': 'How tall is {p} in centimetres? Answer with a number only.'}
SYN = {'association football player': ['soccer', 'footballer', 'football player'], 'film director': ['filmmaker', 'director'],
       'film actor': ['actor', 'actress'], 'actor': ['actress'], 'athletics competitor': ['runner', 'athlete', 'long-distance'],
       'businessperson': ['businessman', 'entrepreneur'], 'military officer': ['marshal', 'general', 'military'],
       'monarch': ['queen'], 'professional wrestler': ['wrestler'], 'competitive diver': ['diver'], 'artistic gymnast': ['gymnast'],
       'ice hockey player': ['hockey player'], 'Formula One driver': ['racing driver', 'formula one', 'formula 1']}
REFUSAL = re.compile(r"do not have|don't have|not able|cannot|can't|no information|not aware|unable|^none$|not publicly|i'm not sure|unknown|please provide|please specify|not available", re.I)
STOP = {'of', 'the', 'de', 'la', 'del', 'y', 'and', 'jr'}


def toks(s):
    return re.sub(r'[^a-z0-9 ]', ' ', unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()).split()


def refusal(ans):
    return bool(REFUSAL.search(ans.strip()))


def correct(rel, gold, ans):
    if refusal(ans): return False
    if rel == 'Height':
        m = re.search(r'\d{3}', ans); return bool(m) and abs(int(m.group()) - int(gold)) <= 3
    g, a = toks(gold), toks(ans)
    if not a: return False
    if rel == 'Occupation':
        return ' '.join(g) in ' '.join(a) or any(s in ans.lower() for s in SYN.get(gold, []))
    if rel == 'BirthPlace':   # city-level answer accepted for districts ("Paris" for "7th arrondissement of Paris")
        return ' '.join(g) in ' '.join(a) or ' '.join(a) in ' '.join(g)
    # Mother / EducatedAt: every content word of the Wikidata label must appear
    gc = [t for t in g if t not in STOP and len(t) > 1] or g
    return all(t in a for t in gc)


def ask(model, key, p, rel, retries=3):
    body = json.dumps(dict(model=model, messages=[{'role': 'user', 'content': Q[rel].format(p=p)}], max_tokens=60,
                           temperature=0)).encode()
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
    for k in range(retries):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=600).read())['choices'][0]['message']['content'] or ''
        except Exception as e:
            err = repr(e); time.sleep(10 * (k + 1))
    return 'ERROR ' + err


def run(model, key, people, truth, workers=8):
    jobs = [(p, r) for p in people for r in Q]
    with ThreadPoolExecutor(workers) as ex:
        ans = list(ex.map(lambda pr: ask(model, key, *pr), jobs))
    return [dict(model=model, person=p, relation=r, gold=truth[p][r], answer=a.strip()[:300],
                 refusal=refusal(a), correct=correct(r, truth[p][r], a)) for (p, r), a in zip(jobs, ans)]
