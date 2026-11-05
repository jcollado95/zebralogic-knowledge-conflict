"""Probe candidate people with the shared closed-book questions (knowledge_probe.Q) and score against ALL Wikidata
values of each relation (multi-valued occupations/schools; birthplace accepts the admin parents in
'place of birth (accepted)'). Resumable: appends to probe_<tag>.jsonl and skips (person, relation) already done."""
import json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import knowledge_probe as K

COLS = {'BirthPlace': 'place of birth (accepted)', 'Occupation': 'occupation', 'Mother': 'mother',
        'EducatedAt': 'educated at', 'Height': 'height (cm)'}


def _mother_partial(gold, ans):
    """'Ann Dunham' for 'Stanley Ann Dunham': >=2 answer words, all in the gold name, surname included."""
    if K.refusal(ans): return False
    g = [t for t in K.toks(gold) if t not in K.STOP and len(t) > 1]; a = [t for t in K.toks(ans) if t not in K.STOP and len(t) > 1]
    return len(a) >= 2 and bool(g) and set(a) <= set(g) and g[-1] in a


def score(rel, golds, ans):
    return any(K.correct(rel, str(g), ans) or (rel == 'Mother' and _mother_partial(str(g), ans)) for g in golds)


def run(model, key, cand, out, workers=24, log=print):
    done = set()
    if os.path.exists(out):
        for l in open(out, encoding='utf-8'):
            r = json.loads(l)
            if not r['answer'].startswith('ERROR'): done.add((r['qid'], r['relation']))
    jobs = [(r, rel) for r in cand.to_dict('records') for rel in K.Q if (r['qid'], rel) not in done]
    lock = threading.Lock(); n = [0]; t0 = time.time()
    log(f'{model}: {len(done)} done, {len(jobs)} to run')

    def job(a):
        r, rel = a; ans = K.ask(model, key, r['person'], rel)
        golds = json.loads(r[COLS[rel]])
        rec = dict(model=model, qid=r['qid'], person=r['person'], relation=rel, gold=golds, answer=ans.strip()[:300],
                   refusal=K.refusal(ans), correct=score(rel, golds, ans))
        with lock:
            with open(out, 'a', encoding='utf-8') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n[0] += 1
            if n[0] % 500 == 0: log(f'{model}: {n[0]}/{len(jobs)} {time.time() - t0:.0f}s')
    with ThreadPoolExecutor(workers) as ex: list(ex.map(job, jobs))
    log(f'{model}: finished {n[0]} in {time.time() - t0:.0f}s')
