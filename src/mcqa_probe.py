"""Model-agnostic multiple-choice (MCQA) knowledge probe for the person population (candidates_v3.csv).

One standard 4-option question per person x fact (birthplace, occupation, birth year):
  * the correct option is the person's displayed value (candidates_v3.csv)
  * 3 distractors that are never true of the person:
      birthplace : other candidates' birthplaces, same country (P17) when >= 3 are available; never equal to or
                   word-contained in any accepted name of the true birthplace (label, aliases, admin parents)
      occupation : other candidates' displayed occupations, sampled by frequency; never equal to / word-contained in
                   any of the person's P106 labels or their aliases
      birth year : years within +-25 of the true year, >= 3 years from it and from each other, not after 2010 (unless the true year is)
  * the correct answer's position (A-D) is exactly balanced per fact (round-robin over a hash order)
  * greedy decoding; answer format from config['answer_mode']: 'brief' (one short recall sentence, then
    'Answer: X' on the last line; used in the experiment), 'letter' or 'text'
Usage:
  python mcqa_probe.py config.json candidates_v3.csv out_dir [--limit N]
config.json: {"endpoint": "...", "api_key_env": "VLLM_API_KEY", "workers": 24, "models": [{"name": ..., "family": ..., "size": ...}]}
"""
import json, os, re, sys, time, hashlib, random, threading, unicodedata, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

FACTS = {'birthplace': 'Where was {p} born?', 'occupation': 'What is the main occupation of {p}?',
         'birth_year': 'In what year was {p} born?'}
LETTERS = 'ABCD'
INSTRS = {'letter': 'Answer with the letter of the correct option only.',
          'text': 'Answer with the correct option only, copied exactly as written.',
          'brief': 'State the answer in one short sentence, then give the letter of the correct option on the last line as "Answer: X".'}
YEAR_SPAN, YEAR_GAP, MAX_YEAR = 25, 3, 2010   # distractor years never after 2010 (unless the true year is)


def _h(*parts): return int(hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()[:12], 16)
def norm(s): return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()).split())
def related(a, b):
    """equal, or one contains the other as a whole-word sequence"""
    a, b = norm(a), norm(b)
    if not a or not b: return a == b
    return a == b or re.search(rf'(^| ){re.escape(a)}( |$)', b) is not None or re.search(rf'(^| ){re.escape(b)}( |$)', a) is not None


def _pick(pool, k, truths, rng, chosen=()):
    out = list(chosen); seen = set()
    for v in pool:
        if len(out) >= k: break
        if v in seen: continue
        seen.add(v)
        if any(related(v, t) for t in truths) or any(related(v, o) for o in out): continue
        out.append(v)
    return out


def build_questions(cand, seed=0):
    C = cand.reset_index(drop=True)
    bp_by_country = {c: g.birthplace.unique().tolist() for c, g in C.groupby('birth_country')}
    all_bp = C.birthplace.unique().tolist()
    occ_counts = Counter(C.occupation)
    occ_vals, occ_w = list(occ_counts), [occ_counts[o] for o in occ_counts]
    Q = []
    for fact in FACTS:
        order = sorted(C.qid, key=lambda q: _h(seed, fact, q))
        pos = {q: i % 4 for i, q in enumerate(order)}                      # balanced correct positions
        for r in C.itertuples():
            rng = random.Random(_h(seed, fact, r.qid, 'opts'))
            if fact == 'birthplace':
                truths = json.loads(r.birthplace_accepted) + [r.birthplace]; true = r.birthplace
                same = [b for b in bp_by_country.get(r.birth_country, []) if b != true]; rng.shuffle(same)
                d = _pick(same, 3, truths, rng)
                if len(d) < 3:
                    rest = list(all_bp); rng.shuffle(rest); d = _pick(rest, 3, truths, rng, d)
                src = 'same country' if len(_pick(same, 3, truths, rng)) >= 3 else 'mixed'
            elif fact == 'occupation':
                truths = json.loads(r.occupations_all) + json.loads(r.occupation_aliases) + [r.occupation]; true = r.occupation
                pool = rng.choices(occ_vals, weights=occ_w, k=400); d = _pick(pool, 3, truths, rng); src = 'frequency'
            else:
                y = int(r.birth_year); true = str(y)
                cands = [x for x in range(y - YEAR_SPAN, min(y + YEAR_SPAN, max(MAX_YEAR, y)) + 1) if abs(x - y) >= YEAR_GAP]
                rng.shuffle(cands); d = []
                for x in cands:
                    if all(abs(x - z) >= YEAR_GAP for z in d): d.append(x)
                    if len(d) == 3: break
                d = [str(x) for x in d]; src = 'year window'
            if len(d) < 3: continue
            rng.shuffle(d); k = pos[r.qid]; opts = d[:k] + [true] + d[k:]
            Q.append(dict(qid=r.qid, person=r.person, fact=fact, options=opts, answer=LETTERS[k], distractor_source=src))
    return Q


def prompt(q, mode='letter'):
    return FACTS[q['fact']].format(p=q['person']) + '\n' + '\n'.join(f'{L}. {o}' for L, o in zip(LETTERS, q['options'])) + '\n' + INSTRS[mode]


def parse_letter(text):
    """letter at the very start ('B', 'B.', '(B)') or after 'answer/option is'; nothing else"""
    t = (text or '').strip()
    m = re.match(r'^\(?([ABCD])(?:[).:]|\s|$)', t) or re.search(r'\b(?:answer|option)\s*(?:is|:)?\s*\(?([ABCD])\b', t, re.I)
    return m.group(1) if m else None


def parse_brief(text):
    """letter from the final 'Answer: X'; otherwise a lone letter on the last non-empty line"""
    t = (text or '').strip()
    m = re.findall(r'answer\s*[:\-]?\s*\**\(?([ABCD])\b', t, re.I)
    if m: return m[-1].upper()
    last = [l.strip(' *.()') for l in t.splitlines() if l.strip()][-1:] or ['']
    return last[0] if last[0] in LETTERS else None


def parse_text(text, options):
    """map a copied option back to its letter: exact (normalised) match first, then the unique option contained in
    the answer; a bare letter is also accepted"""
    a = norm(text)
    for L, o in zip(LETTERS, options):
        if a == norm(o): return L
    hits = [L for L, o in zip(LETTERS, options) if norm(o) and re.search(rf'(^| ){re.escape(norm(o))}( |$)', a)]
    if len(hits) == 1: return hits[0]
    return parse_letter(text)


def ask(endpoint, key, model, text, max_tokens=8, retries=3):
    body = json.dumps(dict(model=model, messages=[{'role': 'user', 'content': text}], max_tokens=max_tokens, temperature=0)).encode()
    req = urllib.request.Request(endpoint, data=body, headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
    err = ''
    for k in range(retries):
        try: return json.loads(urllib.request.urlopen(req, timeout=600).read())['choices'][0]['message']['content'] or ''
        except Exception as e: err = repr(e); time.sleep(10 * (k + 1))
    return 'ERROR ' + err


def run(cfg, questions, out_dir, models=None, log=print, mode=None):
    mode = mode or cfg.get('answer_mode', 'letter')
    key = os.environ[cfg.get('api_key_env', 'VLLM_API_KEY')]; os.makedirs(out_dir, exist_ok=True)
    for m in cfg['models']:
        if models and m['name'] not in models: continue
        path = os.path.join(out_dir, f"probe_mcqa_{m['name'].replace('/', '_')}.jsonl")
        done = set()
        if os.path.exists(path):
            for l in open(path, encoding='utf-8'):
                r = json.loads(l)
                if not r['raw'].startswith('ERROR'): done.add((r['qid'], r['fact']))
        todo = [q for q in questions if (q['qid'], q['fact']) not in done]; lock = threading.Lock(); t0 = time.time()
        log(f"{m['name']}: {len(done)} done, {len(todo)} to run")

        def job(q):
            raw = ask(cfg['endpoint'], key, m['name'], prompt(q, mode), max_tokens={'letter': 8, 'text': 40, 'brief': 80}[mode])
            L = {'letter': parse_letter, 'brief': parse_brief}.get(mode, lambda t: parse_text(t, q['options']))(raw)
            rec = dict(model=m['name'], family=m.get('family'), size=m.get('size'), mode=mode, **q, raw=raw.strip()[:300], letter=L, correct=L == q['answer'])
            with lock:
                with open(path, 'a', encoding='utf-8') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        with ThreadPoolExecutor(cfg.get('workers', 24)) as ex: list(ex.map(job, todo))
        log(f"{m['name']}: finished {len(todo)} in {time.time() - t0:.0f}s")


if __name__ == '__main__':
    cfg = json.load(open(sys.argv[1])); cand = pd.read_csv(sys.argv[2], encoding='utf-8-sig')
    if '--limit' in sys.argv: cand = cand.sample(int(sys.argv[sys.argv.index('--limit') + 1]), random_state=0)
    run(cfg, build_questions(cand), sys.argv[3])
