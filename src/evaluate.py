"""Query an OpenAI-compatible chat endpoint (vLLM) on ZebraLogic puzzles and score the answers (grid mode).

Prompt: ZeroEval-style grid prompt (one worked 3-house example, JSON {reasoning, solution} answer), reconstructed.
Decoding: temperature 0, max 32,768 completion tokens, SSE streaming (the endpoint drops non-streaming requests
that run longer than ~300 s). An answer that hits the budget is scored as given (usually wrong).
Endpoint: VLLM_URL (default: the lab vLLM server); API key passed by the caller (e.g. from VLLM_API_KEY).
"""
import os, json, re, time, threading, http.client, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get('VLLM_URL', 'http://ada01.ujaen.es:8080/v1/chat/completions')

EXAMPLE = '''# Example Puzzle

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{
    "reasoning": "Given Clue 1, Peter is in House 2. By Clue 2, Arnold is directly left of the water drinker; House 3 cannot be left of anyone, so Arnold is in House 1 and Peter drinks water. Eric is in House 3. By Clue 3, Eric drinks milk, so Arnold drinks tea.",
    "solution": {
        "House 1": {"Name": "Arnold", "Drink": "tea"},
        "House 2": {"Name": "Peter", "Drink": "water"},
        "House 3": {"Name": "Eric", "Drink": "milk"}
    }
}
'''


def build_prompt(puzzle, header, n_houses):
    keys = header[1:]
    tmpl = {'reasoning': '___', 'solution': {f'House {h}': {k: '___' for k in keys} for h in range(1, n_houses + 1)}}
    extra = ''
    return (EXAMPLE + '\n# Puzzle to Solve\n\n' + puzzle.strip() + '\n\n# Instruction\n\nNow please solve the above puzzle. '
            'Present your reasoning and solution in the following json format:\n\n' + json.dumps(tmpl, indent=4) + '\n' + extra)


def chat_stream(prompt, model, key, max_tokens=32768, timeout=600, retries=3):
    """Streaming chat completion; `timeout` is the maximum silence between chunks."""
    body = json.dumps(dict(model=model, messages=[{'role': 'user', 'content': prompt}], max_tokens=max_tokens,
                           temperature=0.0, stream=True, stream_options={'include_usage': True})).encode()
    err = None
    for k in range(retries):
        parts, finish, usage = [], None, None
        try:
            req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
            t = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                for raw in r:
                    line = raw.decode('utf-8', 'replace').strip()
                    if not line.startswith('data:'): continue
                    data = line[5:].strip()
                    if data == '[DONE]': break
                    d = json.loads(data)
                    if d.get('usage'): usage = d['usage']
                    for c in d.get('choices') or []:
                        delta = c.get('delta') or {}
                        if delta.get('content'): parts.append(delta['content'])
                        if c.get('finish_reason'): finish = c['finish_reason']
            if finish is None: raise ConnectionError('stream ended without finish_reason')
            return dict(content=''.join(parts), finish=finish, usage=usage, latency=time.time() - t)
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, json.JSONDecodeError) as e:
            err = e
            time.sleep(15 * (k + 1))
    return dict(content='', finish='error', error=repr(err)[:300], usage=None, latency=None)


def _json_candidates(text):
    """JSON objects in the text, scanning from the end (the answer comes last; earlier prose may contain LaTeX braces)."""
    text = re.sub(r'```(?:json)?', '', text); dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r'\{\s*"', text))):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and 'solution' in obj:
            yield obj


def parse(text):
    """-> ('table', {house_int: {key: value}}) | ('fail', None)"""
    for d in _json_candidates(text):
        sol = d.get('solution')
        if isinstance(sol, dict):
            out = {}
            for hk, row in sol.items():
                m = re.search(r'\d+', str(hk))
                if m and isinstance(row, dict): out[int(m.group())] = row
            if out: return 'table', out
    # fallback: the full object may be invalid JSON (e.g. LaTeX backslashes in "reasoning"); decode the
    # value of the last "solution" key on its own
    t2 = re.sub(r'```(?:json)?', '', text); dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r'"solution"\s*:\s*', t2))):
        try: sol, _ = dec.raw_decode(t2, m.end())
        except json.JSONDecodeError: continue
        if isinstance(sol, dict):
            out = {int(re.search(r'\d+', str(k)).group()): r for k, r in sol.items() if re.search(r'\d+', str(k)) and isinstance(r, dict)}
            if out: return 'table', out
    return 'fail', None


norm = lambda s: re.sub(r'\s+', ' ', str(s)).strip().lower()


def score(text, solution):
    """solution: {header, rows}. Returns parse status, whole-puzzle correctness and cell accuracy."""
    kind, table = parse(text)
    header = solution['header']; keys = header[1:]; total = right = 0
    for row in solution['rows']:
        h = int(row[0]); pred = {norm(k): v for k, v in (table or {}).get(h, {}).items()}
        for k, v in zip(keys, row[1:]):
            total += 1; right += norm(pred.get(norm(k), '')) == norm(v)
    return dict(parse=kind, puzzle_correct=right == total, cell_acc=right / total)


def run(items, model, key, out_path, workers=8, log=print, max_tokens=32768):
    """items: dicts with uid, prompt (+ metadata). Appends JSONL; skips uids already present with finish != 'error'."""
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path, encoding='utf-8'):
            try:
                r = json.loads(l)
                if r.get('finish') != 'error': done.add(r['uid'])
            except Exception: pass
    todo = [it for it in items if it['uid'] not in done]; lock = threading.Lock(); t0 = time.time(); n = [0]
    log(f'{len(done)} done, {len(todo)} to run')

    def job(it):
        r = chat_stream(it['prompt'], model, key, max_tokens=max_tokens)
        rec = {k: v for k, v in it.items() if k != 'prompt'}; rec.update(model=model, max_tokens=max_tokens, **r)
        with lock:
            with open(out_path, 'a', encoding='utf-8') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n[0] += 1
            if n[0] % 25 == 0: log(f'{n[0]}/{len(todo)}  {time.time() - t0:.0f}s')
    with ThreadPoolExecutor(workers) as ex: list(ex.map(job, todo))
    log(f'finished {n[0]} in {time.time() - t0:.0f}s')
