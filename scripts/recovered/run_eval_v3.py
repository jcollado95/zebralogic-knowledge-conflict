"""Recovered notebook code that produced `persons_v3_gemma-4-31B-it.jsonl`.

Reconstructed from session lineage, not written as a standalone script.
Expect adjustments (input paths, working directory via WORKDIR) before rerunning.
Steps that restored files from the artifact store are replaced by `pass`;
those files are under src/ and data/ in this repository.
"""
import os, sys, json, shutil, threading
import pandas as pd

W = os.environ.get('WORKDIR', '.')
os.makedirs(W, exist_ok=True)
os.chdir(W)
sys.path.insert(0, W)

# Restore required files from artifacts
for f, v in {
    'zebra_perturb.py': '7bf68009-b752-46d6-b20c-2ee776d4405d',
    'pool_variants.py': '602e36d4-21ba-4815-9ffd-8fe29952b2d0',
    'pool_variants_v2.py': '835bc3df-f4fa-4e48-b664-aba75fd5b42c',
    'zebra_grid.parquet': 'debb54c7-80e2-4f93-9af7-1ca5025b2da9',
    'run_eval.py': 'dc5f86f9-64ac-4af4-83c0-4b9a358336e8',
    'pool_variants_v3.py': '17ed65c8-b856-444e-ab42-5673b5e85bc0',
    'mcqa_probe.py': '50be4489-2dfe-4ea9-9c0b-34d86d6a22b6',
}.items():
    pass  # [repo] artifact-store restore removed; file is in src/ or data/

pass  # [repo] artifact-store restore removed; file is in src/ or data/
os.makedirs('pools', exist_ok=True)
pass  # [repo] artifact-store restore removed; file is in src/ or data/
os.makedirs('out', exist_ok=True)
pass  # [repo] artifact-store restore removed; file is in src/ or data/

cfg = json.load(open('config.json')) if os.path.exists('config.json') else {
    "endpoint": "http://ada01.ujaen.es:8080/v1/chat/completions",
    "api_key_env": "VLLM_API_KEY",
    "workers": 24,
    "answer_mode": "brief",
    "models": [
        {"name": "gemma-4-E4B-it", "family": "gemma-4", "size": "small"},
        {"name": "gemma-4-31B-it", "family": "gemma-4", "size": "large"}
    ]
}

import importlib
import pool_variants as PV0
if hasattr(PV0, '_orig_compatible'):
    del PV0._orig_compatible
importlib.reload(PV0)
import pool_variants_v2 as V2m
importlib.reload(V2m)
import pool_variants_v3 as V3m
importlib.reload(V3m)

people = V3m.load_people('candidates_v3.csv', 'pools/pools_v3_gemma4.csv')

df = pd.read_parquet('out/zebralogic_pools_v3.parquet')

import run_eval as E

items = []
for r in df.itertuples():
    sol = json.loads(r.solution)
    items.append(dict(
        uid=r.id, base_id=r.base_id, size=r.size,
        pool=r.pool, condition=r.condition,
        prompt=E.build_prompt(r.puzzle, sol['header'], len(sol['rows']))
    ))
items.sort(key=lambda it: (it['base_id'], it['pool'], it['condition']))

logf = open('out/eval_log.txt', 'a', encoding='utf-8')
lk = threading.Lock()

def lg(m):
    with lk:
        logf.write(m + '\n')
        logf.flush()

def runm(m):
    path = f"out/persons_v3_{m}.jsonl"
    E.run(items, m, os.environ['VLLM_API_KEY'], path, workers=24, log=lambda s: lg(f'{m}: {s}'))
    recs = [json.loads(l) for l in open(path, encoding='utf-8')]
    if any(r.get('finish') == 'error' for r in recs):
        with open(path, 'w', encoding='utf-8') as f:
            for r in recs:
                if r.get('finish') != 'error':
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
        E.run(items, m, os.environ['VLLM_API_KEY'], path, workers=8, log=lambda s: lg(f'{m} retry: {s}'))

th = [threading.Thread(target=runm, args=(m['name'],)) for m in cfg['models']]
for x in th:
    x.start()
for x in th:
    x.join()

logf.close()

from collections import Counter
for m in cfg['models']:
    recs = [json.loads(l) for l in open(f"out/persons_v3_{m['name']}.jsonl", encoding='utf-8')]
    print(m['name'], len(recs), Counter(r['finish'] for r in recs))

print(open('out/eval_log.txt', encoding='utf-8').read()[-500:])
