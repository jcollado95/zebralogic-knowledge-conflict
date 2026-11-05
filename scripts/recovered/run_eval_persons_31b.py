"""Recovered notebook code that produced `persons_gemma4_31b.jsonl`.

Reconstructed from session lineage, not written as a standalone script.
Expect adjustments (input paths, working directory via WORKDIR) before rerunning.
Steps that restored files from the artifact store are replaced by `pass`;
those files are under src/ and data/ in this repository.
"""
import os, sys, json, shutil, pandas as pd

# Copy dependency files to working directory
shutil.copy('src/run_eval.py', 'run_eval.py')
shutil.copy('data/puzzles/zebralogic_persons.parquet', 'zebralogic_persons.parquet')

sys.path.insert(0, '.')
import run_eval as E

M31 = 'gemma-4-31B-it'
P31 = 'persons_gemma4_31b.jsonl'

var = pd.read_parquet('zebralogic_persons.parquet')
items = []
for r in var.itertuples():
    sol = json.loads(r.solution)
    items.append(dict(uid=r.id, base_id=r.base_id, size=r.size, condition=r.condition,
                      prompt=E.build_prompt(r.puzzle, sol['header'], len(sol['rows']))))
items.sort(key=lambda it: (it['base_id'], it['condition']))

logf = open('run_log_31b.txt', 'a', encoding='utf-8')
def lg(m): logf.write(m + '\n'); logf.flush()

E.run(items, M31, os.environ['VLLM_API_KEY'], P31, workers=24, log=lg)

# one retry pass for transport errors
recs = [json.loads(l) for l in open(P31, encoding='utf-8')]
bad = sum(r.get('finish') == 'error' for r in recs)
if bad:
    with open(P31, 'w', encoding='utf-8') as f:
        for r in recs:
            if r.get('finish') != 'error': f.write(json.dumps(r, ensure_ascii=False) + '\n')
    E.run(items, M31, os.environ['VLLM_API_KEY'], P31, workers=8, log=lg)

logf.close()

from collections import Counter
recs = [json.loads(l) for l in open(P31, encoding='utf-8')]
print(len(recs), Counter(r['finish'] for r in recs), 'retried', bad)
