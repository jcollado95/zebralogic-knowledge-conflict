"""Run the model calls of the experiment against an OpenAI-compatible endpoint (vLLM).

Stages (each resumable: outputs are appended as JSONL and finished items are skipped):
  probe     MCQA knowledge probe, 3 facts x 4,515 candidates per model   -> runs/probe_mcqa_<model>.jsonl
  eval      the 720 true/shuffled puzzles per model                       -> runs/persons_v3_<model>.jsonl
  inpuzzle  in-puzzle MCQA (options = the puzzle's own values)            -> runs/inpuzzle_mcqa_<model>.jsonl
Models, endpoint, workers and answer mode come from config.json; the API key from the env var named in
config['api_key_env'] (default VLLM_API_KEY). VLLM_URL overrides the endpoint.

Usage: python run_experiment.py {probe|eval|inpuzzle} [--config config.json] [--data ../data] [--runs ../runs] [--models m1,m2]
"""
import argparse, json, os, threading
import pandas as pd
import evaluate as E
import mcqa_probe as M
import puzzle_variants as PV


def eval_items(df):
    items = []
    for r in df.itertuples():
        sol = json.loads(r.solution)
        items.append(dict(uid=r.id, base_id=r.base_id, size=r.size, pool=r.pool, condition=r.condition,
                          prompt=E.build_prompt(r.puzzle, sol['header'], len(sol['rows']))))
    items.sort(key=lambda it: (it['base_id'], it['pool'], it['condition']))
    return items


def run_eval(cfg, items, runs, models, key):
    def one(m):
        path = os.path.join(runs, f"persons_v3_{m.replace('/', '_')}.jsonl")
        E.run(items, m, key, path, workers=cfg.get('workers', 24), log=lambda s: print(f'{m}: {s}', flush=True))
        recs = [json.loads(l) for l in open(path, encoding='utf-8')]
        if any(r.get('finish') == 'error' for r in recs):          # drop transport errors and retry them once
            with open(path, 'w', encoding='utf-8') as f:
                for r in recs:
                    if r.get('finish') != 'error': f.write(json.dumps(r, ensure_ascii=False) + '\n')
            E.run(items, m, key, path, workers=8, log=lambda s: print(f'{m} retry: {s}', flush=True))
    th = [threading.Thread(target=one, args=(m,)) for m in models]
    for t in th: t.start()
    for t in th: t.join()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['probe', 'eval', 'inpuzzle'])
    ap.add_argument('--config', default='config.json'); ap.add_argument('--data', default='../data'); ap.add_argument('--runs', default='../runs')
    ap.add_argument('--models', default=None, help='comma-separated subset of config models')
    a = ap.parse_args()
    cfg = json.load(open(a.config)); cfg['endpoint'] = os.environ.get('VLLM_URL', cfg['endpoint']); E.URL = cfg['endpoint']
    models = a.models.split(',') if a.models else [m['name'] for m in cfg['models']]
    os.makedirs(a.runs, exist_ok=True)
    if a.stage == 'probe':
        cand = pd.read_csv(os.path.join(a.data, 'candidates_v3.csv'), encoding='utf-8-sig')
        M.run(cfg, M.build_questions(cand), a.runs, models=models)
    else:
        df = pd.read_parquet(os.path.join(a.data, 'zebralogic_pools_v3.parquet'))
        if a.stage == 'eval':
            run_eval(cfg, eval_items(df), a.runs, models, os.environ[cfg.get('api_key_env', 'VLLM_API_KEY')])
        else:
            PV.run_inpuzzle_mcqa(cfg, PV.inpuzzle_questions(df), os.path.join(a.runs, 'inpuzzle_mcqa_{model}.jsonl'), models=models)
