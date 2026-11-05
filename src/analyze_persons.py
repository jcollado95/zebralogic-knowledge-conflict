"""Score + analyse the real-persons ZebraLogic run (aligned / conflict / control).
python analyze_persons.py out_persons  -> persons_scored.csv, persons_summary.csv, persons_paired_tests.csv,
                                          persons_intrusion.csv, persons_gemma4e4b_results.png"""
import sys, json, itertools
import numpy as np, pandas as pd
from math import comb
def binom_two_sided(k, n):
    # exact two-sided binomial test at p=0.5 (= exact McNemar)
    pk = comb(n, k) / 2 ** n
    return min(1.0, sum(comb(n, i) / 2 ** n for i in range(n + 1) if comb(n, i) / 2 ** n <= pk * (1 + 1e-9)))
import run_eval as E

D = sys.argv[1] if len(sys.argv) > 1 else 'out_persons'
var = pd.read_parquet(f'{D}/zebralogic_persons.parquet').set_index('id')
truth = json.load(open(f'{D}/persons_truth.json', encoding='utf-8'))
recs = [json.loads(l) for l in open(f'{D}/persons_gemma4e4b.jsonl', encoding='utf-8')]
recs = {r['uid']: r for r in recs if r.get('finish') != 'error'}   # last good record per uid
norm = E.norm
rows, intr = [], []
for uid, r in recs.items():
    v = var.loc[uid]; sol = json.loads(v.solution); s = E.score(r['content'], sol)
    u = r.get('usage') or {}
    rows.append(dict(uid=uid, base_id=v.base_id, size=v['size'], condition=v.condition, finish=r['finish'],
                     completion_tokens=u.get('completion_tokens'), **s))
    # knowledge intrusion: in the predicted row of person p, how often attribute k equals p's TRUE value
    _, table = E.parse(r['content']); keys = sol['header'][1:]
    if table is None: continue
    gold_by_name = {row[1]: dict(zip(keys, row[1:])) for row in sol['rows']}
    for h, prow in table.items():
        prow = {norm(k): x for k, x in prow.items()}; p = prow.get('name')
        match = next((n for n in gold_by_name if norm(n) == norm(p or '')), None)
        if match is None: continue
        for k in keys[1:]:
            pred = norm(prow.get(norm(k), ''))
            t = truth.get(match, {}).get(k)          # None for fictional (control) names
            intr.append(dict(uid=uid, condition=v.condition, person=match, attr=k,
                             correct=pred == norm(gold_by_name[match][k]),
                             intrusion=(t is not None and pred == norm(t) and norm(t) != norm(gold_by_name[match][k]))))
S = pd.DataFrame(rows); I = pd.DataFrame(intr)
S.to_csv(f'{D}/persons_scored.csv', index=False); I.to_csv(f'{D}/persons_cells.csv', index=False)

rng = np.random.default_rng(0)
def boot(x, B=5000):
    x = np.asarray(x, float); idx = rng.integers(0, len(x), (B, len(x))); m = x[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])
summ = []
for (c), g in S.groupby('condition'):
    lo, hi = boot(g.puzzle_correct); clo, chi = boot(g.cell_acc)
    summ.append(dict(condition=c, n=len(g), puzzle_acc=g.puzzle_correct.mean(), puzzle_lo=lo, puzzle_hi=hi,
                     cell_acc=g.cell_acc.mean(), cell_lo=clo, cell_hi=chi, parse_fail=(g.parse == 'fail').sum(),
                     truncated=(g.finish == 'length').sum(), median_tokens=g.completion_tokens.median()))
summ = pd.DataFrame(summ)
by_size = S.pivot_table(index='size', columns='condition', values='puzzle_correct', aggfunc='mean')

W = S.pivot_table(index='base_id', columns='condition', values=['puzzle_correct', 'cell_acc'])
tests = []
for a, b in [('aligned', 'control'), ('conflict', 'control'), ('aligned', 'conflict')]:
    pa, pb = W['puzzle_correct'][a], W['puzzle_correct'][b]; ok = pa.notna() & pb.notna()
    pa, pb = pa[ok].astype(bool), pb[ok].astype(bool)
    n10, n01 = int((pa & ~pb).sum()), int((~pa & pb).sum())
    p = binom_two_sided(n10, n10 + n01) if n10 + n01 else 1.0
    d = (W['cell_acc'][a] - W['cell_acc'][b])[ok]; lo, hi = boot(d)
    tests.append(dict(contrast=f'{a} - {b}', n_pairs=int(ok.sum()), puzzle_diff_pp=100 * (pa.mean() - pb.mean()),
                      a_only=n10, b_only=n01, mcnemar_p=p, cell_diff_pp=100 * d.mean(), cell_ci_lo=100 * lo, cell_ci_hi=100 * hi))
T = pd.DataFrame(tests)
order = np.argsort(T.mcnemar_p.values); m = len(T); adj = np.empty(m); run = 0
for r_, i in enumerate(order): run = max(run, min(1, (m - r_) * T.mcnemar_p.values[i])); adj[i] = run
T['holm_p'] = adj

# intrusion: conflict only (in aligned the true value IS the gold); chance = P(random wrong value == true value)
ci = I[I.condition == 'conflict']; wrong = ci[~ci.correct]
intr_tab = pd.DataFrame([dict(cells=len(ci), wrong_cells=len(wrong), intrusions=int(wrong.intrusion.sum()),
                              intrusion_rate_among_wrong=wrong.intrusion.mean() if len(wrong) else np.nan)])
summ.to_csv(f'{D}/persons_summary.csv', index=False); T.to_csv(f'{D}/persons_paired_tests.csv', index=False)
intr_tab.to_csv(f'{D}/persons_intrusion.csv', index=False)
print(summ.round(3).to_string()); print(by_size.round(3).to_string()); print(T.round(4).to_string()); print(intr_tab.to_string())
