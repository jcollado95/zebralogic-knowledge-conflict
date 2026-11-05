"""Recovered notebook code that produced `persons_two_models.png`.

Reconstructed from session lineage, not written as a standalone script.
Expect adjustments (input paths, working directory via WORKDIR) before rerunning.
Steps that restored files from the artifact store are replaced by `pass`;
those files are under src/ and data/ in this repository.
"""
import os, sys, json, shutil, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from math import comb

# --- skill:figure-style helpers ---
META_GREY = "#888888"

def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")
    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
    import matplotlib.pyplot as plt
    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 8) + 1
    s = letter.lower() if case == "lower" else letter.upper()
    ax.text(dx, dy, s, transform=ax.transAxes,
            fontweight="bold", fontsize=fontsize, va="bottom", ha="left")


# --- setup files ---
os.makedirs('out_persons', exist_ok=True)
shutil.copy('src/run_eval.py', 'run_eval.py')
shutil.copy('data/persons/unique_persons.csv', 'unique_persons.csv')
shutil.copy('data/persons/knowledge_probe.csv', 'out_persons/knowledge_probe_both.csv')
shutil.copy('data/puzzles/zebralogic_persons.parquet', 'out_persons/zebralogic_persons.parquet')
shutil.copy('raw/persons_gemma4e4b.jsonl', 'out_persons/persons_gemma4e4b.jsonl')

sys.path.insert(0, '.')
import run_eval as E

REL = {'place of birth': 'BirthPlace', 'occupation': 'Occupation', 'mother': 'Mother',
       'educated at': 'EducatedAt', 'height (cm)': 'Height'}
truth = {r['person']: {REL[c]: r[c] for c in REL}
         for r in csv.DictReader(open('unique_persons.csv', encoding='utf-8-sig'))}
json.dump(truth, open('out_persons/persons_truth.json', 'w', encoding='utf-8'), ensure_ascii=False)

# --- compare_models logic (inline) ---
D = 'out_persons'
# NOTE: persons_gemma4_31b.jsonl must be present; it was produced by running the 31B model in the trace.
# Here we expect it at out_persons/persons_gemma4_31b.jsonl (user must supply it).
MODELS = {'gemma-4-E4B-it': 'persons_gemma4e4b.jsonl', 'gemma-4-31B-it': 'persons_gemma4_31b.jsonl'}
var = pd.read_parquet(f'{D}/zebralogic_persons.parquet').set_index('id')
KP = pd.read_csv(f'{D}/knowledge_probe_both.csv')
known = {m: set(zip(g.person[g.correct], g.relation[g.correct])) for m, g in KP.groupby('model')}
rng = np.random.default_rng(0)
n_ = E.norm


def binom2(k, n):
    if n == 0: return 1.0
    pk = comb(n, k) / 2 ** n
    return min(1.0, sum(comb(n, i) / 2 ** n for i in range(n + 1) if comb(n, i) / 2 ** n <= pk * (1 + 1e-9)))


def boot(x, B=5000):
    x = np.asarray(x, float)
    return np.percentile(x[rng.integers(0, len(x), (B, len(x)))].mean(1), [2.5, 97.5])


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(p) - r) * p[i])); adj[i] = run
    return adj


S, C = [], []
for m, fn in MODELS.items():
    recs = {}
    for l in open(f'{D}/{fn}', encoding='utf-8'):
        r = json.loads(l)
        if r.get('finish') != 'error': recs[r['uid']] = r
    for uid, r in recs.items():
        v = var.loc[uid]; sol = json.loads(v.solution); sc = E.score(r['content'], sol); u = r.get('usage') or {}
        S.append(dict(model=m, uid=uid, base_id=v.base_id, size=v['size'], condition=v.condition, finish=r['finish'],
                      completion_tokens=u.get('completion_tokens'), **sc))
        _, tab = E.parse(r['content']); keys = sol['header'][1:]; n = len(sol['rows'])
        pred = {}
        for h, pr in (tab or {}).items():
            pr = {n_(k): x for k, x in pr.items()}; pred[n_(pr.get('name', ''))] = pr
        dom = {k: {n_(row[j + 1]) for row in sol['rows']} for j, k in enumerate(keys)}
        for row in sol['rows']:
            g = dict(zip(keys, row[1:])); name = g['Name']
            real = name in truth
            pr = pred.get(n_(name), {})
            for k in keys[1:]:
                pv = n_(pr.get(n_(k), ''))
                C.append(dict(model=m, uid=uid, condition=v.condition, n=n, attr=k, person=name if real else None,
                              correct=pv == n_(g[k]), in_domain=pv in dom[k],
                              intrusion=bool(real and pv == n_(truth[name][k]) and n_(truth[name][k]) != n_(g[k])),
                              known=bool(real and (name, k) in known[m])))
S = pd.DataFrame(S); C = pd.DataFrame(C)

# per model x condition summary
summ = []
for (m, c), g in S.groupby(['model', 'condition']):
    lo, hi = boot(g.puzzle_correct); clo, chi = boot(g.cell_acc)
    summ.append(dict(model=m, condition=c, n=len(g), puzzle_acc=g.puzzle_correct.mean(), puzzle_lo=lo, puzzle_hi=hi,
                     cell_acc=g.cell_acc.mean(), cell_lo=clo, cell_hi=chi, parse_fail=int((g.parse == 'fail').sum()),
                     truncated=int((g.finish == 'length').sum()), median_tokens=g.completion_tokens.median()))
summ = pd.DataFrame(summ)

# within-model paired contrasts
T = []
for m, gm in S.groupby('model'):
    W = gm.pivot_table(index='base_id', columns='condition', values=['puzzle_correct', 'cell_acc'])
    rows = []
    for a, b in [('aligned', 'control'), ('conflict', 'control'), ('aligned', 'conflict')]:
        pa, pb = W['puzzle_correct'][a].astype(bool), W['puzzle_correct'][b].astype(bool)
        n10, n01 = int((pa & ~pb).sum()), int((~pa & pb).sum()); d = W['cell_acc'][a] - W['cell_acc'][b]; lo, hi = boot(d)
        rows.append(dict(model=m, contrast=f'{a} - {b}', puzzle_diff_pp=100 * (pa.mean() - pb.mean()), a_only=n10, b_only=n01,
                         mcnemar_p=binom2(n10, n10 + n01), cell_diff_pp=100 * d.mean(), cell_ci_lo=100 * lo, cell_ci_hi=100 * hi))
    for r, p in zip(rows, holm([r['mcnemar_p'] for r in rows])): r['holm_p'] = p
    T += rows
PT = pd.DataFrame(T)

# knowledge-conditioned DiD
K = []
for m in MODELS:
    c = C[(C.model == m) & (C.condition.isin(['aligned', 'conflict']))].copy()
    c['base'] = c.uid.str.split('|').str[0]
    tab = c.groupby(['known', 'condition']).correct.agg(['mean', 'size']).unstack('condition')
    bases = c.base.unique(); grp = {b: g for b, g in c.groupby('base')}; dids = []
    for _ in range(2000):
        s = pd.concat([grp[b] for b in rng.choice(bases, len(bases))]); t = s.groupby(['known', 'condition']).correct.mean()
        try: dids.append((t[(True, 'aligned')] - t[(True, 'conflict')]) - (t[(False, 'aligned')] - t[(False, 'conflict')]))
        except KeyError: pass
    gk = tab.loc[True, ('mean', 'aligned')] - tab.loc[True, ('mean', 'conflict')]
    gu = tab.loc[False, ('mean', 'aligned')] - tab.loc[False, ('mean', 'conflict')]
    lo, hi = np.percentile(dids, [2.5, 97.5])
    K.append(dict(model=m, known_cells_aligned=int(tab.loc[True, ('size', 'aligned')]), known_cells_conflict=int(tab.loc[True, ('size', 'conflict')]),
                  acc_known_aligned=tab.loc[True, ('mean', 'aligned')], acc_known_conflict=tab.loc[True, ('mean', 'conflict')],
                  acc_unknown_aligned=tab.loc[False, ('mean', 'aligned')], acc_unknown_conflict=tab.loc[False, ('mean', 'conflict')],
                  gap_known_pp=100 * gk, gap_unknown_pp=100 * gu, did_pp=100 * (gk - gu), did_ci_lo=100 * lo, did_ci_hi=100 * hi,
                  did_p_boot=2 * min(np.mean(np.array(dids) <= 0), np.mean(np.array(dids) >= 0))))
KD = pd.DataFrame(K).set_index('model')

# cluster-bootstrap gaps for panel c
gaps = {}
for m in ['gemma-4-E4B-it', 'gemma-4-31B-it']:
    c = C[(C.model == m) & C.condition.isin(['aligned', 'conflict'])].copy()
    c['base'] = c.uid.str.split('|').str[0]
    agg = c.groupby(['base', 'known', 'condition']).correct.agg(['sum', 'size']).unstack(['known', 'condition'], fill_value=0)
    bases = agg.index.values

    def gap(a, kn):
        return a[('sum', kn, 'aligned')].sum() / a[('size', kn, 'aligned')].sum() - a[('sum', kn, 'conflict')].sum() / a[('size', kn, 'conflict')].sum()

    for kn in [False, True]:
        est = gap(agg, kn)
        bs = [gap(agg.iloc[rng.integers(0, len(bases), len(bases))], kn) for _ in range(3000)]
        gaps[(m, kn)] = (100 * est, *np.percentile(100 * np.array(bs), [2.5, 97.5]))

# --- plot ---
apply_figure_style()
MOD = ['gemma-4-E4B-it', 'gemma-4-31B-it']
ML = {'gemma-4-E4B-it': 'gemma-4 E4B', 'gemma-4-31B-it': 'gemma-4 31B'}
order = ['control', 'aligned', 'conflict']
lab = {'control': 'fictional names', 'aligned': 'true facts', 'conflict': 'shuffled facts'}
col = {'control': META_GREY, 'aligned': '#2a6fb0', 'conflict': '#d08a1e'}
mcol = {'gemma-4-E4B-it': '#9ab3cf', 'gemma-4-31B-it': '#1f3f66'}

fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw=dict(width_ratios=[1.25, 1.05, 1.0]))

ax = axs[0]; w = 0.26
for i, m in enumerate(MOD):
    for j, c in enumerate(order):
        s = summ[(summ.model == m) & (summ.condition == c)].iloc[0]; x = i + (j - 1) * w
        ax.bar(x, 100 * s.puzzle_acc, width=w * 0.92, color=col[c], label=lab[c] if i == 0 else None)
        ax.errorbar(x, 100 * s.puzzle_acc, yerr=[[100 * (s.puzzle_acc - s.puzzle_lo)], [100 * (s.puzzle_hi - s.puzzle_acc)]], color='k', lw=0.7, capsize=1.5)
p31 = PT[(PT.model == 'gemma-4-31B-it') & (PT.contrast == 'aligned - conflict')].iloc[0]
ya = 100 * summ[(summ.model == 'gemma-4-31B-it') & (summ.condition == 'aligned')].puzzle_hi.iloc[0] + 3
ax.plot([1, 1, 1 + w, 1 + w], [ya, ya + 2, ya + 2, ya], color='k', lw=0.7)
ax.text(1 + w / 2, ya + 3, f'p = {p31.holm_p:.2f}', ha='center', va='bottom', fontsize=6)
ax.set_xticks([0, 1]); ax.set_xticklabels([ML[m] for m in MOD]); ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20))
ax.set_ylabel('Whole-puzzle accuracy (%)')
ax.legend(frameon=False, fontsize=6, loc='upper left', handlelength=1, borderaxespad=0.1)
ax.set_title('Only the 31B model is hurt by\nshuffled facts', loc='left')

ax = axs[1]
rels = ['Occupation', 'BirthPlace', 'Height', 'Mother', 'EducatedAt']
rl = {'Occupation': 'occupation', 'BirthPlace': 'birthplace', 'Height': 'height', 'Mother': 'mother', 'EducatedAt': 'school'}
y = np.arange(len(rels)); h = 0.38
for k, m in enumerate(MOD):
    v = KP[KP.model == m].groupby('relation').correct.mean()[rels] * 100
    yy = y + (k - 0.5) * h
    ax.barh(yy, v, height=h * 0.92, color=mcol[m], label=f"{ML[m]} ({int(KP[KP.model == m].correct.sum())}/190)")
    for yi, vi in zip(yy, v):
        if vi == 0: ax.plot([0, 0.6], [yi, yi], color=mcol[m], lw=2.2, solid_capstyle='butt')
ax.set_yticks(y); ax.set_yticklabels([rl[r] for r in rels]); ax.invert_yaxis(); ax.set_xlim(0, 50)
ax.set_xlabel('Facts answered correctly (%)')
ax.legend(frameon=False, fontsize=6, loc='lower right', handlelength=1)
ax.set_title('31B knows about twice as\nmany facts', loc='left')

ax = axs[2]
for k, m in enumerate(MOD):
    for j, kn in enumerate([False, True]):
        est, lo, hi = gaps[(m, kn)]; x = k + (j - 0.5) * 0.36
        ax.bar(x, est, width=0.32, color=mcol[m] if kn else 'white', hatch=None if kn else '////', edgecolor=mcol[m], lw=0.8)
        ax.errorbar(x, est, yerr=[[est - lo], [hi - est]], color='k', lw=0.7, capsize=1.5)
    r = KD.loc[m]; ax.text(k, 26, f'diff {r.did_pp:+.1f} pp\np = {r.did_p_boot:.2f}', ha='center', va='bottom', fontsize=6)
ax.axhline(0, color='k', lw=0.6)
ax.legend(handles=[Patch(facecolor='white', edgecolor='k', hatch='////', label='facts the model does not know'),
                   Patch(facecolor='0.35', label='facts the model knows')],
          frameon=False, fontsize=6, loc='lower left', handlelength=1.2, borderaxespad=0.1)
ax.set_xticks([0, 1]); ax.set_xticklabels([ML[m] for m in MOD]); ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-14, 34); ax.set_ylabel('Cell accuracy, true − shuffled\nfacts (pp)')
ax.set_title('Gap is larger where 31B\nknows the fact', loc='left')

for a_, l in zip(axs, 'abc'): panel_letter(a_, l)
fig.tight_layout()
fig.savefig('persons_two_models.png', dpi=300)
