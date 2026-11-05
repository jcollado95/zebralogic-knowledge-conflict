"""Recovered notebook code that produced `persons_gemma4e4b_results.png`.

Reconstructed from session lineage, not written as a standalone script.
Expect adjustments (input paths, working directory via WORKDIR) before rerunning.
Steps that restored files from the artifact store are replaced by `pass`;
those files are under src/ and data/ in this repository.
"""
# skill:figure-style kernel.py (auto-injected on skill load)
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


import numpy as np, matplotlib.pyplot as plt, pandas as pd, os

apply_figure_style()

summ = pd.read_csv('results/02_persons_e4b/persons_summary.csv').set_index('condition')
kp = pd.read_csv('data/persons/knowledge_probe.csv')
O = pd.read_csv('results/02_persons_e4b/persons_intrusion.csv').set_index('subset')

order = ['control', 'aligned', 'conflict']
lab = {'control': 'fictional names', 'aligned': 'real people,\ntrue facts', 'conflict': 'real people,\nshuffled facts'}
col = {'control': META_GREY, 'aligned': '#2a6fb0', 'conflict': '#d08a1e'}

fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.1, 1.2, 1]))

ax = axs[0]
for i, c in enumerate(order):
    s = summ.loc[c]; ax.bar(i, s.puzzle_acc * 100, color=col[c], width=0.65)
    ax.errorbar(i, s.puzzle_acc * 100, yerr=[[100 * (s.puzzle_acc - s.puzzle_lo)], [100 * (s.puzzle_hi - s.puzzle_acc)]], color='k', lw=0.8, capsize=2)
    ax.text(i, s.puzzle_hi * 100 + 2, f'{s.puzzle_acc * 100:.1f}%', ha='center', va='bottom', fontsize=7)
ax.set_xticks(range(3)); ax.set_xticklabels([lab[c] for c in order]); ax.set_ylim(0, 80); ax.set_ylabel('Whole-puzzle accuracy (%)')
ax.set_title('Real-person framing does not\nchange accuracy', loc='left')

ax = axs[1]
rels = ['Occupation', 'BirthPlace', 'Height', 'Mother', 'EducatedAt']
rl = {'Occupation': 'occupation', 'BirthPlace': 'birthplace', 'Height': 'height', 'Mother': 'mother', 'EducatedAt': 'school'}
g = kp.groupby('relation'); corr = g.correct.mean() * 100; ref = g.refusal.mean() * 100; wrong = 100 - corr - ref; y = np.arange(len(rels))
ax.barh(y, corr[rels], color='#2a6fb0', label='correct'); ax.barh(y, wrong[rels], left=corr[rels], color='#b9c7d6', label='wrong answer')
ax.barh(y, ref[rels], left=(corr + wrong)[rels], color='#e6e6e6', label='declines')
ax.set_yticks(y); ax.set_yticklabels([rl[r] for r in rels]); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel('Closed-book answers (%)')
ax.legend(frameon=False, fontsize=6, ncol=3, loc='upper center', bbox_to_anchor=(0.45, -0.22), handlelength=1, columnspacing=0.8)
ax.set_title('The model knows few of these facts', loc='left')

ax = axs[2]
for i, nm in enumerate(['unknown', 'known']):
    o = O.loc[nm]; ax.bar(i, 100 * o.obs_rate, color='#d08a1e', width=0.6)
    ax.plot([i - 0.36, i + 0.36], [100 * o.chance_rate] * 2, color='k', lw=1, ls='--')
    ax.text(i, 100 * o.obs_rate + 1.5, f'{100 * o.obs_rate:.0f}%\n(n={int(o.wrong_in_domain)})', ha='center', va='bottom', fontsize=6)
ax.text(1.42, 100 * O.loc['known'].chance_rate, 'chance', fontsize=6, va='center')
ax.set_xticks([0, 1]); ax.set_xticklabels(['unknown facts', 'known facts']); ax.set_xlim(-0.6, 1.9); ax.set_ylim(0, 60)
ax.set_ylabel("Errors that pick the\nperson's true value (%)"); ax.set_title(f"Errors do not track true facts\n(known: p = {O.loc['known'].p_sim:.2f}, n.s.)", loc='left')

for a_, l in zip(axs, 'abc'):
    panel_letter(a_, l)
fig.tight_layout()
fig.savefig('persons_gemma4e4b_results.png', dpi=300)
