"""Recovered notebook code that produced `mcqa_knowledge_vs_fame.png`.

Reconstructed from session lineage, not written as a standalone script.
Expect adjustments (input paths, working directory via WORKDIR) before rerunning.
Steps that restored files from the artifact store are replaced by `pass`;
those files are under src/ and data/ in this repository.
"""
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

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


G = pd.read_csv('PATH_TO_INPUT.csv')

# Since mcqa_by_band_long.csv is not in the dependency mappings, we need to reconstruct it
# from the probe files. However, since the file was saved and read back, we load it directly.
# The file was created from P (the probe dataframe) grouped by model, fact, band.
# We need to load the probe files and candidates to reconstruct G.

import shutil, json, sys

# Load candidates_v3 data - we need band info
# candidates_v3.csv was built in the trace but not saved as an artifact
# We reconstruct mcqa_by_band_long.csv from probe files

# Load probe files for both models
# The probe files are not in the dependency mappings either, so we need to
# use what's available. The mcqa_by_band_long.csv was saved in the v3 directory.

# Since mcqa_by_band_long.csv is the direct input to the figure and was saved
# but not listed as an artifact, we reconstruct from available data:

# Actually, looking at the trace carefully:
# G = pd.read_csv('mcqa_by_band_long.csv')  -- this file was created and saved
# But it's not in the dependency mappings. We need to rebuild it.

# The figure code uses G which was read from mcqa_by_band_long.csv
# That file was created by:
# G = P.groupby(['model','fact','band']).correct.agg(['mean','size']).reset_index()
# where P had band from V3 (candidates_v3.csv)

# Since neither candidates_v3.csv nor probe files are artifacts,
# we load from the probe jsonl files that were saved to disk in the trace.
# The probe files for the full run are in 'probe/' subdirectory.

# Minimal reconstruction: load probe data and candidates_v3
# candidates_v3.csv was built and saved locally in the trace
V3 = pd.read_csv('candidates_v3.csv', encoding='utf-8-sig')

cfg = json.load(open('config.json'))
P = pd.concat([pd.read_json(f"probe/probe_mcqa_{m['name']}.jsonl", lines=True) for m in cfg['models']])
P['band'] = P.qid.map(V3.set_index('qid').band)
P['nonletter'] = P.letter.isna()

G = P.groupby(['model', 'fact', 'band']).correct.agg(['mean', 'size']).reset_index()

bands = ['a_0-4', 'b_5-9', 'c_10-19', 'd_20-49', 'e_50-99', 'f_100+']
bl = ['0–4', '5–9', '10–19', '20–49', '50–99', '100+']
facts = [('birth_year', 'Birth year'), ('occupation', 'Occupation'), ('birthplace', 'Birthplace')]
col = {'gemma-4-E4B-it': '#9ab3cf', 'gemma-4-31B-it': '#1f3f66'}
lab = {'gemma-4-E4B-it': 'gemma-4 E4B', 'gemma-4-31B-it': 'gemma-4 31B'}
apply_figure_style()
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, (f, ft) in zip(axs, facts):
    for m in ['gemma-4-E4B-it', 'gemma-4-31B-it']:
        g = G[(G.model == m) & (G.fact == f)].set_index('band').loc[bands]
        p = g['mean'].values; n = g['size'].values
        se = 1.96 * np.sqrt(p * (1 - p) / n)
        x = np.arange(6)
        ax.fill_between(x, 100 * (p - se), 100 * (p + se), color=col[m], alpha=0.2, lw=0)
        ax.plot(x, 100 * p, '-o', color=col[m], ms=3, lw=1.4, label=lab[m])
    ax.axhline(25, color='0.5', lw=0.8, ls='--')
    ax.set_xticks(range(6)); ax.set_xticklabels(bl)
    ax.set_title(ft, loc='left'); ax.set_ylim(15, 102)
axs[2].text(5.2, 26, 'chance', fontsize=6, ha='right', va='bottom', color='0.4')
axs[0].set_ylabel('Multiple-choice accuracy (%)')
axs[1].set_xlabel('Wikipedia sitelinks (fame band)')
axs[0].legend(frameon=False, fontsize=6, loc='upper left')
fig.suptitle('Knowledge rises with fame; 31B knows more, most clearly for birthplaces', x=0.06, ha='left', fontsize=8, y=1.02)
fig.savefig('mcqa_knowledge_vs_fame.png', dpi=300, bbox_inches='tight')
