"""Recovered notebook code that produced `persons_v3_results.png`.

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


import os, json, shutil
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Load required data files
shutil.copy('src/run_eval.py', 'run_eval.py')

import sys
sys.path.insert(0, '.')
import run_eval as E

# Load analysis outputs
S = pd.read_csv('PATH_TO_INPUT.csv') if False else None

# Since we need to build from the scored data, load directly
# The figure uses: v3_summary.csv, v3_knowledge_gap.csv, v3_scored.csv
# These are produced by analyze_v3.py which reads from out/analysis/
# We need to reconstruct them from the scored data

# Load the scored CSV and knowledge gap CSV
# These were saved to out/analysis/ during the trace
# We read them from there directly

S = pd.read_csv('out/analysis/v3_summary.csv')
K = pd.read_csv('out/analysis/v3_knowledge_gap.csv')
Sc = pd.read_csv('out/analysis/v3_scored.csv')

apply_figure_style(sizes=(8,7,6))
S = S[S.condition.isin(['aligned','conflict'])] if 'condition' in S else S
models=[('gemma-4-E4B-it','Gemma-4 E4B'),('gemma-4-31B-it','Gemma-4 31B')]
pools=[('both','both know'),('only_large','only 31B knows'),('none','neither knows')]
C_T,C_S='#2b6cb0','#d97706'
labs=['both','only\n31B','none']

fig,axs=plt.subplots(1,3,figsize=(7.4,2.9),gridspec_kw=dict(width_ratios=[1.7,0.75,1.2],wspace=0.5))
xt=[]
ax=axs[0]
for i,(m,ml) in enumerate(models):
    for j,(p,pl) in enumerate(pools):
        x=i*3.6+j; xt.append(x)
        v={}
        for cond,col,dx in [('aligned',C_T,-0.13),('conflict',C_S,0.13)]:
            r=S[(S.model==m)&(S.pool==p)&(S.condition==cond)].iloc[0]; v[cond]=100*r.puzzle_acc
            ax.errorbar(x+dx,100*r.puzzle_acc,yerr=[[100*(r.puzzle_acc-r.puzzle_lo)],[100*(r.puzzle_hi-r.puzzle_acc)]],fmt='o',color=col,ms=4,lw=1)
        ax.plot([x-0.13,x+0.13],[v['aligned'],v['conflict']],color='0.6',lw=0.8,zorder=0)
    ax.text(i*3.6+1,101,ml,ha='center',fontsize=7)
ax.set_xticks(xt); ax.set_xticklabels(labs*2,fontsize=6); ax.set_ylabel('Puzzles fully solved (%)'); ax.set_ylim(20,106); ax.set_xlabel('Who knows the people',fontsize=7)
ax.plot([],[],'o',color=C_T,label='true facts'); ax.plot([],[],'o',color=C_S,label='shuffled facts'); ax.legend(loc='lower right',frameon=False,fontsize=7)
ax.set_title('Shuffled facts hurt 31B\nwhen both models know the people',loc='left',fontsize=8)

ax=axs[1]
for i,(m,ml) in enumerate(models):
    for kn,col,dx in [(True,'0.15',-0.14),(False,'0.6',0.14)]:
        r=K[(K.model==m)&(K.recognised.astype(str)==str(kn))].iloc[0]
        ax.errorbar(i+dx,r.gap_pp,yerr=[[r.gap_pp-r.gap_lo],[r.gap_hi-r.gap_pp]],fmt='s',color=col,ms=4,lw=1)
ax.axhline(0,color='0.7',lw=0.6); ax.set_xticks([0,1]); ax.set_xticklabels(['E4B','31B']); ax.set_xlim(-0.5,1.5); ax.set_ylim(-11,22)
ax.set_ylabel('Cell accuracy, true − shuffled (pp)')
ax.plot([],[],'s',color='0.15',label='fact recognised'); ax.plot([],[],'s',color='0.6',label='not recognised'); ax.legend(loc='upper left',frameon=False,fontsize=6)
ax.set_title('Cost tracks\nrecognition in 31B',loc='left',fontsize=8)

T=Sc.groupby(['model','pool','condition']).finish.apply(lambda s:100*(s=='length').mean()).unstack()
ax=axs[2]; w=0.36
for i,(m,ml) in enumerate(models):
    for j,(p,pl) in enumerate(pools):
        x=i*3.6+j; ax.bar(x-w/2,T.loc[(m,p),'aligned'],w,color=C_T); ax.bar(x+w/2,T.loc[(m,p),'conflict'],w,color=C_S)
    ax.text(i*3.6+1,28,ml,ha='center',fontsize=7)
ax.set_xticks(xt); ax.set_xticklabels(labs*2,fontsize=6); ax.set_ylabel('Hit 8,192-token limit (%)'); ax.set_ylim(0,30); ax.set_xlabel('Who knows the people',fontsize=7)
ax.set_title('31B runs out of tokens\nmore often on shuffled facts',loc='left',fontsize=8)

for a,l in zip(axs,'abc'): panel_letter(a,l)

for t in axs[2].texts:
    if t.get_text().startswith('Gemma'): t.set_text(t.get_text().replace('Gemma-4 ',''))

fig.savefig('persons_v3_results.png',dpi=300,bbox_inches='tight')
