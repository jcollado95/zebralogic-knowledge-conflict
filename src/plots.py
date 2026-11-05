"""Figures of the knowledge-conflict experiment, drawn from the result tables.

  candidates_v3_sitelinks.png   fame distribution of the candidate population
  mcqa_knowledge_vs_fame.png    MCQA probe accuracy by fame band, per model and fact
  pools_v3_by_fame.png          knowledge pools by fame band
  persons_v3_results.png        main result: puzzle accuracy, recognition-conditioned gap, truncation
Usage: python plots.py [--data ../data] [--results ../results]
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GREY = '#888888'
BANDS = ['a_0-4', 'b_5-9', 'c_10-19', 'd_20-49', 'e_50-99', 'f_100+']
BAND_LABELS = ['0–4', '5–9', '10–19', '20–49', '50–99', '100+']


def style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
                         'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.linewidth': 0.6, 'axes.spines.top': False,
                         'axes.spines.right': False, 'legend.frameon': False, 'axes.titlelocation': 'left',
                         'savefig.dpi': 300, 'savefig.bbox': 'tight', 'lines.linewidth': 1.2})


def letter(ax, s):
    ax.text(-0.18, 1.02, s, transform=ax.transAxes, fontweight='bold', fontsize=9, va='bottom', ha='left')


def candidates_sitelinks(cand, path):
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.hist(np.log10(cand.sitelinks.clip(lower=1)), bins=np.linspace(0, 2.6, 40), color='#2a6fb0')
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['1', '10', '100'])
    ax.set_xlabel('Wikipedia sitelinks (log scale; 0 shown as 1)'); ax.set_ylabel('People')
    ax.set_title(f'Fame-stratified candidates (n = {len(cand):,})')
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def knowledge_vs_fame(G, path):
    facts = [('birth_year', 'Birth year'), ('occupation', 'Occupation'), ('birthplace', 'Birthplace')]
    col = {'gemma-4-E4B-it': '#9ab3cf', 'gemma-4-31B-it': '#1f3f66'}; lab = {'gemma-4-E4B-it': 'gemma-4 E4B', 'gemma-4-31B-it': 'gemma-4 31B'}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, (f, ft) in zip(axs, facts):
        for m in col:
            g = G[(G.model == m) & (G.fact == f)].set_index('band').loc[BANDS]
            p, n = g['mean'].values, g['size'].values; se = 1.96 * np.sqrt(p * (1 - p) / n); x = np.arange(6)
            ax.fill_between(x, 100 * (p - se), 100 * (p + se), color=col[m], alpha=0.2, lw=0)
            ax.plot(x, 100 * p, '-o', color=col[m], ms=3, lw=1.4, label=lab[m])
        ax.axhline(25, color='0.5', lw=0.8, ls='--'); ax.set_xticks(range(6)); ax.set_xticklabels(BAND_LABELS)
        ax.set_title(ft); ax.set_ylim(15, 102)
    axs[2].text(5.2, 26, 'chance', fontsize=6, ha='right', va='bottom', color='0.4')
    axs[0].set_ylabel('Multiple-choice accuracy (%)'); axs[1].set_xlabel('Wikipedia sitelinks (fame band)')
    axs[0].legend(fontsize=6, loc='upper left')
    fig.suptitle('Knowledge rises with fame; 31B knows more, most clearly for birthplaces', x=0.06, ha='left', fontsize=8, y=1.02)
    fig.savefig(path); plt.close(fig)


def pools_by_fame(B, summ, path):
    B = B.set_index('band').reindex(BANDS); summ = summ.set_index('pool_final')
    pc = {'both': '#1f3f66', 'only_large': '#d08a1e', 'none': GREY}; mk = {'both': 'o', 'only_large': 's', 'none': '^'}
    pl = {'both': 'both know', 'only_large': 'only 31B knows', 'none': 'neither knows'}
    fig, ax = plt.subplots(figsize=(4.2, 2.8)); x = np.arange(6)
    for i, p in enumerate(['both', 'only_large', 'none']):
        ax.plot(x + (i - 1) * 0.18, B[p].values, mk[p], color=pc[p], ms=4.5, ls='none', label=f"{pl[p]} (n={int(summ.loc[p, 'people'])})")
    ax.set_yscale('symlog', linthresh=1); ax.set_yticks([0, 1, 10, 100, 400]); ax.set_yticklabels(['0', '1', '10', '100', '400']); ax.set_ylim(-0.3, 700)
    ax.set_xticks(x); ax.set_xticklabels(BAND_LABELS); ax.set_xlabel('Wikipedia sitelinks (fame band)'); ax.set_ylabel('People (symlog scale)')
    ax.legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, handletextpad=0.3, columnspacing=1.0)
    ax.set_title('Pools separate by fame: only-31B people sit in the middle')
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main_results(S, K, Sc, path):
    models = [('gemma-4-E4B-it', 'Gemma-4 E4B'), ('gemma-4-31B-it', 'Gemma-4 31B')]
    pools = ['both', 'only_large', 'none']; labs = ['both', 'only\n31B', 'none']; C_T, C_S = '#2b6cb0', '#d97706'
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[1.7, 0.75, 1.2], wspace=0.5)); xt = []
    ax = axs[0]
    for i, (m, ml) in enumerate(models):
        for j, p in enumerate(pools):
            x = i * 3.6 + j; xt.append(x); v = {}
            for cond, col, dx in [('aligned', C_T, -0.13), ('conflict', C_S, 0.13)]:
                r = S[(S.model == m) & (S.pool == p) & (S.condition == cond)].iloc[0]; v[cond] = 100 * r.puzzle_acc
                ax.errorbar(x + dx, 100 * r.puzzle_acc, yerr=[[100 * (r.puzzle_acc - r.puzzle_lo)], [100 * (r.puzzle_hi - r.puzzle_acc)]], fmt='o', color=col, ms=4, lw=1)
            ax.plot([x - 0.13, x + 0.13], [v['aligned'], v['conflict']], color='0.6', lw=0.8, zorder=0)
        ax.text(i * 3.6 + 1, 101, ml, ha='center', fontsize=7)
    ax.set_xticks(xt); ax.set_xticklabels(labs * 2, fontsize=6); ax.set_ylabel('Puzzles fully solved (%)'); ax.set_ylim(20, 106)
    ax.set_xlabel('Who knows the people', fontsize=7)
    ax.plot([], [], 'o', color=C_T, label='true facts'); ax.plot([], [], 'o', color=C_S, label='shuffled facts'); ax.legend(loc='lower right', fontsize=7)
    ax.set_title('Shuffled facts hurt 31B\nwhen both models know the people')
    ax = axs[1]
    for i, (m, ml) in enumerate(models):
        for kn, col, dx in [(True, '0.15', -0.14), (False, '0.6', 0.14)]:
            r = K[(K.model == m) & (K.recognised.astype(str) == str(kn))].iloc[0]
            ax.errorbar(i + dx, r.gap_pp, yerr=[[r.gap_pp - r.gap_lo], [r.gap_hi - r.gap_pp]], fmt='s', color=col, ms=4, lw=1)
    ax.axhline(0, color='0.7', lw=0.6); ax.set_xticks([0, 1]); ax.set_xticklabels(['E4B', '31B']); ax.set_xlim(-0.5, 1.5); ax.set_ylim(-11, 22)
    ax.set_ylabel('Cell accuracy, true − shuffled (pp)')
    ax.plot([], [], 's', color='0.15', label='fact recognised'); ax.plot([], [], 's', color='0.6', label='not recognised'); ax.legend(loc='upper left', fontsize=6)
    ax.set_title('Cost tracks\nrecognition in 31B')
    T = Sc.groupby(['model', 'pool', 'condition']).finish.apply(lambda s: 100 * (s == 'length').mean()).unstack()
    ax = axs[2]; w = 0.36
    for i, (m, ml) in enumerate(models):
        for j, p in enumerate(pools):
            x = i * 3.6 + j; ax.bar(x - w / 2, T.loc[(m, p), 'aligned'], w, color=C_T); ax.bar(x + w / 2, T.loc[(m, p), 'conflict'], w, color=C_S)
        ax.text(i * 3.6 + 1, 28, ml.replace('Gemma-4 ', ''), ha='center', fontsize=7)
    ax.set_xticks(xt); ax.set_xticklabels(labs * 2, fontsize=6); ax.set_ylabel('Hit 8,192-token limit (%)'); ax.set_ylim(0, 30)
    ax.set_xlabel('Who knows the people', fontsize=7); ax.set_title('31B runs out of tokens\nmore often on shuffled facts')
    for a, l in zip(axs, 'abc'): letter(a, l)
    fig.savefig(path); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='../data'); ap.add_argument('--results', default='../results'); a = ap.parse_args()
    R_ = lambda f: pd.read_csv(os.path.join(a.results, f)); out = lambda f: os.path.join(a.results, f)
    style()
    candidates_sitelinks(pd.read_csv(os.path.join(a.data, 'candidates_v3.csv'), encoding='utf-8-sig'), out('candidates_v3_sitelinks.png'))
    knowledge_vs_fame(R_('mcqa_by_band_long.csv'), out('mcqa_knowledge_vs_fame.png'))
    pools_by_fame(R_('pools_v3_by_band.csv'), R_('pools_v3_summary.csv'), out('pools_v3_by_fame.png'))
    main_results(R_('v3_summary.csv'), R_('v3_knowledge_gap.csv'), R_('v3_scored.csv'), out('persons_v3_results.png'))
