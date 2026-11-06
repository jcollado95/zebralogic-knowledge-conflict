"""Figures of the knowledge-conflict experiment, drawn from the result tables.

  candidates_v3_sitelinks.png   fame distribution of the candidate population
  mcqa_knowledge_vs_fame.png    MCQA probe accuracy by fame band, per model and fact
  pools_v3_by_fame.png          knowledge pools by fame band
  persons_v3_32k_results.png    main result: puzzle accuracy, recognition-conditioned gap, truncation (32k budget)
  difficulty/difficulty_effects.png   effect of puzzle size (from results/difficulty/*.csv)
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
    ax.set_xticks(xt); ax.set_xticklabels(labs * 2, fontsize=6); ax.set_ylabel('Puzzles fully solved (%)'); ax.set_ylim(15, 106)
    ax.set_xlabel('Who knows the people', fontsize=7)
    ax.plot([], [], 'o', color=C_T, label='true facts'); ax.plot([], [], 'o', color=C_S, label='shuffled facts'); ax.legend(loc='lower right', fontsize=7)
    ax.set_title('Shuffled facts hurt 31B\nwhenever it knows the people')
    ax = axs[1]
    for i, (m, ml) in enumerate(models):
        for kn, col, dx in [(True, '0.15', -0.14), (False, '0.6', 0.14)]:
            r = K[(K.model == m) & (K.recognised.astype(str) == str(kn))].iloc[0]
            ax.errorbar(i + dx, r.gap_pp, yerr=[[r.gap_pp - r.gap_lo], [r.gap_hi - r.gap_pp]], fmt='s', color=col, ms=4, lw=1)
    ax.axhline(0, color='0.7', lw=0.6); ax.set_xticks([0, 1]); ax.set_xticklabels(['E4B', '31B']); ax.set_xlim(-0.5, 1.5); ax.set_ylim(-11, 30)
    ax.set_ylabel('Cell accuracy, true − shuffled (pp)')
    ax.plot([], [], 's', color='0.15', label='fact recognised'); ax.plot([], [], 's', color='0.6', label='not recognised'); ax.legend(loc='upper left', fontsize=6)
    ax.set_title('Cost tracks\nrecognition in 31B')
    T = Sc.groupby(['model', 'pool', 'condition']).finish.apply(lambda s: 100 * (s == 'length').mean()).unstack()
    ax = axs[2]; w = 0.36
    for i, (m, ml) in enumerate(models):
        for j, p in enumerate(pools):
            x = i * 3.6 + j; ax.bar(x - w / 2, T.loc[(m, p), 'aligned'], w, color=C_T); ax.bar(x + w / 2, T.loc[(m, p), 'conflict'], w, color=C_S)
            if m == 'gemma-4-E4B-it': ax.plot([x - w / 2 - 0.15, x + w / 2 + 0.15], [0.4, 0.4], color='0.3', lw=1.2)
        ax.text(i * 3.6 + 1, 37, ml.split()[-1], ha='center', fontsize=7)
    ax.text(1, 3, 'none\ntruncated', ha='center', fontsize=6, color='0.3')
    ax.set_xticks(xt); ax.set_xticklabels(labs * 2, fontsize=6); ax.set_ylabel('Hit 32,768-token limit (%)'); ax.set_ylim(0, 40)
    ax.set_xlabel('Who knows the people', fontsize=7); ax.set_title('31B runs out of tokens\nmore often on shuffled facts')
    for a, l in zip(axs, 'abc'): letter(a, l)
    fig.savefig(path); plt.close(fig)


def difficulty_effects(cur, G, Tk, path):
    from matplotlib.lines import Line2D
    X = np.array([4, 5, 6]); C31, CE = '#1f3b73', '#7aa6d6'; C_T, C_S = '#2b6cb0', '#d97706'; SZ = ['4×4', '5×4', '6×4']
    order = {'4*4': 0, '5*4': 1, '6*4': 2}
    srt = lambda d: d.assign(_o=d['size'].map(order)).sort_values('_o')
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), gridspec_kw=dict(hspace=0.75, wspace=0.38)); axs = axs.ravel()
    ax = axs[0]
    for m, col, lab, mk in [('gemma-4-31B-it', C31, '31B', 'o'), ('gemma-4-E4B-it', CE, 'E4B', 's')]:
        g = srt(cur[cur.model == m])
        ax.errorbar(X, g.acc, yerr=[g.acc - g.lo, g.hi - g.acc], color=col, marker=mk, ms=4, lw=1.2)
        ax.text(6.12, g.acc.iloc[-1], lab, color=col, va='center', fontsize=7)
    ax.set_xticks(X); ax.set_xticklabels(SZ); ax.set_xlim(3.8, 6.5); ax.set_ylim(0, 100)
    ax.set_ylabel('Puzzles fully solved (%)'); ax.set_xlabel('Puzzle size (houses × attributes)', fontsize=7)
    ax.set_title('Without knowledge, E4B collapses one size\nearlier than 31B')
    ax = axs[1]
    for m, col, mk in [('gemma-4-31B-it', C31, 'o'), ('gemma-4-E4B-it', CE, 's')]:
        for p, ls, dx in [('both', '-', -0.06), ('only_large', '--', 0.06)]:
            g = srt(G[(G.model == m) & (G.pool == p)])
            ax.errorbar(X + dx, g.did_pp, yerr=[g.did_pp - g.did_lo, g.did_hi - g.did_pp], color=col, ls=ls, marker=mk, ms=4, lw=1.1,
                        mfc=col if p == 'both' else 'white')
    ax.axhline(0, color='0.7', lw=0.6); ax.set_xticks(X); ax.set_xticklabels(SZ); ax.set_xlim(3.7, 6.3); ax.set_ylim(-45, 75)
    ax.set_ylabel('Knowledge-specific cost (pp)'); ax.set_xlabel('Puzzle size (houses × attributes)', fontsize=7)
    ax.legend(handles=[Line2D([], [], color=C31, marker='o', label='31B'), Line2D([], [], color=CE, marker='s', label='E4B'),
                       Line2D([], [], color='0.3', ls='-', label='both know'),
                       Line2D([], [], color='0.3', ls='--', marker='o', mfc='white', label='only 31B knows')], fontsize=6, loc='upper left', ncol=2)
    ax.set_title("31B's conflict cost is similar at every size;\nE4B shows none")
    ax = axs[2]; g = srt(G[(G.model == 'gemma-4-31B-it') & (G.pool == 'both')]); wd = 0.32
    ax.bar(X - wd / 2, g.facil_pp, wd, color=C_T, yerr=[g.facil_pp - g.facil_lo, g.facil_hi - g.facil_pp], error_kw=dict(lw=0.8, ecolor='0.3'),
           label='true facts vs "neither knows"')
    ax.bar(X + wd / 2, g.interf_pp, wd, color=C_S, yerr=[g.interf_pp - g.interf_lo, g.interf_hi - g.interf_pp], error_kw=dict(lw=0.8, ecolor='0.3'),
           label='shuffled facts vs "neither knows"')
    ax.axhline(0, color='0.5', lw=0.6); ax.set_xticks(X); ax.set_xticklabels(SZ); ax.set_ylim(-45, 65)
    ax.set_ylabel('Accuracy difference (pp)'); ax.set_xlabel('Puzzle size (houses × attributes)', fontsize=7); ax.legend(fontsize=6, loc='upper left')
    ax.set_title('31B, both know: interference on smaller puzzles,\nknowledge shortcut on the hardest')
    ax = axs[3]
    for m, col, mk in [('gemma-4-31B-it', C31, 'o'), ('gemma-4-E4B-it', CE, 's')]:
        for c, ls in [('aligned', '-'), ('conflict', ':')]:
            g = srt(Tk[(Tk.model == m) & (Tk.condition == c)])
            ax.plot(X, g.median_tokens / 1000, color=col, ls=ls, marker=mk, ms=4, lw=1.2, mfc=col if c == 'aligned' else 'white')
    ax.set_yscale('log'); ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(['1k', '2k', '5k', '10k', '20k']); ax.set_ylim(0.8, 32)
    ax.set_xticks(X); ax.set_xticklabels(SZ); ax.set_xlim(3.8, 6.2)
    ax.set_ylabel('Median completion tokens'); ax.set_xlabel('Puzzle size (houses × attributes)', fontsize=7)
    ax.legend(handles=[Line2D([], [], color='0.3', ls='-', marker='o', label='true facts'),
                       Line2D([], [], color='0.3', ls=':', marker='o', mfc='white', label='shuffled facts')], fontsize=6, loc='upper left')
    t31 = Tk[(Tk.model == 'gemma-4-31B-it') & (Tk['size'] == '6*4')].median_tokens.max() / 1000
    te4 = Tk[(Tk.model == 'gemma-4-E4B-it') & (Tk['size'] == '6*4')].median_tokens.min() / 1000
    ax.text(6.0, t31 * 1.25, '31B', color=C31, ha='center', fontsize=7); ax.text(6.0, te4 * 0.72, 'E4B', color=CE, ha='center', fontsize=7)
    ax.set_title('On 6×4, shuffled facts send 31B into\nmuch longer searches')
    for a, l in zip(axs, 'abcd'): letter(a, l)
    fig.savefig(path); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='../data'); ap.add_argument('--results', default='../results'); a = ap.parse_args()
    R_ = lambda f: pd.read_csv(os.path.join(a.results, f)); out = lambda f: os.path.join(a.results, f)
    style()
    candidates_sitelinks(pd.read_csv(os.path.join(a.data, 'candidates_v3.csv'), encoding='utf-8-sig'), out('candidates_v3_sitelinks.png'))
    knowledge_vs_fame(R_('mcqa_by_band_long.csv'), out('mcqa_knowledge_vs_fame.png'))
    pools_by_fame(R_('pools_v3_by_band.csv'), R_('pools_v3_summary.csv'), out('pools_v3_by_fame.png'))
    main_results(R_('v3_summary.csv'), R_('v3_knowledge_gap.csv'), R_('v3_scored.csv'), out('persons_v3_32k_results.png'))
    D_ = lambda f: pd.read_csv(os.path.join(a.results, 'difficulty', f))
    difficulty_effects(D_('diff_knowledge_free_curve.csv'), D_('diff_gaps.csv'), D_('diff_tokens.csv'), os.path.join(a.results, 'difficulty', 'difficulty_effects.png'))
