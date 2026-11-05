"""Recovered notebook code that produced `inpuzzle_mcqa_summary.csv`.

Reconstructed from session lineage, not written as a standalone script.
Expect adjustments (input paths, working directory via WORKDIR) before rerunning.
Steps that restored files from the artifact store are replaced by `pass`;
those files are under src/ and data/ in this repository.
"""
import os
# [lineage] reconstruction failed validation (max_tokens) — raw producing cell below; see Execution Log for full trace
IP=pd.concat([pd.read_json(f"out/inpuzzle_mcqa_{m['name']}.jsonl",lines=True) for m in cfg['models']])
IP['pool']=IP.puzzle_key.str.split('|').str[1]; IP['n_opts']=IP.options.map(len)
T=IP.groupby(['pool','model','fact']).correct.mean().unstack('fact').round(3); print(T.to_string())
print(IP.groupby(['pool','model']).correct.mean().unstack().round(3).to_string())
print('chance by size:', {n: round(1/n,3) for n in sorted(IP.n_opts.unique())}, '| non-letter:', IP.letter.isna().groupby(IP.model).mean().round(3).to_dict())
IP.groupby(['pool','model','fact']).correct.mean().reset_index().to_csv('out/inpuzzle_mcqa_summary.csv',index=False)
V=df.groupby(['pool','condition','size']).valid.agg(['size','sum']).reset_index(); V.to_csv('out/variants_v3_validation.csv',index=False)
