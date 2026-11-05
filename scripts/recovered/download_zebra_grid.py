"""Recovered notebook code that produced `zebra_grid.parquet`.

Reconstructed from session lineage, not written as a standalone script.
Expect adjustments (input paths, working directory via WORKDIR) before rerunning.
Steps that restored files from the artifact store are replaced by `pass`;
those files are under src/ and data/ in this repository.
"""
import os
import urllib.request
import pandas as pd

url = 'https://huggingface.co/datasets/WildEval/ZebraLogic/resolve/main/grid_mode/test-00000-of-00001.parquet'
with urllib.request.urlopen(urllib.request.Request(url), timeout=120) as r:
    data = r.read()
open('zebra_grid.parquet', 'wb').write(data)
g = pd.read_parquet('zebra_grid.parquet')
print(g.shape, g['size'].value_counts().sort_index().to_dict())
