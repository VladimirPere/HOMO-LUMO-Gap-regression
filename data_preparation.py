# load QM9, extract SMILES + gap target, save raw
# output: qm9_raw.csv with columns [smiles, gap]

import deepchem as dc
import numpy as np
import pandas as pd
from pathlib import Path

# csv folder directory
BASE_DIR = Path(__file__).parent
CSV_DIR = BASE_DIR/'csv'

# normalization passed due to inaccuracies in gap numbers
tasks, datasets, transformers = dc.molnet.load_qm9(transformers=[])
train, valid, test = datasets

gap_index = list(tasks).index('gap')

# Combine training and test data, train_test_split will be used instead
all_smiles = np.concatenate([train.ids, valid.ids, test.ids])
all_gaps = np.concatenate([
    train.y[:, gap_index],
    valid.y[:, gap_index],
    test.y[:, gap_index]
])

df = pd.DataFrame({'smiles': all_smiles, 'gap': all_gaps})
print(df.shape)
# (131970, 2) from ~(134000, 2)
print(df.head())
print(df['gap'].describe())

# checking baseline
baseline_mae = np.mean(np.abs(df['gap'] - df['gap'].mean()))
print(f'Baseline MAE (predict mean): {baseline_mae:.4f} Hartree')
exit()
df.to_csv(CSV_DIR/'qm9_raw.csv', index=False)