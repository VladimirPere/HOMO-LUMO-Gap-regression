# load QM9, extract SMILES + gap target, filter and save aromatic molecules
# output: qm9_aromatic.csv with columns [smiles, gap]

import deepchem as dc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import seaborn as sns
from pathlib import Path

# csv folder directory
BASE_DIR = Path(__file__).parent
CSV_DIR = BASE_DIR/'csv'

def is_aromatic(smiles):
    """Filter out non aromatic molecules"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return rdMolDescriptors.CalcNumAromaticRings(mol) > 0

# normalization passed due to inaccuracies in gap numbers
tasks, datasets, transformers = dc.molnet.load_qm9(transformers=[])
train, valid, test = datasets

gap_index = list(tasks).index('gap')

# combine training and test data, train_test_split will be used instead
all_smiles = np.concatenate([train.ids, valid.ids, test.ids])
all_gaps = np.concatenate([
    train.y[:, gap_index],
    valid.y[:, gap_index],
    test.y[:, gap_index]
])

# filter out molecules with no aromatic rings
aromatic_mask = np.array([is_aromatic(smile) for smile in all_smiles])
aromatic_smiles = all_smiles[aromatic_mask]
aromatic_gaps = all_gaps[aromatic_mask]

df = pd.DataFrame({'smiles': aromatic_smiles, 'gap': aromatic_gaps})
print(df.shape)     # (21981, 2)
print(df.head())

# checking baseline
baseline_mae = np.mean(np.abs(df['gap'] - df['gap'].mean()))                # Baseline MAE (predict mean): 0.0404 Hartree
print(f'Baseline MAE (predict mean): {baseline_mae:.4f} Hartree')

# plot distribution of HOMO-LUMO gap values and save image
sns.histplot(df['gap'], bins=50)
plt.title('HOMO-LUMO Gap Distribution — Aromatic QM9 Subset')
plt.xlabel('Gap (Hartree)')
plt.savefig(BASE_DIR/'images'/'aromatic_molecule_distribution.png')
plt.show()

exit()
df.to_csv(CSV_DIR/'qm9_aromatic.csv', index=False)