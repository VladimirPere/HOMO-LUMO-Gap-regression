# SMILES → Morgan fingerprints → save fingerprint
# fingerprints.py output: qm9_fingerprints.csv [smiles, bit_0, bit_1, ... gap]

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from pathlib import Path

# csv folder directory
BASE_DIR = Path(__file__).parent
CSV_DIR = BASE_DIR/'csv'

# morgan generator & variables
radius = 2
fpSize = 1024
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)


def get_fingerprints(smiles):
    mol  = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(morgan_gen.GetFingerprint(mol))

raw = pd.read_csv(CSV_DIR/'qm9_raw.csv')

# apply function and remove None
fp_series = raw['smiles'].apply(get_fingerprints)
valid_mask = fp_series.notna()
fp_array = np.stack(fp_series[valid_mask].values)

# convert to DataFrame
fp_df = pd.DataFrame(fp_array, columns=[f'bit_{i}' for i in range(fp_array.shape[1])])

# add smiles and gap
fp_df.insert(0, 'smiles', raw['smiles'][valid_mask].values)
fp_df['gap'] = raw['gap'][valid_mask].values

# final QA
print(fp_df.shape)  # (131970, 1026) — 1024 bits + smiles + gap
print(fp_df.head())

fp_df.to_csv(CSV_DIR/'qm9_fingerprints(1024).csv', index=False)