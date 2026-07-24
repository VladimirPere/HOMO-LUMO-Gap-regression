# SMILES → physicochemical descriptors → save
# descriptors.py output:  qm9_descriptors.csv  [smiles, MolWt, LogP, ... gap]

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from pathlib import Path

# csv folder directory
BASE_DIR = Path(__file__).parent
CSV_DIR = BASE_DIR/'csv'


def get_descriptors(smiles):
    mol  = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        # molecular size descriptors
        'MolWt': Descriptors.MolWt(mol),
        'NumHeavyAtoms': mol.GetNumHeavyAtoms(),

        # atom types
        'NumN': sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7),
        'NumO': sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8),
        'NumF': sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9),

        # electronic/conjugation
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),             # (num sp3 C atoms) / (total num C atoms)
        'NumDoubleBonds': sum(1 for b in mol.GetBonds() 
                          if b.GetBondTypeAsDouble() == 2.0),
        'NumTripleBonds': sum(1 for b in mol.GetBonds() 
                          if b.GetBondTypeAsDouble() == 3.0),

        # rings
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),              # 'c1ccccc1' (double bonds)
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol),   # may include double bonds but not aromatic
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(mol),   # 'C1CCCCC1' (single bonds only)
        'NumHeterocycles': rdMolDescriptors.CalcNumHeterocycles(mol),       # rings with at least one non-C atom

        # polarity and interactions
        'LogP': Descriptors.MolLogP(mol),                                   # solubility in polar/non-polar solvents
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),                                      # surface area of all polar atoms
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
    }

raw = pd.read_csv(CSV_DIR/'qm9_raw.csv')

desc_df = raw['smiles'].apply(get_descriptors)

# convert list of dicts to dataframe and join with gap column
desc_df = pd.DataFrame(desc_df.tolist())
desc_df['smiles'] = raw['smiles'].values
desc_df['gap'] = raw['gap'].values

# move smiles to front, gap to the end
cols = ['smiles'] + [c for c in desc_df.columns if c not in ['smiles', 'gap']] + ['gap']
desc_df = desc_df[cols]

# drop any rows where mol conversion failed
desc_df = desc_df.dropna()

print(desc_df.shape)
# (131970, 19)
print(desc_df.head())

desc_df.to_csv(CSV_DIR/'qm9_descriptors.csv', index=False)