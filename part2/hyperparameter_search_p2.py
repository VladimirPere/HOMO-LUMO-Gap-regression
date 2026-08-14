import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import gc
import json
from pathlib import Path

# csv folder directory
BASE_DIR = Path(__file__).parent
CSV_DIR = BASE_DIR/'csv'

morgan = pd.read_csv(CSV_DIR/'qm9_fingerprints(1024).csv')

# variables
random_state = 32

# morgan data storage
target_name = 'gap'
mtarget = morgan[target_name]

mdata = morgan.drop(columns=['smiles', target_name])

del morgan
gc.collect()

m_train, m_test, mt_train, mt_test = train_test_split(
    mdata, 
    mtarget, 
    test_size=0.2, 
    random_state=random_state
)