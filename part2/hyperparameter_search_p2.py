import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import shap
import xgboost as xgb
import gc
import json
from pathlib import Path

# csv folder directory
BASE_DIR = Path(__file__).parent
CSV_DIR = BASE_DIR/'csv'

morgan = pd.read_csv(CSV_DIR/'qm9_fingerprints_p2(2048).csv')

# convert to float32 — halves memory vs default float64
bit_cols = [c for c in morgan.columns if c.startswith('bit_')]
morgan[bit_cols] = morgan[bit_cols].astype('float32')

# get cuda device
xgb.collective.init()
local_rank = xgb.collective.get_rank()

# variables
random_state = 71
device = 'cpu'
tree_method = 'hist'
objective = 'reg:squarederror'
n_jobs = -1

# parameter grids
xgb_param = {
    'regressor__n_estimators': [50, 100, 200, 300, 500],
    'regressor__max_depth': [8, 12, 16],
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    'regressor__subsample': [0.6, 0.7, 0.9, 1.0],
    'regressor__colsample_bytree': [0.5, 0.7, 1.0]
}

# data storage
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

# ---Morgan Pipelines---
# XGBoost
morg_XGB = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', xgb.XGBRegressor(
        device=device,
        tree_method=tree_method,
        random_state=random_state,
        objective=objective,
        n_jobs=n_jobs
    ))
])
morg_XGB_search = RandomizedSearchCV(
    morg_XGB,
    param_distributions=xgb_param,
    n_iter=100,
    cv=3,
    n_jobs=2,
    scoring='neg_mean_absolute_error',
    random_state=random_state,
    verbose=2
)

morg_XGB_search.fit(m_train, mt_train)
Morg_XGB_best_params = morg_XGB_search.best_params_
del morg_XGB, morg_XGB_search
gc.collect()

print(f'Morg XGB: {Morg_XGB_best_params}')

# Random Forest
morg_RF = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', xgb.XGBRFRegressor(
        device=device,
        tree_method=tree_method,
        random_state=random_state,
        objective=objective,
        n_jobs=n_jobs
    ))
])
morg_RF_search = RandomizedSearchCV(
    morg_RF,
    param_distributions=xgb_param,
    n_iter=100,
    cv=3,
    n_jobs=2,
    scoring='neg_mean_absolute_error',
    random_state=random_state,
    verbose=2
)

morg_RF_search.fit(m_train, mt_train)
Morg_RF_best_params = morg_RF_search.best_params_

del m_train, m_test, mt_train, mt_test, mdata, morg_RF, morg_RF_search
gc.collect()
print(f'Morg RF: {Morg_RF_best_params}')

# descriptor data
desc = pd.read_csv(CSV_DIR/'qm9_descriptors_p2.csv')
dtarget = desc[target_name]
ddata = desc.drop(columns=['smiles', target_name])

d_train, d_test, dt_train, dt_test = train_test_split(
    ddata, 
    dtarget, 
    test_size=0.2, 
    random_state=random_state
)

# descriptor pipelines
desc_XGB = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', xgb.XGBRegressor(
        device=device,
        tree_method=tree_method,
        random_state=random_state,
        objective=objective,
        n_jobs=n_jobs
    ))
])
desc_RF = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', xgb.XGBRFRegressor(
        device=device,
        tree_method=tree_method,
        random_state=random_state,
        objective=objective,
        n_jobs=n_jobs
    ))
])

desc_XGB_search = RandomizedSearchCV(
    desc_XGB,
    param_distributions=xgb_param,
    n_iter=150,
    cv=3,
    n_jobs=-1,
    scoring='neg_mean_absolute_error',
    random_state=random_state,
    verbose=2
)
desc_RF_search = RandomizedSearchCV(
    desc_RF,
    param_distributions=xgb_param,
    n_iter=100,
    cv=3,
    n_jobs=-1,
    scoring='neg_mean_absolute_error',
    random_state=random_state,
    verbose=2
)

desc_XGB_search.fit(d_train, dt_train)
desc_RF_search.fit(d_train, dt_train)

print(f'Morg XGB: {Morg_XGB_best_params}')
print(f'Morg RF: {Morg_RF_best_params}')

print(f'Desc XGB: {desc_XGB_search.best_params_}')
print(f'desc RF: {desc_RF_search.best_params_}')

# ---Saving best parameters to JSON---
def strip_prefix(params, prefix='regressor__'):
    return {k.replace(prefix, ''): v for k, v in params.items()}

best_params = {
    'morgan_xgb': strip_prefix(Morg_XGB_best_params),
    'morgan_rf': strip_prefix(Morg_RF_best_params),
    'desc_xgb': strip_prefix(desc_XGB_search.best_params_),
    'desc_rf': strip_prefix(desc_RF_search.best_params_)
}

with open(BASE_DIR / 'best_params_p2.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print("Best parameters saved to best_params_p2.json")