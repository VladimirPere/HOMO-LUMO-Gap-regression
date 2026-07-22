# load both CSVs, train/compare all 4 models
# --- LOAD DATA ---
# --- TRAIN/TEST SPLIT ---
# --- HYPERPARAMETER SEARCH (on 20% of data) ---
# --- PRINT HYPERPARAMETERS TO JSON

import gc
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint

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
    test_size=0.8, 
    random_state=random_state
)

m_tune, _, mt_tune, _ = train_test_split(
    m_train, 
    mt_train, 
    test_size=0.3, 
    random_state=random_state
)

del mdata, mtarget, m_test, mt_test, _
gc.collect()

print('completed data collection')

# parameter grids
ridge_param = {
    'regressor__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
}

forests_param = {
    'regressor__n_estimators': randint(50, 300),
    'regressor__max_depth': [2, 4, 8, 16, None],
    'regressor__min_samples_leaf': randint(1, 20)
}

#                   ---Ridge Morgan---
morgan_ridge = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
    ('regressor', Ridge())
])

morgan_ridge_search = GridSearchCV(
    morgan_ridge,
    param_grid=ridge_param,
    cv=3,
    n_jobs=-1,
    scoring='neg_mean_absolute_error',
)

morgan_ridge_search.fit(m_train, mt_train)

morg_rid_params = morgan_ridge_search.best_params_
best_score = -morgan_ridge_search.best_score_

print(f'Morgan Ridge -- Best Alpha: {morg_rid_params}')
print(f'Morgan Ridge -- Best MAE: {best_score}')

#                   ---Random Forests Morgan---
morgan_forests = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=random_state))
])

morg_forests_search = RandomizedSearchCV(
    morgan_forests,
    param_distributions=forests_param,
    n_iter=10,
    cv=3,
    n_jobs=-1,
    scoring='neg_mean_absolute_error',
    random_state=random_state,
    verbose=2
)
morg_forests_search.fit(m_tune, mt_tune)

morg_for_params = morg_forests_search.best_params_
best_score = -morg_forests_search.best_score_

print(f'Morgan Random Forests -- Best Parameters: {morg_for_params}')
print(f'Morgan Random Forests -- Best MAE: {best_score}')


# delete morgan data
del m_train, mt_train, m_tune, mt_tune
gc.collect()

# description data storage
desc = pd.read_csv(CSV_DIR/'qm9_descriptors.csv')

dtarget = desc[target_name]
ddata = desc.drop(columns=['smiles', target_name])

d_train, d_test, dt_train, dt_test = train_test_split(
    ddata, 
    dtarget, 
    test_size=0.2, 
    random_state=random_state
)
d_tune, _, dt_tune, _ = train_test_split(
    d_train, 
    dt_train, 
    test_size=0.8, 
    random_state=random_state)

del ddata, d_test, dt_test
gc.collect()

#                    ---Description Ridge---
desc_ridge = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
    ('regressor', Ridge())
])

desc_ridge_search = GridSearchCV(
    desc_ridge,
    param_grid=ridge_param,
    cv=3,
    n_jobs=-1,
    scoring='neg_mean_absolute_error',
)

desc_ridge_search.fit(d_train, dt_train)

desc_rid_params = desc_ridge_search.best_params_
best_score = -desc_ridge_search.best_score_

print(f'Descriptors Ridge -- Best Alpha: {desc_rid_params}')
print(f'Descriptors Ridge -- Best MAE: {best_score}')


#                   ---Random Forests Descriptors---
desc_forests = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=random_state))
])

desc_forests_search = RandomizedSearchCV(
    desc_forests,
    param_distributions=forests_param,
    n_iter=30,
    cv=3,
    n_jobs=-1,
    scoring='neg_mean_absolute_error',
    random_state=random_state,
    verbose=2
)
desc_forests_search.fit(d_tune, dt_tune)

desc_for_params = desc_forests_search.best_params_
best_score = -desc_forests_search.best_score_

print(f'Descriptors Random Forests -- Best Parameters: {desc_for_params}')
print(f'Descriptors Random Forests -- Best MAE: {best_score}')


# printing parameters to JSON
def strip_prefix(params, prefix='regressor__'):
    return {k.replace(prefix, ''): v for k, v in params.items()}

best_params = {
    'morgan_ridge': strip_prefix(morg_rid_params),
    'desc_ridge': strip_prefix(desc_rid_params),
    'morgan_rf': strip_prefix(morg_for_params),
    'desc_rf': strip_prefix(desc_for_params)
}

with open(BASE_DIR / 'best_params.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print('Best params saved to best_params.json')