import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import gc
import json
import os
from pathlib import Path

# csv folder directory
BASE_DIR = Path(__file__).parent
CSV_DIR = BASE_DIR/'csv'
train_scores = []
test_scores = []

morgan = pd.read_csv(CSV_DIR/'qm9_fingerprints_p2(2048).csv')

# variables
random_state = 96
device = 'cpu'
tree_method = 'hist'
objective = 'reg:squarederror'
n_jobs = -1

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

with open(BASE_DIR / 'best_params_p2.json', 'r') as f:
    best_params = json.load(f)

#                   ---Morgan XGBoost---
morgan_xgb = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
    ('regressor', xgb.XGBRegressor(
        **best_params['morgan_xgb'],
        device=device,
        tree_method=tree_method,
        random_state=random_state,
        objective=objective,
        n_jobs=n_jobs)
        )
    ]
)
morgan_xgb.fit(m_train, mt_train)

cv_scores = cross_val_score(morgan_xgb, m_train, mt_train, cv=5, scoring='neg_mean_absolute_error')
morg_xgb_pred = morgan_xgb.predict(m_test)
test_score = mean_absolute_error(mt_test, morg_xgb_pred)

print(f'Morgan XGBoost -- Training MAE: {-cv_scores.mean():4f} ± {cv_scores.std():4f}')
print(f'Morgan XGBoost -- Test MAE: {test_score:4f}')

# store for graph
train_scores.append(-cv_scores.mean())
test_scores.append(test_score)

#                   ---Morgan Random Forests---
morgan_rf = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
        ('regressor', xgb.XGBRFRegressor(
        **best_params['morgan_rf'],
        device=device,
        tree_method=tree_method,
        random_state=random_state,
        objective=objective,
        n_jobs=n_jobs)
        )
    ]
)
morgan_rf.fit(m_train, mt_train)

cv_scores = cross_val_score(morgan_rf, m_train, mt_train, cv=5, scoring='neg_mean_absolute_error')
morg_rf_pred = morgan_rf.predict(m_test)
test_score = mean_absolute_error(mt_test, morg_rf_pred)

print(f'Morgan Random Forests -- Training MAE: {-cv_scores.mean():4f} ± {cv_scores.std():4f}')
print(f'Morgan Random Forests -- Test MAE: {test_score:4f}')

# store for graph
train_scores.append(-cv_scores.mean())
test_scores.append(test_score)

del m_train, mt_train, mdata, mtarget
gc.collect()

# description data storage
desc = pd.read_csv(CSV_DIR/'qm9_descriptors_p2.csv')

dtarget = desc[target_name]
ddata = desc.drop(columns=['smiles', target_name])

#                   ---SHAP---
# Due to columns low SHAP values; All columns were used: See feature_selection.py

d_train, d_test, dt_train, dt_test = train_test_split(
    ddata, 
    dtarget, 
    test_size=0.2, 
    random_state=random_state
)

#                   ---Descriptors XGBoost---
desc_xgb = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
        ('regressor', xgb.XGBRegressor(
        **best_params['desc_xgb'],
        device=device,
        tree_method=tree_method,
        random_state=random_state,
        objective=objective,
        n_jobs=n_jobs)
        )
    ]
)
desc_xgb.fit(d_train, dt_train)

cv_scores = cross_val_score(desc_xgb, d_train, dt_train, cv=5, scoring='neg_mean_absolute_error')
desc_xgb_pred = desc_xgb.predict(d_test)
test_score = mean_absolute_error(dt_test, desc_xgb_pred)

print(f'Descriptors XGBoost -- Training MAE: {-cv_scores.mean():4f} ± {cv_scores.std():4f}')
print(f'Descriptors XGBoost-- Test MAE: {test_score:4f}')

train_scores.append(-cv_scores.mean())
test_scores.append(test_score)

#                   ---Descriptors Random Forests---
desc_rf = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
        ('regressor', xgb.XGBRFRegressor(
        **best_params['desc_rf'],
        device=device,
        tree_method=tree_method,
        random_state=random_state,
        objective=objective,
        n_jobs=n_jobs)
        )
    ]
)
desc_rf.fit(d_train, dt_train)

cv_scores = cross_val_score(desc_rf, d_train, dt_train, cv=5, scoring='neg_mean_absolute_error')
desc_rf_pred = desc_rf.predict(d_test)
test_score = mean_absolute_error(dt_test, desc_rf_pred)

print(f'Descriptors Random Forests -- Training MAE: {-cv_scores.mean():4f} ± {cv_scores.std():4f}')
print(f'Descriptors Random Forests -- Test MAE: {test_score:4f}')

train_scores.append(-cv_scores.mean())
test_scores.append(test_score)


# 1st graph
# make graph of predictions vs true values
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
titles = ['Morgan XGBoost', 'Morgan Random Forests', 'Descriptor XGBoost', 'Descriptor Random Forests']

datasets = [(mt_test, morg_xgb_pred), (mt_test, morg_rf_pred), (dt_test, desc_xgb_pred), (dt_test, desc_rf_pred)]

for ax, (true, pred), title in zip(axs.flat, datasets, titles):
    ax.scatter(true, pred, color='blue', alpha=0.5)
    ax.set_title(title)

    # create best fit line and calculate R^2
    slope, intercept, r_value, p_value, std_err = linregress(true, pred)

    true_sorted = np.sort(true)
    best_fit_line = slope * true_sorted + intercept

    ax.plot(true_sorted, best_fit_line, color='red', label=f'Best Fit Line: {slope:.2f}x + {intercept:.2f}')

    # annotate the R^2 value on the specific axis
    ax.text(
        0.05,
        0.95,
        f"$R^2 = {r_value**2:.3f}$",
        transform=ax.transAxes,
        verticalalignment="top",
    )
    ax.legend()

fig.supxlabel('True HOMO-LUMO Gap (Hartree)')
fig.supylabel('Predicted HOMO-LUMO Gap (Hartree)')
fig.suptitle('Predictions vs True Values of HOMO-LUMO Gaps for Aromatic Data')

plt.tight_layout()
os.makedirs(BASE_DIR / 'images', exist_ok=True)
plt.savefig(BASE_DIR / 'images/rsquared_comparisons_p2.png', dpi=150, bbox_inches='tight')
plt.show()

#                   ---Graphing MAE---
results_df = pd.DataFrame({
    'Model': ['Morgan XGBoost', 'Morgan RF', 'Descriptor XGBoost', 'Descriptor RF'] * 2,
    'MAE': train_scores + test_scores,
    'Type': ['Train'] * 4 + ['Test'] * 4
})

baseline = .0404

fig, ax = plt.subplots(figsize=(10, 6))

# make bar plot
sns.barplot(data=results_df, x='Model', y='MAE', hue='Type',
            palette={'Train': 'steelblue', 'Test': 'coral'}, ax=ax)

ax.bar_label(ax.containers[0], fmt='%.4f', fontsize=8, padding=2)
ax.bar_label(ax.containers[1], fmt='%.4f', fontsize=8, padding=2)

# add baseline hline
ax.axhline(y=baseline, color='red', linestyle='--', linewidth=1.5,
           label=f'Baseline MAE ({baseline})')

# set labels
ax.set_title('Train vs Test MAE by Model and Feature Set for Aromatic Dataset\n(5 Fold CV Scoring)')
ax.set_ylabel('MAE (Hartree)')
ax.set_xlabel('Model')
ax.set_ylim(0, 0.055)
ax.legend()

# print for README
plt.tight_layout()
os.makedirs(BASE_DIR / 'images', exist_ok=True)
plt.savefig(BASE_DIR / 'images/model_comparison_p2.png', dpi=150, bbox_inches='tight')
plt.show()

# quick test