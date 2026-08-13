import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
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

morgan = pd.read_csv(CSV_DIR/'qm9_fingerprints(1024).csv')

# variables
random_state = 21

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

with open(BASE_DIR / 'best_params.json', 'r') as f:
    best_params = json.load(f)

#                   ---Morgan Ridge---
morgan_ridge = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
    ('regressor', Ridge(**best_params['morgan_ridge']))
])

morgan_ridge.fit(m_train, mt_train)

cv_scores = cross_val_score(morgan_ridge, m_train, mt_train, cv=5, scoring='neg_mean_absolute_error')
morg_rid_pred = morgan_ridge.predict(m_test)
test_score = mean_absolute_error(mt_test, morg_rid_pred)

print(f'Morgan Ridge -- Training MAE: {-cv_scores.mean():4f} ± {cv_scores.std():4f}')
print(f'Morgan Ridge -- Test MAE: {test_score:4f}')

# store for graph
train_scores.append(-cv_scores.mean())
test_scores.append(test_score)

#                   ---Morgan Random Forests---
morgan_rf = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
    ('regressor', RandomForestRegressor(**best_params['morgan_rf'], n_jobs=-1, random_state=random_state))
])

morgan_rf.fit(m_train, mt_train)

morg_rf_pred = morgan_rf.predict(m_test)
train_score = mean_absolute_error(mt_train, morgan_rf.predict(m_train))
test_score = mean_absolute_error(mt_test, morg_rf_pred)

print(f'Morgan Random Forests -- Train MAE: {train_score:.4f}')
print(f'Morgan Random Forests -- Test MAE: {test_score:.4f}')


train_scores.append(train_score)
test_scores.append(test_score)


del m_train, mt_train, mdata, mtarget
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

#                   ---Descriptors Ridge---
desc_ridge = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
    ('regressor', Ridge(**best_params['desc_ridge']))
])

desc_ridge.fit(d_train, dt_train)

cv_scores = cross_val_score(desc_ridge, d_train, dt_train, cv=5, scoring='neg_mean_absolute_error')
desc_rid_pred = desc_ridge.predict(d_test)
test_score = mean_absolute_error(dt_test, desc_rid_pred)

print(f'Descriptors Ridge -- Training MAE: {-cv_scores.mean():4f} ± {cv_scores.std():4f}')
print(f'Descriptors Ridge -- Test MAE: {test_score:4f}')

train_scores.append(-cv_scores.mean())
test_scores.append(test_score)



#                   ---Descriptors Random Forests---
desc_rf = Pipeline(steps=[
    ('preprocessor', StandardScaler()),
    ('regressor', RandomForestRegressor(**best_params['desc_rf'], n_jobs=-1, random_state=random_state))
])

desc_rf.fit(d_train, dt_train)

desc_rf_pred = desc_rf.predict(d_test)
train_score = mean_absolute_error(dt_train, desc_rf.predict(d_train))
test_score = mean_absolute_error(dt_test, desc_rf_pred)

print(f'Descriptors Random Forests -- Training MAE: {train_score:4f}')
print(f'Descriptors Random Forests -- Test MAE: {test_score:4f}')

train_scores.append(train_score)
test_scores.append(test_score)



# 1st graph
# make graph of predictions vs true values
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
titles = ['Morgan Ridge', 'Morgan Random Forests', 'Descriptor Ridge', 'Descriptor Random Forests']

datasets = [(mt_test, morg_rid_pred), (mt_test, morg_rf_pred), (dt_test, desc_rid_pred), (dt_test, desc_rf_pred)]

for ax, (true, pred), title in zip(axs.flat, datasets, titles):
    ax.scatter(true, pred, color='blue', alpha=0.5)
    ax.set_title(title)

    # create best fit line and calculate R^2
    slope, intercept, r_value, p_value, std_err = linregress(true, pred)

    true_sorted = np.sort(true)
    best_fit_line = slope * true_sorted + intercept

    ax.plot(true_sorted, best_fit_line, color='red', label=f'Best Fit Line: {slope:.2f}x + {intercept:.2f}')

    # Annotate the R^2 value on the specific axis
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
fig.suptitle('Predictions vs True Values of HOMO-LUMO Gaps')

plt.tight_layout()
os.makedirs(BASE_DIR / 'images', exist_ok=True)
plt.savefig(BASE_DIR / 'images/rsquared_comparisons.png', dpi=150, bbox_inches='tight')
plt.show()
exit()

#                   ---Graphing MAE---
results_df = pd.DataFrame({
    'Model': ['Morgan Ridge', 'Morgan RF', 'Descriptor Ridge', 'Descriptor RF'] * 2,
    'MAE': train_scores + test_scores,
    'Type': ['Train'] * 4 + ['Test'] * 4
})

baseline = .0416

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
ax.set_title('Train vs Test MAE by Model and Feature Set\n(Ridge train = 5-fold CV mean, RF train = direct)')
ax.set_ylabel('MAE (Hartree)')
ax.set_xlabel('Model')
ax.set_ylim(0, 0.055)
ax.legend()

# print for README
plt.tight_layout()
os.makedirs(BASE_DIR / 'images', exist_ok=True)
plt.savefig(BASE_DIR / 'images/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()