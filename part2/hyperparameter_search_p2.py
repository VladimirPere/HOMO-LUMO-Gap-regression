import matplotlib.pyplot as plt
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

# variables
random_state = 71
device = 'cuda'
tree_method = 'hist'
n_jobs = -1
objective = 'reg:squarederror'

# parameter grids
xgb_param = {
    'regressor__n_estimators': [50, 100, 200, 300],
    'regressor__max_depth': [2, 4, 8],
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'regressor__subsample': [0.5, 0.7, 1.0],
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

# morgan pipelines
morg_XGB = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', xgb.XGBRegressor(
        device=device,
        tree_method=tree_method,
        random_state=random_state,
        objective=objective
    ))
])
morg_RF = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', xgb.XGBRFRegressor(
        device=device,
        tree_method=tree_method,
        random_state=random_state,
        objective=objective
    ))
])

morg_XGB_search = RandomizedSearchCV(
    morg_XGB,
    param_distributions=xgb_param,
    n_iter=100,
    cv=5,
    n_jobs=n_jobs,
    scoring='neg_mean_absolute_error',
    random_state=random_state,
    verbose=2
)
morg_RF_search = RandomizedSearchCV(
    morg_RF,
    param_distributions=xgb_param,
    n_iter=100,
    cv=5,
    n_jobs=n_jobs,
    scoring='neg_mean_absolute_error',
    random_state=random_state,
    verbose=2
)

morg_XGB_search.fit(m_train, mt_train)
morg_RF_search.fit(m_train, mt_train)
Morg_XGB_best_params = morg_XGB_search.best_params_
Morg_RF_best_params = morg_RF_search.best_params_

del m_train, m_test, mt_train, mt_test, mdata, mtarget
gc.collect()

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
disc_XGB = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', xgb.XGBRegressor(
        device=device,
        tree_method=tree_method,
        random_state=random_state,
        objective=objective
    ))
])
disc_RF = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', xgb.XGBRFRegressor(
        device=device,
        tree_method=tree_method,
        random_state=random_state,
        objective=objective
    ))
])

disc_XGB_search = RandomizedSearchCV(
    disc_XGB,
    param_distributions=xgb_param,
    n_iter=100,
    cv=5,
    n_jobs=n_jobs,
    scoring='neg_mean_absolute_error',
    random_state=random_state,
    verbose=2
)
disc_RF_search = RandomizedSearchCV(
    disc_RF,
    param_distributions=xgb_param,
    n_iter=100,
    cv=5,
    n_jobs=n_jobs,
    scoring='neg_mean_absolute_error',
    random_state=random_state,
    verbose=2
)

disc_XGB_search.fit(d_train, dt_train)
disc_RF_search.fit(d_train, dt_train)

# ---Descriptors XGBoost SHAP analysis---
# shap values
XGB_preprocessor = disc_XGB_search.best_estimator_.named_steps['scaler']
d_train_scaled_XGB = XGB_preprocessor.transform(d_train)
xgb_model = disc_XGB_search.best_estimator_.named_steps['regressor']

XGB_explainer = shap.TreeExplainer(xgb_model)
XGB_shap_values = XGB_explainer.shap_values(d_train_scaled_XGB)

# --Plotting SHAP summary for XGBoost data--
# bar plot
plt.figure()
shap.summary_plot(XGB_shap_values, 
                  d_train_scaled_XGB, 
                  feature_names=d_train.columns.tolist(), 
                  plot_type='bar', 
                  show=False
                )

plt.title('XGBoost Feature Importance (SHAP)')
plt.tight_layout()
plt.savefig(BASE_DIR/'images/xgb_shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()

# beeswarm plot
plt.figure()
shap.summary_plot(XGB_shap_values, 
                  d_train_scaled_XGB, 
                  feature_names=d_train.columns.tolist(), 
                  plot_type='dot',
                  show=False
                )

plt.title('XGBoost SHAP Beeswarm')
plt.tight_layout()
plt.savefig(BASE_DIR/'images/xgb_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()

# ---Descriptors Random Forest SHAP analysis---
# shap values
RF_preprocessor = disc_RF_search.best_estimator_.named_steps['scaler']
d_train_scaled_RF = RF_preprocessor.transform(d_train)
rf_model = disc_RF_search.best_estimator_.named_steps['regressor']

RF_explainer = shap.TreeExplainer(rf_model)
RF_shap_values = RF_explainer.shap_values(d_train_scaled_RF)

# --Plotting SHAP summary for Random Forest data--
# bar plot
plt.figure()
shap.summary_plot(RF_shap_values, 
                  d_train_scaled_RF, 
                  feature_names=d_train.columns.tolist(), 
                  plot_type='bar', 
                  show=False
                )

plt.title('Random Forest Feature Importance (SHAP)')
plt.tight_layout()
plt.savefig(BASE_DIR/'images/rf_shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()

# beeswarm plot
plt.figure()
shap.summary_plot(RF_shap_values, 
                  d_train_scaled_RF, 
                  feature_names=d_train.columns.tolist(), 
                  plot_type='dot',
                  show=False
                )

plt.title('Random Forest SHAP Beeswarm')
plt.tight_layout()
plt.savefig(BASE_DIR/'images/rf_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()

# ---Saving best parameters to JSON---
def strip_prefix(params, prefix='regressor__'):
    return {k.replace(prefix, ''): v for k, v in params.items()}

best_params = {
    'morgan_xgb': strip_prefix(Morg_XGB_best_params),
    'morgan_rf': strip_prefix(Morg_RF_best_params),
    'desc_xgb': strip_prefix(disc_XGB_search.best_params_),
    'desc_rf': strip_prefix(disc_RF_search.best_params_)
}

with open(BASE_DIR / 'best_params_p2.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print("Best parameters saved to best_params_p2.json")