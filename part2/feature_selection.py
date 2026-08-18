import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import json
from pathlib import Path


BASE_DIR = Path(__file__).parent
CSV_DIR = BASE_DIR/'csv'

random_state = 234
device = 'cpu'
tree_method = 'hist'
objective = 'reg:squarederror'
n_jobs = -1

# description data storage
desc = pd.read_csv(CSV_DIR/'qm9_descriptors_p2.csv')

target_name = 'gap'
dtarget = desc[target_name]
ddata = desc.drop(columns=['smiles', target_name])

d_train, d_test, dt_train, dt_test = train_test_split(
    ddata, 
    dtarget, 
    test_size=0.2, 
    random_state=random_state
)

with open(BASE_DIR / 'best_params_p2.json', 'r') as f:
    best_params = json.load(f)

#   ---XGBoost---
# pipeline
desc_xgb = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', xgb.XGBRegressor(       
        **best_params['desc_xgb'],
        device=device,
        tree_method=tree_method,
        random_state=random_state,
        objective=objective,
        n_jobs=n_jobs))
    ]
)

desc_xgb.fit(d_train, dt_train)

# shap values
XGB_preprocessor = desc_xgb.named_steps['scaler']
d_train_scaled_XGB = XGB_preprocessor.transform(d_train)
xgb_model = desc_xgb.named_steps['regressor']

XGB_explainer = shap.TreeExplainer(xgb_model)
XGB_shap_values = XGB_explainer.shap_values(d_train_scaled_XGB)

# --Plot--
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

# --Feature importance--
# identify low SHAP features
feature_names = d_train.columns.tolist()
mean_shap = np.abs(XGB_shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    'feature': feature_names,
    'mean_shap': mean_shap
}).sort_values('mean_shap', ascending=False)
print(shap_df)

# identify features below threshold
low_shap = shap_df[shap_df['mean_shap'] < 0.0005]['feature'].tolist()
print(f'Low SHAP features: {low_shap}')

full_mae = mean_absolute_error(dt_test, 
    desc_xgb.predict(d_test))

# reduced feature MAE  
d_train_reduced = d_train.drop(columns=low_shap)
d_test_reduced = d_test.drop(columns=low_shap)

disc_XGB_reduced = Pipeline(steps=[
    ('scaler', StandardScaler()),
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

disc_XGB_reduced.fit(d_train_reduced, dt_train)
reduced_mae = mean_absolute_error(dt_test, disc_XGB_reduced.predict(d_test_reduced))

print(f'Full features MAE: {full_mae:.4f}')
print(f'Reduced features MAE: {reduced_mae:.4f}')
print(f'Difference: {abs(full_mae - reduced_mae):.4f}')

#   ---Random Forests---
desc_rf = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', xgb.XGBRFRegressor(       
        **best_params['desc_rf'],
        device=device,
        tree_method=tree_method,
        random_state=random_state,
        objective=objective,
        n_jobs=n_jobs))
    ]
)

desc_rf.fit(d_train, dt_train)

# shap values
RF_preprocessor = desc_rf.named_steps['scaler']
d_train_scaled_RF = RF_preprocessor.transform(d_train)
rf_model = desc_rf.named_steps['regressor']

RF_explainer = shap.TreeExplainer(rf_model)
RF_shap_values = RF_explainer.shap_values(d_train_scaled_RF)

# --Plot--
# bar plot
plt.figure()
shap.summary_plot(RF_shap_values, 
                  d_train_scaled_RF, 
                  feature_names=d_train.columns.tolist(), 
                  plot_type='bar', 
                  show=False
                )

plt.title('Random Forests Feature Importance (SHAP)')
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

plt.title('Random Forests SHAP Beeswarm')
plt.tight_layout()
plt.savefig(BASE_DIR/'images/rf_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()

# identify low SHAP features
feature_names = d_train.columns.tolist()
mean_shap = np.abs(RF_shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    'feature': feature_names,
    'mean_shap': mean_shap
}).sort_values('mean_shap', ascending=False)
print(shap_df)

# identify features below threshold
low_shap = shap_df[shap_df['mean_shap'] < 0.0005]['feature'].tolist()
print(f'Low SHAP features: {low_shap}')

full_mae = mean_absolute_error(dt_test, 
    desc_rf.predict(d_test))

# reduced feature MAE  
d_train_reduced = d_train.drop(columns=low_shap)
d_test_reduced = d_test.drop(columns=low_shap)

disc_RF_reduced = Pipeline(steps=[
    ('scaler', StandardScaler()),
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

disc_RF_reduced.fit(d_train_reduced, dt_train)
reduced_mae = mean_absolute_error(dt_test, disc_RF_reduced.predict(d_test_reduced))

print(f'Full features MAE: {full_mae:.4f}')
print(f'Reduced features MAE: {reduced_mae:.4f}')
print(f'Difference: {abs(full_mae - reduced_mae):.4f}')




