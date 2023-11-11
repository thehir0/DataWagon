
# %%
import gc
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import catboost as cb
from datetime import datetime
import sys
sys.path.append('..')

from utils import seed_everything
from train.metrics_f1 import calc_f1_score

DATA_ROOT = "data"
WEIGHTS_ROOT = "weights"
os.makedirs(WEIGHTS_ROOT, exist_ok=True)

SEED = 28

seed_everything(SEED)
date = str(datetime.now()).split('.')[0].replace(':','_').replace(' ','__').replace('-','_')

# %% Baseline solution
gt_path = f"../train/prediction/target_predicton_true.csv"
pred_path = f"../train/prediction/target_predicton.csv"

baseline_score = calc_f1_score(gt_path, pred_path)
print(f"{baseline_score = }")

# %%
numerical_features, categorical_features, text_features = [], [], []
months_val = [pd.to_datetime('2022-12-01')]

# %%
target1_df = pd.read_csv(DATA_ROOT + '/train/target/y_train.csv').convert_dtypes()
target2_df = pd.read_csv(DATA_ROOT + '/test/target/y_test.csv').convert_dtypes()

target_df = pd.concat([target1_df, target2_df])
target_df = target_df.sort_values(by=["wagnum", "month"])
target_df = target_df.drop_duplicates(ignore_index=True)
target_df.month = pd.to_datetime(target_df.month)

print(target_df["month"].value_counts())

del target1_df, target2_df
gc.collect()

# %% Список вагонов, по которым известен пробег и тип владения на дату среза
wag_prob1_df = pd.read_parquet('../train/wagons_probeg_ownersip.parquet').convert_dtypes()
wag_prob2_df = pd.read_parquet('../test/wagons_probeg_ownersip.parquet').convert_dtypes()

wag_prob_df = pd.concat([wag_prob1_df, wag_prob2_df])
wag_prob_df = wag_prob_df.sort_values(by=["wagnum", "repdate"])
wag_prob_df = wag_prob_df.drop_duplicates(ignore_index=True)
wag_prob_df = wag_prob_df.drop(columns=["month"])

del wag_prob1_df, wag_prob2_df
gc.collect()

# %% Информация по дислокации
dislok1_df = pd.read_parquet(DATA_ROOT + '/train/dislok_wagons.parquet').convert_dtypes()
dislok2_df = pd.read_parquet(DATA_ROOT + '/test/dislok_wagons.parquet').convert_dtypes()

dislok_df = pd.concat([dislok1_df, dislok2_df])
dislok_df = dislok_df.sort_values(by=["wagnum", "plan_date"])
dislok_df = dislok_df.drop_duplicates(ignore_index=True)

del dislok1_df, dislok2_df
gc.collect()

# %% Данные по характеристикам вагона
wag_param1_df = pd.read_parquet(DATA_ROOT + '/train/wag_params.parquet').convert_dtypes()
wag_param2_df = pd.read_parquet(DATA_ROOT + '/test/wag_params.parquet').convert_dtypes()

wag_param_df = pd.concat([wag_param1_df, wag_param2_df])
wag_param_df = wag_param_df.sort_values(by=["wagnum"])
wag_param_df = wag_param_df.drop_duplicates(ignore_index=True)

del wag_param1_df, wag_param2_df
gc.collect()

# %% Данные по плановым ремонтам
pr_rem1_df = pd.read_parquet(DATA_ROOT + '/train/pr_rems.parquet').convert_dtypes()
pr_rem2_df = pd.read_parquet(DATA_ROOT + '/test/pr_rems.parquet').convert_dtypes()

pr_rem_df = pd.concat([pr_rem1_df, pr_rem2_df])
pr_rem_df = pr_rem_df.sort_values(by=["wagnum", "rem_month"])
pr_rem_df = pr_rem_df.drop_duplicates(ignore_index=True)

del pr_rem1_df, pr_rem2_df
gc.collect()

# %% Данные по текущим ремонтам вагона
tr_rem1_df = pd.read_parquet(DATA_ROOT + '/train/tr_rems.parquet').convert_dtypes()
tr_rem2_df = pd.read_parquet(DATA_ROOT + '/test/tr_rems.parquet').convert_dtypes()

tr_rem_df = pd.concat([tr_rem1_df, tr_rem2_df])
tr_rem_df = tr_rem_df.sort_values(by=["wagnum", "rem_month"])
tr_rem_df = tr_rem_df.drop_duplicates(ignore_index=True)

del tr_rem1_df, tr_rem2_df
gc.collect()

# %% Данные по КТИ
kti_izm1_df = pd.read_parquet(DATA_ROOT + '/train/kti_izm.parquet').convert_dtypes()
kti_izm2_df = pd.read_parquet(DATA_ROOT + '/test/kti_izm.parquet').convert_dtypes()

kti_izm_df = pd.concat([kti_izm1_df, kti_izm2_df])
kti_izm_df = kti_izm_df.sort_values(by=["wagnum", "operation_date_dttm"])
kti_izm_df = kti_izm_df.drop_duplicates(ignore_index=True)
kti_izm_df["operation_date_dttm"] = kti_izm_df["operation_date_dttm"].apply(lambda x: datetime.fromtimestamp(int(str(x)[:-9])).replace(hour=0, minute=0, second=0, microsecond=0))

del kti_izm1_df, kti_izm2_df
gc.collect()

# %% Справочник грузов
freight_info1_df = pd.read_parquet(DATA_ROOT + '/train/freight_info.parquet').convert_dtypes()
freight_info2_df = pd.read_parquet(DATA_ROOT + '/test/freight_info.parquet').convert_dtypes()

freight_info_df = pd.concat([freight_info1_df, freight_info2_df])
freight_info_df = freight_info_df.sort_values(by=["fr_id"])
freight_info_df = freight_info_df.drop_duplicates(ignore_index=True)

del freight_info1_df, freight_info2_df
gc.collect()

# %% Справочник станций
stations1_df = pd.read_parquet(DATA_ROOT + '/train/stations.parquet').convert_dtypes()
stations2_df = pd.read_parquet(DATA_ROOT + '/test/stations.parquet').convert_dtypes()

stations_df = pd.concat([stations1_df, stations2_df])
stations_df = stations_df.sort_values(by=["st_id"])
stations_df = stations_df.drop_duplicates(ignore_index=True)

del stations1_df, stations2_df
gc.collect()

# %%
df = pd.merge(target_df, wag_prob_df, how='left', on=["wagnum"])
df = df.drop(df[df["repdate"] > df["month"]].index)
df = df.sort_values(by=["wagnum", "month", "repdate"])
df = df.groupby(["wagnum", "month"]).last().reset_index()
df = df.fillna(-1)

df["ost_prob"] = df["ost_prob"].astype(int)
df["manage_type"] = df["manage_type"].astype(int)
df["rod_id"] = df["rod_id"].astype(int)
df["reestr_state"] = df["reestr_state"].astype(int)
numerical_features.extend(["ost_prob", "manage_type", "rod_id", "reestr_state"])

# %%
def fit_cb(df, params, months_val, target_name):

    x_train = df[~df["month"].isin(months_val)][numerical_features + categorical_features + text_features]
    x_val = df[df["month"].isin(months_val)][numerical_features + categorical_features + text_features]

    y_train = df[~df["month"].isin(months_val)][target_name]
    y_val = df[df["month"].isin(months_val)][target_name]
    print(f"Train class imbalance: {y_train.value_counts()}")
    print(f"Val class imbalance: {y_val.value_counts()}")

    train_pool = cb.Pool(
        data = x_train,
        label = y_train,
        cat_features = categorical_features,
        text_features = text_features
    )

    eval_pool = cb.Pool(
        data = x_val,
        label = y_val,
        cat_features = categorical_features,
        text_features = text_features
    )

    model = cb.CatBoostClassifier(**params)

    model.fit(
        train_pool,
        eval_set=eval_pool,
        verbose=True
    )
    print("best results (train on train):")
    print(model.get_best_score()["learn"])
    print("best results (on validation set):")
    print(model.get_best_score()["validation"])
    print(model.get_feature_importance(data=train_pool, prettified=True))

    return model

# %%
cb_params  = {
    'iterations': 2000,
    'loss_function': 'CrossEntropy',
    'custom_metric': ['AUC', 'Accuracy', 'F1'],
    'verbose': False,
    'random_seed': SEED,
    "task_type": "GPU",
    "has_time": True,
    "metric_period": 500,
    "save_snapshot": False,
    "use_best_model": True,
}

cb_month_model = fit_cb(df, cb_params, months_val=months_val, target_name="target_month")
cb_10days_model = fit_cb(df, cb_params, months_val=months_val, target_name="target_day")

# cb_model.save_model(f"{WEIGHTS_ROOT}/cb_{date}.txt")

# %% Val pred
x_val = df[df["month"].isin(months_val)][numerical_features + categorical_features + text_features]

pred_month = cb_month_model.predict(x_val)
pred_10days = cb_10days_model.predict(x_val)

# %%
val_df = pd.DataFrame({"wagnum": df[df["month"].isin(months_val)]["wagnum"]})
val_df["target_month"] = pred_month
val_df["target_day"] = pred_10days

# %% 
def my_calc_f1_score(gt_path, pred_df):

    pred_labels = pred_df.sort_values(by=["wagnum"])

    true_labels = pd.read_csv(gt_path)
    true_labels = true_labels.sort_values(by=["wagnum"])
    
    # Таргет для месячного прогноза
    true_labels_month = true_labels['target_month'].values
    pred_labels_month = pred_labels['target_month'].values

    # Таргет для 10 дневного прогноза
    true_labels_day = true_labels['target_day'].values
    pred_labels_day = pred_labels['target_day'].values

    # Посчитаем метрику для месяца и 10 дней
    score_month = f1_score(true_labels_month, pred_labels_month)
    score_day = f1_score(true_labels_day, pred_labels_day)

    # Посчитаем метрику с весом для двух таргетов
    score = 0.5 * score_month + 0.5 * score_day
    return score

cv_score = my_calc_f1_score(gt_path, val_df)
print(f"{cv_score = }")

# %% Test pred
test_df = pd.read_csv(DATA_ROOT + '/train/target/y_predict.csv').convert_dtypes()
test_df_copy = test_df.copy()

test_df = pd.merge(test_df, wag_prob_df, how='left', on=["wagnum"])
test_df = test_df.drop(test_df[test_df["repdate"] > test_df["month"]].index)
test_df = test_df.groupby(["wagnum", "month"]).last().reset_index()
test_df = test_df.fillna(-1)
test_df = test_df_copy.merge(test_df)

test_df["ost_prob"] = test_df["ost_prob"].astype(int)
test_df["manage_type"] = test_df["manage_type"].astype(int)
test_df["rod_id"] = test_df["rod_id"].astype(int)
test_df["reestr_state"] = test_df["reestr_state"].astype(int)

# %%
x_test = test_df[numerical_features + categorical_features + text_features]

pred_month = cb_month_model.predict(x_test)
pred_10days = cb_10days_model.predict(x_test)

# %%
submit_df = test_df[["wagnum", "month"]]
submit_df["target_month"] = pred_month
submit_df["target_day"] = pred_10days

print(submit_df["target_month"].value_counts())
print(submit_df["target_day"].value_counts())

submit_df.to_csv("zalupa.csv", index=False)

# %%
