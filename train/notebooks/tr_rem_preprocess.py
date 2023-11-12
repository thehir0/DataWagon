import pandas as pd
import sys

sys.path.append('..')
from src.timeconverter import date_to_timestamp

def preprocess():
    path_train = r"./"
    tr_rem = pd.read_parquet(path_train + '/tr_rems.parquet').convert_dtypes()

    tr_rem['rem_month'] = tr_rem['rem_month'].astype(str)
    tr_rem['rem_month'] = tr_rem['rem_month'].apply(date_to_timestamp)

    tr_rem.sort_values(by='rem_month', ascending=False, inplace=True)
    tr_rem = tr_rem.drop_duplicates(subset='wagnum', keep='first')
    tr_rem.reset_index(drop=True, inplace=True)

    return tr_rem