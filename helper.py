import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import shutil
import xgboost as xgb


def fn(x):
    return list(x)[0]


def load_data(processed=False):
    if processed:
        return pd.read_hdf('data/processed_data/data.csv')
    item_categories = pd.read_csv('data/all_data/item_categories.csv')
    items = pd.read_csv('data/all_data/items.csv')
    items_categories_merged = items.merge(item_categories, left_on='item_category_id', right_on='item_category_id',
                                          how='inner')
    shops = pd.read_csv('data/all_data/shops.csv')
    train_sales = pd.read_csv('data/all_data/sales_train.csv')
    test_sales = pd.read_csv('data/all_data/test.csv')
    return train_sales, test_sales, items_categories_merged, shops


def save_data(data: pd.DataFrame):
    if os.path.exists('data/processed_data/'):
        shutil.rmtree('data/processed_data/')

    os.mkdir('data/processed_data/')
    data.to_hdf('data/processed_data/data.csv', key='df', index=False)


def exclude_prepositions(x, prepositions_to_exclude):
    x = x.split(' ')
    x = ' '.join(i for i in x if not i in prepositions_to_exclude).strip()
    return x


def create_city_name(x, not_city):
    for i in not_city:
        if i in x:
            return 'unk_city'
    return x.split(' ')[0].strip()


def create_shop_type(x, type_of_shops):
    to_return = 'unk_type'
    for i in type_of_shops:
        regex = re.compile(i)
        if re.search(regex, x):
            to_return = i
    return to_return


def lag_feature(df, lags, col):
    tmp = df[['shop_id', 'item_id', 'date_block_num', col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['shop_id', 'item_id', 'date_block_num', col + '_lag_' + str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return df


def averaged_previous(data, column, target_col, lags=None):
    if not lags:
        lags = [1]
    for i in lags:
        tmp_data = data.copy()
        tmp_data.loc[:, 'date_block_num'] += 1
        if isinstance(column, list):
            to_group = ['date_block_num'] + column
            name = '_'.join(i for i in column)
        else:
            to_group = ['date_block_num'] + [column]
            name = column
        tmp_data = tmp_data.groupby(to_group).agg({target_col: "mean"})
        tmp_data.rename(columns={target_col: target_col + '_previous_averaged_by_' + name + '_lag_' + str(i)},
                        inplace=True)
        data = data.merge(tmp_data, how='left', right_index=True, left_on=to_group)
    return data
