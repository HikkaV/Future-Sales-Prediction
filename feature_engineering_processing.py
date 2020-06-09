from helper import *


def create_tf_idf_feature(items_categories_merged):
    items_categories_merged['type_of_category'] = items_categories_merged['item_category_name'].apply(
        lambda x: x.split(' ')[0].strip())
    dict_types = dict(items_categories_merged['type_of_category'].value_counts())
    cat, _ = zip(*sorted(dict_types.items(), key=lambda x: x[1])[::-1][:5])
    num_features = 10
    symbols_to_exclude = ['[', ']', '!', '.', ',', '*', '(', ')', '"', ':']
    prepositions_to_exclude = ['в', 'на', 'у', 'the', 'a', 'an', 'of', 'для']
    for symbol in symbols_to_exclude:
        items_categories_merged['item_name'] = items_categories_merged['item_name'].str.replace(symbol, '')
    items_categories_merged['item_name'] = items_categories_merged['item_name'].str.lower()
    items_categories_merged['item_name'] = items_categories_merged['item_name'].str.replace('-', ' ')
    items_categories_merged['item_name'] = items_categories_merged['item_name'].str.replace('/', ' ')
    items_categories_merged['item_name'] = items_categories_merged['item_name'].str.strip()
    items_categories_merged['item_name'] = items_categories_merged['item_name'].apply(
        lambda x: exclude_prepositions(x, prepositions_to_exclude))
    vectorizer = TfidfVectorizer(max_features=num_features)
    res = vectorizer.fit_transform(items_categories_merged['item_name'])
    count_vect_df = pd.DataFrame(res.todense(), columns=vectorizer.get_feature_names())
    items_categories_merged = pd.concat([items_categories_merged, count_vect_df], axis=1)
    items_categories_merged.drop(columns=['item_name', 'item_category_name'], inplace=True)
    return items_categories_merged


def create_shop_city_feature(shops):
    not_city = ['Выездная Торговля', 'Интернет-магазин', 'Цифровой склад 1С-Онлайн']
    type_of_shops = ['ТРЦ', 'ТЦ', 'ТРК', 'ТК', 'МТРЦ'] + not_city
    shops['city_name'] = shops['shop_name'].apply(lambda x: create_city_name(x, not_city))
    shops['shop_type'] = shops['shop_name'].apply(lambda x: create_shop_type(x, type_of_shops))
    shops.drop(columns='shop_name', inplace=True)
    return shops


def aggregate_data(train_sales, test_sales):
    train_sales['month'] = train_sales['date'].apply(lambda x: x.split('.')[1])
    from itertools import product
    matrix = []
    cols = ['shop_id', 'item_id', 'date_block_num']
    for i in range(34):
        sales = train_sales[train_sales.date_block_num == i]
        matrix.append(np.array(list(product(test_sales.shop_id.unique(), sales.item_id.unique(), [i])), dtype='int16'))

    matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
    matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
    matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
    matrix['item_id'] = matrix['item_id'].astype(np.int16)
    matrix.sort_values(cols, inplace=True)
    train_sales = train_sales.groupby(['shop_id', 'item_id', 'date_block_num'], as_index=False).agg(
        {'item_cnt_day': np.sum, 'item_price': np.mean,
         'month': fn})
    train_sales = matrix.merge(train_sales, on=['shop_id', 'item_id', 'date_block_num'], how='left')
    train_sales['item_cnt_month'] = train_sales['item_cnt_day'].fillna(0).clip(0, 20)
    train_sales.drop(columns='item_cnt_day', inplace=True)
    test_sales['date_block_num'] = 34
    test_sales.drop(columns='ID', inplace=True)
    data = pd.concat([train_sales, test_sales], ignore_index=True, sort=False,
                     keys=['shop_id', 'item_id', 'date_block_num'])
    month_mapping = data[['month', 'date_block_num']].dropna().drop_duplicates().sort_values(by=['date_block_num']) \
        .set_index('date_block_num').to_dict()['month']
    month_mapping.update({34: '11'})
    data = data.sort_values(by=['date_block_num', 'shop_id', 'item_id'])
    data['item_price'] = data['item_price'].fillna(0)
    data['month'] = data['date_block_num'].map(month_mapping)
    return data


def create_month_feature(data):
    holiday_monthes = ['01', '02', '03', '05', '06', '11']
    data['is_holiday'] = data['month'].apply(lambda x: 1 if x in holiday_monthes else 0)
    return data


def create_lag_features(data):
    data['item_price'] = data['item_price'].fillna(0)
    lower, upper = np.percentile(data['item_price'].values, [1, 99])
    data['item_price'] = data['item_price'].clip(lower, upper)
    data['revenue'] = data['item_price'] * data['item_cnt_month']
    data = lag_feature(data, [1, 3, 6, 7, 8, 9, 10, 12], 'item_cnt_month')
    data = lag_feature(data, [1, 3, 6, 12], 'item_price')
    data = averaged_previous(data, 'shop_id', 'item_cnt_month', [1])
    data = averaged_previous(data, 'item_id', 'item_cnt_month', [1])
    data = averaged_previous(data, 'item_id', 'revenue', [1])
    data.drop(columns=['revenue', 'item_price'], inplace=True)
    data.fillna(0, inplace=True)
    return data


def merge_everything(data, shops, items_categories_merged):
    data = data.merge(shops, on='shop_id', how='left')
    data = data.merge(items_categories_merged, on='item_id', how='left')
    return data


def cast_categorical(data):
    data['is_holiday'] = data['is_holiday'].astype('uint8')
    data['shop_id'] = data['shop_id'].astype('uint8')
    data['month'] = data['month'].astype('uint8')
    data['shop_type'] = data['shop_type'].astype('uint8')
    data['city_name'] = data['city_name'].astype('uint8')
    data['item_category_id'] = data['item_category_id'].astype('uint8')
    data['date_block_num'] = data['date_block_num'].astype('uint8')
    data['item_id'] = data['item_id'].astype('uint16')
    data['type_of_category'] = data['type_of_category'].astype('uint8')


def cast_numerical(data):
    for i in data.columns:
        if 'float' in str(data[i].dtype):
            data[i] = data[i].astype('float16')


def factorize(data):
    to_encode = ['month', 'city_name', 'shop_type', 'type_of_category']
    for i in to_encode:
        data[i] = data[i].factorize()[0]


def execute_pipeline():
    train_sales, test_sales, items_categories_merged, shops = load_data()
    print('Loaded all the data')
    items_categories_merged = create_tf_idf_feature(items_categories_merged)
    print('Created tf-idf feature')
    shops = create_shop_city_feature(shops)
    print('Created shop name and city name features')
    data = aggregate_data(train_sales, test_sales)
    print('Aggregated data')
    data = create_month_feature(data)
    print('Created month feature')
    data = create_lag_features(data)
    print('Created lag features')
    data = merge_everything(data, shops, items_categories_merged)
    print('Merged aggregated data with additional one')
    factorize(data)
    print('Factorized categorical columns')
    cast_categorical(data)
    print('Downcast categorical columns')
    cast_numerical(data)
    print('Downcast numerical columns')
    save_data(data)
    print('Saved data')

if __name__ == '__main__':
    execute_pipeline()