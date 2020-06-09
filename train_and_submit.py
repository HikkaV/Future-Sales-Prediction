from helper import *


def train_and_submit():
    data = load_data(True)
    print('Loaded data')
    train_subset, val_subset, test_subset = data[data.date_block_num < 33], \
                                            data[data.date_block_num == 33], \
                                            data[data.date_block_num == 34]
    Y_train = train_subset['item_cnt_month']
    X_train = train_subset.drop(columns=['item_cnt_month', 'date_block_num'])
    Y_valid = val_subset['item_cnt_month']
    X_valid = val_subset.drop(columns=['item_cnt_month', 'date_block_num'])
    X_test = test_subset.drop(columns=['item_cnt_month', 'date_block_num'])
    print('Created training and validation data')
    model = xgb.XGBRegressor(
        max_depth=4,
        n_estimators=100,
        eta=0.3,
        min_child_weight=0.4,
        n_jobs=4,
        seed=5)

    eval_set = [(X_train, Y_train), (X_valid, Y_valid)]

    model.fit(
        X_train,
        Y_train,
        eval_metric="rmse",
        eval_set=eval_set,
        verbose=True,
        early_stopping_rounds=10)
    print('Trained model')
    prediction = np.clip(model.predict(X_test), 0, 20)
    submission = pd.read_csv('data/all_data/test.csv')
    X_test['item_cnt_month'] = prediction
    sub_to_merge = X_test[['shop_id', 'item_id', 'item_cnt_month']].copy()
    submission = submission.merge(sub_to_merge, how='left', on=['shop_id', 'item_id'])
    submission = submission[['ID', 'item_cnt_month']]
    if not os.path.exists('data/submission/'):
        os.mkdir('data/submission/')
    submission.to_csv('data/submission/submission.csv', index=False)
    print('Made submission file')

if __name__ == '__main__':
    train_and_submit()
