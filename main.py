import pandas as pd
from bestmodelselector import BestModelSelector

if __name__ == '__main__':
    train_data_path = './data/train_data_l90d_daily_balance.csv'
    test_data_path = './data/train_data_n180d_daily_balance.csv'
    growth_data = './data/user_features_l90_growth_20231111.csv'
    stable_data = './data/user_features_l90_stable_20231111.csv'
    x = BestModelSelector(train_data_path, test_data_path, growth_data, stable_data)

    # customise which statistics and plot to visualise, indicate model (lowercase) & subset type
    x.calculate_pred_vs_benchmark('xgb', 'growth')

    best_growth_model_name, best_growth_users, best_stable_model_name, best_stable_users = x.get_best_model()
    print(f'The best stable model is ..... {best_stable_model_name}')
    print(f'The best growth model is ..... {best_growth_model_name}')

    # exports user_ids
    best_stable_users.to_csv('best_stable_users.csv', index=False)
    best_growth_users.to_csv('best_stable_users.csv', index=False)
