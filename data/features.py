features_dct = {
    'lgbm_features_lst_growth': [
        'abs_bal_change_std',
        'beta_normalized',
        'label_by_avg_bal',
        'trend',
        'num_ema_crosses',
        'stationary',
        'num_distinct_recurring_tx',
        'recurring_withdrawals',
        'withdrawal_propn',
    ],

    'lgbm_features_lst_stable': [
        'abs_bal_change_std',
        'beta_normalized',
        'label_by_avg_bal',
        'trend',
        'num_distinct_recurring_tx',
        'recurring_withdrawals',
        'withdrawal_propn'
    ],

    'xgb_features_lst_growth': ['abs_bal_change_std',
                                'beta_normalized',
                                'deposits',
                                'withdrawals',
                                'label_by_avg_bal',
                                'trend',
                                'income',
                                'subscription',
                                'stat_sig_positive_kendall'],

    'xgb_features_lst_stable': ['abs_bal_change_std',
                                'beta_normalized',
                                'label_by_avg_bal',
                                'trend',
                                'income'],

    'logreg_features_lst_growth': ['volatility_stdev', 'volatility_cv',
                                   'abs_bal_change_std', 'trend',
                                   'deposits', 'num_ema_crosses',
                                   'num_distinct_recurring_tx',
                                   'withdrawal_propn'],

    'logreg_features_lst_stable': ['growth_coeff', 'abs_bal_change_std',
                                   'deposits', 'withdrawals', 'ema_7day'],

    'mlp_features_lst_growth': [
        'abs_bal_change_std',
        'beta_normalized',
        'label_by_avg_bal',
        'trend',
        'num_ema_crosses',
        'stationary',
        'num_distinct_recurring_tx',
        'recurring_withdrawals',
        'withdrawal_propn',
    ],

    'mlp_features_lst_stable': [
        'abs_bal_change_std',
        'beta_normalized',
        'label_by_avg_bal',
        'trend',
        'num_distinct_recurring_tx',
        'recurring_withdrawals',
        'withdrawal_propn'
    ],

    'features_lst_growth': [
        'abs_bal_change_std',
        'beta_normalized',
        'label_by_avg_bal',
        'trend',
        'num_ema_crosses',
        'stationary',
        'num_distinct_recurring_tx',
        'recurring_withdrawals',
        'withdrawal_propn',
    ],

    'features_lst_stable': [
        'abs_bal_change_std',
        'beta_normalized',
        'label_by_avg_bal',
        'trend',
        'num_distinct_recurring_tx',
        'recurring_withdrawals',
        'withdrawal_propn',
    ]
}
