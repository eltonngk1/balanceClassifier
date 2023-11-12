import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Union, Dict
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
import keras.backend as K
from bayes_opt import BayesianOptimization
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def grid_search(parameters, model: Union[XGBClassifier, LGBMClassifier, LogisticRegression],
                X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame):
    """Performs GridSearchCV across all the specified params and returns the best model with optimised set of params

    Args:
        parameters (Dict[str, List[Union[int, float]]): Dictionary of parameters with key: parameter name
            and value: list of possible values
        model (Union[XGBClassifier, LGBMClassifier, LogisticRegression]): specified model
        X_train (pd.DataFrame): df for train features
        y_train (pd.DataFrame): df for train labels
        X_test (pd.DataFrame): df for test features

    Returns:
        (Union[XGBClassifier, LGBMClassifier, LogisticRegression], pd.DataFrame): trained model,
            df for predicted y values

    """

    optimised_model = GridSearchCV(estimator=model, cv=StratifiedKFold(), param_grid=parameters, scoring='recall')
    optimised_model.fit(X_train, y_train)

    y_pred = optimised_model.predict(X_test)

    best_model = optimised_model.best_estimator_
    best_params = optimised_model.best_params_
    best_score = optimised_model.best_score_
    return optimised_model, y_pred


def bo_simple_nn(X_train: pd.DataFrame, y_train: pd.DataFrame, scorer, params: dict) -> dict:
    """
    Perform Bayesian Optimization to find the best hyperparameters for a simple neural network.

    Args:
        X_train (pd.DataFrame): df for train features
        y_train (pd.DataFrame): df for train labels
        scorer (callable): Scoring method to evaluate the predictions on the test set.
        params (dict): Hyperparameter space to search in Bayesian Optimization.

    Returns:
        dict: The result of Bayesian Optimization with the best parameters.
    """

    # Define the neural network model for Bayesian Optimization
    def nn_cl_bo(neurons, activation_index, learning_rate, batch_size, epochs, layers1):
        activation_list = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                           'elu', 'exponential']
        neurons = int(round(neurons))
        activation = activation_list[int(round(activation_index)) % len(activation_list)]
        batch_size = int(round(batch_size))
        epochs = int(round(epochs))
        layers1 = int(round(layers1))

        def nn_cl_fun():
            optimizer = Adam(learning_rate=learning_rate)
            model = Sequential()
            model.add(Dense(neurons, input_dim=X_train.shape[1], activation=activation))
            for _ in range(layers1):
                model.add(Dense(neurons, activation=activation))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            return model

        early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', verbose=0, patience=20)
        classifier = KerasClassifier(build_fn=nn_cl_fun, epochs=epochs, batch_size=batch_size, verbose=0)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        score = cross_val_score(classifier, X_train, y_train, scoring=scorer, cv=kfold,
                                fit_params={'callbacks': [early_stopping]}).mean()
        return score

    # Perform Bayesian Optimization
    nn_bo = BayesianOptimization(nn_cl_bo, params, random_state=111)
    nn_bo.maximize(init_points=10, n_iter=10)

    return nn_bo.max


def get_feature_importance(X_train: pd.DataFrame, best_model: Union[LGBMClassifier, XGBClassifier]) -> List[str]:
    """
    Print the feature importances

    Args:
        X_train (pd.DataFrame): df for train features
        best_model (Union[LGBMClassifier, XGBClassifier]): either a LGBMClassifier or a XGBClassifier model

    Returns:
        List[str]: a list of all the features and their relative importances
    """
    feature_importance = best_model.feature_importances_
    feature_names = X_train.columns

    # Pair feature names with their importance scores
    feature_importance_dict = dict(zip(feature_names, feature_importance))

    # Sort feature importance dictionary by values (importance scores)
    sorted_feature_importance = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))

    top_features = []
    for feature, importance in sorted_feature_importance.items():
        print(f"{feature}: {importance}")
        top_features.append(feature)

    return top_features


def get_logreg_feature_importance(X_train: pd.DataFrame, best_model: LogisticRegression) -> None:
    """
    Print the feature importances

    Args:
        X_train (pd.DataFrame): df for train features
        best_model (LogisticRegression): a Logistic Regression model
    """
    coefficients = best_model.coef_[0]

    feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': np.abs(coefficients)})
    feature_importance = feature_importance.sort_values('Importance', ascending=True)
    print(feature_importance)


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale features of given df

    Args:
        df (pd.DataFrame): df that intended for scaling features

    Returns:
        df (pd.DataFrame): df with scaled features
    """
    scaled = StandardScaler()
    scaled.fit(df)
    df_scaled = pd.DataFrame(scaled.transform(df), columns=df.columns)
    return df_scaled


def get_vif(features: pd.DataFrame) -> None:
    """
    Measures degree of multi-collinearity within the features

    Args:
        features (pd.DataFrame): df for features
    """
    vif_data = pd.DataFrame()
    vif_data["Features"] = features.columns
    vif_data["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]

    print(vif_data.sort_values(by='VIF', ascending=False))
    print("\n")


def filter_predicted_growth_subset(growth_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the predicted growth subset of false positives to get 5% weight of entire portfolio

    Args:
        growth_pred (pd.DataFrame): df with predicted growth labels

    Returns:
        pd.DataFrame: final predicted growth subset with 5% weight
    """
    # Filter for users who exhibit positive trend
    growth_pred = growth_pred[(growth_pred['stat_sig_positive_kendall'] == 1) | (growth_pred['trend'] == 2)].copy()

    # Calculate growth rate
    balance_growth_rate = (growth_pred['last_day_balance'] - growth_pred['avg_balance']) / growth_pred['avg_balance']

    # growth rate will be NaN if avg balance = 0
    # For such cases we use growth coefficient as proxy for growth rate
    growth_pred['balance_growth_rate'] = np.where(growth_pred['avg_balance'] != 0,
                                                  balance_growth_rate,
                                                  growth_pred['growth_coeff'])

    growth_pred_sorted = growth_pred.sort_values(by=['balance_growth_rate'], ascending=False)

    # Filter for 5% of subset weight
    growth_pred_final = filter_top_5_percent(growth_pred_sorted)

    return growth_pred_final


def filter_predicted_stable_subset(stable_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the predicted stable subset of false positives to get 5% weight of entire portfolio

    Args:
        stable_pred (pd.DataFrame): df with predicted growth labels

    Returns:
        pd.DataFrame: final predicted growth subset with 5% weight
    """
    stable_pred = stable_pred[(stable_pred['stat_sig_positive_kendall'] == 1) | (stable_pred['trend'] == 2) | (
            (stable_pred['stationary'] == 1) & (stable_pred['trend'] == 1))].copy()

    # Can't sort by stability index because we've already established that
    # the most stable users for the next 180 days were not the most stable for their first 90 days
    stable_pred_sorted = stable_pred.sort_values(by=['trend', 'stat_sig_positive_kendall', 'stationary'],
                                                 ascending=False)

    # Filter for 5% of subset weight
    stable_pred_final = filter_top_5_percent(stable_pred_sorted)

    return stable_pred_final


def filter_top_5_percent(pred: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function that filters for 5% of cumulative weight

    Args:
        pred (pd.DataFrame): df with either growth/stable features and labels

    Returns:
        pd.DataFrame: final predicted subset with 5% weight
    """
    pred['cumulative_weight'] = pred['weight'].cumsum()
    pred_final = pred[pred['cumulative_weight'] <= 0.05].copy()

    final_weight = pred_final['weight'] / pred_final['weight'].sum()
    final_weighted_stability = final_weight * pred_final['stability_index']

    pred_final['final_weight'] = final_weight
    pred_final['weighted_stability'].update(final_weighted_stability)

    return pred


def generate_features(n180d: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function that generates features on the n180d data

    Args:
        n180d (pd.DataFrame): df with complete n180d data

    Returns:
        pd.DataFrame: df with new features
    """
    n180d_copy = n180d.copy()
    n180d_copy = n180d_copy.sort_values(by=['user_id', 'pt_date'])
    n180d_copy = n180d_copy.groupby('user_id').agg(total_balance_std=('total_balance', 'std'),
                                                   avg_balance=('total_balance', 'mean'),
                                                   first_day_balance=('total_balance', 'first'),
                                                   last_day_balance=('total_balance', 'last')).reset_index()
    n180d_copy['weight'] = n180d_copy['avg_balance'] / n180d_copy['avg_balance'].sum()
    n180d_copy['cv'] = n180d_copy['total_balance_std'] / n180d_copy['avg_balance']
    n180d_copy['cv_scaled'] = (n180d_copy['cv'] - min(n180d_copy['cv'])) / (
            max(n180d_copy['cv']) - min(n180d_copy['cv']))
    n180d_copy["stability_index"] = 1 - n180d_copy['cv_scaled']
    n180d_copy['weighted_stability'] = n180d_copy['weight'] * n180d_copy['stability_index']
    return n180d_copy


def get_metrics_df(y_test, y_pred) -> pd.DataFrame:
    """
    Helper function that calculates evaluation statistics

    Args:
        y_test (pd.DataFrame): df with test labels
        y_pred (pd.DataFrame): df with predicted labels

    Returns:
        pd.DataFrame: df with single set of metrics for a model
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    roc_auc = roc_auc_score(y_test, y_pred, average=None)

    metrics_dict = {
        'Class': [0, 1],
        'Accuracy': accuracy.tolist(),
        'Precision': precision.tolist(),
        'Recall': recall.tolist(),
        'ROC-AUC': roc_auc.tolist(),
        'F1-Score': f1.tolist(),
    }

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.set_index('Class', inplace=True)

    return metrics_df


def get_all_metrics_df(y_pred_dct: Dict[str, pd.DataFrame], y_test: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function that calculates evaluation statistics

    Args:
        y_pred_dct (pd.DataFrame): Dict with key: model name & value: predicted labels for model
        y_test (pd.DataFrame): df with test labels

    Returns:
        pd.DataFrame: df with combined metrics for a few models
    """
    model_labels = ["LGBM", "XGB", "LogReg", "MLP", "Voting", "Stacking"]

    all_metrics_dfs = []
    for label, (y_pred, y_test) in zip(model_labels, [(y_pred_dct['LGBM'], y_test), (y_pred_dct['XGB'], y_test),
                                                      (y_pred_dct['LogReg'], y_test), (y_pred_dct['MLP'], y_test),
                                                      (y_pred_dct['Voting'], y_test),
                                                      (y_pred_dct['Stacking'], y_test)]):
        metrics_df = get_metrics_df(y_test, y_pred)
        metrics_df["Model"] = label  # Add a 'Model' column to label the metrics
        all_metrics_dfs.append(metrics_df)

    all_metrics_df = pd.concat(all_metrics_dfs, axis=0)
    all_metrics_df.reset_index(inplace=True)

    return all_metrics_df
