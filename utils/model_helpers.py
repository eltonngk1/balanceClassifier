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
from bayes_opt import BayesianOptimization


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
