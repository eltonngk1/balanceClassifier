import pandas as pd
from imblearn.over_sampling import SMOTE
from typing import List, Union
from utils import *
from sklearn.model_selection import train_test_split


class CfModel:
    """
    A Classification model
    """

    def __init__(self, l90d: pd.DataFrame, n180d: pd.DataFrame, growth_data: pd.DataFrame, stable_data: pd.DataFrame,
                 features: List[str], subset_type: str):
        """Create a base classifier model object

        Args:
            l90d (pd.DataFrame): df for last 90 days training data.
            n180d (pd.DataFrame): df for next 180 days test data.
            growth_data (pd.DataFrame): df for growth data with growth labels
            stable_data (pd.DataFrame): df for growth data with growth labels
            features (List[str]): either growth/stable features
            subset_type (str): string specifying 'growth' or 'stable'
        """
        self.l90d = l90d
        self.n180d = n180d
        self.df270 = pd.concat([self.l90d, self.n180d], ignore_index=True) \
            .drop_duplicates() \
            .sort_values(by=['user_id', 'pt_date'])
        self.subset_type = subset_type
        self.growth_data = growth_data
        self.stable_data = stable_data
        self.data = self.growth_data if subset_type == 'growth' else self.stable_data
        self.features = features

        # model training variables
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_users = None
        self.test_users = None
        self.y_pred = None
        self.model = None
        self.get_x_train_y_train()

    def get_x_train_y_train(self) -> None:
        """
        Prepares data for modeling by conducting train-test split and applying SMOTE.
        """
        data = self.data
        X = data.drop(columns=['label'])
        y = data['label']

        # Train-test split on user level
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

        train_users = X_train['user_id'].tolist()
        test_users = X_test['user_id'].tolist()
        X_train = X_train.drop(columns=['user_id'])
        X_test = X_test.drop(columns=['user_id'])

        # oversample with SMOTE to overcome class imbalance
        smote = SMOTE(sampling_strategy='minority', random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        self.X_train, self.y_train, self.X_test, self.y_test, self.train_users, self.test_users = (
            X_train[self.features], y_train, X_test[self.features], y_test, train_users, test_users)

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

    def show_metrics(self):
        metrics_df = get_metrics_df(self.y_test, self.y_pred)
        print(metrics_df)
        return metrics_df

    def get_results_df(self, user_features: pd.DataFrame) -> pd.DataFrame:
        """
        Adds user features to the test set of users

        Args:
            user_features (pd.DataFrame): Df containing user features

        Returns:
            pd.DataFrame: df with user labels, features and weight
        """
        results_df = pd.DataFrame({
            'user_id': self.test_users,
            'true_label': self.y_test,
            'predicted_label': self.y_pred
        })

        results_df = results_df.merge(user_features, on='user_id', how='left')

        # Weight of each user relative to the 10k validation set users (not entire portfolio of 50k users)
        results_df['weight'] = results_df['avg_balance'] / np.sum(results_df['avg_balance'])
        return results_df

    def get_subset_pred_true(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Returns two separate predicted and true subsets
        """
        results_df = self.get_results_df(self.data)
        # Predicted subset
        subset_pred = results_df[results_df['predicted_label'] == 1].copy()
        subset_pred = self._get_subset_pred_true(subset_pred)

        # Actual subset
        subset_true = results_df[results_df['true_label'] == 1].copy()
        subset_true = self._get_subset_pred_true(subset_true)

        return subset_pred, subset_true

    @staticmethod
    def _get_subset_pred_true(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weight relative to subset & obtain weighted stability
        """
        df['subset_weight'] = df['avg_balance'] / df['avg_balance'].sum()
        df['weighted_stability'] = df['subset_weight'] * df['stability_index']

        return df
