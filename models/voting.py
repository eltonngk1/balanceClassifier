from sklearn.ensemble import VotingClassifier
from utils import filter_predicted_growth_subset, filter_predicted_stable_subset
import pandas as pd
from cfmodel import CfModel
from typing import List
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from scikeras.wrappers import KerasClassifier

class Voting(CfModel):
    """Create a Voting Ensemble model object
    """

    def __init__(self, l90d: pd.DataFrame, n180d: pd.DataFrame, growth_data: pd.DataFrame, stable_data: pd.DataFrame,
                 features: List[str], subset_type: str, xgb: XGBClassifier, lgbm: LGBMClassifier,
                 logreg: LogisticRegression, mlp: KerasClassifier):
        super().__init__(l90d, n180d, growth_data, stable_data, features, subset_type)
        self.xgb = xgb
        self.lgbm = lgbm
        self.logreg = logreg
        self.mlp = mlp
        self.get_x_train_y_train()
        self.model, self.y_pred = self.train()
        self.pred, self.true = self.get_subset_pred_true()
        self.filtered_pred = filter_predicted_growth_subset(self.pred) if self.subset_type == 'growth' else \
            filter_predicted_stable_subset(self.pred)

    def train(self) -> (VotingClassifier, pd.DataFrame):
        voting_ensemble = VotingClassifier(estimators=[
            ('XGBoost', self.xgb),
            ('LightGBM', self.lgbm),
            ('Logistic Regression', self.logreg),
            ('MLP', self.mlp),
        ], voting='soft')

        voting_ensemble.fit(self.X_train, self.y_train)

        y_pred = voting_ensemble.predict(self.X_test)
        return voting_ensemble, y_pred
