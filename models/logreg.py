from utils import filter_predicted_growth_subset, filter_predicted_stable_subset, grid_search
from typing import List
from cfmodel import CfModel
from sklearn.linear_model import LogisticRegression
import pandas as pd


class LogReg(CfModel):
    """
    Create a LogReg model object
    """

    def __init__(self, l90d: pd.DataFrame, n180d: pd.DataFrame, growth_data: pd.DataFrame, stable_data: pd.DataFrame,
                 features: List[str], subset_type: str):
        super().__init__(l90d, n180d, growth_data, stable_data, features, subset_type)
        self.get_x_train_y_train()
        self.model, self.y_pred = self.train()
        self.pred, self.true = self.get_subset_pred_true()
        self.filtered_pred = filter_predicted_growth_subset(self.pred) if self.subset_type == 'growth' else \
            filter_predicted_stable_subset(self.pred)
        print('Log Reg Model has finished training')
    def train(self) -> (LogisticRegression, pd.DataFrame):
        parameters = {
            "C": [0.01],
            "penalty": ['l1']
        }

        model = LogisticRegression(
            max_iter=10000,  # Maximum number of iterations
            random_state=0,  # Random seed for reproducibility
            solver='saga'
        )

        model, y_pred = grid_search(parameters, model, self.X_train, self.y_train,
                                    self.X_test)
        return model, y_pred
