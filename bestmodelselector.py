import pandas as pd
import matplotlib.pyplot as plt
from data import *
from typing import Union
from models import XGB, MLP, LGBM, LogReg, Stacking, Voting
from cfmodel import CfModel
from utils import generate_features
import numpy as np

class BestModelSelector:
    '''
     A class that supervises the prediction of a user's predicted label (Subset/Growth)
     across 4 classification models (XGB, LGBM, LogReg, NN)
     and 2 ensemble techniques (Stacking, Voting) and returns the best model for each subset label
     and
     '''

    def __init__(self, l90d: str, n180d: str, growth_data: str, stable_data: str):
        """Create a CfModel object

        Args:
            l90d (str): Name of csv file containing users' daily bank balances for the first 90 days. 
                        The csv file should have 3 columns: 'pt_date' (data date), 'user_id' (unique identifier of user), 
                        'total_balance' (user's account balance on a given date).

            n180d (str): Name of csv file containing users' daily bank balances for the next 180 days. 
                        The format of this file should be similar to the 'l90d' file.
                        
            growth_data (str): Name of csv file with 'user_id' as index, containing each user's 'label' (growth vs. non-growth) 
                                and user-level features generated from the first 90 days of data.

            stable_data (str): Name of csv file with 'user_id' as index, containing each user's 'label' (stable vs. non-stable) 
                                and user-level features generated from the first 90 days of data.

        """
        self.end_90_days = '2023-03-01'
        self.xgb_growth = self.xgb_stable = self.mlp_stable = self.mlp_growth = None
        self.logreg_stable = self.logreg_growth = self.lgbm_growth = self.lgbm_stable = None
        self.voting_stable = self.voting_growth = self.stacking_stable = self.stacking_growth = None
        self.l90d = pd.read_csv(l90d)
        self.n180d = pd.read_csv(n180d)
        self.df270 = pd.concat([self.l90d, self.n180d], ignore_index=True) \
            .drop_duplicates() \
            .sort_values(by=['user_id', 'pt_date'])
        self.growth_data = pd.read_csv(growth_data)
        self.stable_data = pd.read_csv(stable_data)
        self.generate_models()

    def get_baseline(self) -> (Union[int, float], Union[int, float]):
        """
        Calculates the baseline growth rate and the baseline stability index
        The baseline metrics take into account all users in the dataset provided.

        Returns:
            (Union[int, float], Union[int, float]): growth rate along with the stability index
        """
        growth_baseline_subset = self.n180d.copy()
        growth_baseline_subset = growth_baseline_subset.groupby('pt_date')['total_balance'].sum().reset_index()
        baseline_growth = ((growth_baseline_subset.iloc[-1, 1] - growth_baseline_subset.iloc[0, 1]) /
                           growth_baseline_subset.iloc[0, 1]) * 100

        # stability index of entire portfolio over next 180 days actual data
        stable_baseline_subset = self.n180d.copy()
        stable_baseline_subset = stable_baseline_subset.groupby('user_id').agg(
            total_balance_std=('total_balance', 'std'),
            avg_balance=('total_balance', 'mean')).reset_index()
        stable_baseline_subset['weight'] = stable_baseline_subset['avg_balance'] / np.sum(
            stable_baseline_subset['avg_balance'])
        baseline_stability = self.get_weighted_stability(stable_baseline_subset)

        return baseline_growth, baseline_stability

    def get_weighted_stability(self, df: pd.DataFrame) -> Union[int, float]:
        """
        Calculates a subset's (df) stability with the following steps:
        1)  Calculate each user's stability index and weight, relative to users in 'df'
            The stabilty index of users will range from 0 to 1, and the sum of all user weights is 1.

        2)  Calculated the weighted stability of each user in 'df' by multiplying their weight and stability index.

        3)  The stability of the subset is obtained by summing the weighted stability of all users in 'df'.

        Args:
            df (pd.DataFrame): dataframe of a particular subset

        Returns:
            Union[int, float]: stability index of subset i.e., sum of weighted stability for the users in df
        """

        # Calculate stability_index with next 180 days data
        result = self.n180d.groupby('user_id').agg(total_balance_std=('total_balance', 'std'),
                                                   avg_balance=('total_balance', 'mean')).reset_index()

        result['cv'] = result['total_balance_std'] / result['avg_balance']
        result['cv_scaled'] = (result['cv'] - min(result['cv'])) / (max(result['cv']) - min(result['cv']))
        result["stability_index"] = 1 - result['cv_scaled']

        # sum of final_weight = 1
        # use weight to generate weighted stability
        if 'final_weight' not in df.columns:
            df['final_weight'] = df['weight'] / df['weight'].sum()

        result = df[['user_id', 'final_weight']].merge(result[['user_id', 'stability_index']], on='user_id', how='left')
        result['weighted_stability'] = result['final_weight'] * result['stability_index']

        return result['weighted_stability'].sum()

    def get_benchmark(self) -> (
            pd.DataFrame, pd.DataFrame, Union[int, float], Union[int, float], Union[int, float], Union[int, float]):
        """
        Generates the growth & stable benchmark subsets along with the growth rates and stability indexes
        """
        growth_cf_model = CfModel(self.l90d, self.n180d, self.growth_data, self.stable_data, ['total_balance'],
                                  'growth')
        print('test')
        print(growth_cf_model.test_users)
        test_users = growth_cf_model.test_users
        growth_benchmark = self.growth_data.sort_values(by=['growth_coeff'], ascending=False)
        growth_benchmark = growth_benchmark[growth_benchmark['user_id'].isin(growth_cf_model.test_users)]
        growth_benchmark_growth, growth_benchmark_stability_index = self._get_subset_benchmark(growth_benchmark)

        stable_cf_model = CfModel(self.l90d, self.n180d, self.growth_data, self.stable_data, ['total_balance'],
                                  'stable')
        stable_benchmark = self.stable_data.sort_values(by=['stability_index'], ascending=False)
        stable_benchmark = stable_benchmark[stable_benchmark['user_id'].isin(stable_cf_model.test_users)]
        stable_benchmark_growth, stable_benchmark_stability_index = self._get_subset_benchmark(stable_benchmark)

        return (growth_benchmark, stable_benchmark, growth_benchmark_growth, growth_benchmark_stability_index,
                stable_benchmark_growth, stable_benchmark_stability_index)

    def _get_subset_benchmark(self, subset: pd.DataFrame) -> (Union[int, float], Union[int, float]):
        """
        Helper function for calculating subset statistics i.e, growth rate and stability index
        Args:
            subset (pd.DataFrame): df for a subset
        Returns:
            (Union[int, float], Union[int, float]): growth rate along with the stability index
        """
        df = generate_features(self.n180d)
        subset['weight'] = subset['avg_balance'] / subset['avg_balance'].sum()
        subset['cumulative_weight'] = subset['weight'].cumsum()
        subset = subset[subset['cumulative_weight'] <= 0.05]

        final_df = df[df['user_id'].isin(subset['user_id'])]
        first_day_balance = final_df['first_day_balance'].sum()
        last_day_balance = final_df['last_day_balance'].sum()
        growth_rate = ((last_day_balance - first_day_balance) / first_day_balance) * 100
        stability_index = self.get_weighted_stability(final_df)
        return growth_rate, stability_index

    def get_ideal(self) -> (
            pd.DataFrame, pd.DataFrame, Union[int, float], Union[int, float], Union[int, float], Union[int, float]):
        """
        Generates the growth & stable ideal subsets along with the growth rates and stability indexes
        """
        growth_ideal = self.growth_data[self.growth_data['label'] == 1]
        stable_ideal = self.stable_data[self.stable_data['label'] == 1]
        stable_ideal_growth, stable_ideal_stability_index = self._get_subset_ideal(stable_ideal)
        growth_ideal_growth, growth_ideal_stability_index = self._get_subset_ideal(growth_ideal)

        return (growth_ideal, stable_ideal, growth_ideal_growth, growth_ideal_stability_index,
                stable_ideal_growth, stable_ideal_stability_index)

    def _get_subset_ideal(self, ideal: pd.DataFrame) -> (Union[int, float], Union[int, float]):
        """
        Calculates the ideal growth rate and the ideal stability index
        Args:
            ideal (pd.DataFrame): df for filtered ideal subset on 180 days

        Returns:
            (Union[int, float], Union[int, float]): growth rate along with the stability index
        """
        n180d = generate_features(self.n180d)
        ideal_subset = n180d[n180d['user_id'].isin(ideal['user_id'])]
        first_day_balance = ideal_subset['first_day_balance'].sum()
        last_day_balance = ideal_subset['last_day_balance'].sum()
        ideal_growth = ((last_day_balance - first_day_balance) / first_day_balance) * 100

        return ideal_growth, self.get_weighted_stability(ideal_subset)

    def get_plotting_df(self, subset_pred, subset_benchmark):
        """
            This function takes in 2 user-level dataframes of actual & predicted subset respectively.
            It filters for the subset users' daily bank balance data across 270 days, from that of the full set of users.
            The function then aggregates the daily balances of these subset users to create a time series 
            representing the total daily balance data of the subset across 270 days.

        Args:
            subset_pred (pd.DataFrame): df for users in predicted subset
            subset_benchmark (pd.DataFrame): df for users in true subset

        Returns:
            pd.DataFrame, pd.DataFrame: 2 dataframes in time series format (pt_date & total_balance)
        """

        subset_pred_users = subset_pred.user_id.tolist()
        subset_benchmark_users = subset_benchmark.user_id.tolist()

        subset_pred_270 = self.df270[self.df270['user_id'].isin(subset_pred_users)] \
            .groupby('pt_date')['total_balance'].sum().reset_index()

        subset_benchmark_270 = self.df270[self.df270['user_id'].isin(subset_benchmark_users)] \
            .groupby('pt_date')['total_balance'].sum().reset_index()

        return subset_pred_270, subset_benchmark_270

    def _plot_time_series(self, subset_pred_270, subset_benchmark_270, subset_type):
        plt.figure(figsize=(12, 3))
        plt.plot(subset_pred_270['pt_date'], subset_pred_270['total_balance'],
                 label=f"Predicted {subset_type} subset")
        plt.plot(subset_benchmark_270['pt_date'], subset_benchmark_270['total_balance'],
                 label=f"Benchmark {subset_type} subset")
        plt.xlabel('Date')
        plt.ylabel('Balance')
        plt.title(f'Predicted vs Benchmark {subset_type} Subset Balances Over Time')
        plt.xticks(np.arange(0, len(subset_pred_270['pt_date']), 30), rotation=45)
        plt.axvline(x=self.end_90_days, color='r', linestyle='--', label='End of last 90 days')
        plt.legend()
        plt.show()

    def plot_pred_against_actual(self, subset_pred_270, subset_benchmark_270,
                                 subset_pred, subset_benchmark,
                                 subset_type):
        """
           This function plots the balance time series of the predicted subset against the actual subset across 270 days
           Prints the metrics across the predicted subset and all the benchmarks
        Args:
            subset_pred_270 (pd.DataFrame): time series df for all users
            subset_benchmark_270 (pd.DataFrame): time series df for all users
            subset_pred (pd.DataFrame): df for users in predicted subset
            subset_benchmark (pd.DataFrame): df for users in benchmark subset
            subset_type (str): string indicating 'growth' or 'stable'

        """
        self._plot_time_series(subset_pred_270, subset_benchmark_270, subset_type)
        self.print_comparison_metrics(subset_pred_270, subset_pred, subset_benchmark, subset_type)

    @staticmethod
    def print_metrics(weight, num_users, growth_rate, stability_index):
        print(f"Weight: ", round(weight, 5))
        print(f"No. of users: ", num_users)

        print(f"Growth rate: ", round(growth_rate * 100, 3), "%")
        print(f"Stability index: ", round(stability_index, 5))

    def print_comparison_metrics(self, subset_pred_270,
                                 subset_pred, subset_benchmark,
                                 subset_type):
        # handle data for data reporting
        num_users_pred = len(subset_pred.user_id.tolist())
        num_users_baseline = len(self.n180d)
        num_users_benchmark = len(subset_benchmark)
        if subset_type == 'growth':
            num_users_ideal = len(self.growth_data[self.growth_data['label'] == 1])
        else:
            num_users_ideal = len(self.stable_data[self.stable_data['label'] == 1])

        weight_pred = subset_pred.weight.sum()
        weight_baseline = 1
        weight_benchmark = subset_benchmark.weight.sum()

        growth_ideal_subset, stable_ideal_subset, growth_ideal_growth, growth_ideal_stability_index, \
            stable_ideal_growth, stable_ideal_stability_index = self.get_ideal()
        baseline_growth, baseline_stability = self.get_baseline()
        growth_benchmark, stable_benchmark, growth_benchmark_growth, growth_benchmark_stability_index, \
            stable_benchmark_growth, stable_benchmark_stability_index = self.get_benchmark()

        if subset_type == 'growth':
            weight_ideal = growth_ideal_subset['weight'].sum()
        else:
            weight_ideal = stable_ideal_subset['weight'].sum()

        # Next 180 days metrics
        growth_pred_n180 = (subset_pred_270.iloc[-1]['total_balance'] - subset_pred_270.iloc[90]['total_balance']) / \
                           subset_pred_270.iloc[90]['total_balance']
        weighted_stability_pred_n180 = self.get_weighted_stability(subset_pred)

        # Predicted subset
        print(f"Predicted {subset_type} subset (based on last 90 days data):")
        self.print_metrics(weight_pred, num_users_pred, growth_pred_n180, weighted_stability_pred_n180)

        # Baseline
        print(f'\nBaseline (based on next 180 days data):')
        self.print_metrics(weight_baseline, num_users_baseline, baseline_growth, baseline_stability)

        # Benchmark
        print(f"\nBenchmark {subset_type} subset (based on next 180 days data):")
        if subset_type == 'growth':
            self.print_metrics(weight_benchmark, num_users_benchmark, growth_benchmark_growth,
                               growth_benchmark_stability_index)
        else:
            self.print_metrics(weight_benchmark, num_users_benchmark, stable_benchmark_growth,
                               stable_benchmark_stability_index)

        # Ideal
        print(f'\nIdeal {subset_type} (based on next 180 days data):')
        if subset_type == 'growth':
            self.print_metrics(weight_ideal, num_users_ideal, growth_ideal_growth,
                               growth_ideal_stability_index)
        else:
            self.print_metrics(weight_ideal, num_users_ideal, stable_ideal_growth,
                               stable_ideal_stability_index)

    def generate_models(self):
        self.xgb_growth = XGB(self.l90d, self.n180d, self.growth_data, self.stable_data,
                              features_dct['xgb_features_lst_growth'], 'growth').model
        self.xgb_stable = XGB(self.l90d, self.n180d, self.growth_data, self.stable_data,
                              features_dct['xgb_features_lst_stable'], 'stable').model

        self.lgbm_growth = LGBM(self.l90d, self.n180d, self.growth_data, self.stable_data,
                                features_dct['lgbm_features_lst_growth'], 'growth').model
        self.lgbm_stable = LGBM(self.l90d, self.n180d, self.growth_data, self.stable_data,
                                features_dct['lgbm_features_lst_stable'], 'stable').model

        self.logreg_growth = LogReg(self.l90d, self.n180d, self.growth_data, self.stable_data,
                                    features_dct['logreg_features_lst_growth'], 'growth').model
        self.logreg_stable = LogReg(self.l90d, self.n180d, self.growth_data, self.stable_data,
                                    features_dct['logreg_features_lst_stable'], 'stable').model

        self.mlp_growth = MLP(self.l90d, self.n180d, self.growth_data, self.stable_data,
                              features_dct['mlp_features_lst_growth'], 'growth').model
        self.mlp_stable = MLP(self.l90d, self.n180d, self.growth_data, self.stable_data,
                              features_dct['mlp_features_lst_stable'], 'stable').model

        self.voting_growth = Voting(self.l90d, self.n180d, self.growth_data, self.stable_data,
                                    features_dct['features_lst_growth'], 'growth', xgb=self.xgb_growth,
                                    lgbm=self.lgbm_growth,
                                    logreg=self.logreg_growth, mlp=self.mlp_growth).model
        self.voting_stable = Voting(self.l90d, self.n180d, self.growth_data, self.stable_data,
                                    features_dct['features_lst_stable'], 'stable', xgb=self.xgb_stable,
                                    lgbm=self.lgbm_stable,
                                    logreg=self.logreg_stable, mlp=self.mlp_stable).model

        self.stacking_growth = Stacking(self.l90d, self.n180d, self.growth_data, self.stable_data,
                                        features_dct['features_lst_growth'], 'growth', xgb=self.xgb_growth,
                                        lgbm=self.lgbm_growth,
                                        logreg=self.logreg_growth).model
        self.stacking_stable = Stacking(self.l90d, self.n180d, self.growth_data, self.stable_data,
                                        features_dct['features_lst_stable'], 'stable', xgb=self.xgb_stable,
                                        lgbm=self.lgbm_stable,
                                        logreg=self.logreg_stable).model

    def plot(self, model_name: str, subset_type: str):
        model_map = {
            'stable': {'xgb': self.xgb_stable, 'lgbm': self.lgbm_stable, 'logreg': self.logreg_stable,
                       'mlp': self.mlp_stable, 'voting': self.voting_stable, 'stacking': self.stacking_stable},
            'growth': {'xgb': self.xgb_growth, 'lgbm': self.lgbm_growth, 'logreg': self.logreg_growth,
                       'mlp': self.mlp_growth, 'voting': self.voting_growth, 'stacking': self.stacking_growth},
        }
        model = model_map[subset_type][model_name]
        growth_benchmark, stable_benchmark, growth_benchmark_growth, growth_benchmark_stability_index, \
            stable_benchmark_growth, stable_benchmark_stability_index = self.get_benchmark()
        benchmark_subset = growth_benchmark if subset_type == 'growth' else stable_benchmark
        pred_270_final, benchmark_270 = self.get_plotting_df(model.filtered_pred, benchmark_subset)
        self.plot_pred_against_actual(pred_270_final, benchmark_270, model.filtered_pred, benchmark_subset, subset_type)


    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_data_path = './data/train_data_l90d_daily_balance.csv'
    test_data_path = './data/train_data_n180d_daily_balance.csv'
    growth_data = './data/user_features_l90_growth_20231111.csv'
    stable_data = './data/user_features_l90_stable_20231111.csv'
    x = BestModelSelector(train_data_path, test_data_path, growth_data, stable_data)
    x.plot('xgb', 'growth')
