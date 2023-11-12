from cfmodel import CfModel
from sklearn.metrics import recall_score, make_scorer
from utils import *


class MLP(CfModel):
    """
    Create a MLP model object
    """

    def __init__(self, l90d: pd.DataFrame, n180d: pd.DataFrame, growth_data: pd.DataFrame, stable_data: pd.DataFrame,
                 features: List[str], subset_type: str):
        super().__init__(l90d, n180d, growth_data, stable_data, features, subset_type)
        self.get_x_train_y_train()
        self.model, self.y_pred = self.train()
        self.pred, self.true = self.get_subset_pred_true()
        self.filtered_pred = filter_predicted_growth_subset(self.pred) if self.subset_type == 'growth' else \
            filter_predicted_stable_subset(self.pred)

        print('MLP Model has finished training')

    def find_best_params(self):
        params_simple_nn = {
            'neurons': (10, 100),
            'activation': (0, 8),
            'learning_rate': (0.01, 1),
            'batch_size': (200, 1000),
            'epochs': (20, 100),
            'layers1': (1, 5)
        }

        scorer_rec = make_scorer(recall_score)
        bo_simple_nn(self.X_train, self.y_train, scorer=scorer_rec, params=params_simple_nn)

    def train(self) -> (KerasClassifier, pd.DataFrame):
        params = {
            'stable': {'learning_rate': 0.04374, 'neurons': 95, 'layers1': 4, 'epochs': 94, 'batch_size': 958,
                       'activation': 'softsign'},
            'growth': {'learning_rate': 0.01, 'neurons': 68, 'layers1': 5, 'epochs': 100, 'batch_size': 973,
                       'activation': 'softsign'}
        }

        # Set default values if subset_type is not 'stable'
        chosen_params = params[self.subset_type]

        learning_rate = chosen_params['learning_rate']
        neurons = chosen_params['neurons']
        layers1 = chosen_params['layers1']
        epochs = chosen_params['epochs']
        batch_size = chosen_params['batch_size']
        activation = chosen_params['activation']

        def nn_cl_fun():
            opt = Adam(learning_rate=learning_rate)
            nn = Sequential()
            nn.add(Dense(neurons, input_dim=len(self.X_train.columns), activation=activation))
            for i in range(layers1):
                nn.add(Dense(neurons, activation=activation))
            nn.add(Dense(1, activation='sigmoid'))
            nn.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', recall])
            return nn

        nn = KerasClassifier(build_fn=nn_cl_fun, epochs=epochs, batch_size=batch_size,
                             verbose=0)
        nn.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), verbose=0)
        y_pred = nn.predict(self.X_test)
        return nn, y_pred
