from numpy import mod
import xgboost as xgb
from xgboost import plot_importance, plot_tree
import pandas as pd
import matplotlib.pyplot as plt

class XGBoostModel:
    def __init__(self):
        self.reg = xgb.XGBRegressor(n_estimators=640, max_depth=7, learning_rate=0.008)

    def create_features(self, df, label=None):
        """
        Creates time series features from datetime index
        """

        X = df[['HoursWorked', 'day_of_week',
                'day_of_month', 'day_of_year', 'year']]
        if label:
            y = df[label]
            return X, y
        return X

    def train(self, X_train, y_train, eval_set=None):

        self.reg.fit(X_train, y_train,
                    eval_set=eval_set,
                    verbose=False)

    def test(self, test):

        predictions = self.reg.predict(test)
        return predictions

if __name__ == '__main__':
    pass
