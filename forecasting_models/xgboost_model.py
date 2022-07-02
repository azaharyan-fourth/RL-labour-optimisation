import xgboost as xgb
from xgboost import plot_importance, plot_tree
import pandas as pd

class XGBoostModel:
    def __init__(self,
                **kwargs):
        self.reg = xgb.XGBRegressor(**kwargs)

    def create_features(self, df, label=None):
        """
        Creates time series features from datetime index
        """
        df_copy = df.copy()
        df_copy.drop('date', axis=1, inplace=True)
        if label:
            y = df_copy.pop(label)
            return df_copy, y
            
        return df_copy

    def train(self, X_train, y_train, eval_set=None):

        self.reg.fit(X_train, y_train,
                    eval_set=eval_set,
                    verbose=False)

    def test(self, test):

        predictions = self.reg.predict(test)
        return predictions

if __name__ == '__main__':
    pass
