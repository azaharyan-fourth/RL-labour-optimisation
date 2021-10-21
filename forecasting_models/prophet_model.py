import json
from datetime import datetime
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot, plot_cross_validation_metric
from prophet.utilities import regressor_coefficients

class ProphetModel:

    def __init__(self, 
                data_path: str, 
                output: str="sales",
                regressor: str="",
                location_id: int=None):

        df = pd.read_csv(data_path)

        if location_id != None:
            df = df[df['department_id'] == location_id]

        columns = ['date', output] if regressor == "" else ['date', regressor, output]

        self.df = df[columns].rename(columns={'date': 'ds', output: 'y'})
        self.df['date_index'] = self.df['ds'].copy()
        self.df['date_index'] = pd.to_datetime(self.df['date_index'])
        self.df = self.df.set_index('date_index')


        self.model = Prophet()
        self.regressor = regressor

        if regressor != "":
            self.model.add_regressor(regressor)
    
    def fit(self):
        self.model.fit(self.df)

    def test_predict(self, title_file=""):

        # function that populates the regressor's values in the future DataFrame
        def map_regressor(ds):
            try:
                date = (pd.to_datetime(ds)).date().strftime('%Y-%m-%d')

                if self.df.loc[self.df.ds == date].empty:
                    return future_temp_df.loc[future_temp_df.ds == date][f'future_{self.regressor}'].values[0]
                else:
                    return (self.df.loc[self.df.ds == date][self.regressor]).values[0]
            except Exception as ex:
                return 0


        future = self.model.make_future_dataframe(periods=20)

        if self.regressor != "":
            future_range = pd.date_range(self.split_date, periods=10, freq='D')
            future_temp_df = pd.DataFrame({ 'ds': future_range, f'future_{self.regressor}' : 0})
            future[self.regressor] = future['ds'].apply(map_regressor)

        forecast = self.model.predict(future)

        df_cv = cross_validation(self.model, initial='730 days', period='10 days', horizon = '180 days')
        perf = performance_metrics(df_cv)
        perf.to_csv(title_file)

    def save_model(self, file_name=""):

        with open(f'{file_name}.json', 'w') as fout:
            json.dump(model_to_json(self.model), fout)  # Save model

    @staticmethod
    def load_model(file_name):
        with open(file_name, 'r') as fin:
            m = model_from_json(json.load(fin))  # Load model
            return m

