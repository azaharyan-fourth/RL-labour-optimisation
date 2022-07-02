from environment import Environment
from dataset import Dataset
from forecasting_models.xgboost_model import XGBoostModel
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import params

def compute_prophet():
    location_id = 14922
    dataset = Dataset('./data/all_sales_shifts_14922.csv', location_id)
    env = Environment(dataset)
    df = pd.DataFrame()
    for i in env.iter_dataset(train=True):
        row = {'date': i[1].date }
        for action in env.action_space:
            current_state = env.get_state(i[0])
            action_value = torch.tensor(action)

            current_state.loc[i[0], 'HoursWorked'] += action_value.numpy()
            current_state.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)

            forecast = env.model_sales.predict(current_state).iloc[-1]['yhat']

            row[action] = forecast
        df = df.append(row, ignore_index=True)


    df.to_csv('./data/precomputed_forecasts.csv')

def compute_xgboost():

    sales_params = params.SALES_DAILY_PARAMS

    dataset = pd.read_csv('./data/all_sales_shifts_14922.csv')
    X_train = dataset[dataset['date'] <  '2019-12-01'][['HoursWorked_manager', 'day_of_week',
                             'day_of_month', 'day_of_year', 'year']].to_numpy()
    X_test = dataset[(dataset['date'] >=  '2019-12-01') & (dataset['date'] < '2020-03-01')][['HoursWorked_manager', 'day_of_week',
                             'day_of_month', 'day_of_year', 'year']].to_numpy()

    y_train = dataset[dataset['date'] <  '2019-12-01']['sales'].to_numpy()
    y_test = dataset[(dataset['date'] >= '2019-12-01') & (dataset['date'] < '2020-03-01')]['sales'].to_numpy()

    df = pd.DataFrame()
    model = XGBoostModel(n_estimators=sales_params['n_estimators'],
                        learning_rate=sales_params['learning_rate'],
                        max_depth=sales_params['max_depth'],
                        #reg_alpha=0.05,
                        gamma=sales_params['gamma'],
                        colsample_bytree=sales_params['colsample_bytree'],
                        #min_child_weight=1.0,
                        subsample=sales_params['subsample'],
                        reg_lambda=sales_params['reg_lambda']
                        )
    model.train(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

    X = dataset[dataset['date'] < '2020-03-01'][['HoursWorked_manager', 'day_of_week',
                             'day_of_month', 'day_of_year', 'year']]
    for i in X.iterrows():
        row = { 'index': i[0] }
        row['date'] = dataset.loc[i[0], 'date']
        for action in [-5, 0, 5]:
            current_state = i[1].copy()

            current_state.loc['HoursWorked_manager'] += action

            prediction = model.test(np.expand_dims(current_state.to_numpy(), axis=0))

            row[action] = prediction[0]

        row[-5] = min(row[-5], row[0])
        row[5] = max(row[0], row[5])

        df = df.append(row, ignore_index=True)

    train_accuracy = mean_squared_error(dataset[dataset['date'] < '2019-12-01']['sales'], 
                        df[df['date'] < '2019-12-01'][0],
                        squared=False)

    test_accuracy = mean_squared_error(dataset[(dataset['date'] >= '2019-12-01') & (dataset['date'] < '2020-03-01')]['sales'], 
                        df[(df['date'] >= '2019-12-01') & (df['date'] < '2020-03-01')][0],
                        squared=False)
                        
    print(f"Train set RMSE: {train_accuracy}")
    print(f"Test set RMSE: {test_accuracy}")
    
    df.to_csv('./data/precomputed_forecasts_xgb_5_manager.csv')

def compute_xgboost_labour():
    dataset = pd.read_csv('./data/all_sales_shifts_14922.csv')

    X_train = dataset[dataset['date'] < '2019-12-01'][['sales','day_of_week',
                                'day_of_month','day_of_year', 'year']].to_numpy()
    X_test = dataset[(dataset['date'] >= '2019-12-01') & (dataset['date'] < '2020-03-01')][['sales', 'day_of_week',
                                'day_of_month','day_of_year', 'year']].to_numpy()

    y_train = dataset[dataset['date'] < '2019-12-01']['HoursWorked'].to_numpy()
    y_test = dataset[(dataset['date'] >= '2019-12-01') & (dataset['date'] < '2020-03-01')]['HoursWorked'].to_numpy()

    df = pd.DataFrame()
    model = XGBoostModel(n_estimators=160, 
                        max_depth=10, 
                        learning_rate=0.02,
                        gamma=7, 
                        #reg_alpha=0.75,
                        subsample=0.85,
                        colsample_bytree=0.58,
                        reg_lambda=9.97
                        #min_child_weight=18.6
                        )
    model.train(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
    
    X = dataset[dataset['date'] < '2020-03-01'][['sales', 'day_of_week',
                                'day_of_month','day_of_year', 'year']]

    for i in X.iterrows():
        row = { 'index': i[0] }
        row['date'] = dataset.loc[i[0], 'date']
        for action in [-5, 0, 5]:
            current_state = i[1].copy()

            prediction = model.test(np.expand_dims(current_state.to_numpy(), axis=0))

            row['HoursWorked_forecast'] = prediction[0]
        df = df.append(row, ignore_index=True)

    train_accuracy = mean_squared_error(dataset[dataset['date'] < '2019-12-01']['HoursWorked'], 
                        df[df['date'] < '2019-12-01']['HoursWorked_forecast'],
                        squared=False)

    test_accuracy = mean_squared_error(dataset[(dataset['date'] >= '2019-12-01') & (dataset['date'] < '2020-03-01')]['HoursWorked'], 
                        df[(df['date'] >= '2019-12-01') & (df['date'] < '2020-03-01')]['HoursWorked_forecast'],
                        squared=False)
                        
    print(f"Train set RMSE: {train_accuracy}")
    print(f"Test set RMSE: {test_accuracy}")
    #df.to_csv('./data/precomputed_forecasts_xgb_labour_hptune.csv')

def compute_xgboost_hourly():
    sales_params = params.SALES_HOURLY_PARAMS

    dataset = pd.read_csv('./data/locations/sales_labour_20182019_OctJun_15183.csv')
    X_train = dataset[dataset['date'] <  '2019-05-01'][['HoursWorked_actual', 'day_of_week', 'StartHour',
                             'year','day_of_year', 'day_of_month', 'month']].to_numpy()
    X_test = dataset[(dataset['date'] >=  '2019-05-01') & (dataset['date'] <= '2019-06-02')][['HoursWorked_actual', 'day_of_week',
                             'year', 'day_of_year', 'StartHour', 'day_of_month', 'month']].to_numpy()

    y_train = dataset[dataset['date'] <  '2019-05-01']['sales'].to_numpy()
    y_test = dataset[(dataset['date'] >= '2019-05-01') & (dataset['date'] < '2019-06-02')]['sales'].to_numpy()

    df = pd.DataFrame()
    model = XGBoostModel(n_estimators=sales_params['n_estimators'],
                        learning_rate=sales_params['learning_rate'],
                        max_depth=sales_params['max_depth'],
                        #reg_alpha=sales_params['reg_alpha'],
                        #gamma=sales_params['gamma'],
                        colsample_bytree=sales_params['colsample_bytree'],
                        #min_child_weight=sales_params['min_child_weight'])
                        subsample=sales_params['subsample'],
                        reg_lambda=sales_params['reg_lambda'])
    model.train(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

    X = dataset[['HoursWorked_actual', 'day_of_week', 'StartHour', 'year', 'day_of_year',
                             'day_of_month', 'month']]
    for index, value in X.iterrows():
        row = { 'index': index }
        row['date'] = dataset.loc[index, 'date']
        row['StartHour'] = dataset.loc[index, 'StartHour']
        for action in [-1, 0, 1]:
            current_state = value.copy()

            current_state.loc['HoursWorked_actual'] += action
            #current_state.loc['HoursWorked_manager'] += action

            prediction = model.test(np.expand_dims(current_state.to_numpy(), axis=0))

            row[action] = prediction[0]

        current_state = value.copy()
        current_state.loc['HoursWorked_actual'] = dataset.loc[index, 'HoursWorked_manager']
        row['forecast_manager_labour'] = model.test(np.expand_dims(current_state.to_numpy(), axis=0))[0]

        row[-1] = min(row[-1], row[0])
        row[1] = max(row[0], row[1])

        df = df.append(row, ignore_index=True)

    train_accuracy = mean_squared_error(dataset[dataset['date'] < '2019-05-01']['sales'], 
                        df[df['date'] < '2019-05-01'][0],
                        squared=False)

    test_accuracy = mean_squared_error(dataset[(dataset['date'] >= '2019-05-01') & (dataset['date'] < '2019-06-02')]['sales'], 
                        df[(df['date'] >= '2019-05-01') & (df['date'] < '2019-06-02')][0],
                        squared=False)

    train_mape = mean_absolute_percentage_error(dataset[dataset['date'] < '2019-05-01']['sales'], 
                        df[df['date'] < '2019-05-01'][0])

    test_mape = mean_absolute_percentage_error(dataset[(dataset['date'] >= '2019-05-01') & (dataset['date'] < '2019-06-02')]['sales'], 
                        df[(df['date'] >= '2019-05-01') & (df['date'] < '2019-06-02')][0])
                        
    print(f"Train set RMSE: {train_accuracy}")
    print(f"Test set RMSE: {test_accuracy}")
    
    df.to_csv('./data/precomputed_forecasts_xgb_hourly.csv')

def compute_xgboost_labour_hourly():
    sales_params = params.LABOR_HOURLY_PARAMS

    dataset = pd.read_csv('./data/locations/sales_labour_20182019_OctJun_15183.csv')
    X_train = dataset[dataset['date'] <  '2019-05-01'][['sales', 'day_of_week', 'StartHour',
                             'year', 'day_of_month','day_of_year', 'month']].to_numpy()
    X_test = dataset[(dataset['date'] >=  '2019-05-01') & (dataset['date'] <= '2019-06-02')][['sales', 'day_of_week',
                             'year', 'StartHour', 'day_of_month', 'day_of_year', 'month']].to_numpy()

    y_train = dataset[dataset['date'] <  '2019-05-01']['HoursWorked_actual'].to_numpy()
    y_test = dataset[(dataset['date'] >= '2019-05-01') & (dataset['date'] < '2019-06-02')]['HoursWorked_actual'].to_numpy()

    df = pd.DataFrame()
    model = XGBoostModel(n_estimators=sales_params['n_estimators'],
                        learning_rate=sales_params['learning_rate'],
                        max_depth=sales_params['max_depth'],
                        #reg_alpha=sales_params['reg_alpha'],
                        #gamma=sales_params['gamma'],
                        colsample_bytree=sales_params['colsample_bytree'],
                        #min_child_weight=sales_params['min_child_weight'])
                        subsample=sales_params['subsample'],
                        reg_lambda=sales_params['reg_lambda'])
    model.train(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

    X = dataset[['sales', 'day_of_week', 'StartHour', 'year',
                            'day_of_month','day_of_year', 'month']]
    for i in X.iterrows():
        row = { 'index': i[0] }
        row['date'] = dataset.loc[i[0], 'date']
        row['StartHour'] = dataset.loc[i[0], 'StartHour']

        current_state = i[1].copy()

        prediction = model.test(np.expand_dims(current_state.to_numpy(), axis=0))

        if prediction[0] < 1:
            row['HoursWorked_forecast'] = round(prediction[0]*2)/2
        else:
            row['HoursWorked_forecast'] = round(prediction[0])

        df = df.append(row, ignore_index=True)

    train_accuracy = mean_squared_error(dataset[dataset['date'] < '2019-05-01']['HoursWorked_actual'], 
                        df[df['date'] < '2019-05-01']['HoursWorked_forecast'],
                        squared=False)

    test_accuracy = mean_squared_error(dataset[(dataset['date'] >= '2019-05-01') & (dataset['date'] < '2019-06-02')]['HoursWorked_actual'], 
                        df[(df['date'] >= '2019-05-01') & (df['date'] < '2019-06-02')]['HoursWorked_forecast'],
                        squared=False)

    print(f"Train set RMSE: {train_accuracy}")
    print(f"Test set RMSE: {test_accuracy}")
    
    df.to_csv('./data/precomputed_forecasts_labour_xgb_hourly.csv')

if __name__ == '__main__':
    compute_xgboost()
    #compute_xgboost_labour()
    #compute_xgboost_hourly()
    #compute_xgboost_labour_hourly()