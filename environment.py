from typing import Dict
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
from forecasting_models.xgboost_model import XGBoostModel
from utils import parse_command_args, get_json_params
from dataset import Dataset
from gym import Env
import matplotlib.pyplot as plt
from spaces.evenly_spaced import EvenlySpaced
from sklearn.metrics import mean_absolute_error
from croniter import croniter
from datetime import datetime

class TSEnvironment(Env):

    def __init__(self, 
                file_path: str, 
                start_test_period: str,
                target: str, 
                labor_feature: str,
                number_actions: int,
                start_action: int,
                stop_action: int,
                target_model_params: Dict[str, float],
                labor_model_params: Dict[str, float],
                cost_feature: float,
                window: int,
                cron_expression: str):

        self.dataset = Dataset(file_path, start_test_period)
        

        self.next_dependency = self.dataset.dataset_train.pop('dependency_next')
        self.next_dependency = self.next_dependency.append(self.dataset.dataset_val.pop('dependency_next'))
        self.next_dependency = self.next_dependency.append(self.dataset.dataset_test.pop('dependency_next'))

        self.window = window
        self.action_space = EvenlySpaced(start_action, stop_action, number_actions)
        self.t = window #timestep of the series

        self.target = target
        self.labor_feature = labor_feature

        if target_model_params is not None:
            self.model_target = XGBoostModel(**target_model_params)
        else:
            self.model_target = XGBoostModel(n_estimators=100, learning_rate=0.1)

        if labor_model_params is not None:
            self.model_hours = XGBoostModel(**labor_model_params)
        else:
            self.model_hours = XGBoostModel()

        self.cost_feature = cost_feature

        #cron expression for applying dynamic changes to the forecasts
        if cron_expression:
            first_date = self.dataset.dataset_train.iloc[self.window]['date']
            self.cron_expression = cron_expression
            self.cron_iter = croniter(cron_expression, first_date)
            self.cron_iter.next(datetime)

        self.device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")

    def get_state(self, index=None, is_test=False):
        """ 
            Get the current state of the environment

            Args:
            index (int):
            is_test (bool): 
        """

        if index is None:
            index = self.t
        
        state = None

        if is_test:
            df_concatted = pd.concat([self.dataset.dataset_train.iloc[-self.window:].copy(), self.dataset.dataset_val])
            state = df_concatted.loc[index-self.window:index].copy()
        else:
            state = self.dataset.dataset_train.iloc[index-self.window:index+1].copy()

        return state

    def step(self, action, num_episode=0, is_test=False):
        """ 
            Make step in the environment and return the next state and the reward
            return (nex_state, reward)

            Args:
            action_idx (int): index of the selected action
            is_test (bool): mode of the agent
            Returns:
            next_state
            reward (float): 
            done (bool): True when it is the last element of the dataset
                        (i.e. the series)
        """

        current_state = self.get_state(is_test=is_test)

        reward = self._apply_action_get_reward(action, current_state, num_episode)
        self.t += 1

        done = self.t == len(self.dataset.dataset_train)
        next_state = self.get_state(is_test=is_test)

        if done:
            next_state = None

        return next_state, reward, done

    def reset(self, index=None):
        '''
        Resets the environment, i.e. moves the counter t to the 
        index with offset of size `window`

        Args:
        index (int): 
        '''
        if index != None:
            self.t = index + self.window
        else:   
            self.t = self.window

    def iter_dataset(self, train: bool = True):
        '''
        Wrapper for iteration of a dataset

            Args:
                train (bool): Shows if we should iterate the train dataset

        '''
        if train:
            dataset_to_iterate = self.dataset.dataset_train
        else:
            dataset_to_iterate = self.dataset.dataset_val
            dataset_to_iterate = pd.concat([self.dataset.dataset_train.iloc[-self.window+1:], dataset_to_iterate], axis=0)
            self.t = dataset_to_iterate.iloc[0].name

        for value in dataset_to_iterate[self.window:].iterrows():
            yield value
            
    def transform_data_for_nn(self, df, mode='train') -> torch.Tensor:
        '''
        Drop unnecessary columns for the NN and transform the DataFrame to Tensor 

            Args:
                df (pd.DataFrame): DataFrame of the data

            Returns:
                data (torch.Tensor)
        '''
        if mode == 'train':
            df['forecast_dependency'] = self.precalculate_forecasts_dependency
        elif mode == 'val':
            df['forecast_dependency'] = self.precalculate_forecasts_dependency

        df['dependency_next'] = self.next_dependency

        if 'date' in df.columns:
            df.drop('date', axis=1, inplace=True)

        df.drop(self.labor_feature, axis=1, inplace=True)

        #drop last row which is the current day -> to prevent data leak
        df.drop(df.tail(1).index,inplace=True)

            
        data = torch.tensor(df.values, dtype=torch.float, device=self.device)

        #normalize
        data = torch.nn.functional.normalize(data, p=2.0, dim = 1)
        return data

    def get_predicted_sales(self, 
                    current_state: pd.DataFrame, 
                    num_episode: int = 0,
                    revert: bool = False) -> float:
        '''
            Get forecasted sales for the current state

            Args:
                current_state (DataFrame): the current state of the environment, which
                            consists of the current day + the context window

            Returns:
                prediction(float): 
        '''
        test = self.model_target.create_features(current_state)
        test.drop(self.target, axis=1, inplace=True)
        prediction = self.model_target.test(test.tail(1))[0]
        prediction = np.maximum(0, round(prediction))
        if num_episode > 15 \
            and hasattr(self, 'cron_iter') \
            and self._should_apply_dynamic_change(current_state.tail(1)['date'], revert):
            prediction = 2*prediction

        return prediction

    def get_predicted_labor(self, 
                            current_state: pd.DataFrame, 
                            forecast_target=None) -> float:
        '''
            Get forecast of the feature for the current state

            Args:
                current_state(DataFrame):
                forecast_target (float): if it is not None we should use this as the target
                                for the current day

            Returns:
                prediction(float): prediction of the labor rounded to 1 decimal place
        '''
        state = current_state.copy()
        if forecast_target != None:
            state.loc[-1, self.target] = forecast_target

        test = self.model_hours.create_features(state)
        test.drop(self.labor_feature, axis=1, inplace=True)
        prediction = self.model_hours.test(test.tail(1))[0]
        return np.float64(round(prediction,1))

    def train_environment_and_evaluate(self):
        """ 
            Train helper forecasting models of the environment
        """

        # Create features and train XGBoost models
        X_train, y_train = self.model_target.create_features(self.dataset.dataset_train, label=self.target)
        X_test, y_test = self.model_target.create_features(self.dataset.dataset_val, label=self.target)

        ms = self.model_target.train(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

        X_train_hours, y_train_hours = self.model_target.create_features(self.dataset.dataset_train, label=self.labor_feature)
        X_test_hours, y_test_hours = self.model_target.create_features(self.dataset.dataset_val, label=self.labor_feature)

        m = self.model_hours.train(X_train_hours, y_train_hours, 
                            eval_set=[(X_train_hours, y_train_hours), (X_test_hours, y_test_hours)])


        #Precalculate forecasted dependency and add to the DataFrame
        self._precalculate_forecasted_dependency()

        # remove 1 for the date column
        self.n_observation_space = len(X_train.columns)+2

        #Evaluate the XGBoost models and output the results
        wape_train = self.predict_and_evaluate(X_train, y_train)
        wape_test = self.predict_and_evaluate(X_test, y_test)

        rmse_labor_train = self.predict_and_evaluate(X_train_hours,
                                                    y_train_hours,
                                                    is_target=False,
                                                    metric='rmse')
        rmse_labor_test = self.predict_and_evaluate(X_test_hours,
                                                    y_test_hours,
                                                    is_target=False, 
                                                    metric='rmse')

        print(f"Target train WAPE: {wape_train}")                    
        print(f"Target test WAPE: {wape_test}")
        print(f"Labor train MAE: {rmse_labor_train}")
        print(f"Labor test MAE: {rmse_labor_test}")
        

    def predict_and_evaluate(self, 
                            X: pd.DataFrame,
                            y: pd.Series, 
                            is_target=True, metric='wape'):
        '''
        Evaluate the performance of the trained XGBoost models

        Args:
        X (DataFrame): data frame to get predictions for
        y (Series): true labels of the dataset
        is_target (bool): used to choose which XGBoost model to use
        metric (str): the evaluation metric to be used

        Returns:
        score (float): the score according to the specified metric 
        '''
        if is_target:
            pred = self.model_target.test(X)
        else:
            pred = self.model_hours.test(X)

        pred = np.round(pred)
        if metric == 'wape':
            score = (abs(y - pred)).sum() / y.sum()
        elif metric == 'rmse':
            score = mean_absolute_error(pred, y)
        return score

    def render(self):

        # Plot train predictions
        train_period_to_evaluate = self.dataset.dataset_train[100:200]
        rows = []
        for _, value in train_period_to_evaluate.iterrows():
            row = { }
            for action in self.action_space.values:
                X_test = value.copy()
                X_test.loc[self.labor_feature] += action
                prediction = self.get_predicted_sales(X_test.to_frame().transpose())
                row[action] = prediction

            rows.append(row)

        _ = plt.figure(1)
        plt.plot(range(len(train_period_to_evaluate[self.target])),
                train_period_to_evaluate[self.target],
                marker='o',
                label='actual')
        
        for action in self.action_space.values:
            plt.plot([x[action] for x in rows], 
                    marker='o', 
                    label=f"predictions for {action}")

        plt.title('Train period predictions')
        plt.xlabel('Time preiods')
        plt.ylabel('Sales')
        plt.legend()
        plt.show(block=False)

        # Plot validation predictions
        val_period_to_evaluate = self.dataset.dataset_val
        rows = []
        for _, value in val_period_to_evaluate.iterrows():
            #row = { 'date': value.loc['date'].replace(hour=value.loc['StartHour']) }
            row = { }
            for action in self.action_space.values:
                X_test = value.copy()
                X_test.loc[self.labor_feature] += action
                prediction = self.get_predicted_sales(X_test.to_frame().transpose())
                row[action] = prediction

            rows.append(row)

        _ = plt.figure(2)
        plt.plot(range(len(val_period_to_evaluate['date'])), #  + pd.to_timedelta(val_period_to_evaluate['StartHour'], unit='h')
                val_period_to_evaluate[self.target],
                marker='o',
                label='actual')
        
        for action in self.action_space.values:
            plt.plot([x[action] for x in rows], 
                    marker='o', 
                    label=f"predictions for {action}")

        plt.title('Test period predictions')
        plt.xlabel('Time preiods')
        plt.ylabel('Sales')
        plt.legend()
        plt.xticks(rotation=90)
        plt.show(block=False)


        # Plot dependency predictions
        # Plot train predictions
        preds = []
        for _, value in train_period_to_evaluate.iterrows():
            X_test = value.copy()
            prediction = self.get_predicted_labor(X_test.to_frame().transpose())

            preds.append(prediction)

        _ = plt.figure(3)
        plt.plot(range(len(train_period_to_evaluate[self.labor_feature])),
                train_period_to_evaluate[self.labor_feature],
                marker='o',
                label='actual')
        
        plt.plot(range(len(preds)),
                preds,
                marker='o',
                label='pred')

        plt.title('Train period labour predictions')
        plt.xlabel('Time preiods')
        plt.ylabel('HoursWorked')
        plt.legend()
        plt.show(block=True)

    def reset_cron_iter(self):
        self.cron_iter = croniter(self.cron_expression, self.dataset.dataset_train.iloc[self.window]['date'])
        self.cron_iter.next(datetime)

    def _apply_action_get_reward(self, action, state, num_episode):
        """ Apply action and pass the resulted reward

            Args:
            action_idx (int): index of the selected action in the action space
            state (pd.DataFrame): current state
        """

        action_value = torch.tensor(self.action_space[action])

        forecast_no_action = self.get_predicted_sales(state, num_episode, revert=True) # forecast target from actual dependency
        #forecast_dependency_no_action = self.get_predicted_labor(state, forecast_target=forecast_no_action)

        actual_dependency = state.loc[self.t, self.labor_feature]
        #apply action
        state.loc[self.t, self.labor_feature] = actual_dependency+action_value.cpu().numpy()
        
        forecast_action = self.get_predicted_sales(state, num_episode)
        forecast_action = self._fix_forecasts_minmax(forecast_action, forecast_no_action, action_value)

        forecast_profit = forecast_action - state.loc[self.t][self.labor_feature]*self.cost_feature

        actual_profit = forecast_no_action - actual_dependency*self.cost_feature

        #(forecast target with action-dependency with action) - 
        #(forecast target with 0 - dependency with 0)
        reward = forecast_profit - actual_profit

        return round(reward, 2)


    def _fix_forecasts_minmax(self, forecast_action, forecast_noaction, action):
        """ Fix forecasts, s.t. those for decreased labour do not exceed the one for
            no action and so on.

            Args:
            forecast_action (decimal): forecasted sales with applied action
            forecast_noaction (decimal): forecasted sales for no applied action (0)
            action (int): value of the selected action (not index)
        """

        if action > 0:
            forecast_action = max(forecast_action, forecast_noaction)
        elif action < 0:
            forecast_action = min(forecast_action, forecast_noaction)

        return forecast_action

    def _precalculate_forecasted_dependency(self) -> None:
            
        self.precalculate_forecasts_dependency = pd.Series()

        for df in [self.dataset.dataset_train, self.dataset.dataset_val, self.dataset.dataset_test]:
            for i, value in tqdm(df.iterrows(), total=df.shape[0]):
                forecast_dependency = self.get_predicted_labor(value.to_frame().transpose())
                df.loc[i, 'forecast_dependency'] = forecast_dependency

            self.precalculate_forecasts_dependency = pd.concat([self.precalculate_forecasts_dependency, df.pop('forecast_dependency')])


    def _should_apply_dynamic_change(self, date, revert):
        next_tick = self.cron_iter.get_current(datetime)
        if next_tick == pd.Timestamp(date.values[0]):
            if not revert:
                self.cron_iter.next(datetime)
            return True

        return False

if __name__ == '__main__':
    args = parse_command_args()
    
    target_params = get_json_params(args.target_params)
    labor_params = get_json_params(args.labor_params)

    env = TSEnvironment(args.data_path,
                        args.start_test_period,
                        args.target, 
                        args.dependency_feature,
                        number_actions=int(args.number_actions),
                        start_action=float(args.start_action),
                        stop_action=float(args.stop_action),
                        target_model_params=target_params,
                        labor_model_params=labor_params,
                        cost_feature=args.cost_feature,
                        window=args.window_size)

    env.train_environment_and_evaluate()
    env.render()
