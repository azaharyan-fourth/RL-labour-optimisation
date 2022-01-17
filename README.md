# RL Time Series

RL Time Series is an open source Python library that lets you build environment for Reinforcement learning using Time Series data and forecasting. It inherits from the standard OpenAI [Gym](https://github.com/openai/gym) Environment abstraction which enables the user to easily test different RL agents.

The main idea of the environment is to find the best changes in an important feature that would result in maximizing the series values. This is done using two pillars - XGBoost models used for forecasting. One for the 'target' series and one for the 'feature'.

## Prerequisites
1. Numpy
2. Pandas
3. Gym
4. XGBoost
5. scikit-learn

## API
Creating an instance of the environment and interacting with it is fairly simple. Here is an example:
```python
from environment import TSEnvironment
from utils import parse_command_args, get_json_params

target_params = get_json_params(args.target_params)
labor_params = get_json_params(args.labor_params)

env = TSEnvironment(args.data_path, args.start_test_period,
                        args.target, 
                        args.labor_feature,
                        number_actions=int(args.number_actions),
                        start_action=float(args.start_action),
                        stop_action=float(args.stop_action),
                        target_model_params=target_params,
                        labor_model_params=labor_params,
                        cost_labor=int(args.cost_labor),
                        window=int(args.window_size)
                        )
env.train_environment_and_evaluate()
```
The `train_environment_and_evaluate` method must always be called after instantiating to train the underlying XGBoost models with the passed parameters and dataset. Below we show comparison between an agent interacting with a trained environment versus untrained.

## Training environment

## Parameters
The TSEnvironment is highly parameterizable. Below, we describe the main parameters of the API.
* data_path - the path to the CSV file with the time series data. The CSV should contain only features that would be used for training. There are two mandatory columns - 'index' and 'date'.
* number_actions - the number of actions in the action space. Currently, the action space is defined by the EvenlySpaced class that implements the OpenAI Space abstraction. IT created a space with evenly spaced numbers in a specified interval.
* start_action - defines the start of the action space interval
* stop_action - defines the end of the action space interval
* test_period - the start date of the test period
* target - the name of the target feature in the data
