# RL Time Series

RL Time Series is an open source Python library that lets you build environment for Reinforcement learning using Time Series data and forecasting. It inherits from the standard OpenAI [Gym](https://github.com/openai/gym) Environment abstraction which enables the user to easily test different RL agents.

The main idea of the environment is to find the best changes in an important feature that would result in maximizing the series values. This is done using two pillars - XGBoost models using for forecasting. One for the 'target' series and one for the 'feature'.

# Prerequisites
1. Numpy
2. Pandas
3. Gym

