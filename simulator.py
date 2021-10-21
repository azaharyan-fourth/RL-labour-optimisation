import numpy as np

class Simulator:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def play_one_episode(self, training=True):
        state_history = []
        action_history = []
        reward_history = []

        for index, values in enumerate(self.env.iter_train_dataset()):
            state = self.env.get_state(index)
            action = self.agent.act(state)
            state, reward = self.env.step(action)

            state_history.append(state)
            action_history.append(action)
            reward_history.append(reward)

            #train agent (fit)


if __name__ == '__main__':
    pass