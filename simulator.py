import numpy as np
import matplotlib.pyplot as plt

class Simulator:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def play_one_episode(self, num_episode=0):
        self.env.reset()

        rewards, actions = [], []

        for index, values in enumerate(self.env.iter_train_dataset()):
            state = self.env.get_state(index+self.env.window)
            state_transformed = self.env.transform_data_for_nn(state)
            action = self.agent.act(state_transformed, num_episode)
            next_state, reward = self.env.step(action)

            rewards.append(reward)    
            actions.append(action)

            next_state_transformed = self.env.transform_data_for_nn(next_state)
            self.agent.get_memory().push(state_transformed, action, reward, next_state_transformed)

            self.agent.optimize_model()
        
        with open('rewards.txt', 'a') as fout:
            fout.write(f"Episode rewards: {sum(rewards)}\n")
        with open('actions.txt', 'a') as fout:
            fout.write(f"Episode actions: {actions}\n")

if __name__ == '__main__':
    pass