from os import device_encoding
import random
from numpy import apply_along_axis
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from zmq import device
from agents.DQN.DQN import DQN
from agents.DQN.replay_memory import ReplayMemory, Transition
from agents.DQN.per import PrioritizedExperienceReplay
from policies.eps_decay_greedy import EpsDecayGreedyQPolicy

class DQN_Agent:
    def __init__(self, env, epsilon=0.9):
        self.env = env
        self.policy = EpsDecayGreedyQPolicy(eps_start=1.0, 
                                            eps_end=0.05,
                                            eps_decay=5500)
        self.epsilon = epsilon

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #init policy and target networks, optimizer and replay memory
        self.policy_net = DQN(self.env.n_observation_space*self.env.window, self.env.action_space.n).to(self.device)
        self.target_net = DQN(self.env.n_observation_space*self.env.window, self.env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        #PER params
        self.beta = 0.6
        self.PRIOR_EPS = 1e-6

        self.optimizer = optim.Adam(self.policy_net.parameters())
        #self.memory = ReplayMemory(15000)
        self.memory = PrioritizedExperienceReplay(40000)

        #constants
        self.BATCH_SIZE = 32
        self.GAMMA = 1
        self.TARGET_UPDATE = 200

        self.update_cnt = 0

        random.seed(10)

    def train(self, num_episodes=100):
        """
            Args:
        """
        for i in range(num_episodes):
            self.env.reset()

            rewards, actions, losses = [], [], []
            episode_duration = len(self.env.dataset.dataset_train)-self.env.window

            for index, values in enumerate(self.env.iter_dataset(train=True)):
                state = self.env.get_state(index+self.env.window)
                state_transformed = self.env.transform_data_for_nn(state, mode='train')
                action = self.act(state_transformed, steps_num=index + i*episode_duration)
                next_state, reward, done = self.env.step(action, num_episode=i)

                rewards.append(reward)    
                actions.append(action)

                if isinstance(next_state, pd.DataFrame):
                    next_state = self.env.transform_data_for_nn(next_state, mode='train')

                self.get_memory().push(state_transformed, action, reward, next_state)
                loss = self.optimize_model(i, index)
                losses.append(loss)

                self.update_cnt += 1

            #self.env.reset_cron_iter()
        
            with open('rewards.txt', 'a') as fout:
                fout.write(f"Episode rewards: {sum(rewards)}\n")
            with open('actions.txt', 'a') as fout:
                fout.write(f"Episode actions: {actions}\n")
            with open('losses.txt', 'a') as fout:
                fout.write(f"{sum(losses)}\n")

            torch.save(self.policy_net, './dqn_model.pt')

            if i % 5 == 0:
                self.test()

    def test(self):
        """Test the agent."""
        self.policy_net.eval()

        self.env.reset(index=len(self.env.dataset.dataset_train))
        done = False
        rewards, actions = [], []
        
        for index, values in enumerate(self.env.iter_dataset(train=False), 
                                        start=len(self.env.dataset.dataset_train)):
                state = self.env.get_state(index, is_test=True)
                state_transformed = self.env.transform_data_for_nn(state, mode='val')
                action = self.act(state_transformed, is_test=True)
                next_state, reward, done = self.env.step(action, is_test=True)

                rewards.append(reward)    
                actions.append(action)

                with open('test.txt', 'a') as fout:
                    fout.write(f"{values[1]['date']}: {action} {reward}\n")
        with open('test.txt', 'a') as fout:
                    fout.write(f"-------------\n")
        with open('rewards_test.txt', 'a') as fout:
            fout.write(f"Episode rewards: {sum(rewards)}\n")
        with open('actions_test.txt', 'a') as fout:
            fout.write(f"Episode actions: {actions}\n")


    def act(self, state, is_test=False, steps_num=0):
        #return torch.tensor(1, device=self.device)

        if isinstance(state, pd.DataFrame):
            state = state.to_numpy()
            state = torch.from_numpy(state)

        # Get Q-values
        state = state.unsqueeze(0) #add batch_size dimension
        self.policy_net.eval()
        output = self.policy_net(state.float()).squeeze(0)

        # Get selected action using policy
        action = self.policy.select_action(output, steps_num, is_test)

        return action

    
    def optimize_model(self, num_episode=0, num_step=0):

        if len(self.memory) < self.BATCH_SIZE:
            return 0

        fraction = min(num_step / 420, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)
        
        self.policy_net.train()

        #transitions = self.memory.sample(self.BATCH_SIZE)
        samples = self.memory.sample(self.BATCH_SIZE)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        batch = Transition(*zip(*samples['batch']))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                            device=self.device, dtype=torch.bool)
        
        non_final_next_states = torch.cat([s.clone().detach().unsqueeze(0) for s in batch.next_state
                                            if s is not None])

        state_batch = torch.cat(tuple(map(lambda s: s.clone().detach().unsqueeze(0), batch.state)))
        action_batch = torch.vstack(batch.action)
        reward_batch = torch.tensor(batch.reward, device=self.device)

        state_action_values = self.policy_net(state_batch.float())
        state_action_values = state_action_values.gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)

        # DQN
        #next_state_values[non_final_mask] = self.target_net(non_final_next_states.float()).max(1)[0].detach()

        # Double DQN
        selected_action = self.policy_net(non_final_next_states.float()).argmax(dim=1, keepdim=True)
        next_state_targets = self.target_net(non_final_next_states.float())
        next_state_values[non_final_mask] = next_state_targets.gather(1, selected_action).squeeze(1)

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss(reduction='none')
        elementwise_loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = torch.mean(elementwise_loss * weights)

        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.PRIOR_EPS
        self.memory.update_priorities(indices, new_priorities)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if num_step % 10 == 0:
            print(f"Loss on timestep {num_step}, episode {num_episode}: {loss.item()}")
        
        if self.update_cnt % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def get_memory(self):
        return self.memory

    def reduce_variance(self, reward_batch):
        a = torch.zeros(self.BATCH_SIZE)
        high_var_elements = abs(reward_batch-torch.mean(reward_batch)) < 2*torch.std(reward_batch)
        reduced = reward_batch.where(high_var_elements, torch.mean(reward_batch))
        return reduced