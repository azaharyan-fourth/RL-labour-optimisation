import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from agents.DQN.DQN import DQN
from agents.DQN.replay_memory import ReplayMemory, Transition
from torch_standard_scaler import TorchStandardScaler
from agents.DQN.per import PrioritizedExperienceReplay

class DQN_Agent:
    def __init__(self, env, epsilon=0.9):
        self.env = env
        self.action_space = [-5,0,5]
        self.epsilon = epsilon

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #init policy and target networks, optimizer and replay memory
        self.policy_net = DQN(6*31, len(self.action_space)).to(self.device)
        self.target_net = DQN(6*31, len(self.action_space)).to(self.device)
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
        self.EPS_START = 1.0
        self.EPS_END = 0.05
        self.EPS_DECAY = 1500
        self.TARGET_UPDATE = 200

        self.update_cnt = 0

        random.seed(10)

    def train(self, num_episodes=100):

        for i in range(num_episodes):
            self.env.reset()

            rewards, actions, losses = [], [], []
            episode_duration = len(self.env.dataset.dataset_train)-self.env.window

            start_eps = self.get_eps_threshold(i*episode_duration)

            for index, values in enumerate(self.env.iter_train_dataset()):
                state = self.env.get_state(index+self.env.window)
                state_transformed = self.env.transform_data_for_nn(state)
                action = self.act(state_transformed, steps_num=index + i*episode_duration)
                next_state, reward, done = self.env.step(action)

                rewards.append(reward)    
                actions.append(action)

                if isinstance(next_state, pd.DataFrame):
                    next_state = self.env.transform_data_for_nn(next_state)

                self.get_memory().push(state_transformed, action, reward, next_state)
                loss = self.optimize_model(i, index)
                losses.append(loss)

                self.update_cnt += 1
        
            with open('rewards.txt', 'a') as fout:
                fout.write(f"Episode rewards: {sum(rewards)}\n")
            with open('actions.txt', 'a') as fout:
                fout.write(f"Episode actions: {actions}\n")
            with open('epsilons.txt', 'a') as fout:
                fout.write(f"{start_eps}\n")
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
        
        for index, values in enumerate(self.env.iter_test_dataset(), 
                                        start=len(self.env.dataset.dataset_train)):
                state = self.env.get_state(index+self.env.window, is_test=True)
                state_transformed = self.env.transform_data_for_nn(state)
                action = self.act(state_transformed, is_test=True)
                next_state, reward, done = self.env.step(action, is_test=True)

                rewards.append(reward)    
                actions.append(action)
        
        with open('rewards_test.txt', 'a') as fout:
            fout.write(f"Episode rewards: {sum(rewards)}\n")
        with open('actions_test.txt', 'a') as fout:
            fout.write(f"Episode actions: {actions}\n")


    def act(self, state, is_test=False, steps_num=0):
        #return torch.tensor(2, device=self.device, dtype=torch.long)

        if isinstance(state, pd.DataFrame):
            state = state.to_numpy()
            state = torch.from_numpy(state)

        sample = random.random()

        eps_threshold = self.get_eps_threshold(steps_num)

        if sample > eps_threshold or is_test:
            with torch.no_grad():
                state = state.unsqueeze(0) #add batch_size dimension
                self.policy_net.eval()
                output = self.policy_net(state.float())
                action_idx = torch.argmax(output)
                return action_idx.clone().detach()
                #return torch.tensor(action_idx, device=device, dtype=torch.long)

        else:
            action_idx = random.randint(0, len(self.action_space) - 1)
            return torch.tensor(action_idx, device=self.device, dtype=torch.long)
    
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

    def get_eps_threshold(self, steps_num=0):
        return self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_num / self.EPS_DECAY)

    def reduce_variance(self, reward_batch):
        a = torch.zeros(self.BATCH_SIZE)
        high_var_elements = abs(reward_batch-torch.mean(reward_batch)) < 2*torch.std(reward_batch)
        reduced = reward_batch.where(high_var_elements, torch.mean(reward_batch))
        return reduced