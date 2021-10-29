import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from agents.DQN.DQN import DQN
from agents.DQN.replay_memory import ReplayMemory, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, epsilon=0.9):
        self.action_space = [-1,0,1]
        self.epsilon = epsilon
        
        #init policy and target networks, optimizer and replay memory
        self.policy_net = DQN(3*31, len(self.action_space)).to(device)
        self.target_net = DQN(3*31, len(self.action_space)).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(1000)

        #constants
        self.BATCH_SIZE = 8
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200

    def act(self, state, steps_num=0):
        if isinstance(state, pd.DataFrame):
            state = state.to_numpy()
            state = torch.from_numpy(state)

        sample = random.random()

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_num / self.EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                state = state.unsqueeze(0) #add batch_size dimension
                output = self.policy_net(state.float())
                action_idx = torch.argmax(output)
                return torch.tensor(action_idx, device=device, dtype=torch.long)

        else:
            action_idx = random.randrange(len(self.action_space))
            return torch.tensor(action_idx, device=device, dtype=torch.long)
    
    def optimize_model(self):

        if len(self.memory) < self.BATCH_SIZE:
            return
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                            device=device, dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in map(lambda st: torch.tensor(st).unsqueeze(0), batch.next_state)
                                            if s is not None])

        state_batch = torch.cat(tuple(map(lambda s: torch.tensor(s).unsqueeze(0), batch.state)))
        action_batch = torch.vstack(batch.action)
        reward_batch = torch.tensor(batch.reward, device=device)

        state_action_values = self.policy_net(state_batch.float())
        state_action_values = state_action_values.gather(0, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states.float()).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_memory(self):
        return self.memory
