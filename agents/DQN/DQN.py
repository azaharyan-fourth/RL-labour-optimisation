import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pytorch implementaion from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm

class DQN(nn.Module):

    def __init__(self, input, outputs):
        
        super(DQN, self).__init__()
        self.dense = nn.Linear(input, input*2)
        self.out = nn.Linear(input*2, outputs)

    def forward(self, input):
        x = input.to(device)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.dense(x))
        x = self.out(x)
        #x = F.softmax(x, dim=1)

        # shape = [batch_size, n_actions]
        return x