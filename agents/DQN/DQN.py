import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pytorch implementaion from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm

class DQN(nn.Module):

    def __init__(self, input, outputs, drop_prob=0.3):
        
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input, 128), 
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, outputs)
        )


    def forward(self, input):
        input = input.to(device)
        input = input.reshape(input.size(0), -1)
        x = self.layers(input)
        #x = F.softmax(x, dim=1)

        # shape = [batch_size, n_actions]
        return x