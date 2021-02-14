import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class UserNet(nn.Module):
    def __init__(self, embedding_dim, in_dim):
        super(UserNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, embedding_dim * 2)
        self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, x):
        y = torch.relu(self.fc1(x))
        w = torch.tanh(self.fc2(y))
        return w
    

