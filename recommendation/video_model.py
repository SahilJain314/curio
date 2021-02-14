import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

glove_dir = '/mnt/c/Users/sahil/Desktop/LFS/'
dim = 300

def load_glove(glove_path):
    print('Loading glove')
    glove = pd.read_csv(glove_dir + 'glove.840B.300d.txt', sep=' ', quoting = 3, header = None, index_col=0)
    glove_dict = {k: v.values for k, v in glove.T.items()}
    print('glove loaded')
    
    e_matrix = np.zeros((len(glove_dict), dim))
    word_idx = 0;
    for k in glove_dict.keys():
        e_matrix[word_idx] = glove_dict[k];
        word_idx += 1

    return glove_dict, e_matrix

def create_e_layer(matrix):
    num_e, e_size = matrix.shape;
    emb_layer = nn.Embedding(num_e, e_size)
    emb_layer.load_state_dict({'weight': matrix})
    emb_layer.weight.requires_grad = False 
    return emb_layer, num_e, e_size

class VideoModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, layers, output_dim):
        super().__init__()
        self.embedding, num_e, emb_dim = create_e_layer(embedding_matrix)
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.LSTM = nn.LSTM(emb_dim, hidden_size, self.num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, in_text):
        hidden = self.first_hidden
        out, (hn, cn) = self.LSTM(self.embedding(in_text), hidden.detach())
        return out

    def first_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size))
