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
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x
    
def make_person_feature_vec(survey_results, affinities, video_ids, video_embeddings):
    out_vec = torch.zeros((survey_results.shape[0] + video_embeddings[video_ids[0]].shape[0], 1))
    out_vec[0 : survey_results.shape[0]] = survey_results
    for video in video_ids:
        out_vec[survey_results.shape[0]:] += video_embeddings[video]

