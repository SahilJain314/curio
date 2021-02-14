import user_model as um
import video_model as vm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json

embedding_dim = 50
survey_dim = 45
num_video = 100
num_user = 10
glove_dict, e_matrix = vm.load_glove(vm.glove_dir)

video_mod = vm.VideoModel(e_matrix, 400, 1, embedding_dim)
user_mod = um.UserModel(embedding_dim)
video_embeddings = torch.randn((num_video, embedding_dim), requires_grad = True) 
user_explicit = torch.randn((num_user, survey_dim), requires_grad = False)

with open('data.json') as f:
    video_dataset = json.load(f)

def get_vid_data_by_id(vid_id):
    return video_embeddings[vid_id]

def get_user_explicit_data(user_id):
    return torch.tensor(user_explicit, requires_grad = False)

def loss(affinity, p_embed, v_embed):
    batch_size = p_embed.shape[0]
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    aff_sim = cos(p_embed, v_embed) * affinity
    return -aff_sim / batch_size

def train(interactions, num_iter = 10, batch_size = 10):

    for i in range(num_iter):
        v_embed = {}
        p_features = {}
        
        person_videos = {-1:[]}
        for i in interactions:
            if person_videos.has_key(i[0]):
                person_videos[i[0]].append(get_vid_data_by_id(i[1]))
            else:
                person_videos[i[0]] = [get_vid_data_by_id(i[1])]

        for p in person_videos.keys():
            sum_vid_embed = torch.zeros((1, embedding_dim), requires_grad = True)
            for v in person_videos[p]:
                sum_vid_embed += v
            sum_vid_embed = sum_vid_embed / len(person_videos[p])
            p_features[p] = torch.cat((get_user_explicit_data.detach(), sum_vid_embed))
            
        p_embed = {}
        for p in p_features.keys():
            p_embed = user_mod(p_features[p])

        
        
