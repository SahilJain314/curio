import user_model as um
# import video_model as vm
import torch
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
import json
import pandas as pd
import learner_type_generator as ltg
import video_data_generator as vdg
from operator import itemgetter
import numpy as np

torch.autograd.set_detect_anomaly(True)

embedding_dim = 10
survey_dim = 60
num_user = 203
u_id_offset = 1
# glove_dict, e_matrix = vm.load_glove(vm.glove_dir)

with open('../youtube_video/full_data.json') as f:
    video_dataset = json.load(f)

def get_dataset_size():
    size = 0
    for a in video_dataset.keys():
        size += len(video_dataset[a])
    return size

# video_mod = vm.VideoModel(e_matrix, 400, 1, embedding_dim)
user_mod = um.UserNet(embedding_dim, embedding_dim + survey_dim)
video_embeddings = torch.randn((get_dataset_size(), embedding_dim), requires_grad = True) 
user_explicit = torch.tensor(np.array(pd.read_csv('learner_feature_data.csv'))[:, 1:], requires_grad = False) 
vid_id_to_idx = {}

vid_idx = 0
for a in video_dataset.keys():
    for b in range(len(video_dataset[a])):
        vid_id_to_idx[video_dataset[a][b]['videoId']] = vid_idx
        vid_idx += 1

def get_vid_data_by_id(vid_id):
    return video_embeddings[vid_id_to_idx[vid_id]]

def get_user_explicit_data(user_id):
    return user_explicit[user_id - u_id_offset:user_id - u_id_offset + 1]

def loss(affinity, p_embed, v_embed):
    print(p_embed.shape, v_embed.shape)
    batch_size = p_embed.shape[0]
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    aff_sim = torch.sum(cos(p_embed, v_embed) * affinity)
    return -aff_sim / batch_size

def train(interactions, num_iter = 100, batch_size = 2000, learning_rate = .05):

    params = list(user_mod.parameters())# + list(video_embeddings)
    optimizer = optim.Adam(params, lr=learning_rate)
    for i in range(num_iter):
        v_embed = {}
        p_features = {}
        
        person_videos = {}
        for i in interactions:
            if i[0] in person_videos:
                person_videos[i[0]].append(get_vid_data_by_id(i[1]))
            else:
                person_videos[i[0]] = [get_vid_data_by_id(i[1])]

        for p in person_videos.keys():
            sum_vid_embed = torch.zeros((1, embedding_dim), requires_grad = False)
            for v in person_videos[p]:
                sum_vid_embed += v
            explicit = get_user_explicit_data(p)
            sum_vid_embed = sum_vid_embed / len(person_videos[p])
            p_features[p] = torch.cat((explicit.detach().reshape(1, explicit.shape[1]), sum_vid_embed.detach()), dim=1)
            
        p_embed = {}
        for p in p_features.keys():
            p_embed[p] = user_mod(p_features[p])

        batch_size = len(interactions)
        for i in range(len(interactions) // batch_size):
            p_embed_batch = torch.cat([p_embed[interaction[0]] for interaction in interactions[i * batch_size: (i + 1) * batch_size]]) 
            v_embed_batch = torch.cat([get_vid_data_by_id(interaction[1]).reshape(1, embedding_dim) for interaction in interactions[i * batch_size: (i + 1) * batch_size]]) 
            affinity_batch = torch.tensor([interaction[2] for interaction in interactions[i * batch_size: (i + 1) * batch_size]], requires_grad = False)  
            l = loss(affinity_batch, p_embed_batch, v_embed_batch)
            l.backward()
            optimizer.step()
            print(l)

def train_syn():
    dataset = "learner_feature_data.csv"
    generator = ltg.LearnerTypeGenerator(dataset)
    vid_gen = vdg.VideoDataGenerator("video_data.csv", "learner_feature_data.csv")
    syn_data = []
    for i in range(1, generator.get_number_person() + 1):
        syn_data += vid_gen.generate_video_affinity_data(i)

    train(syn_data)
    
