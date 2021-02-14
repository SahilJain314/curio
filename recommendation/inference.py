import torch
import torch.nn as nn
import personal_vector as pv
import video_model
import user_model as um
import pickle as pkl

with open('../recommendation/vid_embed', 'rb') as f:
    video_embeddings = pkl.load(f) #need to load a dictionary mapping video-id to embedding

with open('../recommendation/vid_idx', 'rb') as e:
    vid_id_to_idx = pkl.load(e)

user_model = torch.load('../recommendation/embed_model.pth')

'''
Takes survey_results(from initial user survey), affinities for videos, the video_ids 
corresponding to those affinities, and candidate videos for search (of the correct topic)
and returns the best n matches by embedding cosine distance.
'''
def inference(survey_results, affinities, video_ids, candidates, n):
    p_vec = make_person_feature_vec(survey_results, affinities, video_ids, video_embeddings, vid_id_to_idx)
    
    p_embed = user_model(p_vec)
    candidate_embeddings = [video_embeddings[vid_id_to_idx[c]] for c in candidates]

    nearest_idx, _ = k_cos_nn(p_embed, candidate_embeddings, n)
    return candidates[nearest_idx]


'''
Takes a torch vector, v, and finds k nearest neighbors in search_list by cosine distance.
'''
def k_cos_nn(v, search_list, k):
    out = 0
    sim = 0
    for i in range(len(search_list)):
        cos_sim = cosine_sim(v, search_list[i])
        if cos_sim > sim:
            sim = cos_sim
            out = i
    return out, sim

'''
Returns cosine similarity between torch vectors v, s along dimension 1
'''
def cosine_sim(v, s):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(v, s)

'''
Returns feature vector for user
'''
def make_person_feature_vec(survey_results, affinities, video_ids, video_embeddings, vid_id_to_idx):
    sum_vid_embed = torch.zeros((1, 10), requires_grad = False)
    for v in video_ids:
        sum_vid_embed += video_embeddings[vid_id_to_idx[v]]
    explicit = torch.tensor(survey_results)
    sum_vid_embed = sum_vid_embed / len(video_ids)
    return torch.cat((explicit.detach().reshape(1, explicit.shape[1]), sum_vid_embed.detach()), dim=1)
