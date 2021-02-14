import torch.nn as nn
import personal_vector as pv
import video_model
import user_model as um

video_embeddings = {} #need to load a dictionary mapping video-id to embedding

'''
Takes survey_results(from initial user survey), affinities for videos, the video_ids 
corresponding to those affinities, and candidate videos for search (of the correct topic)
and returns the best n matches by embedding cosine distance.
'''
def inference(survey_results, affinities, video_ids, candidates, n):
    p_vec = pv.make_person_feature_vec(survey_results, affinities, video_ids, video_embeddings)
    
    p_embed = um.inference(p_vec)
    candidate_embeddings = [video_embeddings[c] for c in candidates]

    nearest_idx = k_cos_nn(p_embed, candidate_embeddings, n)
    return candidates[nearest_idx]


'''
Takes a torch vector, v, and finds k nearest neighbors in search_list by cosine distance.
'''
def k_cos_nn(v, search_list, k):
    out_list = []
    sim_list = []
    for s in search_list:
        cosine_sim = cosine_sim(v, s)

'''
Returns cosine similarity between torch vectors v, s along dimension 1
'''
def cosine_sim(v, s):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(v, s)
