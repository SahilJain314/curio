import numpy as np
import torch

def make_person_feature_vec(survey_results, affinities, video_ids, video_embeddings):
    out_vec = torch.zeros((survey_results.shape[0] + video_embeddings[video_ids[0]].shape[0], 1))
    out_vec[0 : survey_results.shape[0]] = survey_results
    for video in video_ids:
        out_vec[survey_results.shape[0]:] += video_embeddings[video]
