import numpy as np
import torch

from tqdm import tqdm

from utils import topn_recommendations, downvote_seen_items, downvote_seen_items_one_user
from data import data_to_sequences


def get_test_scores(model, data_description, testset_, holdout_, device, effect_test=True):

    if effect_test:
        sasrec_recs = sasrec_model_effect_scoring(model, testset_, data_description, device, topn=50)
    else:
        sasrec_scores = sasrec_model_scoring(model, testset_, data_description, device)
        downvote_seen_items(sasrec_scores, testset_, data_description)
        sasrec_recs = topn_recommendations(sasrec_scores, topn=50)
    test_scores = model_evaluate(sasrec_recs, holdout_, data_description)
    return test_scores


def sasrec_model_effect_scoring(params, data, data_description, device, topn=10):
    model = params
    model.eval()
    test_sequences = data_to_sequences(data, data_description)
    # perform scoring on a user-batch level
    scores = []

    for (test_user_idx, seq) in tqdm(test_sequences.items()):
        with torch.no_grad():
            predictions = model.score(torch.tensor(seq, device=device, dtype=torch.long))
            predictions = predictions.detach().cpu().numpy()
            downvote_seen_items_one_user(predictions, data, data_description, test_user_idx)
            predictions = topn_recommendations(predictions, topn=topn)

        scores.append(predictions)
    return np.concatenate(scores, axis=0)


def sasrec_model_scoring(params, data, data_description, device):
    model = params
    model.eval()
    test_sequences = data_to_sequences(data, data_description)
    # perform scoring on a user-batch level
    scores = []

    for _, seq in tqdm(test_sequences.items()):
        with torch.no_grad():
            predictions = model.score(torch.tensor(seq, device=device, dtype=torch.long))
        scores.append(predictions.detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def calculate_topn_metrics(recommended_items, holdout_items, n_items, n_test_users, topn):
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)

    # HR calculation
    hr = np.mean(hits_mask.any(axis=1))

    # MRR calculation
    hit_rank = np.where(hits_mask)[1] + 1.0
    mrr = np.sum(1 / hit_rank) / n_test_users
   
    #NDCG calculation
    ndcg = np.sum(1 / np.log2(hit_rank + 1.)) / n_test_users

    #COV calculation
    cov = np.unique(recommended_items[:, :topn]).size / n_items

    return {'hr': hr, 'mrr': mrr, 'ndcg': ndcg, 'cov': cov}


def model_evaluate(recommended_items, holdout, holdout_description, topn_list=(1, 5, 10, 20, 50)):
    n_items = holdout_description['n_items']
    itemid = holdout_description['items']
    holdout_items = holdout[itemid].values
    n_test_users = recommended_items.shape[0]
    assert recommended_items.shape[0] == len(holdout_items)

    metrics = {}
    for topn in topn_list:
        metrics = metrics | {f'{key}@{topn}': value for key, value in calculate_topn_metrics(recommended_items, holdout_items, n_items, n_test_users, topn).items()}

    return metrics
