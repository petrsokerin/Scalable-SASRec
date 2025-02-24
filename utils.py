import random, os
import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def transform_indices(data, users, items):
    data_index = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        idx, idx_map = to_numeric_id(data, field)
        data_index[entity] = idx_map
        data.loc[:, field] = idx
    return data, data_index

def to_numeric_id(data, field):
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map

def topidx(a, topn):
    parted = np.argpartition(a, -topn)[-topn:]
    return parted[np.argsort(-a[parted])]

def topn_recommendations(scores, topn=10):
    recommendations = np.apply_along_axis(topidx, 1, scores, topn)
    return recommendations

def downvote_seen_items(scores, data, data_description):
    userid = data_description['users']
    itemid = data_description['items']
    # get indices of observed data
    user_idx = data[userid].values
    item_idx = data[itemid].values
    # downvote scores at the corresponding positions
    user_idx, _ = pd.factorize(user_idx, sort=True)
    seen_idx_flat = np.ravel_multi_index((user_idx, item_idx), scores.shape)
    np.put(scores, seen_idx_flat, -np.inf)

def downvote_seen_items_one_user(scores, data, data_description, selected_user_id):
    userid = data_description['users']
    itemid = data_description['items']
    item_idx = data[data[userid] == selected_user_id][itemid].values
    seen_idx_flat = np.ravel_multi_index((np.array([0]), item_idx), scores.shape)
    np.put(scores, seen_idx_flat, -np.inf)
