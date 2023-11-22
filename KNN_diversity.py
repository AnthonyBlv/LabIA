import cornac
from cornac.utils import cache
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.models import UserKNN, ItemKNN
import pandas as pd
import numpy as np

def distance(movie_idx1,movie_idx2,iknn_cosine):
    return (iknn_cosine.sim_mat[movie_idx1, movie_idx2] + 1)/2

def diversity_list(L,knn):
    lgt = len(L)
    res = 0
    count = 0
    for i in range(lgt):
        for j in range(i+1,lgt):
                res+= distance(L[i],L[j],knn)
                count += 1
    res /= count
    return res

def nearest_neighbor(knn,item_idx,ban_list):
    x = 1
    res = -1
    while res == -1:
        L = np.argsort(knn.sim_mat[item_idx].A.ravel())[-x:]
        for idx in L:
            if idx not in ban_list:
                return idx
        x*=2

def recommend(knn, k, user_idx):
    ban_list = []
    res = []
    rating_mat = knn.train_set.matrix
    rating_arr = rating_mat[user_idx].A.ravel()
    top_rated_items = np.argsort(rating_arr)[-k:]
    ban_list.extend(top_rated_items)
    for idx in top_rated_items:
        add = nearest_neighbor(knn,idx,ban_list)
        res.append(add)
        ban_list.append(add)
    return res