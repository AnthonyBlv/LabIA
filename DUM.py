import cornac
from cornac.utils import cache
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.models import UserKNN, ItemKNN
import pandas as pd
import numpy as np

def distance(movie_idx1,movie_idx2,iknn_cosine):
    return 1-(iknn_cosine.sim_mat[movie_idx1, movie_idx2] + 1)/2

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

def diversity_score_element(L,elem,diversity,knn):
    tmp = L.copy()
    tmp.append(elem)
    return diversity_list(tmp,knn)-diversity


def normalize(L):
    maxi = L.max()
    mini = 1
    return np.apply_along_axis((lambda x: ((x-mini)/(maxi-mini))*(4)+1),0,L)


def recommend(knn,length,user_id):
    (recommendations, scores) = knn.rank(user_id)
    res = []
    tmp = recommendations.tolist()
    first = recommendations[0]
    tmp.remove(first)
    div = 0
    res.append(first)
    for movie_id in tmp:
        if diversity_score_element(res,movie_id,div,knn) > 0:
            res.append(movie_id)
            div = diversity_list(res,knn)
        tmp.remove(movie_id)
        if len(res) >= length:
            break
    return res