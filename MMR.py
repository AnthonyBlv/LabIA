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


def diversity_list_append(L,knn,elem,div):
    lgt = len(L)
    res = 0
    for i in range(lgt):
        res+= distance(L[i],elem,knn)
    return (res+div*(lgt*(lgt-1)/2))/((lgt*(lgt-1)/2)+lgt)

def diversity_score_element(L,elem,diversity,knn):
    if len(L) == 1:
        tmp = L.copy()
        tmp.append(elem)
        return diversity_list(tmp,knn)
    return diversity_list_append(L,knn,elem,diversity)-diversity

def normalize(L):
    maxi = L.max()
    mini = 1
    return np.apply_along_axis((lambda x: ((x-mini)/(maxi-mini))*(4)+1),0,L)

def recommend(user_idx,length,trade_off,knn):
    #check user_id to index
    (recommendations, scores) = knn.rank(user_idx)
    scores = normalize(scores)
    res = []
    tmp = recommendations.tolist()
    first = recommendations[0]
    tmp.remove(first)
    res.append(first)
    div = 0
    for _ in range(length - 1):
        movie_id = max(tmp,key = lambda key: float((1-trade_off)*scores[key] + trade_off*5*(diversity_score_element(res,key,div,knn))))
        div = diversity_list_append(res,knn,movie_id,div)
        res.append(movie_id)
        tmp.remove(movie_id)
    return res