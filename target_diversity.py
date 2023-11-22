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

def diversity_score_element(L,elem,div,knn):
    tmp = L.copy()
    tmp.append(elem)
    return abs(diversity_list(tmp,knn)-div)

def normalize(L):
    maxi = L.max()
    mini = 1
    return np.apply_along_axis((lambda x: ((x-mini)/(maxi-mini))*(4)+1),0,L)


def recommend(user_idx,length,knn):
    #check user_id to index
    (recommendations, scores) = knn.rank(user_idx)
    res = []
    tmp = recommendations.tolist()
    first = recommendations[0]
    tmp.remove(first)
    #Get the best k film for the user
    rating_mat = knn.train_set.matrix
    rating_arr = rating_mat[user_idx].A.ravel()
    top_rated_items = np.argsort(rating_arr)[-length:]
    #compute the diversity of the k-fav movies
    div = diversity_list(top_rated_items,knn)
    #remove worst half of movies
    tmp = tmp[0:len(tmp)//2]
    res.append(first)
    for _ in range(length - 1):
        movie_id = min(tmp,key = lambda key: diversity_score_element(res,key,div,knn))
        res.append(movie_id)
        tmp.remove(movie_id)
    return res