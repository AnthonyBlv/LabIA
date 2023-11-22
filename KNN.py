import numpy as np
import pandas as pd

ratings = pd.read_csv(
    'ml-25M/ratings.csv',
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

movies = pd.read_csv(
    'ml-25M/movies.csv')

ratings = ratings.groupby('userId').filter(lambda x: len(x) >= 55)

movie_list_rating = ratings.movieId.unique().tolist()

movies = movies[movies.movieId.isin(movie_list_rating)]

tags = pd.read_csv(
    'ml-25M/tags.csv',)
tags = tags.drop(columns='timestamp')
tags = tags.drop(columns='userId')
tags = pd.DataFrame(tags[['movieId','tag']].drop_duplicates())

tags['tag'] = tags['tag'] + ' '

aggregation_functions = {'tag': 'sum'}
new_tags = tags.groupby(tags['movieId']).aggregate(aggregation_functions)

mixed = pd.merge(movies, new_tags, on='movieId', how='left')

mixed['metadata'] = mixed['genres'] + '|' + mixed['tag']
mixed = mixed.drop(columns='genres')
mixed = mixed.drop(columns='tag')
mixed = mixed.dropna() 

import math
import re
from collections import Counter

WORD = re.compile(r"\w+")
def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

movie_ids_in_dataframe = mixed['movieId'].unique().tolist()
movies = movies[movies['movieId'].isin(movie_ids_in_dataframe)]
ratings = ratings[ratings['movieId'].isin(movie_ids_in_dataframe)]

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator
    

def find_movie_id(movie_title):
    movie_id = movies.loc[movies['title'] == movie_title, 'movieId'].values[0]
    return movie_id

def search_movies(input_string):
    filtered_df = movies[movies['title'].str.contains(input_string, case=False)]
    return list(filtered_df['title'])

def sim(i1,i2):
    t1 = mixed.loc[mixed['movieId'] == i1, 'metadata'].values[0]
    v1 = text_to_vector(t1)
    t2 = mixed.loc[mixed['movieId'] == i2, 'metadata'].values[0]
    v2 = text_to_vector(t2)
    return  get_cosine(v1,v2)

def sim_movies(movie_id,list_movie):
    nb_row = len(list_movie)
    res_mat = np.zeros((nb_row, 1),dtype=float)
    res = 0.0
    for i in range(nb_row):
            res_mat[i] = sim(list_movie[i],movie_id)
    return res_mat

def find_movie_name_with_Id(movieId):
    movie_id = movies.loc[movies['movieId'] == movieId, 'title'].values[0]
    return movie_id

def get_movie_ids_seen_by_user(user_id):
    user_ratings = ratings[ratings['userId'] == user_id]
    return user_ratings['movieId'].tolist()

def get_movie_ids_not_seen_by_user(user_id):
    all_movie_ids = ratings['movieId'].unique().tolist()
    user_movie_ids = get_movie_ids_seen_by_user(user_id)
    movie_ids_not_seen_by_user = list(set(all_movie_ids) - set(user_movie_ids))
    return movie_ids_not_seen_by_user

def topk_similar_movies(userId, movieId, list_seen,k):
    topk = ((pd.DataFrame(sim_movies(userId,list_seen))).sort_values(by=0,ascending=False))[:k]
    return topk

def predict_grade(userId, movieId, list_seen,k):
    topk = topk_similar_movies(userId, movieId, list_seen,k)
    grade = 0
    for ind in topk.index.tolist():
        res = (ratings[(ratings['userId'] == userId) & (ratings['movieId'] == (get_movie_ids_seen_by_user(userId))[ind])])['rating'].values[0]
        grade+= res
    return grade/k


def insert_rec(recommendation_list,k,grd,mv):
    for i in range(10):
        (val,mid) = recommendation_list[i]
        if val < grd:
            tmp = (val,mid)
            recommendation_list[i] = (grd,mv)
            (grd,mv) = tmp
        
    


def recommendation(userId,k):
    list_unseen = get_movie_ids_not_seen_by_user(userId)
    list_seen =  get_movie_ids_seen_by_user(userId)
    recommendation_list = [(0,0)] * 10
    tot = len(list_unseen)
    count = 0
    for movie in list_unseen:
        count+=1
        print(count,'/',len(list_unseen))
        insert_rec(recommendation_list, k, predict_grade(userId,movie,list_seen,k),movie) 
    return recommendation_list



res = recommendation(3,1)

print(res)