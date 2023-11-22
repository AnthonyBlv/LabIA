import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

# Load the ratings dataset
ratings = pd.read_csv('ml-1m/ratings.dat',
                         delimiter='::', engine='python', header=None,
                         names=['user_id', 'movie_id', 'rating', 'time'])

movies = pd.read_csv('ml-1m/movies.dat',
                        delimiter='::', engine= 'python', header=None,
                        names=['movie_id','movie_name', 'genre'],encoding='latin-1')

movie_list_rating = ratings.movie_id.unique().tolist()

movies = movies[movies.movie_id.isin(movie_list_rating)]

train_df, test_df = train_test_split(ratings, test_size=0.2)

train_watched = pd.DataFrame(columns=['user_id', 'watched'])

for index, user_id in enumerate(range(min(train_df['user_id']), max(train_df['user_id']))):
    d = train_df[train_df['user_id'] == user_id].sort_values(by='time').filter(['movie_id'])
    l = d['movie_id'].tolist()
    st = list(map(str, l))
    train_watched.loc[index] = [user_id, st]

list_row = []
for row in  tqdm(train_watched['watched']):
    list_row.append(row)

model = Word2Vec(window=5, min_count=1, workers=4)
model.build_vocab(list_row)

model.train(list_row,total_examples=model.corpus_count,epochs=30)

def create_avg_user_vector(user_id,df,model):
    movie_id_list = df[df['user_id'] == user_id]['movie_id'].tolist()
    vector_movie_id_list = [model.wv[str(x)] for x in movie_id_list]
    return np.average(vector_movie_id_list, axis=0)

def most_similar_by_vector(vector):
    return [(x, movies[movies['movie_id'] == int(x[0])].iloc[0]['movie_name']) for x in model.wv.similar_by_vector(vector)]

def recommend(user_id,model,df):
    vector = create_avg_user_vector(user_id,df,model)
    return [x[0] for x in model.wv.similar_by_vector(vector,10)]

def diversity_list(L,model):
    lgt = len(L)
    res = 0
    count = 0
    for i in range(lgt):
        for j in range(i+1,lgt):
                res+= model.wv.similarity(L[i],L[j])
                count += 1
    res /= count
    return 1- res

list_user = test_df['user_id'].unique().tolist()
count = 0
tot = 0
avg_div = 0
d = 0
for user in tqdm(list_user):
    d+=1
    test_list = recommend(user,model,train_df)
    ids = test_df[ test_df['user_id'] == user]['movie_id'].tolist()
    tot += len(test_list)
    avg_div += diversity_list(test_list,model)
    for m in test_list:
        if int(m) in ids:
            count+=1
test_res = count/(tot)
avg_div /= d
destFile = ".result_test"
with open(destFile, 'a') as f:
    f.write("Movie2Vec: "+str(test_res)+"\n")
    f.write("average diversity: "+str(avg_div)+"\n")
print('end test')