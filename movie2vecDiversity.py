import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_bis(data):
    # Create a t-SNE object
    tsne = TSNE(n_components=2,perplexity=len(data)-1)

    # Fit and transform your data
    data_tsne = tsne.fit_transform(data)
    # Plot your data
    plt.scatter(data[:, 0], data[:, 1], color='green')

def plot(data,recs,save=False,nb=None):
    # Create a t-SNE object
    tsne = TSNE(n_components=2,perplexity=len(data)-1)

    # Fit and transform your data
    data_tsne = tsne.fit_transform(data)
    frst = data_tsne[:recs*2]
    one = frst[:recs]
    two = frst[recs:]
    test = data_tsne[recs*2:recs*3]
    sec = data_tsne[recs*3:]
    # Plot your data
    plt.scatter(sec[:, 0], sec[:, 1], color='grey')
    #basic
    plt.scatter(one[:, 0], one[:, 1], color='blue')
    #div
    plt.scatter(two[:, 0], two[:, 1], color='red')
    #sample
    plt.scatter(test[:, 0], test[:, 1], color='yellow')
    if save:
        plt.savefig('testplot_sample/plot'+str(nb)+'.png')
    else:
        plt.show()
    plt.clf()

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
for row in  train_watched['watched']:
    list_row.append(row)
    
def create_model(size):
    model = Word2Vec(window=5, min_count=1, workers=4, vector_size = size)
    model.build_vocab(list_row)
    model.train(list_row,total_examples=model.corpus_count,epochs=30)
    return model

def create_avg_user_vector(user_id,df,model):
    movie_id_list = df[df['user_id'] == user_id]['movie_id'].tolist()
    vector_movie_id_list = [model.wv[str(x)] for x in movie_id_list]
    return np.average(vector_movie_id_list, axis=0)

def userPast(user_id,df,model):
    movie_id_list = df[df['user_id'] == user_id]['movie_id'].tolist()
    return [model.wv[str(x)] for x in movie_id_list]

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

def most_similar_by_vector(vector):
    return [(x, movies[movies['movie_id'] == int(x[0])].iloc[0]['movie_name']) for x in model.wv.similar_by_vector(vector)]

def build_sample(user_id,model,df,k):
    movie_id_list = df[df['user_id'] == user_id]['movie_id'].tolist()
    sample = []
    v_sample = []
    sample.append(movie_id_list.pop(0))
    v_sample.append(model.wv[str(sample[0])])
    while len(sample) != k:
        weights = []
        for m in movie_id_list:
            weights.append(distance(v_sample,m,model))
        s = sum(weights)
        weights = [x / s for x in weights]
        choice = random.choices(movie_id_list, weights=weights, k=1)[0]
        sample.append(choice)
        v_sample.append(model.wv[str(choice)])
        movie_id_list.remove(choice)
    #plot_bis(v_sample)
    return (sample,v_sample) 

def build_sample2(user_id,model,df,k):
    movie_id_list = df[df['user_id'] == user_id]['movie_id'].tolist()
    sample = []
    v_sample = []
    sample.append(movie_id_list.pop(0))
    v_sample.append(model.wv[str(sample[0])])
    while len(sample) != k:
        weights = []
        for m in movie_id_list:
            weights.append(distance2(v_sample,m,model))
        s = sum(weights)
        weights = [x / s for x in weights]
        choice = random.choices(movie_id_list, weights=weights, k=1)[0]
        sample.append(choice)
        v_sample.append(model.wv[str(choice)])
        movie_id_list.remove(choice)
    #plot_bis(v_sample)
    return sample

def build_sample3(user_id,model,df,k):
    histo = train_df[(train_df['user_id'] == user_id)]
    movie_id_list = df[df['user_id'] == user_id]['movie_id'].tolist()
    sample = []
    v_sample = []
    sample.append(movie_id_list.pop(0))
    v_sample.append(model.wv[str(sample[0])])
    while len(sample) != k:
        weights = []
        for m in movie_id_list:
            weights.append(grade(m,histo) * distance2(v_sample,m,model))
        s = sum(weights)
        weights = [x / s for x in weights]
        choice = random.choices(movie_id_list, weights=weights, k=1)[0]
        sample.append(choice)
        v_sample.append(model.wv[str(choice)])
        movie_id_list.remove(choice)
    #plot_bis(v_sample)
    return sample
    
def build_sample4(user_id,model,df,k,list,grades):
    movie_id_list = df[df['user_id'] == user_id]['movie_id'].tolist()
    sample = list.copy()
    v_sample = []
    for m in sample:
        v_sample.append(model.wv[str(m)])
    while len(sample) != k:
        weights = []
        for m in movie_id_list:
            weights.append(grades[m] * distance2(v_sample,m,model))
        s = sum(weights)
        weights = [x / s for x in weights]
        choice = random.choices(movie_id_list, weights=weights, k=1)[0]
        sample.append(choice)
        v_sample.append(model.wv[str(choice)])
        movie_id_list.remove(choice)
    #plot_bis(v_sample)
    return sample

def build_sample5(user_id,model,df,k,grades):
    movie_id_list = df[df['user_id'] == user_id]['movie_id'].tolist()
    sample = []
    v_sample = []
    sample.append(max(grades,key=grades.get))
    v_sample.append(model.wv[str(sample[0])])
    while len(sample) != k:
        weights = []
        for m in movie_id_list:
            weights.append(grades[m]**2 * distance2(v_sample,m,model))
        s = sum(weights)
        weights = [x / s for x in weights]
        choice = random.choices(movie_id_list, weights=weights, k=1)[0]
        sample.append(choice)
        v_sample.append(model.wv[str(choice)])
        movie_id_list.remove(choice)
    #plot_bis(v_sample)
    return sample

def build_sample6(user_id,model,df,k,grades):
    movie_id_list = df[df['user_id'] == user_id]['movie_id'].tolist()
    sample = []
    v_sample = []
    sample.append(max(grades,key=grades.get))
    v_sample.append(model.wv[str(sample[0])])
    while len(sample) != k:
        weights = []
        for m in movie_id_list:
            weights.append(2**grades[m] * distance2(v_sample,m,model))
        s = sum(weights)
        weights = [x / s for x in weights]
        choice = random.choices(movie_id_list, weights=weights, k=1)[0]
        sample.append(choice)
        v_sample.append(model.wv[str(choice)])
        movie_id_list.remove(choice)
    #plot_bis(v_sample)
    return sample

def grade(movie_id,df):
    return df[(df['movie_id'] == movie_id)]['rating'].values[0]

def distance(L,elem,model):
    vec = model.wv[str(elem)]
    distances = [euclidean(x, vec) for x in L]
    return sum(distances)/len(distances)
def distance2(L,elem,model):
    vec = model.wv[str(elem)]
    distances = [euclidean(x, vec) for x in L]
    return min(distances)
                
    
def recommend_div(user_id,model,df,k):
    (sample,v_sample) = build_sample(user_id,model,df,k)
    #plot(sample)
    movie_id_list = df[df['user_id'] == user_id]['movie_id'].tolist()
    res = []
    for m in sample:
        L = [x[0] for x in model.wv.most_similar(str(m),k+1)]
        for m in L:
            if not (m in movie_id_list) and not (m in res):
                res.append(m)
                break
    if len(res) < k:
        print('problem')
    return res

def recommend_div4(user_id,model,df,k,list,grades):
    sample = build_sample4(user_id,model,df,k- len(list),list,grades)
    #plot(sample)
    movie_id_list = df[df['user_id'] == user_id]['movie_id'].tolist()
    res = []
    for m in sample:
        L = [x[0] for x in model.wv.most_similar(str(m),k+1)]
        for m in L:
            if not (m in movie_id_list) and not (m in res):
                res.append(m)
                break
    res = list + res
    if len(res) < k:
        print('problem')
    return res



def recommend_div3(user_id,model,df,k):
    sample = build_sample3(user_id,model,df,k)
    #plot(sample)
    movie_id_list = df[df['user_id'] == user_id]['movie_id'].tolist()
    res = []
    for m in sample:
        L = [x[0] for x in model.wv.most_similar(str(m),k+1)]
        for m in L:
            if not (m in movie_id_list) and not (m in res):
                res.append(m)
                break
    if len(res) < k:
        print('problem')
    return res

def recommend_div5(user_id,model,df,k,grades):
    sample = build_sample5(user_id,model,df,k,grades)
    #plot(sample)
    movie_id_list = df[df['user_id'] == user_id]['movie_id'].tolist()
    res = []
    for m in sample:
        L = [x[0] for x in model.wv.most_similar(str(m),k+1)]
        for m in L:
            if not (m in movie_id_list) and not (m in res):
                res.append(m)
                break
    if len(res) < k:
        print('problem')
    return res

def recommend_div6(user_id,model,df,k,grades):
    sample = build_sample6(user_id,model,df,k,grades)
    #plot(sample)
    movie_id_list = df[df['user_id'] == user_id]['movie_id'].tolist()
    res = []
    for m in sample:
        L = [x[0] for x in model.wv.most_similar(str(m),k+1)]
        for m in L:
            if not (m in movie_id_list) and not (m in res):
                res.append(m)
                break
    if len(res) < k:
        print('problem')
    return res

def recommend(user_id,model,df,k):
    vector = create_avg_user_vector(user_id,df,model)
    movie_id_list = df[df['user_id'] == user_id]['movie_id'].tolist()
    L =  [x[0] for x in model.wv.similar_by_vector(vector,2*k)]
    res = []
    for m in L:
            if not(m in movie_id_list):
                res.append(m)
            if len(res) == k:
                break
    if len(res) < k:
        print("error in rec length")
    return res

def get_genres(movie_ids):
    genres = []
    proportion = []
    for movie_id in movie_ids:
        here = (movies[movies['movie_id'] == movie_id]['genre'].values[0])
        genres_list = here.split('|')
        for g in genres_list:
            if g not in genres:
                genres.append(g)
                proportion.append(1)
            else:
                proportion[genres.index(g)] += 1
    return (genres,proportion)

def ratio_genre(genres,gp,L):
    dem = sum(gp)
    test = gp.copy()
    count = 0
    for m in L:
        here = (movies[movies['movie_id'] == int(m)]['genre'].values[0])
        genres_list = here.split('|')
        for g in genres_list:
            if g in genres:
                test[genres.index(g)] -= 1
    return sum([x for x in test if x > 0])/dem * 100




def test(size_embd,size_rec,model):
    list_user = test_df['user_id'].unique().tolist()
    count = 0
    count_div = 0
    c3 = 0
    c4 = 0
    d3 = 0
    d4 = 0
    tot = 0
    diversity_list1 = 0
    diversity_list2 = 0
    g1 = 0
    g4 = 0
    g2 = 0
    g3 = 0
    num = 0
    for user in tqdm(list_user):
        ids = train_df[ train_df['user_id'] == user]['movie_id'].tolist()
        if len(ids) < size_rec:
            continue
        user_ratings = train_df[train_df['user_id'] == user]
        (genres,proportion) = get_genres(ids)
        movie_dict = dict(zip(user_ratings['movie_id'], user_ratings['rating']))
        num+=1
        test_list = recommend(user,model,train_df,size_rec)
        test_list_div = recommend_div(user,model,train_df,size_rec)
        t3 = recommend_div5(user,model,train_df,size_rec,movie_dict)
        t4 = recommend_div6(user,model,train_df,size_rec,movie_dict)
        #diversity_list1+=diversity_list(test_list,model)
        #diversity_list2+=diversity_list(test_list_div,model)
        #d3 +=diversity_list(t3,model)
        #d4 +=diversity_list(t4,model)
        g1 += ratio_genre(genres,proportion,test_list)
        g2 += ratio_genre(genres,proportion,test_list_div)
        g3 += ratio_genre(genres,proportion,t3)
        g4 += ratio_genre(genres,proportion,t4)
        #tot += size_rec
        '''
        ids = test_df[ test_df['user_id'] == user]['movie_id'].tolist()
        for m in test_list:
            if int(m) in ids:
                count+=1
        for n in test_list_div:
            if int(n) in ids:
                count_div+=1
        for n in t3:
            if int(n) in ids:
                c3+=1
                
        for n in t4:
            if int(n) in ids:
                c4+=1
    test_res = count/tot
    test_res_div = count_div/tot
    pres3 = c3/tot
    pres4 = c4/tot
    diversity_list1 /= num
    diversity_list2 /= num
    d3 /= num
    d4 /= num
    '''
    tg1 = g1 / num
    tg2 = g2/ num
    tg3 = g3 / num
    tg4 = g4 / num
    destFile = ".result_test6"
    with open(destFile, 'a') as f:
        f.write("---------\n")
        f.write("Test settings: embeding size = "+str(size_embd)+", recommendations size = "+str(size_rec)+"\n")
        f.write("\tBasic Movie2Vec:\n")
        #f.write("\t\tPrecision: "+str(test_res)+"\n")
        #f.write("\t\tdiversity: "+str(diversity_list1)+"\n")
        f.write(str(tg1)+"\n")
        f.write("\tDiversity based Movie2Vec with:\n")
        #f.write("\t\tPrecision: "+str(test_res_div)+"\n")
        #f.write("\t\tdiversity: "+str(diversity_list2)+"\n")
        f.write(str(tg2)+"\n")
        f.write("\tDiversity based Movie2Vec with grade^2:\n")
        f.write(str(tg3)+"\n")
        #f.write("\t\tPrecision: "+str(pres3)+"\n")
        #f.write("\t\tdiversity: "+str(d3)+"\n")
        f.write("\tMovie 2 Vec 2^grade:\n")
        f.write(str(tg4)+"\n")
        #f.write("\t\tPrecision: "+str(pres4)+"\n")
        #f.write("\t\tdiversity: "+str(d4)+"\n")
        f.write("---------\n")


print("begin test")
'''
print("test 20")
model = create_model(20)
test(20,5,model)
test(20,10,model)
test(20,15,model)
test(20,20,model)

print("test 40")
model = create_model(40)
test(40,5,model)
test(40,10,model)
test(40,15,model)
test(40,20,model)

print("test 60")
model = create_model(60)
test(60,5,model)
test(60,10,model)
test(60,15,model)
test(60,20,model)

print("test 80")
model = create_model(80)
test(80,5,model)
test(80,10,model)
test(80,15,model)
test(80,20,model)

'''
print("test 100")
model = create_model(100)
test(100,5,model)
test(100,10,model)
test(100,15,model)
test(100,20,model)

print("end test")

def testplot():
    model = create_model(100)
    list_user = test_df['user_id'].unique().tolist()
    for user in tqdm(list_user):
        ids = train_df[ train_df['user_id'] == user]['movie_id'].tolist()
        if len(ids) < 20:
            continue
        list = recommend(user,model,train_df,20)
        (listdiv,v_sample) = recommend_div(user,model,train_df,20)
        vects = []
        for m in list + listdiv:
            vects.append(model.wv[str(m)])
        l = vects + v_sample + userPast(user,test_df,model)
        plot(np.array(l),20,True,str(user))


def testrepartition():
    res = [0]*20
    model = create_model(100)
    list_user = test_df['user_id'].unique().tolist()
    for user in list_user:
        ids = train_df[ train_df['user_id'] == user]['movie_id'].tolist()
        if len(ids) < 20:
            continue
        (list,vect) = recommend_div(user,model,train_df,20)
        ids = test_df[ test_df['user_id'] == user]['movie_id'].tolist()
        for i in range(len(list)):
            if(int(list[i]) in ids):
                res[i]+=1
    print(res)
    