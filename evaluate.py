import DUM
import MMR
import KNN_diversity
import target_diversity

import cornac
from cornac.eval_methods import RatioSplit
from cornac.models import ItemKNN
import pandas as pd
import numpy as np

from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP
ml_100k = cornac.datasets.movielens.load_feedback(variant="1M")
rs = RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)
iknn_cosine =ItemKNN(
  k=50, similarity="cosine", mean_centered=True, name="ItemKNN-AdjustedCosine", verbose=False
).fit(rs.train_set)

curr = 0
end = len(rs.test_set.user_data)
print('begin test')
count = 0
tot = 0
for k in rs.test_set.user_data:
    curr+=1
    print(str(curr)+"/"+str(end))
    (arr,s) = rs.test_set.user_data[k]
    if len(s) < 10:
        continue
    res = MMR.recommend(k,10,0.5,iknn_cosine)
    tot +=len(res)
    for m in res:
        if m in arr:
            ind = arr.index(m)
            if s[ind] >= 1:
                count+=1
test_res = count/(tot)
destFile = ".result_test"
with open(destFile, 'a') as f:
    f.write("MMR: "+str(test_res)+"\n")
print('end test')