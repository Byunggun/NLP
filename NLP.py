
#1. One-Hot Encording - by DictVectorizer()
#cf. 트하나키, 파이썬 자연어 처리의 이론과 실제, p.216(cf.https://github.com/jalajthanaki/NLPython/blob/master/ch5/onehotencodingdemo/OHEdemo.py)
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


df = pd.DataFrame([['rick','young'],['phil','old']],columns=['name','age-group'])
print(df)
print("\n----By using Panda ----\n")
print(pd.get_dummies(df)) #Q)?

X = pd.DataFrame({'income': [100000,110000,90000,30000,14000,50000],
                  'country':['US', 'CAN', 'US', 'CAN', 'MEX', 'US'],
                  'race':['White', 'Black', 'Latino', 'White', 'White', 'Black']})

print("\n----By using Sikit-learn ----\n")
v = DictVectorizer()
qualitative_features = ['country'] #country 행 제목
X_qual = v.fit_transform(X[qualitative_features].to_dict('records')) #Q)?
print(v.vocabulary_)
print(X_qual.toarray())
