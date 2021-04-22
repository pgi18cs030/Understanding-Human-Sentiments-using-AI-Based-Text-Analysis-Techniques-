import pandas as pd
import numpy as np
df=pd.read_csv('balanced_reviews.csv')

df.head()

df['overall'].value_counts()

df.isnull().any(axis=0)

df.dropna(inplace=True)

df['overall']!=3

df=df[df['overall']!=3]

df['Positivity']=np.where(df['overall']>3,1,0)

from sklearn.model_selection import train_test_split

features_train,features_test,labels_train,label_test=train_test_split(df['reviewText'],df['Positivity'],random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer().fit(features_train)

len(vect.get_feature_names())
vect.get_feature_names()[10000:10010]


features_train_vectorized=vect.transform(features_train)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression(solver='liblinear')

model.fit(features_train_vectorized,labels_train)

predictions=model.predict(vect.transform(features_test))

from sklearn.metrics import confusion_matrix
confusion_matrix(label_test,predictions)

from sklearn.metrics import roc_auc_score

roc_auc_score(label_test,predictions)


#model version-02

from sklearn.model_selection import train_test_split

features_train,features_test,labels_train,label_test=train_test_split(df['reviewText'],df['Positivity'],random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vect=TfidfVectorizer(min_df=5).fit(features_train)

len(vect.get_feature_names())
vect.get_feature_names()[10000:10010]


features_train_vectorized=vect.transform(features_train)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression(solver='liblinear')

model.fit(features_train_vectorized,labels_train)

predictions=model.predict(vect.transform(features_test))

from sklearn.metrics import confusion_matrix
confusion_matrix(label_test,predictions)

from sklearn.metrics import roc_auc_score  

roc_auc_score(label_test,predictions)

import pickle

pkl_filename="pickle_model.pkl"

file=open(pkl_filename,"wb")

pickle.dump(model,file)

file.close()


#---------------------------------

file=open('pickle_model.pkl',"rb")

recreated_model=pickle.load(file)

predictions=recreated_model.predict(vect.transform('I loved this product.'))

from sklearn.metrics import confusion_matrix
confusion_matrix(label_test,predictions)

from sklearn.metrics import roc_auc_score

roc_auc_score(label_test,predictions)

