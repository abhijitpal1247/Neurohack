#%%
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#%%
df = pd.read_csv('..\\Results\\l1_tagged_data.csv')

#%%
model = Doc2Vec.load('..\\Results\\doc2vec')
doc_vec = np.zeros((df.shape[0], 400))
#%%
for i in range(df.shape[0]):
    doc_vec[i] = model.dv[[i]]
# %%
label = df['l1_topic'].values
print(label.shape)
print(doc_vec.shape)
#%%
X_train, X_test, Y_train, Y_test = train_test_split(doc_vec, label, test_size=0.33, random_state=42)
# %%
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
#%%
from sklearn.metrics import accuracy_score, f1_score
print('Testing accuracy %s' % accuracy_score(Y_test, Y_pred))
print('Testing F1 score: {}'.format(f1_score(Y_test, Y_pred, average='weighted')))