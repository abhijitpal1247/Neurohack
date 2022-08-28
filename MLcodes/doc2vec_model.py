#%%
from re import L
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
import pandas as pd
from tqdm import tqdm
import numpy as np
#%%
df = pd.read_csv('..\\Results\\topic_added.csv')
#%%
df.dropna(inplace=True)
#%%
class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.epoch = 0
    def on_epoch_begin(self, model):
        if self.epoch==0:
            self.pb = tqdm(self.total_epochs)

    def on_epoch_end(self, model):
        self.epoch += 1
        if self.epoch == self.total_epochs:
            self.pb.close()

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df['preprocessed'])]
#%%
model = Doc2Vec(documents, vector_size=400, 
window=5, min_count=1, epochs=30, 
callbacks=[EpochSaver(30)])
# %%
model.save('..\\Results\\doc2vec')

# %%
model = Doc2Vec.load('..\\Results\\doc2vec')
#%% 
topic_doc = [[] for i in range(1980)] 
for row in df[['preprocessed', 'topic_num']].iterrows():
    i, doc, topic = row[0], row[1]['preprocessed'], row[1]['topic_num']
    topic_doc[int(topic)].append(i)
#%%

top_vec = np.zeros((1980, 400))

for i in range(top_vec.shape[0]):
    top_vec[i] = np.mean(model.dv[topic_doc[i]], axis=0)
    
# %%
import scipy.cluster.hierarchy as shc
from matplotlib import pyplot
pyplot.figure(figsize=(10, 7))  
pyplot.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(top_vec, method='ward'))
# %%
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', 
compute_full_tree=True, distance_threshold=2)

#%%
# Cluster the data
res = cluster.fit_predict(top_vec)

#%%
print(f"Number of clusters = {1+np.amax(cluster.labels_)}")

# Display the clustering, assigning cluster label to every datapoint 
print("Classifying the points into clusters:")
print(cluster.labels_)

# Display the clustering graphically in a plot
print(f"SK Learn estimated number of clusters = {1+np.amax(cluster.labels_)}")
print(" ")
#%%
# %%

l1_dict = {}
for i in range(res.shape[0]):
    if res[i] in l1_dict.keys():
        l1_dict[res[i]].append(i)
    else:
        l1_dict[res[i]] = [i]

# %%
print(l1_dict[0])
# %%
import sklearn
for i in np.arange(0.1, 10.0, 0.1):
    cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', 
    compute_full_tree=True, distance_threshold=i)
    res = cluster.fit_predict(top_vec)
    print(f"{i}, Number of clusters = {1+np.amax(cluster.labels_)}")
    try:
        print(sklearn.metrics.silhouette_score(top_vec, res))
    except Exception as e:
        print(e)
# %%
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', 
compute_full_tree=True, distance_threshold=1.2000000000000002)

#%%
# Cluster the data
res = cluster.fit_predict(top_vec)

#%%
print(f"Number of clusters = {1+np.amax(cluster.labels_)}")

# Display the clustering, assigning cluster label to every datapoint 
print("Classifying the points into clusters:")
print(cluster.labels_)

# Display the clustering graphically in a plot
print(f"SK Learn estimated number of clusters = {1+np.amax(cluster.labels_)}")
print(" ")
#%%
# %%

l1_dict = {}
for i in range(res.shape[0]):
    if res[i] in l1_dict.keys():
        l1_dict[res[i]].append(i)
    else:
        l1_dict[res[i]] = [i]

# %%
print(l1_dict[0])
# %%
l1_dict_per_topic = {}
for i, j in l1_dict.items():
    for k in j:
        l1_dict_per_topic[k] = i
# %%
l1_topic = np.zeros((df.shape[0]))
for row in df[['topic_num']].iterrows():
    i, topic = row[0], row[1]['topic_num']
    l1_topic[i] = l1_dict_per_topic[int(topic)]
# %%
df['l1_topic'] = l1_topic
# %%
df.to_csv('..\\Results\\l1_tagged_data.csv')
# %%
