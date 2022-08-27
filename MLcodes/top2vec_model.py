#%%
from top2vec import Top2Vec
import pandas as pd

df_combined = pd.read_csv('../Results/processed_combined.csv')

df_combined.dropna(inplace=True)
documents = df_combined['preprocessed'].values.astype(str)
#%%
model = Top2Vec(documents, embedding_model='doc2vec',  speed='fast-learn')
model.save("top2vec.model")

#%%
model = Top2Vec.load('../Results/top2vec')
#%%
docs = model.documents
#%%
topic_sizes, topic_nums = model.get_topic_sizes()
#%%
import numpy as np
doc_topics = np.zeros(docs.shape[0])
for i in range(docs.shape[0]):
    doc_topics[i] = model.get_documents_topics([i])[0][0]
# %%
df_combined['topic_num'] = doc_topics
# %%
df_combined.to_csv('../Results/topic_added.csv')
# %%
