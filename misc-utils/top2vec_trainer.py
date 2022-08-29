#%%
from top2vec import Top2Vec
import pandas as pd
emb_model = "paraphrase-multilingual-MiniLM-L12-v2"
df_combined = pd.read_csv('../Results/processed_combined.csv')

df_combined.dropna(inplace=True)
documents = df_combined['preprocessed'].values.astype(str)
#%%
model = Top2Vec(documents, embedding_model=emb_model,  speed='fast-learn')
print('model created')
model.save(f"../Results/top2vec-{emb_model}.model")
print('saved model')
#%%
model = Top2Vec.load(f'../Results/top2vec-{emb_model}.model')
#%%
docs = model.documents
#%%
topic_sizes, topic_nums = model.get_topic_sizes()
#%%
import numpy as np
doc_topics = model.get_documents_topics(doc_ids=model.document_ids)[0]
# %%
df_combined['topic_num'] = doc_topics
# %%
df_combined.to_csv(f'../Results/topic_added_top2vec-{emb_model}.csv')
print('saved to csv')
# %%