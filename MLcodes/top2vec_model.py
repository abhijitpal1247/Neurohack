#%%
from top2vec import Top2Vec
import pandas as pd
#%%
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
df_combined['data_feeded'] = docs
# %%
df_combined.to_csv('../Results/topic_added_sanity_check.csv')
# %%
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from matplotlib import pyplot as plt

for i in range(3):
    stopwords_set = set(stopwords.words())
    words_to_add=['hi','team']
    stopwords_set.update(words_to_add)
    stopwords_set.update(STOPWORDS)
    word_cloud = WordCloud(
                            background_color='blue',
                            stopwords=stopwords_set,
                            max_font_size=38,
                            max_words=38, 
                            random_state=42
                            ).generate(str(df_combined.loc[df_combined['topic_num']==i]))
    fig = plt.figure(figsize=(20,16))
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()
    fig.savefig(f'../Results/worlcloud_{i}.png')
# %%
#%%
for i in range(3):
    documents, _, _ = model.search_documents_by_topic(topic_num=i, num_docs=10)
    #%%
    stopwords_set = set(stopwords.words())
    words_to_add=['hi','team']
    stopwords_set.update(words_to_add)
    stopwords_set.update(STOPWORDS)
    word_cloud = WordCloud(
                            background_color='blue',
                            stopwords=stopwords_set,
                            max_font_size=38,
                            max_words=38, 
                            random_state=42
                            ).generate(' '.join(documents))
    fig = plt.figure(figsize=(20,16))
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()
    fig.savefig(f'../Results/worlcloud_alt_{i}.png')
# %%
