#%%
from top2vec import Top2Vec
import pandas as pd, numpy as np
import pandas as pd
import os
from Encoding import countvectorizer
#%%
model = Top2Vec.load('../Results/top2vec')
#%%
df_combined = pd.read_csv('../Results/processed_combined.csv')
df_combined.dropna(inplace=True)
docs = model.documents
topic_sizes, topic_nums = model.get_topic_sizes()
#%%
#%% ngram analysis
#Write your code here to find the top 30 unigram frequency among the complaints in the cleaned datafram(df_clean). 


# df_topic = pd.read_csv(r'..\Results\topic_added.csv')

def get_top_unigram(text, n=5):
    vector = countvectorizer(text, 'english', True, (1,1), None)
    bag_of_words = vector.transform(text)
    sum_of_words = bag_of_words.sum(axis=0) 
    word_freq = [(word, sum_of_words[0, idx]) for word, idx in vector.vocabulary_.items()]
    word_freq =sorted(word_freq, key = lambda x: x[1], reverse=True)
    return word_freq[:n]
    #Write your code here to find the top 30 bigram frequency among the complaints in the cleaned datafram(df_clean).

def get_top_bigram(text, n=5):
    vector = countvectorizer(text, 'english', True, (2,2), None)
    bag_of_words = vector.transform(text)
    sum_of_words = bag_of_words.sum(axis=0) 
    word_freq = [(word, sum_of_words[0, idx]) for word, idx in vector.vocabulary_.items()]
    word_freq =sorted(word_freq, key = lambda x: x[1], reverse=True)
    return word_freq[:n]

def get_top_trigram(text, n=5):
    vector = countvectorizer(text, 'english', True, (3,3), None)
    bag_of_words = vector.transform(text)
    sum_of_words = bag_of_words.sum(axis=0) 
    word_freq = [(word, sum_of_words[0, idx]) for word, idx in vector.vocabulary_.items()]
    word_freq =sorted(word_freq, key = lambda x: x[1], reverse=True)
    return word_freq[:n]

def n_gram_analysis(docs):
    try:
        top_common_words = get_top_unigram(docs)
    except:
        top_common_words = [('','')]
    unigrams = ','.join([i[0] for i in top_common_words])
    try:
        top_common_words = get_top_bigram(docs)
    except:
        top_common_words = [('','')]
    bigrams = ','.join([i[0] for i in top_common_words])
    try: 
        top_common_words = get_top_trigram(docs)
    except:
        top_common_words = [('','')]
    trigrams = ','.join([i[0] for i in top_common_words])

    return {'unigrams':unigrams,'bigrams':bigrams,'trigrams':trigrams}

#%%
data = []
for i in topic_nums:
    try: 
        processed_documents, _, doc_ids = model.search_documents_by_topic(topic_num=i, num_docs=5)
        documents = df_combined.iloc[doc_ids]['description']
        
    except:
        documents = ['number of documets less than 5']*5
        processed_documents = ['number of documets less than 5']*5
    docs = {f"doc{i}":doc for i,doc in enumerate(documents)}
    pro_docs = {f"pro_doc{i}":doc for i,doc in enumerate(processed_documents)}
    n_grams = n_gram_analysis(processed_documents)
    data.append({'topic_num':i}|n_grams|docs|pro_docs)

# %%
df = pd.DataFrame(data)
df.to_csv('../Results/top_5_docs_per_topic.csv')
# %%
tag_groups = model.hierarchical_topic_reduction(100)
#%%
import pickle
with open('../Results/tag_groups.pkl','wb') as f:
    pickle.dump(tag_groups,f)

#%% read pickle
with open('../Results/tag_groups.pkl','wb') as f:
    tag_groups = pickle.load(f)