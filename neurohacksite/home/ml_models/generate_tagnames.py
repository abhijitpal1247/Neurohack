#%%
from top2vec import Top2Vec
import pickle, os, pandas as pd, numpy as np
from top2vec import Top2Vec
from Encoding import countvectorizer

#%%
df_combined = pd.read_csv('../Results/topic_added.csv')
df_combined.dropna(inplace=True)
documents = df_combined['preprocessed'].values.astype(str)
df_combined['topic_num'] = df_combined['topic_num'].astype(int)
#%%
model = Top2Vec.load('../Results/top2vec')
#%%
tag_groups = model.hierarchical_topic_reduction(100)
#%%
import pickle
with open('../Results/tag_groups.pkl','wb') as f:
    pickle.dump(tag_groups,f)

#%%
  
# Open the file in binary mode
with open('../Results/tag_groups.pkl', 'rb') as file:
    # Call load method to deserialze
    l1_tag_list = pickle.load(file)

# %%
l2_to_l1_dict = {}
for i in range(len(l1_tag_list)):
    for j in l1_tag_list[i]:
        l2_to_l1_dict[j] = i

# %%
l2_to_l1_dict.keys()
# %%
df_combined['l1_tag'] = df_combined.apply(lambda row: l2_to_l1_dict[int(row['topic_num'])], axis=1)

# %%
df_combined['l1_tag']
# %%
df_combined.to_csv('../Results/complete_tagged.csv')

#%%
model = Top2Vec.load('../Results/top2vec')
#%%
df_combined = pd.read_csv('../Results/complete_tagged.csv')
# df_combined.dropna(inplace=True)
docs = model.documents
topic_sizes, topic_nums = model.get_topic_sizes()
#%%
#%% ngram analysis

def get_top_n_gram(n,text):
    vector = countvectorizer(text, 'english', True, (n,n), None)
    bag_of_words = vector.transform(text)
    sum_of_words = bag_of_words.sum(axis=0) 
    word_freq = [(word, sum_of_words[0, idx]) for word, idx in vector.vocabulary_.items()]
    word_freq =sorted(word_freq, key = lambda x: x[1], reverse=True)
    return word_freq[0][0]

#%%
# find L2 tags
l2_tags = {}
for i in topic_nums:
    l2_tags[i] = get_top_n_gram(3,
    df_combined.loc[df_combined['topic_num']==i]['preprocessed'].values.astype(str)
    )
#%%
# find L1 tags
l1_tags = {}
for i in range(len(l1_tag_list)):
    l1_tags[i] = get_top_n_gram(3,
    df_combined.loc[df_combined['l1_tag']==i]['preprocessed'].values.astype(str)
    )
#%%
with open('../Results/tag_names.pkl','wb') as f:
    pickle.dump({'l1_tags':l1_tags,'l2_tags':l2_tags},f)
# %%
df_combined['L1_Tag'] = df_combined.apply(lambda row: l1_tags[row['l1_tag']], axis=1)
df_combined['L2_Tag'] = df_combined.apply(lambda row: l2_tags[row['topic_num']], axis=1)
#%%
# ['Number','L1_Tag','L2_Tag','Short description']
df_combined.to_csv('../Results/complete_named_tags.csv')
df = df_combined[['Number','L1_Tag','L2_Tag','description']]
df.rename(columns = {'description':'Short description'}, inplace = True)
df.to_csv('../Results/complete_named_tags_upload.csv')

# %%
