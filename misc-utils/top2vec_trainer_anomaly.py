#%% import 
from top2vec import Top2Vec
import pandas as pd
import sys, os
sys.path.insert(1, '../MLcodes')
from Encoding import countvectorizer
import preprocessing as custom_preprocessing
import pickle, os, pandas as pd, numpy as np
#%%
emb_model = "doc2vec"
ticket_dump_file = '../official_data/Ticket Trend Data/Trend_Data_1.xlsx'
preprocessed_ticket_dump_file = '../Results/anomaly-tickettrend-preprocessed.csv'
l2_topics_file = f'../Results/topic_added_top2vec_anomaly-{emb_model}.csv'
model_file = f'../Results/top2vec_anomaly-{emb_model}.model'
tag_groups_file = f'../Results/tag_groups_anomaly_{emb_model}.pkl'
tag_names_file = f'../Results/tag_names_anomaly_{emb_model}.pkl'
final_tagged_file = f'../Results/final_tagged_anomaly_{emb_model}.csv'
#%%
def preprocess_text(sample):
    sample=custom_preprocessing.contraction_expander(sample)
    sample=custom_preprocessing.url_remover(sample)
    sample=custom_preprocessing.email_remover(sample)
    sample=custom_preprocessing.removal_html_tags(sample)
    sample=custom_preprocessing.digit_remover(sample)
    sample=custom_preprocessing.special_character_removal(sample)
    sample=custom_preprocessing.stopwords_removal(sample)
    sample=custom_preprocessing.lemmatize(sample)
    return sample

#%% load data
if not os.path.exists(preprocessed_ticket_dump_file):
    print('pre-processing')
    df = pd.read_excel(ticket_dump_file)
    df['Description'] = pd.Series([' '.join(row.astype(str)) for row in df[['Short description','Task type','Assignment group',]].fillna('').values], index=df.index)
    print('data loaded')
    df_combined = df[['Number','Description','Created','Resolved','Resolution time','Reassignment count']].copy()
    df_combined['preprocessed'] = df_combined['Description'].apply(lambda x: preprocess_text(x))
    df_combined.dropna(inplace=True)
    #save preprocessed text
    df_combined.to_csv(preprocessed_ticket_dump_file)
    print('pre-processing completed')
#%% load preprocessed data
df_combined = pd.read_csv(preprocessed_ticket_dump_file)
documents = df_combined['preprocessed'].values.astype(str)
#%%
model = Top2Vec(documents, embedding_model=emb_model,  speed='fast-learn')
print('model created')
model.save(model_file)
print('saved model')
#%%
model = Top2Vec.load(model_file)
#%%
topic_sizes, topic_nums = model.get_topic_sizes()
#%%
doc_topics = model.get_documents_topics(doc_ids=model.document_ids)[0]
df_combined['topic_num'] = doc_topics
df_combined.to_csv(l2_topics_file)
print('saved to csv')
# %%

#Generate tag names


#%%
df_combined = pd.read_csv(l2_topics_file)
df_combined.dropna(inplace=True)
documents = df_combined['preprocessed'].values.astype(str)
df_combined['topic_num'] = df_combined['topic_num'].astype(int)
#%%
model = Top2Vec.load(model_file)
#%%
tag_groups = model.hierarchical_topic_reduction(min(30,len(topic_nums)-1))
#%%
import pickle
with open(tag_groups_file,'wb') as f:
    pickle.dump(tag_groups,f)
#%%
# Open the file in binary mode
with open(tag_groups_file, 'rb') as file:
    # Call load method to deserialze
    l1_tag_list = pickle.load(file)

# %%
l2_to_l1_dict = {}
for i in range(len(l1_tag_list)):
    for j in l1_tag_list[i]:
        l2_to_l1_dict[j] = i
l2_to_l1_dict.keys()
# %%
df_combined['l1_tag'] = df_combined.apply(lambda row: l2_to_l1_dict[int(row['topic_num'])], axis=1)
# df_combined.dropna(inplace=True)
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
with open(tag_names_file,'wb') as f:
    pickle.dump({'l1_tags':l1_tags,'l2_tags':l2_tags},f)
# %%
df_combined['L1_Tag'] = df_combined.apply(lambda row: l1_tags[row['l1_tag']], axis=1)
df_combined['L2_Tag'] = df_combined.apply(lambda row: l2_tags[row['topic_num']], axis=1)
#%%
# ['Number','L1_Tag','L2_Tag','Short description']
df_combined.to_csv(final_tagged_file)
print('completed successfully')
