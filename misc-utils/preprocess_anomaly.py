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
preprocessed_ticket_dump_file = '../Results/anomaly-tickettrend-preprocessed-bak.csv'
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