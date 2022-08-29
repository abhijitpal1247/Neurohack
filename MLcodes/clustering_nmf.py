#%%
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from preprocessing import contraction_expander
from tqdm import tqdm
from preprocessing import contraction_expander,url_remover,email_remover,removal_html_tags,digit_remover,special_character_removal,stopwords_removal,lemmatize, noun_extraction
from matplotlib import pyplot as plt                 
from preprocessing import special_character_removal
from nltk.corpus import stopwords
from Encoding import tfidfvectorizer, word2vecmodel
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf
#%%
tqdm.pandas()
#%%
df_combined = pd.read_csv('..\official_data\Tag_Data\data_combined.csv')
df_combined.head()
#%%

df_combined.dropna(inplace=True)

def preprocess(sample):
    sample=contraction_expander(sample)
    sample=url_remover(sample)
    sample=email_remover(sample)
    sample=(removal_html_tags(sample))
    sample=digit_remover(sample)
    sample=special_character_removal(sample)
    sample=stopwords_removal(sample)
    sample=lemmatize(sample)
    #sample=noun_exctraction(sample)

    return sample
# %%
df_combined['preprocessed'] = df_combined.progress_apply(lambda row: preprocess(row['description']), axis=1)

# %%
df_combined.head()
# %%
df_combined['noun_extracted'] = df_combined.progress_apply(lambda row: noun_extraction(row['preprocessed']), axis=1)
df_combined.head()
#%%
df_combined.isnull().sum()
#%%
df_combined.to_csv('..\Results\processed_combined.csv')
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
                         ).generate(str(df_combined['noun_extracted']))

fig = plt.figure(figsize=(20,16))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()
plt.savefig('..\Results\worlcloud.jpg')
#%%
#Write your code here to initialise the TfidfVectorizer 
df_combined.head()
#%%
df_combined.isnull().sum()
#%%
df_combined = pd.read_csv('../Results/processed_combined.csv')
#%%
df_combined.dropna(inplace=True)
#%%
tfidfvect =  tfidfvectorizer(max_df = 0.95, min_df=2,corpus = df_combined['preprocessed'], stopwords=stopwords_set, lowercase=True, ngram_range=(1, 3), max_features=None)
#%%
dtm=tfidfvect.transform(df_combined['preprocessed'])
# %%
print(dtm.shape)
#%%
#from sklearn.decomposition import TruncatedSVD
#svd = TruncatedSVD(n_components=5000, n_iter=7, random_state=42)
#svd.fit(dtm)
#%%
#print(sum(svd.explained_variance_))
# %%
tfidf_feature_names = tfidfvect.get_feature_names()
#%%
no_topics = 0 #@param {type:"integer"}

no_top_words = 5 #@param {type:"integer"}

no_top_documents = 3 #@param {type:"integer"}
#%%
# Run NMF
from sklearn.decomposition import NMF, LatentDirichletAllocation
nmf_model = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(dtm)
nmf_W = nmf_model.transform(dtm)
nmf_H = nmf_model.components_
#%%
def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print("Topic %d:" % (topic_idx))
        print(" ".join([ (feature_names[i] + " (" + str(topic[i].round(2)) + ")")
          for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print(str(doc_index) + ". " + documents[doc_index])

# %%
print("NMF Topics")
display_topics(nmf_H, nmf_W, tfidf_feature_names, df_combined['preprocessed'], no_top_words, no_top_documents)
print("--------------")
# %%
import pyLDAvis.sklearn

pyLDAvis.enable_notebook()

pyLDAvis_data = pyLDAvis.sklearn.prepare(nmf_model, dtm, tfidfvect)
# Visualization can be displayed in the notebook
pyLDAvis.display(pyLDAvis_data)
# %%
df_combined['tokens'] = df_combined.apply(lambda row: row['preprocessed'].split(), axis=1)
texts = df_combined['tokens']
#%%
# Create a dictionary
# In gensim a dictionary is a mapping between words and their integer id
dictionary = Dictionary(texts)
#%%
# Filter out extremes to limit the number of features
dictionary.filter_extremes(
    no_below=3,
    no_above=0.85,
    keep_n=5000
)
#%%
# Create the bag-of-words format (list of (token_id, token_count))
corpus = [dictionary.doc2bow(text) for text in texts]
#%%
# Create a list of the topic numbers we want to try
topic_nums = list(np.arange(5, 75 + 1, 5))

# Run the nmf model and calculate the coherence score
# for each number of topics
coherence_scores = []

for num in tqdm(topic_nums):
    nmf = Nmf(
        corpus=corpus,
        num_topics=num,
        id2word=dictionary,
        chunksize=2000,
        passes=5,
        kappa=.1,
        minimum_probability=0.01,
        w_max_iter=300,
        w_stop_condition=0.0001,
        h_max_iter=100,
        h_stop_condition=0.001,
        eval_every=10,
        normalize=True,
        random_state=42
    )
    
    
    # Run the coherence model to get the score
    cm = CoherenceModel(
        model=nmf,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    
    coherence_scores.append(round(cm.get_coherence(), 5))
#%%
# Get the number of topics with the highest coherence score
from operator import itemgetter
scores = list(zip(topic_nums, coherence_scores))
best_num_topics = sorted(scores, key=itemgetter(1), reverse=True)[0][0]

print(best_num_topics)

# %%
fig = plt.figure(figsize=(15, 7))

plt.plot(
    topic_nums,
    coherence_scores,
    linewidth=3,
    color='#4287f5'
)

plt.xlabel("Topic Num", fontsize=14)
plt.ylabel("Coherence Score", fontsize=14)
plt.title('Coherence Score by Topic Number - Best Number of Topics: {}'.format(best_num_topics), fontsize=18)
plt.xticks(np.arange(5, max(topic_nums) + 1, 5), fontsize=12)
plt.yticks(fontsize=12)

plt.show()
# %%
texts = df_combined['tokens']

# Create the tfidf weights
tfidf_vectorizer = tfidfvect

tfidf = dtm

# Save the feature names for later to create topic summaries
tfidf_fn = tfidf_feature_names
#%%
from sklearn.decomposition import NMF, LatentDirichletAllocation
# Run the nmf model
nmf = NMF(
    n_components=best_num_topics,
    init='nndsvd',
    max_iter=500,
    l1_ratio=0.0,
    solver='cd',
    alpha=0.0,
    tol=1e-4,
    random_state=42
).fit(tfidf)

#%%
def top_words(topic, n_top_words):
    return topic.argsort()[:-n_top_words - 1:-1]  
#%%
def topic_table(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        t = (topic_idx)
        topics[t] = [feature_names[i] for i in top_words(topic, n_top_words)]
    return pd.DataFrame(topics)
#%%
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
def whitespace_tokenizer(text): 
    pattern = r"(?u)\b\w\w+\b" 
    tokenizer_regex = RegexpTokenizer(pattern)
    tokens = tokenizer_regex.tokenize(text)
    return tokens

#%%
# Funtion to remove duplicate words
def unique_words(text): 
    ulist = []
    [ulist.append(x) for x in text if x not in ulist]
    return ulist

#%%
docweights = nmf.transform(tfidf_vectorizer.transform(df_combined['preprocessed']))

n_top_words = 8

topic_df = topic_table(
    nmf,
    tfidf_fn,
    n_top_words
).T

# Cleaning up the top words to create topic summaries
topic_df['topics'] = topic_df.apply(lambda x: [' '.join(x)], axis=1) # Joining each word into a list
topic_df['topics'] = topic_df['topics'].str[0]  # Removing the list brackets
topic_df['topics'] = topic_df['topics'].apply(lambda x: whitespace_tokenizer(x)) # tokenize
topic_df['topics'] = topic_df['topics'].apply(lambda x: unique_words(x))  # Removing duplicate words
topic_df['topics'] = topic_df['topics'].apply(lambda x: [' '.join(x)])  # Joining each word into a list
topic_df['topics'] = topic_df['topics'].str[0]  # Removing the list brackets
#%%
topic_df.head(30)

# %%
# word2vec
df_combined.dropna(inplace=True)
w2vmodel=word2vecmodel(df_combined['preprocessed'],
stopwords=[],lowercase=True,skip_gram=1,
vector_size=500,window_size=5, epochs=30)
# %%
w2vmodel.save("../Results/word2vec.model")
# %%
word_vectors = w2vmodel.wv
word_vectors.save("../Results/word2vec.wordvectors")
# %%
