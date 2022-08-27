#%%
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from preprocessing import contraction_expander
from tqdm import tqdm
from preprocessing import contraction_expander,url_remover,email_remover,removal_html_tags,digit_remover,special_character_removal,stopwords_removal,lemmatize, noun_extraction
from matplotlib import pyplot as plt                 
from preprocessing import special_character_removal
from nltk.corpus import stopwords
from Encoding import tfidfvectorizer
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
df_combined = pd.read_csv('..\Results\processed_combined.csv')
#%%
df_combined.dropna(inplace=True)
tfidfvect =  tfidfvectorizer(max_df = 0.95, min_df=2,corpus = df_combined['preprocessed'], stopwords=stopwords_set, lowercase=True, ngram_range=(1, 3), max_features=None)
#%%
dtm=tfidfvect.transform(df_combined['preprocessed'])
# %%
print(dtm.shape)
#%%
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=5000, n_iter=7, random_state=42)
svd.fit(dtm)
#%%
#print(sum(svd.explained_variance_))
# %%
tfidf_feature_names = tfidfvect.get_feature_names()
#%%
# Run NMF
from sklearn.decomposition import NMF, LatentDirichletAllocation
nmf_model = NMF(n_components=5, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(dtm)
nmf_W = nmf_model.transform(dtm)
nmf_H = nmf_model.components_

# %%
