#Write your code here to find the top 30 unigram frequency among the complaints in the cleaned datafram(df_clean). 
import pandas as pd
import os
from Encoding import countvectorizer

df_topic = pd.read_csv(r'..\Results\topic_added.csv')

def get_top_unigram(text, n=30):
    vector = countvectorizer(text, 'english', True, (1,1), None)
    bag_of_words = vector.transform(text)
    sum_of_words = bag_of_words.sum(axis=0) 
    word_freq = [(word, sum_of_words[0, idx]) for word, idx in vector.vocabulary_.items()]
    word_freq =sorted(word_freq, key = lambda x: x[1], reverse=True)
    return word_freq[:n]
    #Write your code here to find the top 30 bigram frequency among the complaints in the cleaned datafram(df_clean).

def get_top_bigram(text, n=30):
    vector = countvectorizer(text, 'english', True, (2,2), None)
    bag_of_words = vector.transform(text)
    sum_of_words = bag_of_words.sum(axis=0) 
    word_freq = [(word, sum_of_words[0, idx]) for word, idx in vector.vocabulary_.items()]
    word_freq =sorted(word_freq, key = lambda x: x[1], reverse=True)
    return word_freq[:n]

def get_top_trigram(text, n=30):
    vector = countvectorizer(text, 'english', True, (3,3), None)
    bag_of_words = vector.transform(text)
    sum_of_words = bag_of_words.sum(axis=0) 
    word_freq = [(word, sum_of_words[0, idx]) for word, idx in vector.vocabulary_.items()]
    word_freq =sorted(word_freq, key = lambda x: x[1], reverse=True)
    return word_freq[:n]

def n_gram_analysis(df, topic_num):
    top_common_words = get_top_unigram(df.values.astype('U'))
    df_unigram = pd.DataFrame(top_common_words, columns = ['unigram' , 'count'])
    df_unigram.to_csv(f'..\\Results\\topic_num_{topic_num}\\unigram.csv')

    top_common_words = get_top_bigram(df.values.astype('U'))
    df_bigram = pd.DataFrame(top_common_words, columns = ['bigram' , 'count'])
    df_bigram.to_csv(f'..\\Results\\topic_num_{topic_num}\\bigram.csv')

    top_common_words = get_top_trigram(df.values.astype('U'))
    df_trigram = pd.DataFrame(top_common_words, columns = ['trigram' , 'count'])
    df_trigram.to_csv(f'..\\Results\\topic_num_{topic_num}\\trigram.csv')

for i in df_topic['topic_num'].unique():
    os.mkdir(f'..\\Results\\topic_num_{i}')
    n_gram_analysis(df_topic.loc[df_topic['topic_num']==i]['preprocessed'], i)