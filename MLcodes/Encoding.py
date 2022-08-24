import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def countvectorizer(corpus, stopwords, lowercase, ngram_range, max_features):
    vectorizer = CountVectorizer(stop_words=stopwords, lowercase=lowercase, 
    ngram_range=ngram_range, max_features=max_features)
    vectorizer.fit(corpus)
    return vectorizer

def tfidfvectorizer(corpus, stopwords, lowercase, ngram_range, max_features):
    vectorizer = TfidfVectorizer(stop_words=stopwords, lowercase=lowercase, 
    ngram_range=ngram_range, max_features=max_features)
    vectorizer.fit(corpus)
    return vectorizer
