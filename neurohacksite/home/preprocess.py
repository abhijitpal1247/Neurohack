import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from textblob import TextBlob
 
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

 
def contraction_expander(text):
    contractions = { 
        "ain\\'t": "am not / are not / is not / has not / have not",
        "aren\\'t": "are not / am not",
        "can\\'t": "cannot",
        "can\\'t\\'ve": "cannot have",
        "\\'cause": "because",
        "could\\'ve": "could have",
        "couldn\\'t": "could not",
        "couldn\\'t\\'ve": "could not have",
        "didn\\'t": "did not",
        "doesn\\'t": "does not",
        "don\\'t": "do not",
        "hadn\\'t": "had not",
        "hadn\\'t\\'ve": "had not have",
        "hasn\\'t": "has not",
        "haven\\'t": "have not",
        "he\\'d": "he had / he would",
        "he\\'d\\'ve": "he would have",
        "he\\'ll": "he shall / he will",
        "he\\'ll\\'ve": "he shall have / he will have",
        "he\\'s": "he has / he is",
        "how\\'d": "how did",
        "how\\'d\\'y": "how do you",
        "how\\'ll": "how will",
        "how\\'s": "how has / how is / how does",
        "I\\'d": "I had / I would",
        "I\\'d\\'ve": "I would have",
        "I\\'ll": "I shall / I will",
        "I\\'ll\\'ve": "I shall have / I will have",
        "I\\'m": "I am",
        "I\\'ve": "I have",
        "isn\\'t": "is not",
        "it\\'d": "it had / it would",
        "it\\'d\\'ve": "it would have",
        "it\\'ll": "it shall / it will",
        "it\\'ll\\'ve": "it shall have / it will have",
        "it\\'s": "it has / it is",
        "let\\'s": "let us",
        "ma\\'am": "madam",
        "mayn\\'t": "may not",
        "might\\'ve": "might have",
        "mightn\\'t": "might not",
        "mightn\\'t\\'ve": "might not have",
        "must\\'ve": "must have",
        "mustn\\'t": "must not",
        "mustn\\'t\\'ve": "must not have",
        "needn\\'t": "need not",
        "needn\\'t\\'ve": "need not have",
        "o\\'clock": "of the clock",
        "oughtn\\'t": "ought not",
        "oughtn\\'t\\'ve": "ought not have",
        "shan\\'t": "shall not",
        "sha\\'n\\'t": "shall not",
        "shan\\'t\\'ve": "shall not have",
        "she\\'d": "she had / she would",
        "she\\'d\\'ve": "she would have",
        "she\\'ll": "she shall / she will",
        "she\\'ll\\'ve": "she shall have / she will have",
        "she\\'s": "she has / she is",
        "should\\'ve": "should have",
        "shouldn\\'t": "should not",
        "shouldn\\'t\\'ve": "should not have",
        "so\\'ve": "so have",
        "so\\'s": "so as / so is",
        "that\\'d": "that would / that had",
        "that\\'d\\'ve": "that would have",
        "that\\'s": "that has / that is",
        "there\\'d": "there had / there would",
        "there\\'d\\'ve": "there would have",
        "there\\'s": "there has / there is",
        "they\\'d": "they had / they would",
        "they\\'d\\'ve": "they would have",
        "they\\'ll": "they shall / they will",
        "they\\'ll\\'ve": "they shall have / they will have",
        "they\\'re": "they are",
        "they\\'ve": "they have",
        "to\\'ve": "to have",
        "wasn\\'t": "was not",
        "we\\'d": "we had / we would",
        "we\\'d\\'ve": "we would have",
        "we\\'ll": "we will",
        "we\\'ll\\'ve": "we will have",
        "we\\'re": "we are",
        "we\\'ve": "we have",
        "weren\\'t": "were not",
        "what\\'ll": "what shall / what will",
        "what\\'ll\\'ve": "what shall have / what will have",
        "what\\'re": "what are",
        "what\\'s": "what has / what is",
        "what\\'ve": "what have",
        "when\\'s": "when has / when is",
        "when\\'ve": "when have",
        "where\\'d": "where did",
        "where\\'s": "where has / where is",
        "where\\'ve": "where have",
        "who\\'ll": "who shall / who will",
        "who\\'ll\\'ve": "who shall have / who will have",
        "who\\'s": "who has / who is",
        "who\\'ve": "who have",
        "why\\'s": "why has / why is",
        "why\\'ve": "why have",
        "will\\'ve": "will have",
        "won\\'t": "will not",
        "won\\'t\\'ve": "will not have",
        "would\\'ve": "would have",
        "wouldn\\'t": "would not",
        "wouldn\\'t\\'ve": "would not have",
        "y\\'all": "you all",
        "y\\'all\\'d": "you all would",
        "y\\'all\\'d\\'ve": "you all would have",
        "y\\'all\\'re": "you all are",
        "y\\'all\\'ve": "you all have",
        "you\\'d": "you had / you would",
        "you\\'d\\'ve": "you would have",
        "you\\'ll": "you shall / you will",
        "you\\'ll\\'ve": "you shall have / you will have",
        "you\\'re": "you are",
        "you\\'ve": "you have"
    }
    for word in text.split():
        if word.lower().strip() in contractions.keys():
            text = text.replace(word, contractions[word.lower().strip()])
    return text

def stopwords_removal(text):
    filtered_text = []
    stopwords_set = set(stopwords.words())
    words_to_add=['hi','team']
    stopwords_set.update(words_to_add)
    for word in text.split():
        if word.lower() not in stopwords_set:
            filtered_text.append(word)
    return " ".join(filtered_text)


def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text=[]
    for word in text.lower().split():
        lemmatized_text.append(lemmatizer.lemmatize(word))
    return " ".join(lemmatized_text)

def removal_html_tags(text):
    return BeautifulSoup(text, 'html.parser').get_text()

def email_remover(text):
    pattern = r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+"
    email_removed_text = re.sub(pattern=pattern, repl=' ', string=text)
    return email_removed_text

def incident_id_remover(text):
    pattern = r"INC\d*"
    incident_id_removed_text = re.sub(pattern, ' ', text)
    return incident_id_removed_text

def digit_remover(text):
    pattern = r"\d+"
    digit_removed_text = re.sub(pattern, ' ', text)
    return digit_removed_text

def url_remover(text):
    filtered_url_text = [l for l in text.split() if not urlparse(l).scheme]
    url_removed_txt = ' '.join(filtered_url_text)
    return url_removed_txt

def noun_extraction(text):
    blob = TextBlob(text)
    noun_phrases = []
    for nouns in blob.noun_phrases:
        noun_phrases.append(nouns)
    return " ".join(noun_phrases)

def special_character_removal(text):
    cleanString = re.sub('\W+',' ', text )
    return cleanString
