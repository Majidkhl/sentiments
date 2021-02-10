import pandas as pd
import json
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import codecs



import nltk
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from nltk import word_tokenize, sent_tokenize
from gensim.models.word2vec import Word2Vec
import string
import numpy as np
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer

# Download stopwords
# nltk.download()

file_encoding = 'cp1252'        # set file_encoding to the file encoding (utf8, latin1, etc.)


#Display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

# path = '//sentiments/data/'
path = 'C:/Users/a_khl/PycharmProjects/sentiments/'

hotel_infos = 'infos.json'
hotels_general = 'hotels.json'
hotel_reviews = 'reviews.json'
#
hotel_infos_fd = f"{path}{hotel_infos}"
hotels_general_fd = f"{path}{hotels_general}"
hotel_reviews_fd = f"{path}{hotel_reviews}"

file_hotels = 'C:/Users/a_khl/PycharmProjects/sentiments/data/hotels.json'
file_infos = 'C:/Users/a_khl/PycharmProjects/sentiments/data/infos.json'
file_reviews = 'C:/Users/a_khl/PycharmProjects/sentiments/data/reviews.json'

# final_stopwords_list = stopwords.words('english') + stopwords.words('french')
# stop_words = set(stopwords.words('french'))

# print(stopwords.fileids())
stop_words = nltk.corpus.stopwords.words('french')
stop_words += ['a', 'à', 'chambre', 'chambres', 'sejour', 'séjour', 'séjourné', 'séjourner', 'très', 'hotel', 'hôtel',
               'cet', 'nuit', "l'", "n'", 'nous', "j'", "d'", 'autre', 'si', "c'", 'bâtiment', 'las', 'vegas', 'nellis',
               'afb', 'endroit', 'prix', 'bon', 'strip', 'où', 'situé', 'bâtiments', 'plus', 'loin', 'jamais', 'gens',
               'chez', 'ny', 'non', 'rester', 'beaucoup', 'rv', 'jai', 'cest', 'tout', 'encore', 'lhôtel', 'ici',
                'personnel']



# # Reviews sentiment analysis using TextBlob
pol = lambda x: TextBlob(x, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment[0]
sub = lambda x: TextBlob(x, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment[1]


text1 = "Quelle belle matinée"
text2 = "C'est une voiture horrible"

pol = TextBlob(text2, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment[0]
sub = TextBlob(text2, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment[1]

print(pol)
print(sub)
#
def remove_stopwords (stopw, text_tf):
    filtered_titles = []
    stopped = []

    text_lw = [w.lower() for w in text_tf]
    for word in text_lw:
        if word in stop_words:
            stopped.append(word)
        else:
            filtered_titles.append(word)
        text = filtered_titles

    return text







