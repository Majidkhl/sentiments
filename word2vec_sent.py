import gensim
import nltk
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from nltk import word_tokenize, sent_tokenize
from gensim.models.word2vec import Word2Vec
import string
import numpy as np
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer




documents = 'test'

# build vocabulary and train model
model = gensim.models.Word2Vec(
            documents,
            size=150, window=10,
            min_count=2,
            workers=10,
            iter=10)