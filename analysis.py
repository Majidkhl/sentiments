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

# df_hotel = pd.read_json(hotels_general _fd, lines=True)
# df_info = pd.read_json(hotel_infos_fd, lines=True)
# df_review = pd.read_json(hotel_reviews_fd, lines=True)

df_hotel = pd.read_json(codecs.open(file_hotels, 'r', 'utf-8'), lines=True)
df_info = pd.read_json(codecs.open(file_infos, 'r', 'utf-8'), lines=True)
df_review = pd.read_json(codecs.open(file_reviews, 'r', 'utf-8'), lines=True)


# Merging the datframes
hotel_all = df_info.merge(df_hotel, left_on=['name'], right_on=['name'], how="left")

df_review.dropna(inplace=True)


# # Distributions of the ratings: general
rating_percent = round(100 * df_review['rating'].value_counts()/len(df_review['comment']), 1)
plt.title('Distribution of ratings: general appreciation', fontsize=14, fontweight='bold')
rating_percent.plot.bar()
plt.show()

# Classification of the ratings: satisfait, pas satisfait, neutre
conditions = [
    (df_review['rating'] > 3),
    (df_review['rating'] == 3),
    (df_review['rating'] < 3)
]
classify = ['satisfait', 'neutre', 'pas satisfait']

df_review['user_sentiment'] = np.select(conditions, classify)

# Distributions of the ratings: cleanliness, service, value_for_money
rating = 'value_for_money'
rating_percent = round(100 * df_info[rating].value_counts()/len(df_info[rating]), 1)
print(rating_percent)
plt.title(f"Distribution of ratings: {rating}", fontsize=14, fontweight='bold')
rating_percent.plot.bar()
plt.show()

# # Grouping the entries by hotel names and joining the titles together
df_titles = df_review[['name', 'title']]
df_titles = df_titles.groupby(['name'], as_index = False).agg({'title': ' '.join})


# Grouping the entries by hotel names and joining the comments together
df_comments = df_review[['name', 'comment']]
df_comments = df_comments.groupby(['name'], as_index = False).agg({'comment': ' '.join})

d_frame = df_comments
column_text = 'comment'

print(d_frame.head())

# Create and generate a word cloud image based on the titles
# text = d_frame[column_text][2]
hotel_name = 'ARIA Resort & Casino'
text = d_frame.loc[d_frame['name'] == hotel_name][column_text][0]


# Removing punctuation
text = text.translate(str.maketrans('','',string.punctuation))

# Using NLTK and removing stopwords
toknizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
text_token = toknizer.tokenize(text)
text_token_lw = [w.lower() for w in text_token]

# filtered_titles = [w.lower() for w in text_token if w not in stop_words]
filtered_titles = []
stopped = []

for word in text_token_lw:
    if word in stop_words:
        stopped.append(word)
    else:
        filtered_titles.append(word)

new_text = ' '.join([word for word in filtered_titles])
allWordDist = nltk.FreqDist(filtered_titles)


# Generating a wordcloud
wordcloud = WordCloud(width=1400, height=800, max_words=100).generate(new_text)

# Display the generated image:
plt.figure()
plt.title(hotel_name, fontsize=14, fontweight='bold')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



def tokenizer_function(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    toknizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
    text_token = toknizer.tokenize(text)

    text_token_lw = [w.lower() for w in text_token]
    filtered_titles = []
    stopped = []

    for word in text_token_lw:
        if word in stop_words:
            stopped.append(word)
        else:
            filtered_titles.append(word)

    new_text = ' '.join([word for word in filtered_titles])

    allWordDist = nltk.FreqDist(filtered_titles)

    return allWordDist.most_common(1)[0][0]


# Updating datframe adding most frequent word in titles

# Based on titles
df_titles['most_fw'] = df_titles['title'].apply(tokenizer_function)

# Based on reviews
df_comments['most_fw'] = df_comments['comment'].apply(tokenizer_function)

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









