
#' ---
#' title: Text Viz w/ 2016 Election Tweets
#' output: html_document
#' ---


#+ setup

from sklearn.preprocessing import StandardScaler
#import keras


import scattertext
import spacy
import textacy
from textacy.corpus import Corpus
from textacy.tm import topic_model
from textacy.vsm import vectorizers
from sklearn import cluster

import os
#import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import functools
import pickle


#+ load, results='show'

nlp = spacy.load("en_core_web_md")
print("loaded")


#+ load_data, results='hide'

######### Load Data ##################

#with open(".tweets.spacy", "rb") as f:
#    tweets = pickle.load(f)

tweets = Corpus.load(nlp, ".tweets.spacy")

tw_df = pd.read_pickle(".tweets.df")



#+ quant, results='hide'

################ Quant vars ############

followers = np.array(tw_df.followers_count)
favorites = list(tw_df.favorite_count)
retweets = list(tw_df.retweet_count)

# normalize numeric vars!
sc = StandardScaler()
followers, favorites, retweets = sc.fit_transform([followers, favorites,
                                                  retweets])

predictand = np.array(list(zip(favorites, retweets)))

print(predictand.shape)

# check
#sns.heatmap(pd.DataFrame({'fav': favorites, 'fol':followers}).corr(),
#           annot=True)
# .15 corr



#+ simple_analysis

########### Analysis ###############


# Show some of the parsed doc/token attributes
sample = tweets[0][:]
print(sample.ents)
print(sample.root)

# lemmas?

# grab some top
#   named entities
#   words

# in: list (tweets) of lists (string words)
# out: sorted str : freq map
def most_common(corpus, n, strings=False):
    word_freq = dict()
    # count
    for tw in corpus:
        for w in tw:
            try:
                word_freq[w] += 1
            except KeyError:
                word_freq[w] = 1
    # filter stop words
    real_words = list(filter(lambda x: not (nlp.vocab[x[0]].is_stop or
                                            nlp.vocab[x[0]].is_punct or
                                            nlp.vocab[x[0]].is_space),
                             word_freq.items()))
    # normalize freqs
    tot = len(corpus)
    real_words = [(word[0], float(word[1] / tot)) for word in real_words]
    # sort
    top_words = sorted(real_words, key=lambda x: x[1], reverse=True)[:n]
    if strings:
        top_words = [x[0] for x in top_words]
    # pull text view (already done)
    #top_words = list(map(lambda x: nlp.vocab[x[0]].text, top_words))
    return top_words


#+ topic_mod

# we want both!!!
# _.to_terms_list helper
def lower_lemma(token):
    return token.lemma_.lower()

tok_tweets = [list(doc._.to_terms_list(ngrams=1, as_strings=True,
                                       normalize=lower_lemma,
                                       filter_stops=True, filter_punct=True,
                                       filter_nums=True))
              for doc in tweets]


vectorizer = vectorizers.Vectorizer(apply_idf=True, tf_type='binary', max_df=10)
tweet_matrix = vectorizer.fit_transform(tok_tweets)

model = topic_model.TopicModel('lda', n_topics=30)
model.fit(tweet_matrix)
topic_matrix = model.get_doc_topic_matrix(tweet_matrix)
topics = model.top_topic_terms(vectorizer.id_to_term)

# fuck... half of topics are non-english
# I could... toss tweets whose contents are mostly out of vocabulary?
#   do that in parse.py

# I tried to drop tweets w/ less than two recognizable english words
# now it's... worse?

# tried dropping tw w/ max 2 unrecognizable words
# still v bad

# maybe... parse as token; check if not symbol and in vocab?
# wait, top words are still the same and in english. why is topic model wack??
#   we're refitting vectorizer...

# this is all w/ NMF. try LSA
# LSA is same; try LDA
# LDA gives english terms, nice
# holy shit, so different

#+ cluster
# Annnyway, k-means

kmeans = cluster.KMeans(n_clusters=2)
# docs are rows in topic matrix, nice
#labels = kmeans.fit_predict(topic_matrix)
labels = kmeans.fit_predict(tweet_matrix)

# I suspect the clustering ATMO
#   try term-doc r/t topic-doc?
#     gives tiny category 0...


# split corpora
class0 = [tok_tweets[i] for i in np.where(labels == 0)[0]]
class1 = [tok_tweets[i] for i in np.where(labels == 1)[0]]

# re-check most common words
print("0:")
print(most_common(class0, 20, strings=True))
print("1:")
print(most_common(class1, 20, strings=True))

# scattertext vis
# ugh scattertext abstracts out some of the spacy stuff

#+ viz
# diy viz???
# compute term frequency in both corpora
common0 = pd.DataFrame(most_common(class0, 200), columns=["word", "freq0"])
common1 = pd.DataFrame(most_common(class1, 200), columns=["word", "freq1"])
# stitch
common = pd.merge(common0, common1, how='inner')
# toss most common terms
#common = common.query('freq0 < .1')
# vis
sns.relplot(data=common, x='freq0', y='freq1')
plt.show()


