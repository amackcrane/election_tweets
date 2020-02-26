
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

import os
#import seaborn as sns
import numpy as np
import pandas as pd
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



#+ analysis

########### Analysis ###############


# Show some of the parsed doc/token attributes
sample = tweets[0][:]
print(sample.ents)
print(sample.root)

# lemmas?

# grab some top
#   named entities
#   words

# get list of words from vocab
# orth : freq map
word_freq = dict()
# count
for tw in tweets:
    for w in tw:
        try:
            word_freq[w.orth] += 1
        except KeyError:
            word_freq[w.orth] = 1

# filter stop words
real_words = list(filter(lambda x: not (nlp.vocab[x[0]].is_stop or
                                        nlp.vocab[x[0]].is_punct or
                                        nlp.vocab[x[0]].is_space),
                         word_freq.items()))

# sort
top_words = sorted(real_words, key=lambda x: x[1], reverse=True)[:50]
# pull text view
top_words = list(map(lambda x: nlp.vocab[x[0]].text, top_words))
print(top_words)


# get a list of tf-idf outliers?
#   no I think we need clusters first...
# hrm... you could get tf-idf outliers
#   but here, because docs are so short, that'll just be rare words
# recall tf-idf is text-wise

# try a k-medoids alg w/ similarity()???
# actually duh that's a continuous space, use k-means





