
#' ---
#' title: Text Viz w/ 2016 Election Tweets
#' output: html_document
#' ---


#+ setup

from sklearn.preprocessing import StandardScaler

import spacy
import textacy
from textacy.corpus import Corpus
from textacy.tm import topic_model
from textacy.vsm import vectorizers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys


#+ load

nlp = spacy.load("en_core_web_md")
print("loaded")


#+ load_data

######### Load Data ##################


tweets = Corpus.load(nlp, ".tweets.spacy")


#+ quant, eval=FALSE

################ DON'T USE until filtering fixed in parse.py  ############
tw_df = pd.read_pickle(".tweets.df")

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
############################################################################


# load in functions
#+ functions
exec(open('functions.py', 'r').read())


################### Specify Models #####################

# setup caching
# list of cluster_tweets objects
cachename = '.cluster.cache'

unigrams = get_unigram_lists(tweets)

#' # Soft Clustering w/ Biterms
#+
sb = Clusterer(hash(tweets), k=2, soft=True, biterm=True, idf=True)
sb = get_clusterer(sb, cachename)

labels = sb.labels
topics = sb.topics

#' ### Topic Terms
#+ results='show'
print_topics(topics, top_n=10)

#' ### Category Breakdown
#+ results='show'
print(labels.cluster.describe())

#' ### Inspect Category Exemplars
#+ results='show'
peek_clusters(tweets, sb, top_n=20)

#' ### Characteristic Terms
#+ results='show', warning=FALSE
print_top_cluster_terms(top_terms_by_cluster(unigrams, labels, 300), 3)



#' # Soft Clustering Sans idf
#+
ssi = Clusterer(hash(tweets), k=2, soft=True, biterm=True, idf=False)
ssi = get_clusterer(ssi, cachename)

labels = ssi.labels
topics = ssi.topics

#' ### Topic Terms
#+ results='show'
print_topics(topics, top_n=10)

#' ### Category Breakdown
#+ results='show'
print(labels.cluster.describe())

#' ### Inspect Category Exemplars
#+ results='show'
peek_clusters(tweets, ssi, top_n=30)

#' ### Characteristic Terms
#+ results='show', warning=FALSE
print_top_cluster_terms(top_terms_by_cluster(unigrams, labels, 300), 3)


#' # Hard Clustering w/ Biterms
#+
hb = Clusterer(hash(tweets), k=2, soft=False, biterm=True, idf=False)
hb = get_clusterer(hb, cachename)

labels = hb.labels
topics = hb.topics

#' ### Topic Terms
#+ results='show'
print_topics(topics, top_n=10)

#' ### Category Breakdown
#+ results='show'
print(labels.cluster.describe())

#' ### Inspect Category Exemplars
#+ results='show'
peek_clusters(tweets, hb, top_n=20)

#' ### Characteristic Terms
#+ results='show', warning=FALSE
print_top_cluster_terms(top_terms_by_cluster(unigrams, labels, 300), 3)


