

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering
from sklearn.decomposition import TruncatedSVD
from textacy import Corpus
from textacy.tm import TopicModel
import spacy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# fix wd
path = ""
if os.path.basename(os.getcwd()) == "scripts":
    path = "../"

# helpers
exec(open(f"{path}scripts/topic_helpers.py", 'r').read())
exec(open(f"{path}scripts/graph_helpers.py", 'r').read())
exec(open(f"{path}scripts/cluster_helpers.py", 'r').read())
    
tweets = Corpus.load("en_core_web_sm", f"{path}.data/tweets.spacy")
print("Tweets loaded")
    
# draw in dataframe
tw_df = pd.read_pickle(f"{path}.data/tweets.df")
print("df loaded")

# Vectorize
binary=False
unigrams = get_unigram_lists(tweets, binary)
rec_handles = get_recurring_handles(tw_df, 3)
user_bags = word_bags_by_handle(unigrams, tw_df, binary)

# 100k words w/o min_df
# 3000 w/ min_df = 30
cv = CountVectorizer(analyzer=lambda x: x, min_df = 30)

doc_term = cv.fit_transform(unigrams)

doc_term_rec = get_matrix(rec_handles, user_bags, cv)

# reduce dimensionality
#topic = TopicModel("lsa", n_topics=100)

#topic.fit(doc_term)
#doc_topic = topic.transform(doc_term)

# co-clustering
cospec = SpectralCoclustering(n_clusters=6)

# Doesn't work; suspect too sparse? something breaks w/ k-means on eigenvectors
#cospec.fit(doc_term_rec.todense())

x = np.random.random([100,200])
cospec.fit(x)
draw_matrix(cospec, x)
plt.savefig("visualization/cospec")

# visualize

draw_matrix(cospec, doc_term_rec, cv)
plt.savefig("visualization/cospec")

# bi-clustering

bispec = SpectralBiclustering(n_clusters=(10,15), method='log')

bispec.fit(doc_term_rec.todense())

_,ax = plt.subplots(tight_layout=True)
draw_image_matrix(bispec, doc_term_rec, cv, ax)
plt.savefig("visualization/bispec", dpi=500)
plt.close('all')




