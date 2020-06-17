

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

# helpers from topic.py
exec(open(f"{path}scripts/topic_helpers.py", 'r').read())
    
tweets = Corpus.load("en_core_web_sm", f"{path}.data/tweets.spacy")
print("Tweets loaded")
    
# draw in dataframe
tw_df = pd.read_pickle(f"{path}.data/tweets.df")
print("df loaded")

# Vectorize
unigrams = get_unigram_lists(tweets)
cv = TfidfVectorizer(analyzer=lambda x: x)

doc_term = cv.fit_transform(unigrams)

# reduce dimensionality
topic = TopicModel("lsa", n_topics=100)

topic.fit(doc_term)
doc_topic = topic.transform(doc_term)

# spectral clustering
cospec = SpectralCoclustering(n_clusters=10)

cospec.fit(doc_topic)


def sum_cluster(clusterer, doc=True, n=10):
    matrix = clusterer.rows_ if doc else clusterer.columns_
    title = "Docs" if doc else "Topics"
    clusters = range(len(matrix))
    doc_indices = np.arange(len(matrix[0]))
    for c in clusters:
        # bool indicating cluster membership
        doc_where = matrix[c]
        c_indices = doc_indices[doc_where]
        try:
            c_indices = np.random.choice(c_indices, size=n, replace=False)
        except ValueError:
            pass # tried to sample w/ 'size' > len
        to_print = f"*** {title} in Cluster {c} ***\n\n"
        for i,ind in enumerate(c_indices):
            if doc:
                thing = tweets[ind]
            else:
                thing = list(topic.top_topic_terms(cv.get_feature_names(), top_n=6))[ind]
            to_print += f"{i}. {thing}\n"
        print(to_print)


# visualize






