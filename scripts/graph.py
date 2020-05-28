

#' ---
#' title: Semantic and Pragmatic Relatedness from Matrices and Graphs
#' output:
#'   pdf_document:
#'     latex_engine: xelatex
#' bibliography: twt.bib
#' ---
#'

#'
#' General-purpose language models learn embeddings for words that correspond to their syntactic niche and lexical meaning, since they're trained on broad corpora where words are used in a variety of contexts. This works well for tasks such as sentiment analysis and text classification where words can be largely understood in their common sense. But how do we approach a situation where, say, social actors marshal support for a political agenda using class-coded appeals to emotion? Pragmatics and social meanings -- how language is interpreted in particular social contexts; the ways it encodes identities and mediates relations -- subimpose beneath lexical meaning to sublimate delicate negotiations.
#'
#' As representative of a general-purpose language model, we'll consider the word vectors included with Spacy's 'en-core-web-md'^[what are these], and we'll compare these with a few representations based on document-term matrices intended to capture the social situation of words. Our data comprise a corpus of tweets scraped based on the hashtag #Election2016 ^[over what time interval] and filtered to english.
#'
#'
#' # Models for Pragmatics
#'
#' A central aspect of a word embedding model is the variance it attends to. Most language models ^[afaik...] are based on patterns of word co-occurrence in small textual envelopes -- a sentence, say, or a handful of words on either side of the target word. Alternatively, some models look at word-document coincidences. These distinctions don't correspond to social distinctions: this sentence vs. that sentence in a document is generally not a social distinction, nor is this wikipedia article vs. that one.
#'
#' 
#' Networks are [@HellstenLeydesdorff2019]
#'
#' 
#'
#' 
#' One motivation for looking at two-mode graphs is that later on we can try embedding handles and words in the same 2-d space; this is a nice way of making more concrete the social valence induced in these embeddings without being heavy-handed.
#'
#' 

                                        #+ setup

import spacy
from sklearn.manifold import TSNE, MDS, Isomap, SpectralEmbedding
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from textacy.corpus import Corpus
from umap import UMAP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import hashlib

# fix wd
path = ""
if os.path.basename(os.getcwd()) == "scripts":
    path = "../"

# checksums for cache invalidation
graph_helpers = open(path+"scripts/graph_helpers.py").read()
topic_helpers = open(path+"scripts/topic_helpers.py").read()
check_helpers = hash(graph_helpers) ^ hash(topic_helpers)

                                        #+ helpers, cache.extra=py$check_helpers
# load helper functions
exec(graph_helpers)
exec(topic_helpers)

                                        #+ load_corpus

temp = Corpus.load("en_core_web_sm", path+".data/tweets.spacy")
print("corpus loaded")

check_corpus = hash(temp)

                                        #+ corpus, cache.extra=py$check_corpus
tweets = temp

                                        #+ unigrams

# grab a list of interesting words
unigrams = get_unigram_lists(tweets)
#n_top = 50
#ref_words = common_words(tweets, n_top)

#'
#' # Expectations
#'
#' Since this is an open-ended task, we need to be clear about what results count as meaningful so that we're not just fitting our metrics to our favorite representations. Some things we should expect to see following from this conception of lexical and pragmatic meanings are:
#'
#' * vector embeddings show some syntactic clustering, while pragmatic embeddings cluster syntactically distinct words together when they share a political valence
#' * differences emerge for words with similar generic/topical meaning but particular partisan sense
#'
#' Let's pick out some words to pay particular attention to. We'll want some that have similar syntax but divergent meaning; some similar words with and without political overtones. Something we'll want to avoid is words that simply have a distinct *lexical* meaning in the context of elections.
#'
                                        #+ peek_words, results='show'
                                        
#pd.set_option('display.max_rows', 100)
peek = pd.DataFrame({"all": common_words(tweets, 100),
                     "verbs": common_verbs(tweets, 100),
                     "nouns": common_nouns(tweets, 100),
                     "objects": common_objects(tweets, 100),
                     "roots": common_roots(tweets, 100),
                     "desc": common_descriptors(tweets, 100),
                     "mentions": common_mentions(tweets, 100)})
print(peek)

ref_words = ["hate", "hope", "trump", "hillary", "clinton", "duty", "family", "justice", "imwithher", "poll", "obama", "change", "woman", "white", "black", "maga", "pray", "man", "history", "law", "news", "job", "democracy", "future", "fact", "ass", "war", "free", "stupid", "racist", "scared", "scary", "nervous", "historic", "dumb", "civic", "fucked", "smart"]

print(ref_words)

#'
#' Some patterns we'd expect to see in these words are:
#'
#' * 'maga' and 'imwithher' should roughly index trumpian and liberal regions of the space
#' * 'racist' may occupy a more liberal position in a pragmatic embedding
#' * 'duty' and 'justice' spread apart in pragmatic embedding vs. lexical
#' 
#'
#'
#' ## Baseline
#' 
#' Start with word vectors from spacy. How do these vectors cluster key words from this election tweet corpus?
#'
#' ### t-SNE vs MDS
#'
                                        #+ plot_vec, results='show'

top_vec = get_spacy_vectors(ref_words)
top_dist = cosine_distances(top_vec)

tsne = TSNE(perplexity=5, metric='precomputed')
mds = MDS(dissimilarity='precomputed', n_init=20)
umap = UMAP(metric='precomputed')
svd = TruncatedSVD(algorithm='arpack')

_, ax = plt.subplots(2,2, figsize=(12,12))

plot_embed(top_dist, tsne, ax[0][0], "t-SNE", filter=False)
plot_embed(top_dist, mds, ax[0][1], "MDS", filter=False)
plot_embed(top_dist, umap, ax[1][0], "UMAP", filter=False)
plot_embed(top_vec, svd, ax[1][1], "SVD (on plain data)", filter=False)

plt.show()


#'
#' t-SNE and UMAP exhibit more clustering than MDS. In the latter two we see that names of political figures cluster together; man/woman/black/white demographic features cluster; there's a 'civics' cluster of civic/democracy/justice/war/law/duty (more distinct in t-SNE); and there's sort of an emotion/epithet cloud of nervous/scary/hate/stupid/racist/dumb
#'
#' Some differences between t-SNE and UMAP: racist + imwithher; placement of historic/history
#'
#' MDS shows less distinct clustering; it has a similar configuration of 'civic' terms and family/news/pray/change; it intermingles demographic and epithetic terms more than the others; and it places 'maga' and 'imwithher' off to the side.
#'
#'
#'
#' 
#' # Handle-Word Matrices
#'
#' Now let's get into attempts at pragmatic word embeddings. We'll start by looking at the matrix of word frequencies for each twitter handle, aggregating each user's tweets (and standardizing counts). The simplest way of looking at this is to apply UMAP to the vectors given by each word's occurrence in different users. We'll compare a version using all handles with one using handles which recur some number of times in the dataset, in case the reduced sparsity is helpful.
#' 
                                        #+ load_df

temp = pd.read_pickle(path+".data/tweets.df")
# cache invalidation
check_df = hashlib.md5(pd.util.hash_pandas_object(temp).values).hexdigest()

                                        #+ df, cache.extra=py$check_df
tw_df = temp

                                        #+ prep_df
# How many repeated handles?
tw_df.handle.value_counts().hist(bins=100)
# count 3+ occurrences
tw_df.handle.value_counts().where(lambda x: x > 2, np.nan).hist(bins=100)
# around 2000 (from 100k)
# pull out recurring handles:
rec_handles = pd.DataFrame({'count': tw_df.handle.value_counts()})
rec_handles['handle'] = rec_handles.index
rec_handles = rec_handles.query('count > 2').handle.values


# handle-word matrix
user_text = {}
for i in range(tw_df.shape[0]):
    handle = tw_df["handle"].iloc[i]
    text = unigrams[i]
    if not handle in user_text:
        user_text[handle] = text
    else:
        user_text[handle].update(text)

handles = user_text.keys()

# What min_df gives us the right number of words?
                                        #+ cv_test, eval=FALSE
cv_test = CountVectorizer(analyzer=lambda x: x).fit_transform([user_text[h] for h in handles])
df = np.array(np.sum(cv_test.sign(), axis=0)).reshape(-1)
# fuginc ram
df = df[np.random.choice(range(df.shape[0]), size=1000, replace=False)]

                                        #+ cv_test2, results='show', eval=FALSE
plt.hist(df, cumulative=True, bins=3000)
# this took like 5m at 10,000... am I fucking up types?? shouldn't be hard??
plt.gca().set_xlim([0, 50])
plt.show()

# Looks like min_df=3 should be adequate. (min_df = 5 yields 8000)
# Seems wrong though. Why should 80% of words occur only once? TODO

                                        #+ twomode_setup


cv = TfidfVectorizer(analyzer = lambda x: x)

fit_cv(min_df=10)
ref_inds = reference_indices()

umap.n_neighbors=3

word_user = get_matrix(handles).transpose()
word_user_rec = get_matrix(rec_handles).transpose()
word_dist = cosine_distances(word_user)
word_dist_rec = cosine_distances(word_user_rec)


#'
#' ### UMAP on cosine distance
#' 
                                        #+ twomode_plot, results='show'
_, ax = plt.subplots(1,2, figsize=(12,7))

plot_embed(word_dist, umap, ax[0], "All Handles")
plot_embed(word_dist_rec, umap, ax[1], "Recurring Handles")

plt.show()

#'
#' ### SVD on term-user matrix
#' 
                                        #+ twomode_svd, results='show'

_, ax = plt.subplots(1, 2, figsize=[12,7])

plot_embed(word_user, svd, ax[0], "All Handles (SVD)")
plot_embed(word_user_rec, svd, ax[1], "Recurring Handles (SVD)")

plt.show()


#'
#' ### UMAP on euclidean distance
#'
                                        #+ twomode_plot2, results='show', eval=FALSE

_, ax = plt.subplots(1,2, figsize=(12,7))

umap.metric='euclidean'
# gets an error for low n_neighbors?
umap.n_neighbors=10

plot_embed(word_user, umap, ax[0], "All Handles")
plot_embed(word_user_rec, umap, ax[1], "Recurring Handles")

plt.show()

#'
#' # Graph Methods
#'
#' In addition to looking at the handle-word matrix as such, we can see this matrix of co-occurrences between words and users as implying a network of relations among words [@Breiger1974], following the "distributional" assumption that co-occurring words are similar^[Of course, following our interest in different scales of social variance, we must elaborate at some point what kind of similarity we infer from co-occurrence.].
#' 

                                        #+ onemode_setup

# switch back to counts -- more meaningful when we're trying to relate words via walks...
cv = CountVectorizer(analyzer = lambda x: x)


# Setup
which_handles = handles
fit_cv(min_df=30, _handles=which_handles)

user_word = get_matrix(which_handles)
word_word = get_one_mode(user_word) # counts of two-step paths
word_word_bin = get_one_mode_binary(user_word, threshold=5)
word_word_dist = word_word.copy()
convert_to_distances(word_word_dist) # "distance" as inverse of path count

#'
#' Isomap is a manifold-based dimensionality-reduction algorithm sort of close to what we want to do: it does MDS on shortest paths in a weighted graph of euclidean distances to nearest neighbors. First, we check how different that is to a simpler MDS on shortest paths from dissimilarities. The difference should be that Isomap draws the graph based on just a few nearest neighbors, whereas the DIY way potentially draws all edges.
#'
#' I think both of these methods are sort of confused, because taking distance as 'inverse of co-occurrence rate' is unlikely to yield a coherent set of distances, nor a situation where shortest path length is meaningful. We might inspect
#'
#' * MDS on this distance-as-inverse-co-occurence metric
#' * see what kind of variance we have in distance and shortest path length
#'
#' We could also see how these compare with the unweighted graph method at different thresholds..
#'


                                        #+ test_nx, results='show'

# SLOW
G = nx.convert_matrix.from_numpy_array(word_word_bin.todense())
layout_dict = nx.drawing.layout.spring_layout(G)

plot_graph(layout_dict, plt, "Spring Layout of Binarized Graph")

plt.show()

                                        #+ onemode_plot, results='show', eval=FALSE
# Isomap vs. DIY
_, ax = plt.subplots(1, 2, figsize=[12,7])

# isomap
iso = Isomap(metric="precomputed")
plot_embed(word_word_dist, iso, ax[0], "Isomap")

# DIY
paths = shortest_path(word_word_dist)
plot_embed(paths, mds, ax[1], "MDS on Shortest Paths")

plt.show()


                                        #+ iso_spec, results='show', eval=FALSE
# Isomap vs Spectral

_, ax = plt.subplots(1, 2, figsize=[12,7])

# MDS
plot_embed(word_word_dist.toarray(), mds, ax[0], "MDS on pseudo-distances")

# spectral embed
spec = SpectralEmbedding(affinity="precomputed")
plot_embed(word_word, spec, ax[1], "Spectral Embedding (logged)", log=True)
# warning: graph not fully connected, SE may not work great
# - this was for rec_handles. all good w/ all handles

plt.show()


                                        #+ iso_umap, results='show', eval=FALSE

_, ax = plt.subplots(1, 2, figsize=[12,7])

# Isomap on tfidf
#  should up-weight co-occurrences of rare words...
cv = TfidfVectorizer(analyzer = lambda x: x)
fit_cv(min_df=30, _handles=which_handles)
user_tfidf = get_matrix(which_handles)
word_tfidf = get_one_mode(user_tfidf)
convert_to_distances(word_tfidf)

plot_embed(word_tfidf, iso, ax[0], "Isomap - tf-idf")


# UMAP on shortest paths
umap.n_neighbors=15
umap.metric="precomputed"
plot_embed(paths, umap, ax[1], "UMAP on psuedo path lengths from tf-idf")

plt.show()


