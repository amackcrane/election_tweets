

#' ---
#' title: Visualizing Semantic and Pragmatic Relatedness
#' output: pdf_document
#' ---


                                        #+ setup

import spacy
from sklearn.manifold import TSNE, MDS
from sklearn.feature_extraction.text import CountVectorizer
from textacy.corpus import Corpus
import scipy
import scipy.sparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

                                        #+ load
# load functions
exec(open("functions.py").read())

tweets = Corpus.load("en", ".tweets.spacy")
print("corpus loaded")

                                        #+ vec

# grab a list of interesting words
n_top = 50
unigrams = get_unigram_lists(tweets)
ref_words = common_words(tweets, n_top)

# grab vectors from spacy
def get_spacy_vectors(words):
    try:
        top_vec = np.load(".top_vec")
    except NameError:
        nlp = spacy.load("en_core_web_md")
        top_vec = [nlp.vocab[string].vector for string in words]
        top_vec = np.array(top_vec).reshape(n_top, -1)
        np.save(".top_vec", top_vec)
    return top_vec


def plot_embed(data, reducer, ax, title, indices=[]):
    global ref_words
    # dimensionality reduction
    red_vec = reducer.fit_transform(data)
    if len(indices) > 0:
        red_vec = red_vec[indices]
    # plot
    ax.scatter(red_vec[:,0], red_vec[:,1])
    for i,w in enumerate(ref_words):
        ax.annotate(w, red_vec[i])
    try:
        ax.set_title(title)
    except AttributeError:
        pass

#'
#' Start with word vectors from spacy. How do these vectors cluster top words from this election tweet corpus? Since these vectors are general-purpose (this is en_core_web_md), we might expect words to cluster on syntactic features more than political valence...
#'
#' 
                                        #+ plot_vec, results='show'

top_vec = get_spacy_vectors(ref_words)
tsne = TSNE(n_components=2, perplexity=5)
mds = MDS(n_components=2)

_, ax = plt.subplots(1,2, figsize=(12,7))

plot_embed(top_vec, tsne, ax[0], "t-SNE")
plot_embed(top_vec, mds, ax[1], "MDS")

plt.show()



# embeddings from pragmatics
# grab twitter handles...
                                        #+ load_df

tw_df = pd.read_pickle(".tweets.df")

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

#'
#' What min_df gives us the right number of words?
#'
                                        #+
cv_test = CountVectorizer(analyzer=lambda x: x).fit_transform([user_text[h] for h in handles])
df = np.array(np.sum(cv_test.sign(), axis=0)).reshape(-1)
# fuginc ram
df = df[np.random.choice(range(df.shape[0]), size=1000, replace=False)]

                                        #+ cv_test, results='show'
plt.hist(df, cumulative=True, bins=3000)
# this took like 5m at 10,000... am I fucking up types?? shouldn't be hard??
plt.gca().set_xlim([0, 50])
plt.show()
#'
#' Looks like min_df=3 should be adequate. (min_df = 5 yields 8000)
#' Seems wrong though. Why should 80% of words occur only once?
#'
                                        #+ twomode_funs


cv = CountVectorizer(analyzer = lambda x: x)
# pre-fit for vocab safety...
def fit_cv(min_df=1, max_df=1.0, _handles=[]):
    global cv, handles, user_text
    cv.set_params(min_df=min_df, max_df=max_df)
    # If _handles omitted, this is fit on the global data
    if len(_handles) == 0:
        _handles = handles
    cv.fit([user_text[h] for h in _handles])

def get_matrix(handles):
    global cv, user_text
    word_bags = [user_text[h] for h in handles]
    user_word = cv.transform(word_bags)
    return user_word


def twomode_common(user_word):
    global cv, ref_words
    voc = cv.vocabulary_
    word_user = user_word.transpose()
    common_word_user = [word_user[voc[w]].toarray() for w in ref_words]
    common_word_user = np.array(common_word_user).reshape(n_top, -1)
    return common_word_user


                                        #+ twomode_plot, results='show'
fit_cv(10)
_, ax = plt.subplots(1,2, figsize=(12,7))


plot_embed(twomode_common(*get_matrix(handles)), tsne, ax[0], "All Handles")
plot_embed(twomode_common(*get_matrix(rec_handles.handle)), tsne, ax[1], "Recurring Handles")

plt.show()

#'
#' Ok, restricting to recurring handles seems to help get some sense from it...
#' 


# ok, re-do with actually computing dissimilarities as path length...

                                        #+ onemode_funs

def apply_threshold(mat, thresh):
    """In-place"""
    mat.data = np.where(np.less(mat.data, thresh), 0, 1)

def get_one_mode(user_word, threshold=1):
    word_user = user_word.transpose()
    # possible to impose threshold in this step??
    word_word = word_user @ user_word
    # make binary
    #word_word = word_word.sign()
    apply_threshold(word_word, threshold)
    word_word.eliminate_zeros()
    return word_word

# 10s on n=1500 w/ 170k values (min_df=50, thresh=5)
# 25s on n=5000 w/ 55k values
# 5s on n=1100 w/ 140k
def shortest_path(word_word):
    print("shape: " + str(word_word.shape))
    print("nonzero: " + str(word_word.nnz))
    start = time.time()
    paths = scipy.sparse.csgraph.shortest_path(word_word, directed=False, method='D')
    # output should be dense...
    vmax = np.max(np.where(np.isfinite(paths), paths, 0))
    paths = np.where(np.isinf(paths), vmax, paths)
    print("time: " + str(time.time() - start))
    return paths

def approx_path(word_word, degree):
    """Approximate path length by inverting # of walks up to a given degree"""
    walks = word_word
    # Note the word-word matrix will have ones on diagonal
    #   so exponentiating gives cumulative walks up to length 'degree'
    for d in range(degree-1):
        walks = walks @ (word_word / (d+1))
    # convert to dense
    walks = walks.todense()
    # invert (leaving absent paths as 0)
    dists = np.where(np.isclose(walks, 0), 0, 1 / walks)
    # scale and recode
    vmax = np.max(walks)
    vmean = np.mean(walks)
    dists = np.where(np.isclose(walks, 0), vmax / vmean, walks / vmean)
    return dists

def onemode_common(distances):
    global cv, ref_words
    top_indices = [cv.vocabulary_[w] for w in ref_words]
    print(top_indices)
    common_dist = distances[top_indices][:,top_indices]
    return common_dist

def common_indices():
    global cv, ref_words
    common_inds = [cv.vocabulary_[w] for w in ref_words]

#'


                                        #+ approx_degree, results='show'
# New dimension-reducers to take distance/dissimilarity matrix
tsne = tsne.set_params(metric='precomputed', learning_rate = 100)
mds = mds.set_params(dissimilarity='precomputed', metric=False)

_, ax = plt.subplots(2,2, figsize=[12,7])

ax = list(itertools.chain(*ax))

for i,t in enumerate([1,2,3,4]):
    plot_embed(
        onemode_common(
            approx_path(
                get_one_mode(get_matrix(handles),
                             5),
                t)),
        tsne,
        ax[i],
        "Degree "+str(t))

plt.show()


# quick test code
which_handles = handles
fit_cv(min_df=5, max_df=.2, _handles=which_handles)
user_word = get_matrix(which_handles)
word_word = get_one_mode(user_word, 10)
#dist = approx_path(word_word, 5)
dist = shortest_path(word_word)
ref_words = common_mentions(tweets, 70, cv.vocabulary_.keys())
#c_dist = onemode_common(dist)
plt.figure(figsize=[9, 9])
plot_embed(dist, mds, plt, "", indices=common_indices())
plt.show()


