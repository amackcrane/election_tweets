

import time
import scipy
import scipy.sparse
from sklearn.metrics.pairwise import manhattan_distances
import numpy as np
import functools
import community as louvain

#'
#' NOTE: these use a lot of global variables defined in graph.py. they may not work in other contexts
#' 


############################### Vectors ####################################

# grab vectors from spacy
def get_spacy_vectors(words):
    try:
        top_vec = np.load(path+".data/top_vec")
    except FileNotFoundError:
        nlp = spacy.load("en_core_web_md")
        top_vec = [nlp.vocab[string].vector for string in words]
        top_vec = np.array(top_vec).reshape(len(words), -1)
        np.save(path+".data/top_vec", top_vec)
        del nlp
    return top_vec


# general-use distance dag
def normalized_manhattan(word_user):
    dist = manhattan_distances(word_user)
    try:
        norms = scipy.sparse.linalg.norm(word_user, axis=1, ord=1)
    except TypeError:
        norms = np.linalg.norm(word_user, axis=1, ord=1)
    # get matrix of norm sums for all pairs of indices
    denoms = functools.reduce(lambda x,y: x+y, np.meshgrid(norms, norms))
    return dist / denoms


########################## Two-mode ####################################
    
# Fit vectorizer instance (as global 'cv')
#   Need to re-fit on each corpus to avoid out-of-vocab errors
def fit_cv(min_df=1, max_df=1.0, _handles=[]):
    global cv, handles, user_text
    cv.set_params(min_df=min_df, max_df=max_df)
    # If _handles omitted, this is fit on the global data
    if len(_handles) == 0:
        _handles = handles
    cv.fit([user_text[h] for h in _handles])

# Note 'cv' must have analyzer=lambda x: x; it defaults to tokenizing/lemmatizing etc. itself
def get_matrix(handles):
    global cv, user_text
    word_bags = [user_text[h] for h in handles]
    user_word = cv.transform(word_bags)
    return user_word


def twomode_reference(user_word):
    global cv, ref_words
    voc = cv.vocabulary_
    word_user = user_word.transpose()
    common_word_user = [word_user[voc[w]].toarray() for w in ref_words]
    common_word_user = np.array(common_word_user).reshape(len(ref_words), -1)
    return common_word_user



######################### Graph Functions ###############################

# unweighted one-mode graph
def apply_threshold(mat, thresh):
    """In-place"""
    mat.data = np.where(np.less(mat.data, thresh), 0, 1)
def get_one_mode_binary(user_word, threshold=1):
    word_user = user_word.transpose()
    word_word = word_user @ user_word
    # make binary
    apply_threshold(word_word, threshold)
    word_word.eliminate_zeros()
    return word_word


# construct weighted one-mode graph
def get_one_mode(user_word):
    word_user = user_word.transpose()
    word_word = word_user @ user_word
    return word_word

def convert_to_distances(mat):
    """in place"""
    # min-max scale
    min = np.min(mat.data)
    max = np.max(mat.data)
    range = max - min
    # avoid edge cases
    min -= range / 1000
    max += range / 1000
    # transform
    mat.data = (mat.data - min) / range
    mat.data = np.sqrt(-1 * np.log(mat.data))
    

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

def onemode_reference(distances):
    global cv, ref_words
    top_indices = [cv.vocabulary_[w] for w in ref_words]
    print(top_indices)
    common_dist = distances[top_indices][:,top_indices]
    return common_dist

def reference_indices():
    global cv, ref_words
    common_inds = [cv.vocabulary_[w] for w in ref_words]
    return common_inds


#################### Commm Detection ################################


# louvain modularity on word-word matrix
def get_partition(word_word):
    G = nx.convert_matrix.from_scipy_sparse_matrix(word_word)
    #tree = louvain.generate_dendrogram(G)
    part = louvain.best_partition(G)
    # defaults to getting 'weight' attr from G
    # invert partition (group ids ordered as word_word)
    comms = np.full(word_word.shape[:1], -1)
    for community, nodes in part.entries():
        for node in nodes:
            comms[node] = community
    return comms







############################## Plotting ###############################

# embed using dimensionality reduction model 'reducer', and plot
def plot_embed(data, reducer, ax, title, filter=True, log=False):
    global ref_words
    # dimensionality reduction
    red_vec = reducer.fit_transform(data)
    if filter:
        red_vec = red_vec[reference_indices()]
    # axis limits
    x=red_vec[:,0]
    y=red_vec[:,1]
    sx=(np.max(x) - np.min(x)) * .1
    sy=(np.max(y) - np.min(y)) * .1
    if log:
        x = np.log(x - np.min(x) + sx)
        y = np.log(y - np.min(y) + sy)
    ax.set_xlim(np.min(x) - sx, np.max(x) + sx)
    ax.set_ylim(np.min(y) - sy, np.max(y) + sy)
    # plot
    ax.scatter(red_vec[:,0], red_vec[:,1])
    for i,w in enumerate(ref_words):
        ax.annotate(w, red_vec[i])
    try:
        ax.set_title(title)
    except AttributeError:
        pass

# plot graph-style
def plot_graph(user_word, mds, ax, title):
    global ref_words
    # similarities
    word_word_sim = get_one_mode(user_word)
    max_sim = np.amax(word_word_sim.data)
    # dissimilarities
    word_word_dist = manhattan_distances(user_word.transpose())
    # layout
    word_points = mds.fit_transform(word_word_dist)
    # indices to filter on
    inds = reference_indices()
    # draw edges
    for inds in np.array(word_word_sim.nonzero()).transpose():
        # line plot from first to second
        node_coords = word_points[inds]
        ax.plot(node_coords[:,0], node_coords[:,1],
                linewidth=word_word_sim[inds[0], inds[1]]*10/max_sim)
    # draw nodes
    ax.scatter(word_points[:,0], word_points[:,1])
    for i,w in enumerate(ref_words):
        ax.annotate(w, (word_points[i]))
    try:
        ax.set_title(title)
    except AttributeError:
        pass
        
