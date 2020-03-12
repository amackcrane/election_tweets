
import sklearn.cluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functools
import pickle
import os


########### Text Clustering Functions ###############


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

# _.to_terms_list helper from textacy
def lower_lemma(token):
    return token.lemma_.lower()

def is_stop(tok):
    return (tok.is_stop or tok.is_punct or tok.is_space or
            lower_lemma(tok) in ["election2016", "rt"])

# custom biterm grabber
def get_biterm_lists(corpus):
    biterm_lists = []
    unique_biterms = [] # to standardize order of terms in biterms
    for doc in corpus:
        biterms = []
        for w1 in doc:
            if is_stop(w1): # filter stops
                continue
            for w2 in doc[w1.i+1 : len(doc)]: # for all subsequent words
                if is_stop(w2): # filter stops again
                    continue
                # compose w/ order-independence
                if lower_lemma(w1) > lower_lemma(w2):
                    biterms.append(lower_lemma(w1) + " " + lower_lemma(w2))
                else:
                    biterms.append(lower_lemma(w2) + " " + lower_lemma(w1))
        # add doc's biterm list
        biterm_lists.append(biterms)
    return biterm_lists

# list (along docs) of term lists
def get_unigram_lists(corpus):
    return [list(doc._.to_terms_list(ngrams=1, as_strings=True,
                                     normalize=lower_lemma,
                                     filter_stops=True, filter_punct=True,
                                     filter_nums=True,
                                     entities=False))
            for doc in corpus]


# Class for holding hyperparameters & results for clustering models
class Clusterer:
    def __init__(self, corpus_hash, k=2, soft=True, biterm=True, idf=True):
        try:
            int(corpus_hash)
        except TypeError:
            raise TypeError("Don't pass corpus to clusterer, just its hash value")
            
            
        self.corpus_hash=corpus_hash
        self.k=k
        self.soft=soft
        self.biterm=biterm
        self.idf=idf
        self.fitted=False

    def fit(self, corpus):
        # Get term lists
        if self.biterm:
            tm_input = get_biterm_lists(corpus)
            # topic model takes a few minutes with this version
        else:
            tm_input = get_unigram_lists(corpus)
        # Convert term lists to doc-term matrix
        vectorizer = vectorizers.Vectorizer(apply_idf=self.idf,
                                            idf_type='smooth', tf_type='binary')
        doc_term = vectorizer.fit_transform(tm_input)
        # fit topic model on doc-term
        n_topics = self.k if self.soft else 20
        model = topic_model.TopicModel('lda', n_topics=n_topics)
        model.fit(doc_term)
        # get doc-topic matrix and top terms by topic
        doc_topic = model.get_doc_topic_matrix(doc_term)
        topics = list(model.top_topic_terms(vectorizer.id_to_term, top_n=30, weights=True))
        # Get cluster labels
        kmeans=None
        if self.soft:
            labels = [(x[0], x[1][0]) for x in  model.top_doc_topics(doc_topic, top_n=1)]
            labels = pd.DataFrame(labels, columns=['doc', 'cluster'], dtype='category')
        else:
            kmeans = sklearn.cluster.KMeans(n_clusters=self.k)
            labels = kmeans.fit_predict(doc_topic)
            labels = pd.DataFrame(zip(range(len(labels)),
                                      labels),
                                  columns=['doc', 'cluster'])
        self.fitted = True
        self.topics = topics
        self.doc_topic = doc_topic
        self.labels = labels
        self.topic_model = model
        self.kmeans = kmeans


    # Test whether parameters, not outputs, are equal
    def __eq__(self, other):
        return (self.corpus_hash == other.corpus_hash and
                self.k == other.k and
                self.soft == other.soft and
                self.biterm == other.biterm and
                self.idf == other.idf)


# input: clusterer, not yet fit
# look up clusterer in cached models
# only compute if doesn't exist
def get_clusterer(clusterer, cachename):
    recompute = False
    # create empty cache file if it doesn't exist
    if not os.path.exists(cachename):
        pickle.dump([], open(cachename, 'wb'))
    # Load cache
    with open(cachename, 'rb') as readcache:
        cache = pickle.load(readcache)
        try:
            # If there's a matching instance in cache, grab it
            # (overridden equality method tests for equality of hyperparams)
            ind = cache.index(clusterer)
            clusterer = cache[ind]
        except ValueError:
            # If no chached match, compute fit
            clusterer.fit(tweets)
            recompute = True
            cache.append(clusterer)

    # If we've computed something new, dump it
    if recompute:
        with open(cachename, 'wb') as writecache:
            pickle.dump(cluster_cache, writecache)

    return clusterer

# in: textacy corpus of tweets
# out: dict
#    - topics: lists of top terms for topics in LDA model
#    - doc_topic: document-topic matrix from topic model
#    - labels: DataFrame mapping doc index to cluster label
#    - model: textacy TopicModel
#    - kmeans: scikit KMeans model, if soft=False
def deprecated_cluster_tweets(corpus, k=2, soft=True, biterm=True, idf=True):
    # Get term lists
    if biterm:
        tm_input = get_biterm_lists(corpus)
        # topic model takes a few minutes with this version
    else:
        tm_input = get_unigram_lists(corpus)
    # Convert term lists to doc-term matrix
    vectorizer = vectorizers.Vectorizer(apply_idf=idf, idf_type='smooth', tf_type='binary')
    doc_term = vectorizer.fit_transform(tm_input)
    # fit topic model on doc-term
    n_topics = k if soft else 20
    model = topic_model.TopicModel('lda', n_topics=n_topics)
    model.fit(doc_term)
    # get doc-topic matrix and top terms by topic
    doc_topic = model.get_doc_topic_matrix(doc_term)
    topics = list(model.top_topic_terms(vectorizer.id_to_term, top_n=30, weights=True))
    # Get cluster labels
    kmeans=None
    if soft:
        labels = [(x[0], x[1][0]) for x in  model.top_doc_topics(doc_topic, top_n=1)]
        labels = pd.DataFrame(labels, columns=['doc', 'cluster'], dtype='category')
    else:
        kmeans = sklearn.cluster.KMeans(n_clusters=k)
        labels = kmeans.fit_predict(doc_topic)
        labels = pd.DataFrame(zip(range(len(labels)),
                                  labels),
                              columns=['doc', 'cluster'])
    # Return
    to_return = {}
    to_return['topics'] = topics
    to_return['doc_topic'] = doc_topic
    to_return['labels'] = labels
    to_return['topic_model'] = model
    to_return['kmeans'] = kmeans
    to_return['params'] = {'corpus': hash(corpus),
                           'k': k,
                           'soft': soft,
                           'biterm': biterm,
                           'idf': idf}
    return to_return


# in:
#    term lists along documents
#    dataframe w/ doc index & label
#    number of terms to include
# out:
#    DataFrame with cols [word, freq[cluster1, cluster1, ...]]
def top_terms_by_cluster(term_list, doc_labels, n_most_common):
    # get list of top terms by label
    common = {}
    clusters = pd.unique(doc_labels['cluster'])
    for topic in clusters:
        partition = [term_list[i] for i in 
                             doc_labels['doc'].loc[doc_labels['cluster'] == topic]]
        common[topic] = most_common(partition, n_most_common)
    # convert to data.frame
    common_df = {}
    for topic in common.keys():
        common_df[topic] = pd.DataFrame(common[topic], columns=['word', 'freq'])
    # concatenate and reshape
    common_df = pd.concat(common_df, names=['cluster'])
    common_df.reset_index(level='cluster', inplace=True)
    common_df = common_df.pivot(columns='cluster', index='word', values=['freq']).reset_index()
    return common_df


# For now this just pertains to soft clustering i.e. LDA w/ two topics
# returns topic : spacy_doc_list dict
def peek_clusters(corpus, clusterer, top_n=3):
    # If this is just soft clustering from topic model...
    if clusterer.soft:
        # get top docs for each cluster
        #top_docs = dict(model.top_topic_docs(doc_topic, top_n=top_n))
        top_docs = {}
        for i in range(clusterer.doc_topic.shape[1]):
            top_docs[i] = np.argsort(clusterer.doc_topic[:,i])[-top_n-1 : -1]
    else:
        # get distances to cluster centers by doc
        doc_cluster = clusterer.kmeans.transform(clusterer.doc_topic)
        # find minima for each cluster
        top_docs = {}
        for i in range(doc_cluster.shape[1]):
            top_docs[i] = np.argsort(doc_cluster[:,i])[0:top_n]
    # slot in docs
    top_topic_docs = {}
    for topic, docs in top_docs.items():
        top_topic_docs[topic] = [corpus[i] for i in docs]
    return top_topic_docs


# in:
#    top_terms: DataFrame of terms with frequency by cluster
#    outlier_ratio: criterion for labeling words characteristic of one category
def scatter_text(top_terms, outlier_ratio, cluster_labs=[0,1]):
    x,y = cluster_labs
    # label outliers
    freq = top_terms['freq']
    top_terms['ratio'] = np.maximum(freq[x] / freq[y], freq[y] / freq[x])
    top_terms['label'] = np.where(top_terms['ratio'] > outlier_ratio,
                                  top_terms['word'],
                                  "")
    # filter NaNs
    top_terms = top_terms.dropna()
    # mpl style
    fig, ax = plt.subplots(1, 1, figsize=[5, 5])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ylims = [.8 * min(top_terms['freq', y]), 1.1 * max(top_terms['freq', y])]
    xlims = [.8 * min(top_terms['freq', x]), 1.1 * max(top_terms['freq', x])]
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    ax.scatter(x=top_terms['freq', x], y=top_terms['freq', y])
    # annotate points
    for i in range(top_terms.shape[0]):
        lab = top_terms['label'].iloc[i]
        _x,_y = top_terms.iloc[i].loc['freq']
        ax.annotate(lab, (_x, _y), xytext=(_x / 1.05, _y * 1.05))
    # print
    plt.show()



def print_top_cluster_terms(term_df, top_n):
    # df comes to us with cols word, freq[0, 1]
    # gross vv is there not a way to do this?
    clusters = list(filter(lambda x: x != '', term_df.columns.unique('cluster')))
    # creat ratio vars
    for c in clusters:
        # compute ratio for cluster c as frequency in c over mean frequency in other clusters
        other_clusters = np.setdiff1d(clusters, [c])
        term_df[('ratio',c)] = (term_df['freq',c] /
                                np.mean([ term_df['freq',n]
                                          for n in other_clusters ],
                                        axis=0))
    # create vars for words unique to cluster
    for c in clusters:
        other_clusters = np.setdiff1d(clusters, [c])
        #other_na = term_df.apply(lambda x: , axis=1)
        other_na = functools.reduce(lambda x,y: x and y,
                                    [ pd.isna(term_df['freq',oc])
                                             for oc in other_clusters ])
        term_df[('unique',c)] = np.where(other_na,
                                           term_df['freq',c], np.NaN)
    # print
    for c in clusters:
        print("\nCluster " + str(c) + " outliers\n")
        print(term_df
              .sort_values(by=('ratio',c), ascending=False)
              .loc[:,[('word',''), ('ratio',c)]].head(top_n))
        # meh, vv these vv are misleading & don't contribute much
        #print("\nCluster " + str(c) + " uniques\n")
        #print(term_df
        #      .sort_values(by=('unique',c), ascending=False)
        #      .loc[:,[('word',''),('freq',c)]].head(top_n))




    
