

import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from textacy.corpus import Corpus
from textacy.tm import TopicModel
import scipy
import scipy.sparse
import aspect_based_sentiment_analysis as absa

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import itertools
import time
import os

# fix wd
path = ""
if os.path.basename(os.getcwd()) == "scripts":
    path = "../"

# timer context manager (for blocks)
class Timer():
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.time = time.time() - self.start



                                        #+ load
# load functions
exec(open(f"{path}scripts/topic_helpers.py").read())

tweets = Corpus.load("en_core_web_md", f"{path}.data/tweets.spacy")
print("corpus loaded")

# settings
n_jobs = 4
# keep consistent for caching expensive results
np.random.seed(1203742)


# load up absa
nlp = absa.load()

# make smol
all_tweets = tweets
tweets = [all_tweets[n] for n in np.random.choice(np.arange(len(tweets)), size=1000, replace=False)]

# test
def display_absa(task):
    print(f"\n### {task.text}")
    for aspect,subtask in task.subtasks.items():
        print("{} -- {}".format(aspect, subtask.sentiment.name))

def test_absa(n):
    for tw in tweets[:n]:
        words = tw._.to_terms_list(ngrams=1, as_strings=True, filter_stops=True,
                                   normalize=lower_lemma)
        tasks = nlp(tw.text, aspects=list(words))
        display_absa(tasks)

#+ vec

# construct vector shits
unigrams = get_unigram_lists(tweets)
cv = CountVectorizer(analyzer=lambda x: x, binary=True)
doc_word = cv.fit_transform(unigrams)

# get common words as aspects for sentiment analysis
ref_words = np.array(common_nouns(tweets, 500))
ref_dict = {k: i for i,k in enumerate(ref_words)}


# do function-style
# shouldn't use any globals
def get_sentiment(indices, tweets, ref_words):
    global nlp
    #nlp = absa.load()
    #del all_tweets
    some_tweets = [tweets[i] for i in indices]
    del tweets
    # construct list of arrays of ref_words occurring in each tweet
    tweet_ref_words = []
    for tw in some_tweets:
        tw_words = np.array([x for x in tw._.to_terms_list(ngrams=1,
                                                           as_strings=True,
                                                           normalize=lower_lemma)])
        tweet_ref_words.append(np.intersect1d(ref_words, tw_words))
    print("Matched aspects")
    # fill in doc-aspect matrix with sentiments from absa
    doc_aspect = scipy.sparse.lil_matrix((len(some_tweets), len(ref_words)))
    for i,tw in enumerate(some_tweets):
        if i % 20 == 0:
            print("tweet {} of {}".format(i, len(indices)))
        aspects = tweet_ref_words[i]
        if len(aspects) > 0:
            sents = nlp(tw.text, aspects=aspects)
            for j in range(len(aspects)):
                sent_enum = sents[aspects[j]].sentiment
                if sent_enum == absa.Sentiment.positive:
                    sentiment = 1
                elif sent_enum == absa.Sentiment.negative:
                    sentiment = -1
                else:
                    sentiment = 0
                doc_aspect[i,ref_dict[aspects[j]]] = sentiment
    return doc_aspect
    


# compute conditionally -- absa takes awhile!
data_hash = hash(str(ref_words)) ^ hash(str(tweets))
filename = f"{path}.data/sentiment_{data_hash}"
try:
    with open(filename, 'rb') as f:
        doc_aspect = np.load(f)
    print("Found saved sentiment data")
except FileNotFoundError:
    print("Computing sentiment")
    with Timer() as timer:
        # split into indices
        tot = len(tweets)
        loi = np.array_split(np.arange(tot), n_jobs)
        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=3)(
            delayed(get_sentiment)(inds, tweets, ref_words) for inds in loi)
        doc_aspect = scipy.sparse.vstack(results)
    # 1:20 for 100 tweets
    # 13m for 1000 in csr format
    # 12.5m for 1000 in lil
    # 9m for 1000 w/ 4 threads
    print("took {} seconds for {} tweets in {} jobs".format(timer.time, tot, n_jobs))
    # save
    with open(filename, 'wb') as f:
        np.save(f, doc_aspect)

    

# break out into positive and negative matrices
doc_aspect = doc_aspect.tocsr()
doc_aspect_positive = doc_aspect.copy()
doc_aspect_positive.data = np.where(np.equal(doc_aspect.data, 1), 1, 0)
doc_aspect_negative = doc_aspect.copy()
doc_aspect_negative.data = np.where(np.equal(doc_aspect.data, -1), 1, 0)

doc_aspect_positive.eliminate_zeros()
doc_aspect_negative.eliminate_zeros()

# hang onto vocabularies
ref_positive = list(map(lambda x: "+"+x, ref_words))
ref_negative = list(map(lambda x: "-"+x, ref_words))

# remove rows from neutral matrix
#   assuming that words aren't mentioned with multiple valences in a single tweet
# iterate over ref_words
for i,w in enumerate(ref_words):
    delete = doc_aspect_positive[:,i].todense() + doc_aspect_negative[:,i].todense()
    doc_word = doc_word.multiply(1 - delete)
    

# construct
doc_all = scipy.sparse.hstack((doc_word, doc_aspect_positive, doc_aspect_negative))
vocabulary = cv.get_feature_names() + ref_positive + ref_negative


# topic modellll

topic = TopicModel("lsa", n_topics=100)
topic.fit(doc_all)
doc_topic = topic.transform(doc_all)

def print_docs(inds):
    toprint = ""
    for i in inds:
        toprint += f"\n>> {tweets[i].text}"
    print(toprint+"\n")

top_terms = list(topic.top_topic_terms(vocabulary, top_n=6))
top_docs = list(topic.top_topic_docs(doc_topic, top_n=5))
for t,d in zip(top_terms, top_docs):
    print(t)
    print_docs(d[1])


