

import spacy
from sklearn.manifold import TSNE, MDS, Isomap, SpectralEmbedding
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from textacy.corpus import Corpus
import scipy
import scipy.sparse
from umap import UMAP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import os

# fix wd
path = ""
if os.path.basename(os.getcwd()) == "scripts":
    path = "../"


                                        #+ load
# load functions
exec(open(path+"scripts/functions.py").read())

tweets = Corpus.load("en_core_web_md", path+".data/tweets.spacy")
print("corpus loaded")

                                        #+ vec

# grab a list of interesting words
unigrams = get_unigram_lists(tweets)
#n_top = 50
#ref_words = common_words(tweets, n_top)



# read in sentiment valences
sentiment_df = pd.read_csv(path+"sentiment.csv")
sentiment = dict(zip(sentiment_df["Word"], sentiment_df["V.Mean.Sum"]))


# helper from kaggle
def get_dep(doc, dependencies):
    try:
        token = [tok for tok in doc if tok.dep_ in dependencies]
    except IndexError:
        token = []
    return token

def is_negated(token):
    return "neg" in [t.dep_ for t in token.children]

    
# for a given target, get the sentiment with which it's inflected
def noun_inflection(token):
    deps = list(token.children)
    if token.dep_ == "nsubj": # if we're interested in subj, verb and obj pertain
        deps += get_dep(token.doc, "ROOT") + get_dep(token.doc, ("dobj", "pobj", "obj", "acomp", "attr"))
    return compose_inflection(deps)

# consider: just bagging sentiments for whole tweet?
# improving hashtag tokenization

#

def compose_inflection(tokens):
    if tokens: # non-empty
        # get sentiments
        valences = np.array([sentiment.get(lower_lemma(t), 5) for t in tokens])
        # flip if negated
        negations = np.array([-1 if is_negated(t) else 1 for t in tokens])
        # center
        valences = valences - 5
        # apply flip
        valences = valences * negations
        valence = np.mean(valences)
    else:
        valence = 0
    if valence > 1:
        return 1
    elif valence < -1:
        return -1
    else:
        return 0
    

def bagged_inflection(token):
    



def test_inflection(doc):
    for tok in doc:
        print(tok.text + " --- " + str(noun_inflection(tok)))
