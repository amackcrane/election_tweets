
#from keras.models import Model
#from keras.layers import Input, Dense, Embedding, LSTM
#from keras.layers import Bidirectional, TimeDistributed
#from keras.layers import Flatten
from sklearn.preprocessing import StandardScaler
#import keras.backend as K
#import keras

#import json
import functools
import scattertext
import spacy
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from spacy.tokens import Doc
from spacy.language import Language

from python_version_of_glove_twitter_preprocess_script import tokenize

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        _

#import seaborn as sns
import numpy as np
import pandas as pd

#nlp = spacy.load("en_core_web_sm")
nlp = Language(Vocab())

############################ F X N S ###########################




###################### T W I T T E R ###########################

# draw in twitter dataz
path = "../input/election-day-tweets/election_day_tweets.csv"

tw_df = pd.read_csv(path)


tw_df["followers_count"] = tw_df["user.followers_count"]
tw_df = tw_df.loc[:, ["text", "retweeted", "created_at",
                      "followers_count", "favorite_count", "retweet_count"]]

#print(tw_df.head())

#print(tw_df.retweet_count.describe())
#print(tw_df.favorite_count.describe())
#print(tw_df.followers_count.describe())

# quick filter so this doesn't take forever
tw_df = tw_df.sample(frac=.02)
num_samples = tw_df.shape[0]


##################### E M B E D D I N G S #################################

# draw in pretrained embeddings
glove_vectors = pd.read_csv(
    "../input/glove-global-vectors-for-word-representation/glove.twitter.27B.200d.txt", 
                      sep=" ", header=None)
#print(glove_vectors.head())


# get vector of words
keys = list(glove_vectors.iloc[:,0])
# get embed matrix
embeddings = np.array(glove_vectors.iloc[:,1:])

# create a vector for unknown stuff
unk_vector = glove_vectors.sample(300).mean(axis=0)
# and a zero vector
zero_vector = np.zeros(embeddings.shape[1])

# add in
keys.insert(0, "__unk__")
keys.insert(1, "__zero__")
embeddings = np.append([unk_vector, zero_vector], embeddings, axis=0)

# slot into spacy
nlp.vocab = Vocab(strings=keys)

# test
print(nlp.vocab.strings[1:10])

#vectors = Vectors(data=embeddings, keys=nlp.vocab.strings)
#nlp.vocab.vectors = vectors




#################### F I N A L    I / O #########################

# tokenize using GloVe syntax
tweet_text = np.array(tw_df.text.apply(lambda x: tokenize(x).split(" ")))

# create Docs
docs = Docs()
max_len = 0
for twt in tweet_text:
    tweet_tokens = []
    for word in twt:
        try: 
            tweet_tokens.append(dictionary[word])
        except KeyError:
            tweet_tokens.append(dictionary["__unk__"])
    if len(tweet_tokens) > max_len:
        max_len = len(tweet_tokens)
    tokens.append(tweet_tokens)


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
# .15 
