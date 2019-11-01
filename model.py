

from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM
from keras.layers import Bidirectional, TimeDistributed
from keras.layers import Flatten
from sklearn.preprocessing import StandardScaler
import keras.backend as K
import keras
import json
import functools

from python_version_of_glove_twitter_preprocess_script import tokenize

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        _

#import seaborn as sns
import numpy as np
import pandas as pd

############################ F X N S ###########################

def r2_score(y_true, y_pred):
    """via Fred Navruzov on kaggle"""
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))



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

#print("test:")
#print(keys[0] == "__unk__")
#print(unk_vector is embeddings[0])
#print("zeros:")
#print(embeddings[1])
#print(keys[1])
# working atmo

# make dict word -> index
dictionary = dict(zip(keys, range(len(keys))))

vocab_size = embeddings.shape[0]
embedding_len = embeddings.shape[1]
#print(vocab_size, embedding_len)



#################### F I N A L    I / O #########################

# convert tweet text to list of wordsz
# for now, stripping out non-alnum chars
#tweet_text = np.array(
#    tw_df.text.apply(
#        lambda x: ''.join(filter(lambda y: y not in '@#!.,?:;()$%&*+-/', x
#                          )).split(" ")
#    )
#)

# got the real preprocessing script now!
tweet_text = np.array(tw_df.text.apply(lambda x: tokenize(x).split(" ")))

# quick tokenize test
#print(tw_df.iloc[1:10].loc[:,"text"].apply(tokenize))


#tweet_lengths = np.vectorize(len)(tweet_text)
#print(pd.DataFrame(tweet_lengths).describe())
# 75% < 20 tokens
# max 31 median 14

# then list of word IDs
# setting unknown words to __unk__
tokens = []
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

# pad w/ zero vectors
# truncate to 12 tokens
tokens = keras.preprocessing.sequence.pad_sequences(tokens, 
                            padding='pre', truncating='post',
                            value=dictionary["__zero__"])

followers = np.array(tw_df.followers_count)
favorites = list(tw_df.favorite_count)
retweets = list(tw_df.retweet_count)
predictand = np.array(list(zip(favorites, retweets)))

# normalize numeric vars!
sc = StandardScaler()
followers, favorites, retweets = sc.fit_transform([followers, favorites,
                                                  retweets])


print(predictand.shape)

# check
#sns.heatmap(pd.DataFrame({'fav': favorites, 'fol':followers}).corr(),
#           annot=True)
# .15 corr


########### S P E C I F I C A T I O N ######################

# spec fixed embedding layer
words_input = Input(shape=(None,), name="words_input")
e_layer = Embedding(vocab_size, embedding_len, trainable=False,
                   weights=[embeddings])
x = e_layer(words_input)


# connect it to a BLSTM
x = Bidirectional(LSTM(units=embedding_len), merge_mode='concat')(x)
#x = LSTM(25)(x)

x = Dense(50)(x)

#x = TimeDistributed(Dense(10))(x)
#x = Flatten()(x)

# toss in a control (followers_count)
ctrl = Input(shape=(1,), name="aux_input")
x = keras.layers.concatenate([x, ctrl])

x = Dense(50)(x)

# output likes & RTs for now
output = Dense(1, activation='relu')(x)

output2 = Dense(1, activation='relu', kernel_initializer='identity',
               bias_initializer='zeros')(ctrl)

m = Model(inputs=[words_input, ctrl], outputs=output)
m.summary()

m2 = Model(inputs=ctrl, outputs=output2)


##################### Z O O M #################################

print(1)

sgd = keras.optimizers.SGD(lr=.1, momentum=.2, clipnorm=1.)

m.compile(optimizer='sgd', loss='mse', metrics=[r2_score])

print("hi")

m.fit({'words_input': tokens, 'aux_input': followers}, favorites, 
      epochs=5, verbose=2, batch_size=num_samples)

#m2.summary()
#m2.compile(optimizer='sgd', loss='mse', metrics=[r2_score])
#m2.fit(followers, favorites, epochs=5, verbose=2)

print("doneeee")
