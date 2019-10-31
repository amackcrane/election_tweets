


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, Bidirectional
import keras
import json
import functools

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        _


###################### T W I T T E R ###########################

# draw in twitter dataz
path = "../input/twitter-sample/twitter_samples/tweets.20150430-223406.json"
with open(path) as f:
    tw_data = []
    for line in f:
        tw_data.append(json.loads(line))
        
        
#print(len(tw_data))
#print(tw_data[0])

# prune tw_data
tw_pruned = []
for tw in tw_data:
    # toss if RT
    if "retweeted_status" in tw:
        continue
        
    keys = ["id", "created_at", "text", "favorite_count", "retweet_count"]
    new_tw = {k:tw[k] for k in keys}
    new_tw["followers_count"] = tw["user"]["followers_count"]
    tw_pruned.append(new_tw)


    
#print(tw_pruned[0])

# reduce list of dicts to data.frame
tw_df = functools.reduce(lambda x,y: x.append(y, ignore_index=True), 
                         tw_pruned, pd.DataFrame())


##################### E M B E D D I N G S #################################

# draw in pretrained embeddings
glove_vectors = pd.read_csv("../input/glove-twitter/glove.twitter.27B.25d.txt", 
                      sep=" ", header=None)
#print(glove_vectors.head())


# get vector of words
keys = list(glove_vectors.iloc[:,0])
# get embed matrix
embeddings = np.array(glove_vectors.iloc[:,1:])

# create a vector for unknown stuff
unk_vector = glove_vectors.sample(300).mean(axis=0)

# add in
keys.insert(0, "__unk__")
embeddings = np.append([unk_vector], embeddings, axis=0)

#print("test:")
#print(keys[0] == "__unk__")
#print(unk_vector is embeddings[0])
# working atmo

# make dict word -> index
dictionary = dict(zip(keys, range(len(keys))))

vocab_size = embeddings.shape[0]
embedding_len = embeddings.shape[1]
#print(vocab_size, embedding_len)


#################### F I N A L    I / O #########################

# convert tweet text to list of wordsz
# for now, stripping out non-alnum chars
tweet_text = np.array(
    tw_df.text.apply(
        lambda x: ''.join(filter(lambda y: y not in '@#!.,?:;()$%&*+-/', x
                          )).split(" ")
    )
)


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

# Gotta make fixed dimensions for training
# leaving this hackish for now; revisit later
tmp = np.zeros((len(tokens), max_len))
for i in range(len(tmp)):
    for j in range(max_len):
        try:
            tmp[i][j] = tokens[i][j]
        except IndexError:
            tmp[i][j] = dictionary["__unk__"]
            # fill in blank space w/ unknowns :\ :\ :\
            
tokens = tmp

followers = np.array(tw_df.followers_count)
favorites = list(tw_df.favorite_count)
retweets = list(tw_df.retweet_count)
predictand = np.array(list(zip(favorites, retweets)))

print(predictand.shape)

#print("favorites:")
#print(pd.DataFrame(favorites).describe())
#print("RTs:")
#print(pd.DataFrame(retweets).describe())
# these.......... are all zeros

# whatever, let's see if i can at least spec it


########### S P E C I F I C A T I O N ######################

# spec fixed embedding layer
words_input = Input(shape=(None,), name="words_input")
e_layer = Embedding(vocab_size, embedding_len, trainable=False,
                   weights=[embeddings])
#e_layer.set_weights(np.transpose(embeddings))
#e_layer.trainable=False
e = e_layer(words_input)


# connect it to a BLSTM
bigboi = Bidirectional(LSTM(units=30), merge_mode='sum')(e)
# toss in a control (followers_count)
ctrl = Input(shape=(1,), name="aux_input")
both = keras.layers.concatenate([bigboi, ctrl])

# output likes & RTs for now
output = Dense(2, activation='relu')(both)


m = Model(inputs=[words_input, ctrl], outputs=output)
#m.summary()


##################### Z O O M #################################

print(1)
m.compile(optimizer='sgd', loss='mae', metrics=['mae'])

print("hi")

m.fit({'words_input': tokens, 'aux_input': followers}, predictand, epochs=1, verbose=2)

print("doneeee")

