
import time
start = time.time()

import pandas as pd
import pickle
import sys

import spacy
import textacy
from textacy.corpus import Corpus


nlp = spacy.load("en_core_web_md")

print("loaded")


# draw in twitter dataz
path = "election-day-tweets/election_day_tweets.csv"

tw_df = pd.read_csv(path)


print(tw_df.head())
print("Initial shape")
print(tw_df.shape)

# filter to english
tw_df = tw_df.query("lang == 'en'")

tw_df["followers_count"] = tw_df["user.followers_count"]
tw_df["description"] = tw_df["user.description"]
tw_df["handle"] = tw_df["user.screen_name"]
tw_df = tw_df.loc[:, ["text", "retweeted", "created_at", "handle", "description",
                      "followers_count", "favorite_count", "retweet_count"]]


# quick filter so this doesn't take forever
tw_df = tw_df.sample(frac=.3)
print("Shape after sample")
print(tw_df.shape)


# draw into Spacy
# this is the slow bit
chunk_size = 10_000
tweets = Corpus(nlp)
for i in range((tw_df.shape[0] // chunk_size) + 1):
    tweets.add(tw_df.text.iloc[chunk_size * i : chunk_size * (i+1)])
    print("Parsed " + str(chunk_size * (i+1)) + " of " + str(tw_df.shape[0]))
#tweets = Corpus(nlp, list(tw_df.text))


# persist
tweets.save(".data/tweets.spacy")
print("made .tweets.spacy")

# persist data.frame format
tw_df.to_pickle(".data/tweets.df")
print("made .tweets.df")
# df.read_pickle


end = time.time()
print("time (s):")
print(end - start)
