
import pandas as pd
import pickle

import spacy
import textacy
from textacy.corpus import Corpus

nlp = spacy.load("en_core_web_md")
print("loaded")


# draw in twitter dataz
path = "election-day-tweets/election_day_tweets.csv"

tw_df = pd.read_csv(path)


tw_df["followers_count"] = tw_df["user.followers_count"]
tw_df = tw_df.loc[:, ["text", "retweeted", "created_at",
                      "followers_count", "favorite_count", "retweet_count"]]

print(tw_df.head())

# quick filter so this doesn't take forever
tw_df = tw_df.sample(frac=.02)


# persist data.frame format
tw_df.to_pickle(".tweets.df")
print("made .tweets.df")
# df.read_pickle

# draw into Spacy
tweets = Corpus(nlp, list(tw_df.text))
# persist
tweets.save(".tweets.spacy")
print("made .tweets.spacy")

'''
# get list of docs
tweets = []
for tw_text in tw_df.text:
    tweets.append(nlp(tw_text))
#tweets = list(map(nlp, tw_df.text))

# persist
with open(".tweets.spacy", "wb") as f:
    pickle.dump((tweets, nlp.vocab), f)
'''
