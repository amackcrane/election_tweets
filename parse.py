
import time
start = time.time()

import pandas as pd
import pickle
import sys

import spacy
import textacy
from textacy.corpus import Corpus
from spacy_langdetect import LanguageDetector


nlp = spacy.load("en_core_web_md")
# add language detection
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

print("loaded")


# draw in twitter dataz
path = "election-day-tweets/election_day_tweets.csv"

tw_df = pd.read_csv(path)


tw_df["followers_count"] = tw_df["user.followers_count"]
tw_df = tw_df.loc[:, ["text", "retweeted", "created_at",
                      "followers_count", "favorite_count", "retweet_count"]]

print(tw_df.head())
print("Initial shape")
print(tw_df.shape)

# filter to english based on spacy vocabulary
# hackishly, cuz tokenizing all properly would take too long


# this doesn't work very well...
# keep tweets with less than three unrecognizable tokens
#tw_df.loc[:,"eng"] = [
#    [nlp.vocab.strings[word] in nlp.vocab
#     for word in tw.split()].count(False) < 3
#    for tw in tw_df.text]
#test = tw_df.groupby('eng').agg({'text':'count'})
#print(test)
#tw_df = tw_df.iloc[tw_df.eng.values]
#print("Shape filtered to english-ish")
#print(tw_df.shape)

# quick filter so this doesn't take forever
tw_df = tw_df.sample(frac=.05)
print("Shape after sample")
print(tw_df.shape)

# persist data.frame format
#tw_df.to_pickle(".tweets.df")
#print("made .tweets.df")
# df.read_pickle
########## Don't do that for now b/c we've lost alignment w/ the spacy corpus d/t lang filtering

# draw into Spacy
tweets = Corpus(nlp, list(tw_df.text))
all_len = len(tweets)
# filter english
tweets.remove(lambda doc: doc._.language['language'] != 'en')
#tweets = [doc for doc in tweets if doc._.language == 'en']
en_len = len(tweets)
pct = 100 - 100 * float(en_len) / all_len
print("filtering to english dropped " + str(pct) + "% of tweets")
# looks like 12%

# persist
tweets.save(".tweets.spacy")
print("made .tweets.spacy")


end = time.time()
print("time (s):")
print(end - start)
