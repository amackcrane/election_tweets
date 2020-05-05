
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

""" Lolll the data.frame has a 'lang' column already
# filter english
all_len = len(tweets)
is_english = np.vectorize(lambda doc: doc._.language['lanugage'] == 'en')
keep = is_english(tweets)
to_remove = np.where(np.logical_not(keep))
for i in to_remove:
    del tweets[i]

#tweets.remove(lambda doc: doc._.language['language'] != 'en')
#tweets = [doc for doc in tweets if doc._.language == 'en']

# filter english in data.frame too
tw_df = tw_df.loc[keep]

en_len = len(tweets)
pct = 100 - 100 * float(en_len) / all_len
print("filtering to english dropped " + str(pct) + "% of tweets")
# looks like 12%
"""

# persist
tweets.save(".tweets.spacy")
print("made .tweets.spacy")

# persist data.frame format
tw_df.to_pickle(".tweets.df")
print("made .tweets.df")
# df.read_pickle


end = time.time()
print("time (s):")
print(end - start)
