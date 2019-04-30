import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import string
from nltk import FreqDist as fd
from nltk import pos_tag
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize as token
from nltk import NaiveBayesClassifier as nb
from nltk.stem import WordNetLemmatizer as wl
tweet_token=TweetTokenizer()
from sklearn.feature_extraction.text import CountVectorizer as cv
df=pd.read_csv("training_twitter_x_y_train.csv")
df_test=pd.read_csv("test_twitter_x_test.csv")
print(df.shape)
print(df.columns.values)
tweets=list(df["text"])
test_tweets=list(df_test['text'])
sentiments=list(df["airline_sentiment"])
# for tweet in tweets[0:3]:
#     print("---------------",tweet,"-----------------------")
# print(len(tweets))
stops=set(stopwords.words("english"))
lemmatizer=wl()
punc=list(string.punctuation)
stops.update(punc)
tweet_words=[]
test_tweet_words=[]
for tweet in test_tweets:
    test_tweet_words.append((tweet_token.tokenize(tweet)))
for tweet,sentiment in zip(tweets,sentiments):
    tweet_words.append((tweet_token.tokenize(tweet),sentiment))
def get_simple_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN
def clean_tweets(tweet):
    output_words=[]
    for t in tweet:
        if t.lower() not in stops:
            pos=pos_tag([t])
            clean_word=lemmatizer.lemmatize(t,pos=get_simple_pos(pos[0][1]))
            output_words.append(clean_word.lower())
    return output_words
# print(tweet_words[0])
clean_words_train=[ (clean_tweets(tweet),sentiment) for tweet,sentiment in tweet_words]
clean_words_test=[ (clean_tweets(tweet)) for tweet in test_tweet_words]
print(clean_words_train)
all_words=[]
for tweet in clean_words_train:
    all_words+=tweet[0]
freq=fd(all_words)
common=freq.most_common(200)
features=[i[0] for i in common]
def get_feature_dict(words):
    current_features={}
    words_set=set(words)
    for w in features:
        current_features[w]=w in words_set
    return current_features
training_data=[(get_feature_dict(tweet),sentiment) for tweet, sentiment in clean_words_train]
testing_data=[(get_feature_dict(tweet)) for tweet in clean_words_test]
print(training_data)
print(testing_data[0])
classifier=nb.train(training_data)
output=[]
# for tweet_words in testing_data:
#     print("--------------------------------")
#     print(tweet_words)
output=[classifier.classify(tweet_words) for tweet_words in testing_data]
print(output)
np.savetxt("predictions_twitter_sentimental.csv",output,fmt="%s",delimiter=" ")
