import pandas as pd
import random
import re
from nltk import WordPunctTokenizer
from snowballstemmer import TurkishStemmer
import pickle


def readLineByLine(data):
    tweets = []
    for row in range(data.shape[0]):
        tweet = []

        for col in range(data.shape[1]):
            tweet.append(data.iat[row, col])

        tweets.append(tweet)

    return tweets


def createCsv():
    train_tweets = readLineByLine(pd.read_excel("train_tweets.xlsx"))
    test_tweets = readLineByLine(pd.read_excel("test_tweets.xlsx"))

    all_tweets = train_tweets + test_tweets
    random.shuffle(all_tweets)

    data = pd.DataFrame(all_tweets, columns=['Sentence', 'Sentiment'])
    data.to_csv('17k-tweets.csv', index=False)


def get_external_stopwords(path):
    file = open(path, "r", encoding='utf8')
    stop_words = [word.strip() for word in file]
    file.close()
    return stop_words


def find_max_length(features):
    length = 0
    for sentence in features:
        if len(sentence) > length:
            length = len(sentence)
    return length
