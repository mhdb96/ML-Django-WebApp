from .IProcessor import IProcessor
from .IDataset import IDataset
# from Hurriyet import Hurriyet


import pandas as pd
import random
import re
from nltk import WordPunctTokenizer
from snowballstemmer import TurkishStemmer
from pathlib import Path


class TurkishProcessor(IProcessor):

    def __init__(self, dataset: IDataset):
        self.dataset = dataset

    def process(self):
        params = self.dataset.getParameters()
        features = self.dataset.getFeatures()
        processed_features = []
        for index in range(0, len(features)):

            # Convert to lower case
            processed_feature = str(features[index]).lower()

            if params["tweet"] == True:
                # Replace emojis with either EMO_POS or EMO_NEG
                processed_feature = self.handleEmojis(processed_feature)
                # Cleaning sentence for tweets
                processed_feature = self.cleanForTweet(processed_feature)

            # Cleaning sentence for normal texts
            processed_feature = self.cleanNormalText(processed_feature)
            # Cleaning stop words
            processed_feature = self.filter_stop_words(
                processed_feature, self.get_external_stopwords())

            if params["stemming"] == True:
                # Stemming words
                processed_feature = self.stemming_words(processed_feature)

            processed_features.append(processed_feature)
            print(index)
        return processed_features

    def handleEmojis(self, tweet):
        # Smile -- :), : ), :-), (:, ( :, (-:, :')
        tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
        # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
        tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
        # Love -- <3, :*
        tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
        # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
        tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
        # Sad -- :-(, : (, :(, ):, )-:
        tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
        # Cry -- :,(, :'(, :"(
        tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)

        return tweet

    def cleanForTweet(self, tweet):
        # Replaces URLs with the word URL
        tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', tweet)
        # Replace @handle with the word USER_MENTION
        tweet = re.sub(r'@[\S]+', '', tweet)
        # Replaces #hashtag with hashtag
        tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
        # Remove RT (retweet)
        tweet = re.sub(r'\brt\b', '', tweet)
        # Replace 2+ dots with space
        tweet = re.sub(r'\.{2,}', ' ', tweet)
        # Strip space, " and ' from tweet
        tweet = tweet.strip(' "\'')

        return tweet

    def cleanNormalText(self, sentence):
        # Remove all the special characters
        sentence = re.sub(r'\W', ' ', sentence)
        # Remove all digit characters
        sentence = re.sub(r'\d', '', sentence)
        # remove all single characters
        sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
        # Remove single characters from the start
        sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', sentence)
        # Substituting multiple spaces with single space
        sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)

        return sentence

    def get_external_stopwords(self):
        path = Path(__file__).parent / \
            "../Data/stop_words.txt"
        file = open(path, "r", encoding='utf8')
        stop_words = [word.strip() for word in file]
        file.close()

        return stop_words

    def filter_stop_words(self, text, stop_words):
        wpt = WordPunctTokenizer()
        tokenized_words = wpt.tokenize(text)
        processed_words = [
            word for word in tokenized_words if not word in stop_words]
        text = ' '.join([str(word) for word in processed_words])
        return text

    def stemming_words(self, text):
        wpt = WordPunctTokenizer()
        words = wpt.tokenize(text)
        turkishStemmer = TurkishStemmer()
        stemmed_words = []
        for word in words:
            stemmed_words.append(turkishStemmer.stemWord(word))
            # try:
            #     # stemmed_words.append(turkishStemmer.stemWord(word))
            #     stemmed_words.append(word[0:5])
            # except:
            #     # stemmed_words.append(turkishStemmer.stemWord(word))
            #     stemmed_words.append(word)
        text = ' '.join([str(word) for word in stemmed_words])
        return text

    def find_max_length(self, features):
        length = 0
        for sentence in features:
            if len(sentence) > length:
                length = len(sentence)
        return length


# H = Hurriyet(False, True)
# tp = TurkishProcessor(H)
# data = tp.process()
# print(data)
