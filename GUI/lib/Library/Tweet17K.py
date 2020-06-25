import random
from pathlib import Path
import os
import io
import pandas as pd
from .IDataset import IDataset


class Tweet17K (IDataset):

    def getDataset(self):
        try:
            path = Path(__file__).parent / \
                "../Data/tweet17k/dataset.csv"
            dataset = pd.read_csv(path, encoding='iso-8859-9')
        except:
            path = Path(__file__).parent / \
                "../Data/tweet17k"
            train_tweets = self.readLineByLine(
                pd.read_excel(f"{path}/train_tweets.xlsx"))
            test_tweets = self.readLineByLine(
                pd.read_excel(f"{path}/test_tweets.xlsx"))
            dataset = train_tweets + test_tweets
            random.shuffle(dataset)
            random.shuffle(dataset)
            dataset = pd.DataFrame(dataset, columns=['Sentence', 'Sentiment'])
            # dataset.dropna(inplace=True)
            path = Path(__file__).parent / \
                "../Data/tweet17k/dataset.csv"
            dataset.to_csv(path, index=False)
            print("No csv file was found!, new file was created :)")
        # dataset = dataset.sample(frac=1).reset_index(drop=True)
        return dataset

    def getParameters(self):
        return {"tweet": self.tweet, "stemming": self.stemming, "classes_num": 3}

    def __init__(self, tweet=False, stemming=False):
        self.tweet = tweet
        self.stemming = stemming

    def __str__(self):
        return '17K Tweet'

    def getClasses(self):
        return self.getDataset().iloc[:, 1].values

    def getFeatures(self):
        return self.getDataset().iloc[:, 0].values

    def getPath(self):
        return Path(__file__).parent / \
            "../Data/tweet17k"

    def readLineByLine(self, data):
        tweets = []
        for row in range(data.shape[0]):
            tweet = []
            for col in range(data.shape[1]):
                tweet.append(data.iat[row, col])
            tweets.append(tweet)
        return tweets


# H = Tweet17K()
# H.getDataset()
# print(H.getDataset())
