import random
from pathlib import Path
import os
import io
import pandas as pd

from .IDataset import IDataset


class Tweet3K (IDataset):

    def getDataset(self):
        try:
            path = Path(__file__).parent / \
                "../Data/tweet3k/dataset.csv"
            dataset = pd.read_csv(path, encoding='iso-8859-9')
        except:
            dataset = []
            path = Path(__file__).parent / \
                "../Data/tweet3k/raw_texts"
            for Root, Dirs, Files in os.walk(f"{path}"):
                for di in Dirs:
                    for root, dirs, files in os.walk(f"{path}/{di}"):
                        for file in files:
                            sub_data = []
                            with io.open(f"{path}/{di}/"+file, 'r', encoding='iso-8859-9') as f:
                                text = f.read()
                                sub_data.append(text)
                                sub_data.append(str(int(di)-1))
                            dataset.append(sub_data)
                random.shuffle(dataset)
            dataset = pd.DataFrame(dataset, columns=['Sentence', 'Sentiment'])
            # dataset.dropna(inplace=True)
            path = Path(__file__).parent / \
                "../Data/tweet3k/dataset.csv"
            dataset.to_csv(path, index=False, encoding='iso-8859-9')
            print("No csv file was found!, new file was created :)")
        #dataset = dataset.sample(frac=1).reset_index(drop=True)
        return dataset

    def getParameters(self):
        return {"tweet": self.tweet, "stemming": self.stemming, "classes_num": 3}

    def __init__(self, tweet=False, stemming=False):
        self.tweet = tweet
        self.stemming = stemming

    def __str__(self):
        return '3K Tweet'

    def getClasses(self):
        return self.getDataset().iloc[:, 1].values

    def getFeatures(self):
        return self.getDataset().iloc[:, 0].values

    def getPath(self):
        return Path(__file__).parent / \
            "../Data/tweet3k"

# H = Tweet3K()
# H.getDataset()
# print(H.getDataset())
