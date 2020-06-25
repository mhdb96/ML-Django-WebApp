import random
from pathlib import Path

import pandas as pd

from .IDataset import IDataset


class Hurriyet (IDataset):

    def getDataset(self):
        try:
            path = Path(__file__).parent / \
                "../Data/hurriyet/dataset.csv"
            dataset = pd.read_csv(path, encoding='iso-8859-9')
        except:
            path = Path(__file__).parent / \
                "../Data/hurriyet/csv_result-hurriyet6c1k-smooth_Corpus.csv"
            raw_texts = pd.read_csv(path, encoding='iso-8859-9')
            columnsData = raw_texts.iloc[:, 0].values
            dataset = []
            for i in columnsData:
                record = []
                words = i.split(';')
                record.append(words[2])
                record.append(str(int(words[1])-1))
                dataset.append(record)
            random.shuffle(dataset)
            path = Path(__file__).parent / \
                "../Data/hurriyet/dataset.csv"
            dataset = pd.DataFrame(dataset, columns=['Sentence', 'Category'])
            # dataset.dropna(inplace=True)
            dataset.to_csv(path, index=False, encoding='iso-8859-9')
            print("No csv file was found!, new file was created :)")
        #dataset = dataset.sample(frac=1).reset_index(drop=True)
        return dataset

    def getParameters(self):
        return {"tweet": self.tweet, "stemming": self.stemming, "classes_num": 6}

    def __init__(self, tweet=False, stemming=False):
        self.tweet = tweet
        self.stemming = stemming

    def __str__(self):
        return 'Hurriyet'

    def getClasses(self):
        return self.getDataset().iloc[:, 1].values

    def getFeatures(self):
        return self.getDataset().iloc[:, 0].values

    def getPath(self):
        return Path(__file__).parent / \
            "../Data/hurriyet"

# H = Hurriyet()
# H.getDataset()
# print(H.getDataset())
