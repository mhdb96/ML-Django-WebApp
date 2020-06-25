import random
from pathlib import Path
import os
import io
import pandas as pd

from .IDataset import IDataset


class MiniNews (IDataset):

    def getDataset(self):
        try:
            path = Path(__file__).parent / \
                "../Data/mini-news/dataset.csv"
            dataset = pd.read_csv(path)
        except:
            dataset = []
            path = Path(__file__).parent / \
                "../Data/mini-news/data"
            for Root, Dirs, Files in os.walk(f"{path}"):
                for di in Dirs:
                    for root, dirs, files in os.walk(f"{path}/{di}"):
                        for file in files:
                            sub_data = []
                            with io.open(f"{path}/{di}/"+file, 'r') as f:
                                text = f.read()
                                sub_data.append(text)
                                sub_data.append(di)
                            dataset.append(sub_data)
                random.shuffle(dataset)
            for i in range(2000):
                text = dataset[i][0]
                sentences = text.split('\n')
                b = False
                new = []
                for sen in sentences:
                    if sen.startswith('Lines'):
                        b = True
                        continue
                    if b == True:
                        new.append(sen)
                text = ' '.join([str(word) for word in new])
                dataset[i][0] = text
            dataset = pd.DataFrame(dataset, columns=['Sentence', 'Category'])
            # dataset.dropna(inplace=True)
            path = Path(__file__).parent / \
                "../Data/mini-news/dataset.csv"
            dataset.to_csv(path, index=False)
            print("No csv file was found!, new file was created :)")
        #dataset = dataset.sample(frac=1).reset_index(drop=True)
        return dataset

    def getParameters(self):
        return {"tweet": self.tweet, "stemming": self.stemming, "classes_num": 20}

    def __init__(self, tweet=False, stemming=False):
        self.tweet = tweet
        self.stemming = stemming

    def __str__(self):
        return 'Mini News'

    def getClasses(self):
        return self.getDataset().iloc[:, 1].values

    def getFeatures(self):
        return self.getDataset().iloc[:, 0].values

    def getPath(self):
        return Path(__file__).parent / \
            "../Data/mini-news"


# H = MiniNews(False, True)
# H.getDataset()
# H.getDataset()
