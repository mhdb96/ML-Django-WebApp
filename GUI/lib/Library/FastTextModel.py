from .IModel import IModel
from .IProcessor import IProcessor
from .IDataset import IDataset
import numpy as np

# from Hurriyet import Hurriyet
# from Aahaber import Aahaber
# from Tweet3K import Tweet3K
# from Tweet17K import Tweet17K
# from Milliyet import Milliyet
# from TurkishProcessor import TurkishProcessor
from .helpers import get_external_stopwords, find_max_length
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Dropout, LSTM, SpatialDropout1D, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import to_categorical
import pickle
from nltk import WordPunctTokenizer
from gensim.models.fasttext import FastText
import matplotlib.pyplot as plt


class FastTextModel (IModel):

    EPOCHS = 30
    BATCH_SIZE = 100
    ACTIVATION = 'sigmoid'
    LOSSFUNC = 'binary_crossentropy'
    TEST_SIZE = 0.5
    NUM_WORDS = 900

    def __init__(self, processor: IProcessor, dataset: IDataset, test_size=None, batch=None):
        self.processor = processor
        self.dataset = dataset
        if test_size != None:
            self.TEST_SIZE = test_size
        if batch != None:
            self.BATCH_SIZE = batch

    def __str__(self):
        return 'FastText'

    def evaluate(self):

        path = self.dataset.getPath()
        try:
            features = pickle.load(open(f"{path}/preprocessed.p", "rb"))
        except:
            features = self.processor.process()
            pickle.dump(features, open(f"{path}/preprocessed.p", "wb"))
        word_punctuation_tokenizer = WordPunctTokenizer()
        word_tokenized_corpus = [
            word_punctuation_tokenizer.tokenize(sent) for sent in features]
        # print(word_tokenized_corpus)
        embedding_size = 64
        window_size = 3
        min_word = 5
        down_sampling = 1e-2
        ft_model = FastText(word_tokenized_corpus,
                            size=embedding_size,
                            window=window_size,
                            min_count=min_word,
                            sample=down_sampling,
                            sg=1,
                            iter=100)
        # pickle.dump(ft_model, open("ft_model.p", "wb"))
        # ft_model = pickle.load(open("ft_model.p", "rb"))
        # print(ft_model.wv['gün'])
        embedding_matrix = np.zeros((len(ft_model.wv.vocab) + 1, 64))
        for i, vec in enumerate(ft_model.wv.vectors):
            embedding_matrix[i] = vec
        vocab_size = len(ft_model.wv.vocab)+1
        # semantically_similar_words = {words: [item[0] for item in ft_model.wv.most_similar(
        #     [words], topn=5)]for words in ['gün', 'katil', 'ekonomi', 'haber', 'başbakan', 'siyaset']}

        # for k, v in semantically_similar_words.items():
        #     print(k+":"+str(v))
        # # print(ft_model.wv.similarity(w1='siyaset', w2='futbol'))
        # from sklearn.decomposition import PCA

        # all_similar_words = sum(
        #     [[k] + v for k, v in semantically_similar_words.items()], [])

        # # print(all_similar_words)
        # # print(type(all_similar_words))
        # # print(len(all_similar_words))

        # word_vectors = ft_model.wv[all_similar_words]

        # pca = PCA(n_components=2)

        # p_comps = pca.fit_transform(word_vectors)
        # word_names = all_similar_words

        # plt.figure(figsize=(18, 10))
        # plt.scatter(p_comps[:, 0], p_comps[:, 1], c='red')

        # for word_names, x, y in zip(word_names, p_comps[:, 0], p_comps[:, 1]):
        #     plt.annotate(word_names, xy=(x+0.06, y+0.03),
        #                  xytext=(0, 0), textcoords='offset points')
        # plt.show()

        labels = self.dataset.getClasses()
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(labels)
        labels = to_categorical(labels)
        return self.ft_model(features, labels, embedding_matrix,
                             vocab_size, ft_model)

    def setParameters(self):
        pass

    def ft_model(self, processed_features, labels, embedding_matrix, vocab_size, model):
        classes_num = self.dataset.getParameters()["classes_num"]
        X_train, X_test, y_train, y_test = train_test_split(
            processed_features, labels, test_size=self.TEST_SIZE, random_state=0)
        tokenizer = Tokenizer(num_words=self.NUM_WORDS)
        tokenizer.fit_on_texts(X_train)
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        lenth = find_max_length(X_train)
        # vocab_size = len(tokenizer.word_index) + 1
        X_train = pad_sequences(X_train, padding='post', maxlen=lenth)
        X_test = pad_sequences(X_test, padding='post', maxlen=lenth)
        model = Sequential()
        model.add(Embedding(vocab_size, 64, input_length=lenth,
                            weights=[embedding_matrix], trainable=True))
        model.add(Dropout(0.5))
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes_num, activation=self.ACTIVATION))
        model.compile(optimizer='adam',
                      loss=self.LOSSFUNC,
                      metrics=['accuracy'])
        es_callback = EarlyStopping(
            monitor='val_loss', patience=4)
        model.summary()

        history = model.fit(X_train, y_train,
                            epochs=self.EPOCHS,
                            verbose=1,
                            validation_data=(X_test, y_test),
                            batch_size=self.BATCH_SIZE, callbacks=[es_callback])
        loss, accuracy = model.evaluate(X_train, y_train, verbose=1)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
        return history


# H = Tweet17K(False, True)
# tp = TurkishProcessor(H)
# mm = FastTextModel(tp, H)
# mm.evaluate()
