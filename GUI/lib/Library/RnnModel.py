from .IModel import IModel
from .IProcessor import IProcessor
from .IDataset import IDataset

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
from tensorflow.keras.layers import Dense, Flatten, Embedding, Dropout, SimpleRNN
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import to_categorical
import pickle


class RnnModel (IModel):

    ACTIVATION = 'sigmoid'
    LOSSFUNC = 'binary_crossentropy'
    TEST_SIZE = 0.5
    NUM_WORDS = 2500
    EMBEDING_DIM = 64
    EPOCHS = 200
    BATCH_SIZE = 400
    VOCAB_SIZE = 0
    INPUT_LENGTH = 0

    def __init__(self, processor: IProcessor, dataset: IDataset, test_size=None):
        self.processor = processor
        self.dataset = dataset
        if test_size != None:
            self.TEST_SIZE = test_size

    def __str__(self):
        return 'RNN'

    def evaluate(self):

        path = self.dataset.getPath()
        try:
            features = pickle.load(open(f"{path}/preprocessed.p", "rb"))
        except:
            features = self.processor.process()
            pickle.dump(features, open(f"{path}/preprocessed.p", "wb"))
        labels = self.getLables()
        x_train, x_test, y_train, y_test = self.prepareData(features, labels)
        return self.rnn_model(x_train, x_test, y_train, y_test)

    def setParameters(self):
        pass

    def getLables(self):
        labels = self.dataset.getClasses()
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(labels)
        labels = to_categorical(labels)
        return labels

    def prepareData(self, processed_features, labels):
        # Split train & test
        x_train, x_test, y_train, y_test = train_test_split(
            processed_features, labels, test_size=self.TEST_SIZE, random_state=1)
        # Tokenize and transform to integer index
        tokenizer = Tokenizer(num_words=self.NUM_WORDS)
        tokenizer.fit_on_texts(x_train)
        x_train = tokenizer.texts_to_sequences(x_train)
        x_test = tokenizer.texts_to_sequences(x_test)
        self.INPUT_LENGTH = find_max_length(x_train)
        x_train = pad_sequences(x_train, maxlen=self.INPUT_LENGTH)
        x_test = pad_sequences(x_test, maxlen=self.INPUT_LENGTH)
        # Adding 1 because of reserved 0 index
        self.VOCAB_SIZE = len(tokenizer.word_index) + 1
        return x_train, x_test, y_train, y_test

    def rnn_model(self, x_train, x_test, y_train, y_test):
        classes_num = self.dataset.getParameters()["classes_num"]
        model = Sequential()
        model.add(Embedding(
            self.VOCAB_SIZE, self.EMBEDING_DIM, input_length=self.INPUT_LENGTH))
        model.add(SimpleRNN(128))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(classes_num, activation=self.ACTIVATION))
        model.compile(
            loss=self.LOSSFUNC,
            optimizer='adam',
            metrics=['accuracy']
        )
        es_callback = EarlyStopping(
            monitor='val_loss', patience=3)
        model.summary()

        history = model.fit(x_train, y_train,
                            epochs=self.EPOCHS,
                            verbose=1,
                            validation_data=(x_test, y_test),
                            batch_size=self.BATCH_SIZE, callbacks=[es_callback])
        loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
        return history


# H = Tweet3K(False, True)
# tp = TurkishProcessor(H)
# mm = RnnModel(tp, H)
# history = mm.evaluate()
# print(history.history['val_loss'])
