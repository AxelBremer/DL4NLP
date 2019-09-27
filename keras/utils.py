import numpy as np

import matplotlib.pyplot as plt

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from keras.utils import to_categorical
from attention import Attention

def load_data(seq_length, max_features):
    print('Loading data...')
    # save np.load
    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    # call load_data with allow_pickle implicitly set to true
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)


    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    # restore np.load for future normal usage
    np.load = np_load_old

    x_train = sequence.pad_sequences(x_train, maxlen=seq_length)
    x_test = sequence.pad_sequences(x_test, maxlen=seq_length)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)

def get_model(seq_length, embed_dim, hidden_dim, n_layers, dropout,  recurrent_dropout, bidirectional, max_features, attention):
    model = Sequential()
    model.add(Embedding(max_features, 300, input_length=seq_length))
    for i in range(n_layers-1):
        if bidirectional:
            model.add(Bidirectional(LSTM(units=hidden_dim, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)))
        else:
            model.add(LSTM(units=hidden_dim, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))
    if bidirectional:
        model.add(Bidirectional(LSTM(units=hidden_dim, return_sequences=attention, dropout=dropout, recurrent_dropout=recurrent_dropout)))
    else:
        model.add(LSTM(units=hidden_dim, return_sequences=attention, dropout=dropout, recurrent_dropout=recurrent_dropout))
    if attention: model.add(Attention(seq_length))
    model.add(Dropout(rate=1-dropout))
    model.add(Dense(2, activation='sigmoid'))
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    return model