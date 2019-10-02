from __future__ import print_function
import numpy as np
import os
import json
import argparse

import matplotlib.pyplot as plt

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from attention import Attention

from utils import load_data, get_model

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def train(config):
    if not os.path.exists(f'runs/{config.name}'):
        os.makedirs(f'runs/{config.name}')

    print(config.__dict__)

    with open(f'runs/{config.name}/args.txt', 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(config.seq_length, config.max_features) 

    print(x_train.shape)

    model = get_model(seq_length=config.seq_length, embed_dim=config.embed_dim, hidden_dim=config.hidden_dim, 
                      n_layers=config.n_layers, dropout=config.dropout, bidirectional=config.bidirectional, 
                      recurrent_dropout=config.recurrent_dropout, max_features=config.max_features, attention=config.attention, weight_decay=config.weight_decay)

    print(model.summary())

    checkpoint = ModelCheckpoint(f'runs/{config.name}/model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    print('Train...')
    history = model.fit(x_train, y_train,
            batch_size=config.batch_size,
            epochs=config.train_epochs,
            validation_data=[x_val, y_val],
            shuffle=True,
            callbacks=[checkpoint])


    with open(f'runs/{config.name}/history.json', 'w') as f:
        json.dump(history.history, f)


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.savefig(f'runs/{config.name}/loss.png')

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['acc', 'val_acc'])
    plt.savefig(f'runs/{config.name}/acc.png')

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--name', type=str, required=True, help='name of run')
    parser.add_argument('--seq_length', type=int, default=200, help='Dimensionality of input sequence')
    parser.add_argument('--max_features', type=int, default=20000, help='Max vocab size')
    parser.add_argument('--embed_dim', type=int, default=300, help='Dimensionality of the embeddings')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Number of hidden unit')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for Adam')
    parser.add_argument('--dropout', type=float, default=0.5, help='Drop out rate')
    parser.add_argument('--recurrent_dropout', type=float, default=0.5, help='recurrent dropout rate')
    parser.add_argument('--train_epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--attention', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")


    config = parser.parse_args()

    # Train the model
    train(config)