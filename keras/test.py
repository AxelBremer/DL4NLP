'''
This model predicts the sentiment for the 12500 reviews in the test set. It uses the model
specified by the --name argument. It predicts each review 100 times using dropout to approximate
a Bayesian approximation. The means and stds of both classes and the review text are saved 
in a csv file for further analysis.
'''

from __future__ import print_function
import numpy as np
import os
import pandas as pd
import json
import argparse

from utils import load_data, get_model

import matplotlib.pyplot as plt

import keras
from keras.preprocessing import sequence
from keras.models import load_model
from attention import Attention
from dropout_prediction import KerasDropoutPrediction
from tqdm import tqdm

B = 100
batch_size = 1250

d = {'target':[], 'mean0':[], 'std0':[], 'mean1':[], 'std1':[], 'text':[]}

# Create id to word dictionary to get the review text
word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+3) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
word_to_id["<UNUSED>"] = 3
id_to_word = {value:key for key,value in word_to_id.items()}
id_to_word[0] = ""

def test(config):
    with open(f'runs/{config.name}/args.txt', 'r') as f:
        config.__dict__ = json.load(f)

    _, _, (x_test, y_test) = load_data(config.seq_length, config.max_features) 

    # Load model
    model = load_model(f'runs/{config.name}/model.h5', custom_objects={'Attention':Attention})

    print(model.summary())

    # Add prediction dropout wrapper
    kdp = KerasDropoutPrediction(model)

    results = np.zeros((B,12500,2))

    num = 12500 / batch_size

    for i in tqdm(range(B)):
        for j in range(int(num)):
            joe = kdp.predict(x_test[batch_size*j:batch_size*(j+1),:], 1)
            results[i,batch_size*j:batch_size*(j+1),:] = joe

    mean = results.mean(axis=0)
    std = results.std(axis=0)

    d['target'].extend(y_test.tolist())
    d['mean0'].extend(mean[:,0].tolist())
    d['std0'].extend(std[:,0].tolist())
    d['mean1'].extend(mean[:,1].tolist())
    d['std1'].extend(std[:,1].tolist())
    for sent in x_test:
        d['text'].append(' '.join([id_to_word[x] for x in sent]))

    pd.DataFrame(d).to_csv(f'runs/{config.name}/results.csv')



if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--name', type=str, required=True, help='name of the run')

    config = parser.parse_args()

    # Train the model
    test(config)