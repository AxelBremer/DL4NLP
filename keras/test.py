from __future__ import print_function
import numpy as np
import os
import pandas as pd
import json
import argparse

from utils import load_data, get_model

import matplotlib.pyplot as plt

from keras.preprocessing import sequence
from keras.models import load_model
from attention import Attention
from tqdm import tqdm

B = 100

d = {'target':[], 'mean0':[], 'std0':[], 'mean1':[], 'std1':[]}

def test(config):
    with open(f'runs/{config.name}/args.txt', 'r') as f:
        config.__dict__ = json.load(f)

    _, (x_test, y_test) = load_data(config.seq_length, config.max_features) 

    model = load_model(f'runs/{config.name}/model.h5', custom_objects={'Attention':kapre.time_frequency.Melspectrogram})

    print(model.summary())

    results = np.zeros((B,25000,2))

    for i in tqdm(range(B)):
        results[i,:,:] = model.predict(x_test, batch_size=1028, verbose=1)

    mean = results.mean(axis=0)
    std = results.std(axis=0)

    d['target'].extend(y_test.tolist())
    d['mean0'].extend(mean[:,0].tolist())
    d['std0'].extend(std[:,0].tolist())
    d['mean1'].extend(mean[:,1].tolist())
    d['std1'].extend(std[:,1].tolist())

    pd.DataFrame(d).to_csv(f'runs/{config.name}/results.csv')



if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--name', type=str, required=True, help='name of the run')

    config = parser.parse_args()

    # Train the model
    test(config)