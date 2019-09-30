from tensorflow.python.util import deprecation

from __future__ import print_function
import numpy as np
import os
import json
import argparse

from utils import load_data, get_model

import matplotlib.pyplot as plt

from keras.preprocessing import sequence
from keras.models import load_model


def test(config):
    with open(f'runs/{config.name}/args.txt', 'r') as f:
        config.__dict__ = json.load(f)

    (x_train, y_train), (x_test, y_test) = load_data(config.seq_length, config.max_features) 

    model = load_model(f'runs/{config.name}/model.h5')



if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--name', type=str, required=True, help='name of the run')

    config = parser.parse_args()

    # Train the model
    test(config)