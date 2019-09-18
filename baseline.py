from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from imdb_dataset import IMDBDataset


def train_baseline(config):  
    # Initialize the dataset and data loader (note the +1)
    train_set = IMDBDataset(train_or_test='train', seq_length=config.seq_length)
    test_set = IMDBDataset(train_or_test='test', seq_length=config.seq_length)

    text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    # text_classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    
    text_classifier.fit(train_set.get_reviews(), train_set.get_targets())

    predictions = text_classifier.predict(test_set.get_reviews())

    print(confusion_matrix(test_set.get_targets(), predictions))
    print(classification_report(test_set.get_targets(), predictions))
    print(accuracy_score(test_set.get_targets(), predictions))


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--seq_length', type=int, default=200, help='Dimensionality of input sequence')

    config = parser.parse_args()

    # Train the model
    train_baseline(config)