# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import math
import numpy as np
import torch
import torch.utils.data as data

from matplotlib import pyplot as plt

from collections import Counter

import json
import pickle
import pandas as pd
from tqdm import tqdm

from string import punctuation

# remove annoying characters
chars = {
    '\xc2\x82' : '',        # High code comma
    '\xc2\x84' : '',       # High code double comma
    '\xc2\x85' : '',      # Tripple dot
    '\xc2\x88' : '',        # High carat
    '\xc2\x91' : '',     # Forward single quote
    '\xc2\x92' : '',     # Reverse single quote
    '\xc2\x93' : '',     # Forward double quote
    '\xc2\x94' : '',     # Reverse double quote
    '\xc2\x95' : '',
    '\xc2\x96' : '',        # High hyphen
    '\xc2\x97' : '',       # Double hyphen
    '\xc2\x99' : '',
    '\xc2\xa0' : '',
    '\xc2\xa6' : '',        # Split vertical bar
    '\xc2\xab' : '',       # Double less than
    '\xc2\xbb' : '',       # Double greater than
    '\xc2\xbc' : '',      # one quarter
    '\xc2\xbd' : '',      # one half
    '\xc2\xbe' : '',      # three quarters
    '\xca\xbf' : '',     # c-single quote
    '\xcc\xa8' : '',         # modifier - under curve
    '\xcc\xb1' : ''          # modifier - under line
}

def pad_and_truncate(reviews, labels, seq_length):
    new_reviews = np.zeros((len(reviews), seq_length), dtype=int)
    new_labels = []
    for i, review in enumerate(reviews):
        len_review = len(review)
        
        if len_review < seq_length:
            zero = list(np.zeros(seq_length-len_review))
            new_review = zero+review
        elif len_review > seq_length:
            new_review = review[:seq_length]
        else:
            new_review = review
            
        new_reviews[i,:] = np.array(new_review)
        new_labels.append(labels[i])
    
    return new_reviews, np.array(new_labels)

def load_data(mypath):
    '''
    Reads in data from the given path and returns a list of reviews and a list of labels
    '''
    f = []
    labels = []
    for sentiment in ['pos','neg']:
        path = mypath+'/'+sentiment
        for (_, _, filenames) in os.walk(path):
            f.extend([path + '/' + filename for filename in filenames])
            if sentiment == 'pos':
                label = 1
            else:
                label = 0
            labels.extend([label for i in range(len(filenames))])
    
    reviews = []

    print(f'Reading reviews from {mypath}')
    for file in tqdm(f):
        with open(file, 'r', encoding='utf-8') as fb:
            reviews.append(fb.read())
            
    reviews = '\n'.join(reviews)
    reviews = reviews.lower()
    all_text = ''.join([c for c in reviews if c not in punctuation])
    all_text = ''.join([i if ord(i) < 128 else ' ' for i in all_text])
    reviews = all_text.split('\n')

    return reviews, labels


class IMDBDataset(data.Dataset):
    def __init__(self, train_or_test, seq_length):
        self._train_or_test = train_or_test

        self.seq_length = seq_length

        with open('data/imdb.vocab', 'r', encoding='utf-8') as f:
            vocab = f.read()

        vocab = vocab.split('\n')

        self._vocab_size = len(vocab)+2

        self._word_to_ix = {w:i+1 for i, w in enumerate(vocab)}
        self._ix_to_word = {value:key for key, value in self._word_to_ix.items()}
        
        # self._unknown_ix = len(self._word_to_ix.keys()) + 1
        self._unknown_ix = 0

        try:
            print('Checking for file...')
            with open(f'data/pckls/{train_or_test}_{seq_length}.pckl', 'rb') as f:
                d = pickle.load(f)
            print('found file!')
        
        except Exception as e:
            print(e)
            print(f'File not found. Creating {train_or_test}ing file for {seq_length} seq length')

            reviews, labels = load_data(f'data/{train_or_test}')
            # Tokenize reviews
            tokenized_reviews = []
            for review in reviews:
                tokenized_reviews.append([self._word_to_ix.get(w, self._unknown_ix) for w in review.split()])

            reviews, labels = pad_and_truncate(tokenized_reviews, labels, self.seq_length)

            d = {'reviews': reviews, 'labels': labels}

            with open(f'data/pckls/{train_or_test}_{seq_length}.pckl', 'wb') as f:
                pickle.dump(d, f)

        self._reviews = d['reviews']
        self._labels = d['labels']
        self._data_size = self._reviews.shape[0]



    def __getitem__(self, item):
        inputs = self._reviews[item]
        target = self._labels[item]
        return inputs, target

    def string_to_ix(self, sent):
        return [self._word_to_ix.get(w, self._unknown_ix) for w in sent.split()]

    def convert_to_string(self, word_ix):
        st = ''
        for ix in word_ix:
            if ix != 0:
                st += self._ix_to_word[ix] + ' '
        return st

    def __len__(self):
        return self._data_size

    def get_reviews(self):
        return self._reviews

    def get_targets(self):
        return self._labels

    @property
    def vocab_size(self):
        return self._vocab_size