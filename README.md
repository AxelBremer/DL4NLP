# Sentiment Analysis

This repository contains scripts to train and test a sentiment analysis LSTM

## code/train.py
This script is used to train an lstm on the imdb dataset. 

### Usage:

python code/train.py --name $name_of_model [--arg arg]

Train.py always requires the --name argument. These are the other arguments and their default values:

| Name                | Explanation                              | Default value |
|---------------------|------------------------------------------|---------------|
| --seq_length        | Length of input sequence                 | 200           |
| --max_features      | Max vocab size                           | 20000         |
| --embed_dim         | Dimension of word embedding              |  300          |
| --hidden_dim        | Dimension of hidden layer                |  512          |
| --n_layers          | Number of LSTM layers                    | 2             |
| --batch_size        | Number of examples to process in a batch | 128           |
| --learning_rate     | Learning rate of the optimizer           | 0.005         |
| --dropout           | Dropout rate of the non-lstm layers      | 0.5           |
| --recurrent_dropout | Dropout rate of the recurrent dropout    | 0.5           |
| --train_epochs      | Number of training epochs                | 15            |
| --bidirectional     | Use bidirectional LSTM or not            | False         |
| --attention         | Use an attention layer or not            | False         |
| --weight_decay      | Parameter for L2 regularization          | 0.0001        |

A folder runs/$name_of_model is made and the model, metrics and configuration are saved in that folder.

## code/test.py

This script uses the given model to classify the test reviews. It predicts 100 times per review with dropout and uses the mean and std of the prediction to approximate Bayesian approximation.

### Usage:

python code/test.py --name $name_of_model

Test.py always requires the --name argument. These are the other arguments and their default values:


| Name                | Explanation                              | Default value |
|---------------------|------------------------------------------|---------------|
| --B                 | Number of times to classify each review  | 100           |
| --batch_size        | Number of examples to process in a batch | 1024          |

After running the mean and std of the predictions, the actual target and the review text are saved in runs/$name_of_model/results.csv

These are the only scripts needed to train and test a model. The contents of the other scripts are explained in their respective comments.

## Team Members

Axel Bremer

Rochelle Choenni

Tim de Haan

Shane Koppers
