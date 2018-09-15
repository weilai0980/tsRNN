# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sys


def load_specific_data(data, _from, _to):
    data = data[_from: _to]
    features = np.array(data.iloc[:, : -1].values.tolist())
    target = np.array(data.iloc[:, -1: ].values.tolist())
    return np.concatenate([features, target], axis=-1)


def load_data():
    
    data = pd.read_csv('../../../../dataset/dataset_ts/nasdaq100_padding.csv')
    train = load_specific_data(data, 0, 35100)
    val = load_specific_data(data, 35100, 35100 + 2730)
    test = load_specific_data(data, 35100 + 2730, 35100 + 2730 * 2)
    return train, val, test


def preprocess_data(data, stat=None):
    if stat is None:
        # do the statistics
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)

        # in case the stdev=0, then we will get NaN.
        for i in range(len(data_std)):
            if data_std[i] < 0.00000001:
                data_std[i] = 1

        stat = (data_mean, data_std)
    else:
        data_mean, data_std = stat

    # norm.
    data = (data - data_mean) * 1.0 / data_std
    return data, stat


def process_data_dual(data, window_size):
    # get data.
    _features, _labels = data[:, : -1], data[:, -1]

    # init.
    features, history_labels, labels = [], [], []
    num_points = len(_features)
    num_valid_points = num_points - window_size + 1

    # split data.
    for ind in range(num_valid_points):
        features.append(_features[ind: ind + window_size - 1])
        history_labels.append(_labels[ind: ind + window_size - 1])
        labels.append(_labels[ind + window_size - 1])

    features = np.array(features)
    history_labels = np.array(history_labels)
    labels = np.array(labels)
    return features, history_labels, labels

def save_data_dual(data, label):
    
    features, history_labels, labels = data
    features.dump("./x{}_dual.dat".format(label))
    history_labels.dump("./hy{}_dual.dat".format(label))
    labels.dump("./y{}_dual.dat".format(label))

    
def process_data_plain(data, window_size):
    # get data.
    _features, _labels = data[:, : -1], data[:, -1]

    # init.
    features, history_labels, labels = [], [], []
    num_points = len(_features)
    num_valid_points = num_points - window_size + 1

    # split data.
    for ind in range(num_valid_points):
        
        concat_feat = np.append( _features[ind: ind + window_size - 1], 
                                np.expand_dims(_labels[ind: ind + window_size - 1], axis=1), 
                                axis = 1 )
        
        features.append( concat_feat )
        labels.append(_labels[ind + window_size - 1])

    features = np.array(features)
    labels = np.array(labels)
    return features, labels

def save_data_plain(data, label):
    
    features, labels = data
    features.dump("./x{}_nasdaq.dat".format(label))
    labels.dump("./y{}_nasdaq.dat".format(label))

def main(method_str):
    # load data.
    train, val, test = load_data()
    
    print np.shape(train)
    
    if method_str == 'dual':
        
        # preprocess data.
        train, stat = preprocess_data(train)
        test, _ = preprocess_data(test, stat)

        # process data.
        train = process_data_dual(train, window_size=10 + 1)
        test = process_data_dual(test, window_size=10 + 1)

        # save data.
        save_data_dual(train, 'train')
        save_data_dual(test, 'test')
    
    elif method_str == 'plain':
        
        # preprocess data.
        train, stat = preprocess_data(train)
        test, _ = preprocess_data(test, stat)

        # process data.
        train = process_data_plain(train, window_size=10 + 1)
        test = process_data_plain(test, window_size=10 + 1)

        # save data.
        save_data_plain(train, 'train')
        save_data_plain(test, 'test')



if __name__ == '__main__':
    
    method_str = str(sys.argv[1])

    main(method_str)