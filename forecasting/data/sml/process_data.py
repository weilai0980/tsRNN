# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing


def load_specific_data(data, _from, _to):
    data = data[_from: _to]
    features = data[[
        '3:Temperature_Comedor_Sensor',
        '5:Weather_Temperature', '6:CO2_Comedor_Sensor',
        '7:CO2_Habitacion_Sensor', '8:Humedad_Comedor_Sensor',
        '9:Humedad_Habitacion_Sensor', '10:Lighting_Comedor_Sensor',
        '11:Lighting_Habitacion_Sensor', '13:Meteo_Exterior_Crepusculo',
        '14:Meteo_Exterior_Viento', '15:Meteo_Exterior_Sol_Oest',
        '16:Meteo_Exterior_Sol_Est', '17:Meteo_Exterior_Sol_Sud',
        '18:Meteo_Exterior_Piranometro', '22:Temperature_Exterior_Sensor',
        '23:Humedad_Exterior_Sensor']]
    features = np.array(features.values.tolist())

    target = data[['4:Temperature_Habitacion_Sensor']]
    target = np.array(target.values.tolist())
    return np.concatenate([features, target], axis=-1)


def load_data():
    # load original data.
    data1 = pd.read_csv('../../../../dataset/dataset_ts/NEW-DATA-1.T15.txt', sep=" ")
    data2 = pd.read_csv('../../../../dataset/dataset_ts/NEW-DATA-2.T15.txt', sep=" ")
    data = pd.concat([data1, data2])

    # load specific data.
    train = load_specific_data(data, 0, 3200)
    val = load_specific_data(data, 3200, 3200 + 400)
    test = load_specific_data(data, 3200 + 400, 3200 + 400 + 537)
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
    features.dump("../../../../dataset/dataset_ts/x{}_sml.dat".format(label))
    labels.dump("../../../../dataset/dataset_ts/y{}_sml.dat".format(label))
    
    
def main():
    # load data.
    train, val, test = load_data()

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
    main()
