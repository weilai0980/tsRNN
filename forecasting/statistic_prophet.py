#!/usr/bin/python

# data processing packages
import numpy as np   
import pandas as pd 

from pandas import *
from numpy import *
from scipy import *

import random
import sys

# local packages 
from utils_data_prep import *
from ml_models import *


dataset_str = str(sys.argv[1])
print "Load dataset %s"%dataset_str

# ---- DATA ----
file_dic = {}
    
file_addr = ["../../dataset/dataset_ts/air_xtrain.dat", \
                 "../../dataset/dataset_ts/air_xtest.dat",\
                 "../../dataset/dataset_ts/air_ytrain.dat", \
                 "../../dataset/dataset_ts/air_ytest.dat"]
file_dic.update( {"air": file_addr} )
    
file_addr = ["../../dataset/dataset_ts/energy_xtrain.dat", \
                 "../../dataset/dataset_ts/energy_xtest.dat",\
                 "../../dataset/dataset_ts/energy_ytrain.dat", \
                 "../../dataset/dataset_ts/energy_ytest.dat"]
file_dic.update( {"energy": file_addr} )
    
file_addr = ["../../dataset/dataset_ts/pm25_xtrain.dat", \
                 "../../dataset/dataset_ts/pm25_xtest.dat",\
                 "../../dataset/dataset_ts/pm25_ytrain.dat", \
                 "../../dataset/dataset_ts/pm25_ytest.dat"]
file_dic.update( {"pm25": file_addr} )

file_addr = ["../../dataset/dataset_ts/plant_xtrain.dat", \
                 "../../dataset/dataset_ts/plant_xtest.dat",\
                 "../../dataset/dataset_ts/plant_ytrain.dat", \
                 "../../dataset/dataset_ts/plant_ytest.dat"]
file_dic.update( {"plant": file_addr} )

file_addr = ["../../dataset/dataset_ts/temp_xtrain.dat", \
             "../../dataset/dataset_ts/temp_xtest.dat",\
             "../../dataset/dataset_ts/temp_ytrain.dat", \
             "../../dataset/dataset_ts/temp_ytest.dat"]
file_dic.update( {"temp": file_addr} )

file_addr = ["../../dataset/dataset_ts/syn_xtrain.dat", \
             "../../dataset/dataset_ts/syn_xtest.dat",\
             "../../dataset/dataset_ts/syn_ytrain.dat", \
             "../../dataset/dataset_ts/syn_ytest.dat"]
file_dic.update( {"syn": file_addr} )


# ---- normalization ----
xtrain, ytrain, xtest, ytest, _, _ = \
prepare_train_test_data( False, file_dic[dataset_str] )

print np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)


# ---- DATA power plant ----

para_win_size = 40

file_name = ['plant-irradiance.csv', 'plant-temperature.csv', 'weather-cloudcover.csv', \
             'weather-dewpoint.csv', 'weather-humidity.csv', 'weather-pressure.csv', \
             'weather-temperature.csv', 'weather-windbearing.csv', 'weather-windspeed.csv']

# load training data
addr = "../../dataset/ts/power_plant/training/"

idx = range(19)
cols = ['id', 'day'] + idx


file_path = addr + 'plant-power.csv'
dta_df = pd.read_csv( file_path, sep=',', names = cols, header = 0 )

tmp = np.asarray( dta_df[ dta_df['id']==1 ].drop(['id', 'day'], axis = 1) )
stack_ts = np.expand_dims( tmp.flatten(), -1 )

print "number of y values:", np.shape(stack_ts)

for i in file_name:
    
    file_path = addr + i
    dta_df = pd.read_csv( file_path ,sep=',', names = cols, header = 0 )
    
    tmp_ts = np.asarray( dta_df[ dta_df['id']==1 ].drop(['id', 'day'], axis=1) )
    var_ts = np.expand_dims( tmp_ts.flatten(), -1 )
    
    print 'Shape of the dataset: ', dta_df.shape, np.shape(var_ts)
    
    stack_ts = np.concatenate( (stack_ts, var_ts), axis=1 )

vari_name = ['power', 'irradiance', 'p-temperature', 'cloudcover', 'dewpoint', 'humidity', 'pressure', \
             'temperature', 'windbearing', 'windspeed']


dta_df = pd.DataFrame( stack_ts, columns = vari_name )

print dta_df.shape
print dta_df.columns

# ---- ----

target_col = 'power'
feature_cols = ['p-temperature', 'cloudcover', 'dewpoint', 'humidity', 'temperature', 'windbearing', 'windspeed']
# 'irradiance'
# 'pressure'

print "\n ---- Start to train Prophet"

from fbprophet import Prophet

# df['y'] = np.log(df['y'])
# df.head()

para_train_range = (0, 100)
para_test_range  = (5400, 6954)

y = dta_df[target_col].iloc[para_train_range[0]:para_train_range[1]]

dt = [i for i in range(para_train_range[0], para_train_range[1])]

df = pd.DataFrame( columns = ['y', 'ds'])
    
df['y'] = y
df['ds'] = np.asarray(dt)

print df.columns, df.shape

# ---- begin to train models ----

#  Fitting
prop = Prophet()
prop.fit(df);

# Prediction
future = prop.make_future_dataframe( periods = 1 )
print '---- prediction tail: ',future.tail()

forecast = prop.predict(future)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()




