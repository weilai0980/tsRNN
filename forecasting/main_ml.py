#!/usr/bin/python

# data processing packages
import numpy as np   
#import pandas as pd 

#from pandas import *
#from numpy import *
#from scipy import *

import random
import json

# machine leanring packages
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import xgboost as xgb
import sys

# local packages 
from ml_models import *

''' 
Arguments:

dataset_str: name of the dataset

'''

# ---- load data ----

dataset_str = str(sys.argv[1])
print("Load dataset %s"%dataset_str)

with open('config.json') as f:
    file_dict = json.load(f)
    
print(" ---- Loading files at", file_dict[dataset_str]) 
files_list = file_dict[dataset_str]


# ---- normalization ----

xtrain = np.load(files_list[0], encoding='latin1')
xtest = np.load(files_list[1], encoding='latin1')
ytrain = np.load(files_list[2], encoding='latin1')
ytest = np.load(files_list[3], encoding='latin1')

orig_shape = np.shape(xtrain)

tmplen = np.shape(xtrain)[0]
xtrain = np.reshape(xtrain, [tmplen, -1])

tmplen = np.shape(xtest)[0]
xtest = np.reshape(xtest, [tmplen, -1])


print(np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest))


# ---- log files ----
# hyperparameter log files
with open("../../ts_results/tsML.txt", "a") as text_file:
    text_file.write("\n---- %s, data shape %s \n"%(dataset_str, orig_shape))
        

# ---- begin to train models ----

# -- Bayesian regression

print("\n ---- Start to train Bayesian Regression")

from sklearn import linear_model

bayesian_reg = linear_model.BayesianRidge( normalize=True, fit_intercept=True )
bayesian_reg.fit(xtrain, ytrain)

# BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
#        fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
#        normalize=False, tol=0.001, verbose=False)

y_pred = bayesian_reg.predict ( xtest )
tmp_rmse = np.sqrt(np.mean( [(ytest[i]-ytmp)**2 for i, ytmp in enumerate( y_pred )] ))
tmp_mae = (np.mean( [abs(ytest[i]-ytmp) for i, ytmp in enumerate( y_pred )] ))

print(tmp_rmse, tmp_mae, '\n', ytest[:10], '\n', y_pred[:10])

with open("../../ts_results/tsML.txt", "a") as text_file:
    text_file.write( "BAYESIAN REGRESSION %f %f \n"%(tmp_rmse, tmp_mae))

    
# -- ElasticNet

print("\n ---- Start to train ElasticNet")

print(np.shape(xtrain), np.shape(ytrain))

err_min, err_list = enet_alpha_l1([0, 0.001, 0.01, 0.1, 1, 2], [0.0, 0.1, 0.3, 0.5, 0.7, 1.0], xtrain, ytrain,\
                                  xtest, ytest)
# 0  0.01 0.1 1 10 100

enet = ElasticNet(alpha = err_min[0], l1_ratio = err_min[1] )
enet.fit(xtrain, ytrain)
            
y_pred = enet.predict( xtest )
tmp_rmse = np.sqrt(np.mean( [(ytest[i]-ytmp)**2 for i,ytmp in enumerate( y_pred )]) )
tmp_mae = (np.mean([abs(ytest[i]-ytmp) for i,ytmp in enumerate( y_pred )]))
print(err_min[0], err_min[1], " : ", tmp_rmse, tmp_mae, '\n', ytest[:5], '\n', y_pred[:5])

with open("../../ts_results/tsML.txt", "a") as text_file:
    text_file.write( "ELASTIC NET %s\n"%str(err_min))



# -- Random forest performance
print("\n ---- Start to train Random Forest")

n_err, n_err_list = rf_n_depth_estimatior( 130, 25, xtrain, ytrain, xtest, ytest, False)

print("n_estimator, RMSE:", n_err)

with open("../../ts_results/tsML.txt", "a") as text_file:
    text_file.write( "RANDOM FOREST %s\n" %str(n_err) )

    
# -- GBT performance
print("\n ---- Start to train GBT")

fix_lr = 0.25

n_err, n_err_list = gbt_n_estimatior(301, xtrain, ytrain, xtest, ytest, fix_lr, False)

print("n_estimator, RMSE:", n_err)

depth_err, depth_err_list = gbt_tree_para(xtrain, ytrain, xtest, ytest, range(3,16), fix_lr, n_err[0], False)

print("depth, RMSE:", depth_err)

with open("../../ts_results/tsML.txt", "a") as text_file:
    text_file.write( "GBT %s\n" %str(depth_err) ) 


# -- XGBoosted performance
print("\n ---- Start to train XGBoosted")

fix_lr = 0.2


n_depth_err, n_depth_err_list = xgt_n_depth(fix_lr, 16, 51, xtrain, ytrain, xtest, ytest, False, 0)
print(" depth, number of rounds, RMSE:", n_depth_err)


l2_err, l2_err_list = xgt_l2(fix_lr, n_depth_err[0], n_depth_err[1], xtrain, ytrain, xtest, ytest,\
                             [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], False, 0)
print(" l2, RMSE:", l2_err)

print(l2_err_list)


with open("../../ts_results/tsML.txt", "a") as text_file:
    text_file.write( "XGBOOSTED %s\n" %str(l2_err) )

    

