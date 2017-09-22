#!/usr/bin/python

# data processing packages
import numpy as np   
import pandas as pd 

from pandas import *
from numpy import *
from scipy import *

import random

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
sys.path.append('../tsRNN/forecasting/')
from ml_models import *


addr = "../dataset/dataset_huarui/diagnosis.txt"
dta_df = pd.read_csv( addr ,sep=' ', header=None )
# print 'Shape of the dataset: ', dta_df.shape, dta_df.columns
dta_df[48] = dta_df[48]-1

x_col = range(48)
y_col = 48
    
x_all = np.array( dta_df[x_col]  )
y_all = np.array( dta_df[y_col] )

#     x_all = x_all[ total_idx[: int(1.0*len(total_idx))] ]
#     y_all = y_all[ total_idx[: int(1.0*len(total_idx))] ]
    
xtrain, xtest, ytrain, ytest = \
    train_test_split( x_all, y_all, test_size = 0.2, random_state = 20)
    
print np.shape(xtrain), np.shape(xtest), np.shape(ytrain), np.shape(ytest)

bool_clf = True

# GBT performance
#print "\nStart to train GBT"

#fix_lr = 0.25

#n_err, n_err_list = gbt_n_estimatior(301, xtrain, ytrain, xtest, ytest, fix_lr, bool_clf)

#print "n_estimator, RMSE:", n_err

#depth_err, depth_err_list = gbt_tree_para( xtrain, ytrain, xtest, ytest, range(3,16), fix_lr, n_err[0], bool_clf)

#print "depth, RMSE:", depth_err

#with open("res/diagnosis.txt", "a") as text_file:
#   text_file.write( "GBT %s\n" %str(depth_err) ) 


# XGBoosted performance
print "\nStart to train XGBoosted"

fix_lr = 0.2

n_depth_err, n_depth_err_list = xgt_n_depth( fix_lr, 16, 51, xtrain, ytrain, xtest, ytest, bool_clf, 11)

print " depth, number of rounds, RMSE:", n_depth_err

l2_err, l2_err_list = xgt_l2( fix_lr, n_depth_err[0], n_depth_err[1], xtrain, ytrain, xtest, ytest,\
                    [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], bool_clf, 11)

print " l2, RMSE:", l2_err

with open("res/diagnosis.txt", "a") as text_file:
    text_file.write( "XGBOOSTED %s\n" %str(l2_err) )


    
# Random forest performance
print "\nStart to train Random Forest"

n_err, n_err_list = rf_n_depth_estimatior( 130, 25, xtrain, ytrain, xtest, ytest, bool_clf)

print "n_estimator, RMSE:", n_err

with open("res/diagnosis.txt", "a") as text_file:
    text_file.write( "RANDOM FOREST %s\n" %str(n_err) )
