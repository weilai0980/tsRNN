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

# local packages 
from utils_data_prep import *
from ml_models import *

import sys


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

# ---- normalization ----
xtrain, ytrain, xtest, ytest, _, _ = \
prepare_train_test_data( False, file_dic[dataset_str] )

print np.shape(xtrain), np.shape(ytrain), np.shape(xtest), np.shape(ytest)


# ---- begin to train models ----

#with open("res/tsML.txt", "w") as text_file:
#    text_file.close()


# GBT performance
print "\nStart to train GBT"

fix_lr = 0.25

n_err, n_err_list = gbt_n_estimatior(301, xtrain, ytrain, xtest, ytest, fix_lr, False)

print "n_estimator, RMSE:", n_err

depth_err, depth_err_list = gbt_tree_para( xtrain, ytrain, xtest, ytest, range(3,16), fix_lr, n_err[0],False )

print "depth, RMSE:", depth_err

with open("res/tsML.txt", "a") as text_file:
    text_file.write( "GBT %s\n" %str(depth_err) ) 


# XGBoosted performance
print "\nStart to train XGBoosted"

fix_lr = 0.2

n_depth_err, n_depth_err_list = xgt_n_depth( fix_lr, 16, 51, xtrain, ytrain, xtest, ytest, False,0)

print " depth, number of rounds, RMSE:", n_depth_err

l2_err, l2_err_list = xgt_l2( fix_lr, n_depth_err[0], n_depth_err[1], xtrain, ytrain, xtest, ytest,\
                    [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], False, 0)

print " l2, RMSE:", l2_err

with open("res/tsML.txt", "a") as text_file:
    text_file.write( "XGBOOSTED %s\n" %str(l2_err) )


    
# Random forest performance
print "\nStart to train Random Forest"

n_err, n_err_list = rf_n_depth_estimatior( 130, 25, xtrain, ytrain, xtest, ytest,False)

print "n_estimator, RMSE:", n_err

with open("res/tsML.txt", "a") as text_file:
    text_file.write( "RANDOM FOREST %s\n" %str(n_err) )

    
    
# Bayesian regression
print "\nStart to train Bayesian Regression"

from sklearn import linear_model

bayesian_reg = linear_model.BayesianRidge( normalize=True, fit_intercept=True )
bayesian_reg.fit(xtrain, ytrain)

# BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
#        fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
#        normalize=False, tol=0.001, verbose=False)

y_pred = bayesian_reg.predict ( xtest )

print np.shape(ytest), np.shape(xtest)
tmpval= sqrt(mean( [(ytest[i]-ytmp)**2 for i,ytmp in enumerate( y_pred )] ))

with open("res/tsML.txt", "a") as text_file:
    text_file.write( "BAYESIAN REGRESSION %f\n"%tmpval)

    
    
# ElasticNet
print "\nStart to train ElasticNet"


err_min, err_list = enet_alpha_l1( [0, 0.001, 0.01, 0.1, 1] , [0.7] , xtrain, ytrain, xtest, ytest)
# 0  0.01 0.1 1 10 100
print err_min

with open("res/tsML.txt", "a") as text_file:
    text_file.write( "ELASTIC NET %s\n"%str(err_min))
