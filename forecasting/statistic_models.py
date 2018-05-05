#!/usr/bin/python
from numpy import prod
import math

import sys
import os

# statiscal models
import statsmodels.api as sm

# local packages
from utils_libs import *
from utils_data_prep import *

# ---- LOADING DATA ----

if len(sys.argv)<=1:
    print '----- [ERROR] specifiy dataset'
    

dataset_str = str(sys.argv[1])
print "Load dataset %s"%dataset_str

file_dic = {}

file_addr = ["../../dataset/dataset_ts/energy_xtrain.dat", \
             "../../dataset/dataset_ts/energy_xtest.dat", \
             "../../dataset/dataset_ts/energy_ytrain.dat", \
             "../../dataset/dataset_ts/energy_ytest.dat"]
file_dic.update( {"energy": file_addr} )

file_addr = ["../../dataset/dataset_ts/pm25_xtrain.dat", \
             "../../dataset/dataset_ts/pm25_xtest.dat", \
             "../../dataset/dataset_ts/pm25_ytrain.dat", \
             "../../dataset/dataset_ts/pm25_ytest.dat"]
file_dic.update( {"pm25": file_addr} )

file_addr = ["../../dataset/dataset_ts/plant_xtrain.dat", \
             "../../dataset/dataset_ts/plant_xtest.dat", \
             "../../dataset/dataset_ts/plant_ytrain.dat", \
             "../../dataset/dataset_ts/plant_ytest.dat"]
file_dic.update( {"plant": file_addr} )

file_addr = ["../../dataset/dataset_ts/syn_xtrain.dat", \
             "../../dataset/dataset_ts/syn_xtest.dat", \
             "../../dataset/dataset_ts/syn_ytrain.dat", \
             "../../dataset/dataset_ts/syn_ytest.dat"]
file_dic.update( {"syn": file_addr} )

files_list = file_dic[dataset_str]

# ---- Statistical input  ----
# input shape
# xts: [T 1]
# exts: [T D]

def oneshot_prediction_arimax( xtr, extr, xts, exts, arima_order, bool_add_ex ):
    
    xts_hat = []
        
    if bool_add_ex == True:
        mod = sm.tsa.statespace.SARIMAX(endog = xtr, exog = extr, order = arima_order)
            
        fit_res = mod.fit(disp=False)
        predict = fit_res.get_forecast( len(xts), exog = exts )
        predict_ci = predict.conf_int()
    
    else:
        mod = sm.tsa.statespace.SARIMAX(endog = xtr, order = arima_order)
            
        fit_res = mod.fit(disp=False )
        predict = fit_res.get_forecast( len(xts) )
        predict_ci = predict.conf_int()
        
    tr_predict = fit_res.get_prediction()
    tr_predict_ci = predict.conf_int()
    
    xtr_hat = tr_predict.predicted_mean 
    xts_hat = predict.predicted_mean
    
    # out-sample forecast, in-sample rmse, out-sample rmse      
    return xts_hat, xtr_hat, sqrt(mean((xts - np.asarray(xts_hat))*(xts - np.asarray(xts_hat)))), \
sqrt(mean((xtr - np.asarray(xtr_hat))*(xtr - np.asarray(xtr_hat))))

def oneshot_prediction_strx( xtr, extr, xts, exts, bool_add_ex ):
    
    xts_hat = []
    
    if bool_add_ex == True:
        roll_mod = sm.tsa.UnobservedComponents(endog = xtr, exog = extr, level= 'local linear trend', trend = True )
            
        fit_res = roll_mod.fit(disp=False)
        predict = fit_res.get_forecast(len(xts), exog = exts)
        predict_ci = predict.conf_int()
            
    else:
        roll_mod = sm.tsa.UnobservedComponents(endog = xtr, level= 'local linear trend', trend = True )
            
        fit_res = roll_mod.fit(disp=False)
        predict = fit_res.get_forecast(len(xts))
        predict_ci = predict.conf_int()
            
    tr_predict = fit_res.get_prediction()
    tr_predict_ci = predict.conf_int()
            
    xts_hat = predict.predicted_mean
    xtr_hat = tr_predict.predicted_mean    
    
    # out-sample forecast, in-sample rmse, out-sample rmse      
    return xts_hat, xtr_hat, sqrt(mean((xts - np.asarray(xts_hat))*(xts - np.asarray(xts_hat)))), \
sqrt(mean((xtr - np.asarray(xtr_hat))*(xtr - np.asarray(xtr_hat))))



def roll_prediction_arimax( xtr, extr, xts, exts, training_order, arima_order, bool_add_ex, log_file ):
    roll_x = xtr
    roll_ex= extr
    
    exdim = len(exts[0])
    xts_hat = []
    
    for i in range(len(xts)):
        
        tmp_x = roll_x[-training_order:]
        tmp_ex = roll_ex[-training_order:]
        
        print 'test on: ', i 
        
        if bool_add_ex == True:
            roll_mod = sm.tsa.statespace.SARIMAX(endog = tmp_x, exog = tmp_ex, order = arima_order)
            
            fit_res = roll_mod.fit(disp=False)
            predict = fit_res.get_forecast(1, exog = np.reshape(exts[i], [1, exdim]))
            predict_ci = predict.conf_int()
            
        else:
            roll_mod = sm.tsa.statespace.SARIMAX(endog = tmp_x, order = arima_order)
            
            fit_res = roll_mod.fit(disp=False)
            predict = fit_res.get_forecast(1)
            predict_ci = predict.conf_int()
            
        xts_hat.append(predict.predicted_mean)
        
        roll_x  = np.concatenate( (roll_x,   xts[i:i+1]) )
        roll_ex = np.concatenate( (roll_ex, exts[i:i+1]) )
        
        with open(log_file, "a") as text_file:
            text_file.write( "%f %f\n"%(xts[i], predict.predicted_mean) )
        
    # return rooted mse    
    return xts_hat, sqrt(mean((xts - np.asarray(xts_hat))*(xts - np.asarray(xts_hat))))

def roll_prediction_strx( xtr, extr, xts, exts, training_order, bool_add_ex ):
    roll_x = xtr
    roll_ex= extr
    
    exdim = len(exts[0])
    xts_hat = []
    
    for i in range(len(xts)):
        
        tmp_x = roll_x[-training_order:]
        tmp_ex = roll_ex[-training_order:]
        
        # --- test 
        tmp_x1  = np.expand_dims(np.asarray(roll_x[-training_order-100:-100]), 0)
        tmp_ex1 = np.asarray(roll_ex[-training_order-100:-100])
        tmp_x   = np.expand_dims(np.asarray(tmp_x), 0)
        tmp_ex  = np.asarray(tmp_ex)
        
        tmp_x = np.concatenate( [tmp_x, tmp_x1], 0 )
        print '!! test shape: ', np.shape(tmp_x)
        # ---
        
        print 'test on: ', i 
        
        if bool_add_ex == True:
            
            roll_mod = sm.tsa.UnobservedComponents(endog = tmp_x, exog = tmp_ex, level= 'local linear trend', trend = True )
            
            fit_res = roll_mod.fit(disp=False)
            predict = fit_res.get_forecast(1, exog = np.reshape(exts[i], [1, exdim]))
            predict_ci = predict.conf_int()
            
        else:
            roll_mod = sm.tsa.UnobservedComponents(endog = tmp_x, level= 'local linear trend', trend = True )
            
            fit_res = roll_mod.fit(disp=False)
            predict = fit_res.get_forecast(1)
            predict_ci = predict.conf_int()
            
        xts_hat.append(predict.predicted_mean)
        
        roll_x  = np.concatenate( (roll_x,   xts[i:i+1]) )
        roll_ex = np.concatenate( (roll_ex, exts[i:i+1]) )
        
    # return rooted mse    
    return xts_hat, sqrt(mean((xts - np.asarray(xts_hat))*(xts - np.asarray(xts_hat))))



# ---- Machine learning way input ----

# input data shape
# xts: [B T 1]
# exts: [B T D]
def roll_arimax( xts, exts, yts, arima_order, bool_add_ex ):
    
    import statsmodels.api as sm
    
    exdim = len(exts[0][0])
    xts_hat = []
    
    for i in range(len(xts)):
        
        tmp_x = xts[i]
        tmp_ex = exts[i]
        
        if i%100 == 0:
            print '--- finish by: ', i
        
        if bool_add_ex == True:
            roll_mod = sm.tsa.statespace.SARIMAX(endog = tmp_x, exog = tmp_ex, order = arima_order)
            
            fit_res = roll_mod.fit(disp=False)
            predict = fit_res.get_forecast(1, exog = np.reshape(exts[i][-1], [1, exdim]))
            predict_ci = predict.conf_int()
            
        else:
            roll_mod = sm.tsa.statespace.SARIMAX(endog = tmp_x, order = arima_order)
            
            fit_res = roll_mod.fit(disp=False)
            predict = fit_res.get_forecast(1)
            predict_ci = predict.conf_int()
            
        xts_hat.append(predict.predicted_mean[0])
        
        #roll_x  = np.concatenate( (roll_x,   xts[i:i+1]) )
        #roll_ex = np.concatenate( (roll_ex, exts[i:i+1]) )
        
        #with open(log_file, "a") as text_file:
        #    text_file.write( "%f %f\n"%(xts[i], predict.predicted_mean) )
        
    # return rooted mse
    return xts_hat, sqrt(mean((yts - np.asarray(xts_hat))*(yts - np.asarray(xts_hat)))),\
           mean( abs(yts - np.asarray(xts_hat)) ), mean( abs(yts - np.asarray(xts_hat))/(yts+1e-5) )

def roll_strx( xts, exts, yts, bool_add_ex ):
    
    import statsmodels.api as sm
    
    exdim = len(exts[0][0])
    xts_hat = []
    
    for i in range(len(xts)):
        
        tmp_x = xts[i]
        tmp_ex = exts[i]
        
        if i%100 == 0:
            print '--- finish by: ', i
        
        if bool_add_ex == True:
            
            roll_mod = sm.tsa.UnobservedComponents(endog = tmp_x, exog = tmp_ex, level= 'local linear trend', trend = True )
            
            fit_res = roll_mod.fit(disp=False)
            predict = fit_res.get_forecast(1, exog = np.reshape(tmp_ex[-1], [1, exdim]))
            predict_ci = predict.conf_int()
            
        else:
            roll_mod = sm.tsa.UnobservedComponents(endog = tmp_x, level= 'local linear trend', trend = True )
            
            fit_res = roll_mod.fit(disp=False)
            predict = fit_res.get_forecast(1)
            predict_ci = predict.conf_int()
            
        xts_hat.append(predict.predicted_mean)
        
    # return rooted mse    
    return xts_hat, sqrt(mean((yts - np.asarray(xts_hat))*(yts - np.asarray(xts_hat)))),\
           mean( abs(yts - np.asarray(xts_hat)) ), mean( abs(yts - np.asarray(xts_hat))/(yts+1e-5) )


# --- main process ---

#print 'Number of arguments:', len(sys.argv), 'arguments.'
print '--- Argument List:', str(sys.argv)

xtr = np.load(files_list[0])
xts = np.load(files_list[1])
ytr = np.load(files_list[2])
yts = np.load(files_list[3])

num_var = np.shape(xts)[2]

# split target and exogenous series 
[ex_ts, x_ts] = np.split(xts, [num_var-1], axis=2)

print '--- Shape of training data: ', np.shape(x_ts), np.shape(ex_ts)

with open("../../ts_results/tsML.txt", "a") as text_file:
    text_file.write( dataset_str+'\n' )

# --- arimax 


#order selection
errors = []
for ar_order in range(3, 5):
    for ma_order in range(1, 2):
        
        _, rmse, mae, mape = roll_arimax( x_ts, ex_ts, yts, [ar_order, 0, ma_order], True )
        
        print "--- ARIMAX parameters: ", ar_order, ma_order, rmse, mae, mape
        
        errors.append( [ar_order, ma_order, rmse, mae, mape] )

error_min = min(errors, key = lambda x: x[2])
print "--- ARIMAX: ", error_min

with open("../../ts_results/tsML.txt", "a") as text_file:
    text_file.write( "ARIMAX %s\n"%str(error_min))

'''
# --- structural time series models
_, rmse, mae, mape = roll_strx( x_ts, ex_ts, yts, True )

print "--- STRX: ", rmse, mae, mape
error_min = [rmse, mae, mape]

with open("../../ts_results/tsML.txt", "a") as text_file:
    text_file.write( "STRX %s\n"%str(error_min))
'''
