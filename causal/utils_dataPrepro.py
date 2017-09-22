# data processing packages
import numpy as np   
import pandas as pd 

from scipy import stats
from scipy import linalg
from scipy import *

import random

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def conti_normalization_train_dta(dta_df):
    
    return preprocessing.scale(dta_df)

def conti_normalization_test_dta(dta_df, train_df):
    
    mean_dim = np.mean(train_df, axis=0)
    std_dim = np.std(train_df, axis=0)
    
#    print '--test--', mean_dim, std_dim
    
    df=pd.DataFrame()
    cols = train_df.columns
    idx=0
    
#    print '--test--', cols
    
    for i in cols:
        df[i] = (dta_df[i]- mean_dim[idx])*1.0/std_dim[idx]
        idx=idx+1
        
    return df.as_matrix()


# for both univeriate and multi-variate cases
def instance_extraction( list_ts, win_size ):
    
    n = len(list_ts)
    if n < win_size:
        print "ERROR: SIZE"
        return
    
    listX = []
    listY = []
    for i in range(win_size, n):
        listX.append( list_ts[i-win_size:i] )
        listY.append( list_ts[i] )
        
    return listX, listY


# for the case of multiple independent and one target series 
# argu: list
# return: list
def instance_extraction_multiple_one( list_target, list_indepen, win_size ):
    
    n = len(list_target)
    if n < win_size:
        print "ERROR: SIZE"
        return
    
    listX = []
    listY = []
    for i in range(win_size, n):
        
        tmp  = list_indepen[i-win_size:i]
        tmp1 = np.reshape( list_target[i-win_size:i], [-1,1] )
        
        tmp  = np.append( tmp, tmp1 , axis = 1 )
                    
        listX.append( tmp )
        listY.append( list_target[i] )
        
    return listX, listY

def y_normalization( ylist ):
    tmpy = []
    for i in ylist:
        tmpy.append( math.log1p( i ) )
    return tmpy
        

    
#   to handle unbalance on Y distribution 
def upsampling_onY( xlist, ylist ):
    return 0


# Utilities 

# argu: np.matrix
# return: np.matrix
def normalize_y( y_mat, m, v ):
    
    tmpy = y_mat.reshape( (len(y_mat),) )
    tmpy = list(tmpy)
    
    resy = []
    
    for i in tmpy:
        resy.append( (i - m)*1.0/sqrt(v + 0.001) )
    
    return np.array(resy)        


def expand_x_local( local_size, data):
    
    list_data = list(data)
    cnt = len(list_data)
    steps = len(list_data[0])
    
    tmp_dta = []
    
    for i in range(cnt):
        tmp_dta.append([])
        for j in range(local_size-1,steps):
            tmp_dta[-1].append( list_data[i][j-local_size+1:j+1] )
    
    return np.array(tmp_dta)

# expand y, y_t -> y_1,...y_t
# argu: np.matrix
# return: np.matrix
def expand_y( x, y ):
    cnt = len(x)
    expand_y = []
    
    tmpx = list(x)
    tmpy = list(y)
    
    for i in range(cnt):
        tmp = tmpx[i][1:]
        tmp = np.append( tmp, tmpy[i] )
        
        expand_y.append( tmp )
    
    return np.array( expand_y )


def expand_x_trend( x ):
    
    shape = np.shape(x)
    cnt   = shape[0]
    steps = shape[1]
    
    tmp_x = list(x)
    if len( shape ) == 2:
      
        res_x = []
        for i in range(cnt):
            res_x.append( [ (tmp_x[i][j], tmp_x[i][j] - tmp_x[i][j-1]) for j in range(1,steps)] )          
            
    elif len( shape ) == 3:
        
        n_dim = shape[2]
    
        res_x = []
        for i in range(cnt):
            res_x.append([])
            for j in range(1, steps):
                res_x[-1].append( [ (tmp_x[i][j][k], tmp_x[i][j][k] - tmp_x[i][j-1][k]) for k in range(n_dim)] )    
                        
    res_x = np.array(res_x)
    
    return np.reshape(res_x, [cnt, steps-1, -1])


def prepare_train_test_data(bool_add_trend, files_list):
    
    PARA_ADD_TREND = bool_add_trend
                      
    xtr = np.load(files_list[0])
    xts = np.load(files_list[1])
    ytr = np.load(files_list[2])
    yts = np.load(files_list[3]) 
                    
    cnt_tr = len(xtr)
    cnt_ts = len(xts)   
    
    original_shape_tr = np.shape(xtr)
    original_shape_ts = np.shape(xts)
                      
    # integrate trends
    if PARA_ADD_TREND == True:
        
        trend_xtrain = expand_x_trend( xtr )
        trend_xtest  = expand_x_trend( xts )
    
        tmp_xtrain = np.reshape( trend_xtrain, [cnt_tr, -1 ] )
        tmp_xtest  = np.reshape( trend_xtest,  [cnt_ts, -1 ] )
    
        xtrain_df = pd.DataFrame( tmp_xtrain )
        xtest_df  = pd.DataFrame( tmp_xtest )
        
    else:
        
        tmp_xtrain = np.reshape( xtr, [cnt_tr, -1 ] )
        tmp_xtest  = np.reshape( xts, [cnt_ts, -1 ] )
    
        xtrain_df = pd.DataFrame( tmp_xtrain )
        xtest_df  = pd.DataFrame( tmp_xtest )
        
#   normalize x in training and testing datasets

#    print '--test--', xtest_df.shape, xtrain_df.shape

#    print '--test--', xtrain_df.iloc[:10]
    
#    print '--test--', np.mean(xtrain_df, axis=0)
    
#   test
#    xtest = xtest_df.as_matrix()
#    xtrain = xtrain_df.as_matrix()

    xtest = conti_normalization_test_dta(  xtest_df, xtrain_df )
    xtrain= conti_normalization_train_dta( xtrain_df )
        
    return xtrain, ytr, xtest, yts, original_shape_tr, original_shape_ts


def flatten_features(dta):
    tmplen = np.shape(dta)[0]
    return np.reshape(dta, [tmplen,-1])


def build_training_testing_data_4learning( dta_df, target_col, indep_col, \
                                para_uni_variate, para_train_test_split, para_win_size, \
                                para_train_range, para_test_range):
        
# univariate
    if para_uni_variate == True:
        
        x_all, y_all = instance_extraction( \
                       list(dta_df[target_col][ para_train_range[0]:para_train_range[1] ]), para_win_size )

# multiple independent and one target series
    else:
        dta_mat = dta_df[ indep_col ][ para_train_range[0]:para_train_range[1] ].as_matrix()
        x_all, y_all = instance_extraction_multiple_one( \
                        list(dta_df[ target_col ][ para_train_range[0]:para_train_range[1] ]),\
                                                 list(dta_mat),para_win_size )
# multivariate 
# x_all, y_all = instance_extraction( list(dta_df[['Open','High','Low','Volume']][:4000]), 100 )

# downsample the whole data
    total_idx = range(len(x_all))
    np.random.shuffle(total_idx)

    x_all = np.array(x_all)
    y_all = np.array(y_all)

    x_all = x_all[ total_idx[: int(1.0*len(total_idx))] ]
    y_all = y_all[ total_idx[: int(1.0*len(total_idx))] ]
    
    tmp_x_train, tmp_x_test, tmp_y_train, tmp_y_test = \
    train_test_split( x_all, y_all, test_size = 0.2, random_state = 20)

    
# training and testing data

# by extracting from subsequent data
    if para_train_test_split == False:
        
        if para_uni_variate == False:
            
            dta_mat = dta_df[ indep_col ][ para_test_range[0]:para_test_range[1] ].as_matrix()

            x_test, y_test = instance_extraction_multiple_one(\
                             list(dta_df[ target_col ][ para_test_range[0]:para_test_range[1] ]),\
                                                           list(dta_mat),para_win_size )
        else:
            x_test, y_test = instance_extraction( \
                             list(dta_df[ target_col ][ para_test_range[0]:para_test_range[1] ]), para_win_size )
            
        
        x_train = np.array(x_all)
        y_train = np.array(y_all)
        
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        
        return tmp_x_train, x_test, tmp_y_train, y_test

# by randomly split
    else:
        return  tmp_x_train, tmp_x_test, tmp_y_train, tmp_y_test



def build_training_testing_data_4statistics( dta_df, target_col, indep_col, \
                                para_uni_variate, \
                                para_train_range, para_test_range):
        
# univariate
    if para_uni_variate == True:
        
        x_train = dta_df[target_col][ para_train_range[0]:para_train_range[1] ].as_matrix()
        
        x_test  = dta_df[target_col][ para_test_range[0]:para_test_range[1] ].as_matrix()


# multiple independent and one target series
    else:
        
        indep_col.append(target_col)
        
        x_train = dta_df[ indep_col ][ para_train_range[0]:para_train_range[1] ].as_matrix()
        
        x_test  = dta_df[ indep_col ][ para_test_range[0]:para_test_range[1] ].as_matrix()
        
    return x_train, x_test

############################################
#TO DO

def prepare_trend_train_test_data( steps, bool_add_trend, xtrain_df, xtest_df, ytrain_df, ytest_df):
    
    PARA_STEPS = steps
    PARA_ADD_TREND = bool_add_trend
    
    # integrate trends
    if PARA_ADD_TREND == True:
        
        trend_xtrain = expand_x_trend( xtrain_df.as_matrix() )
        trend_xtest  = expand_x_trend( xtest_df.as_matrix() )
    
        tmp_xtrain = np.reshape( trend_xtrain, [-1, (PARA_STEPS-1)*2 ] )
        tmp_xtest  = np.reshape( trend_xtest,  [-1, (PARA_STEPS-1)*2 ] )
    
        xtrain_df = pd.DataFrame( tmp_xtrain )
        xtest_df  = pd.DataFrame( tmp_xtest )
    
#   normalize x in training and testing datasets
        xtest = conti_normalization_test_dta(xtest_df, xtrain_df)
        xtrain= conti_normalization_train_dta(xtrain_df)

#   trend enhanced
        xtest  = np.reshape( xtest,  [-1, (PARA_STEPS-1), 2 ] )
        xtrain = np.reshape( xtrain, [-1, (PARA_STEPS-1), 2 ] )
        
    else:
#   normalize x in training and testing datasets
        xtest = conti_normalization_test_dta(xtest_df, xtrain_df)
        xtrain= conti_normalization_train_dta(xtrain_df)

    ytrain = ytrain_df.as_matrix()
    ytest  = ytest_df.as_matrix()
        
    return xtrain, ytrain, xtest, ytest


def prepare_lastPoint_train_test_data( steps, bool_add_trend, xtrain_df, xtest_df, ytrain_df, ytest_df):
    
    PARA_STEPS = steps
    PARA_ADD_TREND = bool_add_trend
    
    # integrate trends
    if PARA_ADD_TREND == True:
        
        trend_xtrain = expand_x_trend( xtrain_df.as_matrix() )
        trend_xtest  = expand_x_trend( xtest_df.as_matrix() )
    
        tmp_xtrain = np.reshape( trend_xtrain, [-1, (PARA_STEPS-1)*2 ] )
        tmp_xtest  = np.reshape( trend_xtest,  [-1, (PARA_STEPS-1)*2 ] )
    
        xtrain_df = pd.DataFrame( tmp_xtrain )
        xtest_df  = pd.DataFrame( tmp_xtest )
    
#   normalize x in training and testing datasets
        xtest = conti_normalization_test_dta(xtest_df, xtrain_df)
        xtrain= conti_normalization_train_dta(xtrain_df)

#   trend enhanced
        xtest  = np.reshape( xtest,  [-1, (PARA_STEPS-1), 2 ] )
        xtrain = np.reshape( xtrain, [-1, (PARA_STEPS-1), 2 ] )
        
    else:
#   normalize x in training and testing datasets
        xtest = conti_normalization_test_dta(xtest_df, xtrain_df)
        xtrain= conti_normalization_train_dta(xtrain_df)

    ytrain = ytrain_df.as_matrix()
    ytest  = ytest_df.as_matrix()
        
    return xtrain, ytrain, xtest, ytest










