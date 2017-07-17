# data processing packages
import numpy as np   
import pandas as pd 

from scipy import stats
from scipy import linalg
from scipy import *
 
import matplotlib as mplt
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
# matplotlib.style.use('ggplot')
import itertools

import random

from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def conti_normalization_train_dta(dta_df):
    
    return preprocessing.scale(dta_df)

def conti_normalization_test_dta(dta_df, train_df):
    
    mean_dim = np.mean(train_df, axis=0)
    std_dim = np.std(train_df, axis=0)
        
    df=pd.DataFrame()
    cols = train_df.columns
    idx=0
    
    for i in cols:
        df[i] = (dta_df[i]- mean_dim[idx])*1.0/std_dim[idx]
        idx=idx+1
        
    return df.as_matrix()


# ddata preprocessing utilities
def y_distribution_plot( ylist, title_str ):
    
    fig = plt.figure()
    n, bins, patches = plt.hist(ylist, normed=1, facecolor='green', alpha=0.75)
#   y = mlab.normpdf( bins, mu, sigma)
#   l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('Values')
    plt.ylabel('Probability')
    plt.title(title_str)
#   plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
#   fig.savefig('./results/classDis.jpg', format='jpg', bbox_inches='tight')


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


def multivariate_ts_plot( dta_df, title_str ):
        
    matplotlib.rcParams.update({'font.size': 15})
    figure_size = (15.4,7)
    legend_font = 8.5
    fig = plt.figure()
    fig.set_size_inches( figure_size )
    
    tmpt = range(dta_df.shape[0])
    for i in dta_df.columns:
        
        tmpx = list(dta_df[i])    
        plt.plot( tmpt, tmpx, label= i )

#     axes = plt.gca()
#     axes.set_xlim([1, tdf.shape[0]+10])
#     axes.set_ylim([yrange[0], yrange[1]])

    # plt.plot( list( clean_tdf['value']), color='g' )

    # plt.xticks([1,2,3,4,5], ['5','10','15','20','25'] )
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # axes.xaxis.set_ticks( ['5m', '10m', '15m','20m'] ) 
    # axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title( title_str )
    plt.ylabel('Value')
    plt.xlabel('Time')
    # plt.legend( loc='upper left',fontsize=12 )
    plt.legend(loc='upper left')
    #     bbox_to_anchor=(0., 1.0, 1., .10),
    #            loc=0,
    #            ncol=5, mode="expand", borderaxespad=0., fontsize= legend_font , numpoints=1 )
    
    
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
    if len( shape_arr ) == 2:
      
        res_x = []
        for i in range(cnt):
            res_x.append( [(tmp_x[i][j], tmp_x[i][j] - tmp_x[i][j-1])  for j in range(1,steps)] )          
#            res_x.append([])
#            for j in range(1,colcnt):
#                res_x[-1].append( (tmp_x[i][j], tmp_x[i][j] - tmp_x[i][j-1]) )
            
    elif len( shape_arr ) == 3:
        
        n_dim = shape_arr[2]
    
        res_x = []
        for i in range(cnt):
            res_x.append([])
            for j in range(steps):
                res_x[-1].append( [ (tmp_x[i][j][k], tmp_x[i][j][k] - tmp_x[i][j-1][k])  for k in range(1,n_dim)] )    
                        
    res_x = np.array(res_x)
    
    return np.reshape(res_x, [cnt, steps, -1]


def prepare_train_test_data(steps, bool_add_trend, xtrain, xtest, ytrain, ytest, files_list):
    
    
    PARA_STEPS = steps
    PARA_ADD_TREND = bool_add_trend
                      
    xtr = np.load(files_list[0])
    xts = np.load(files_list[1])
    ytr = np.load(files_list[2])
    yts = np.load(files_list[3]) 
                      
    xtrain_df = pd.DataFrame( xtr )
    xtest_df  = pd.DataFrame( xts )
#    ytrain_df = pd.DataFrame( flatten_features(np.load(files_list[2])) )
#    ytest_df  = pd.DataFrame( flatten_features(np.load(files_list[3])) )
                    
    cnt_tr = len(xtr)
    cnt_ts = len(xts)      
                      
    # integrate trends
    if PARA_ADD_TREND == True:
        
        trend_xtrain = expand_x_trend( xtr )
        trend_xtest  = expand_x_trend( xts )
    
        tmp_xtrain = np.reshape( trend_xtrain, [cnt_tr, -1 ] )
        tmp_xtest  = np.reshape( trend_xtest,  [cnt_ts, -1 ] )
    
        xtrain_df = pd.DataFrame( tmp_xtrain )
        xtest_df  = pd.DataFrame( tmp_xtest )
        
#   normalize x in training and testing datasets
    xtest = conti_normalization_test_dta( xtest_df, xtrain_df)
    xtrain= conti_normalization_train_dta(xtrain_df)

#   trend enhanced
#        xtest  = np.reshape( xtest,  [-1, (PARA_STEPS-1), 2 ] )
#        xtrain = np.reshape( xtrain, [-1, (PARA_STEPS-1), 2 ] )
        
#    else:
#   normalize x in training and testing datasets
#        xtest = conti_normalization_test_dta(xtest_df, xtrain_df)
#        xtrain= conti_normalization_train_dta(xtrain_df)

#    ytrain = ytrain_df.as_matrix()
#    ytest  = ytest_df.as_matrix()
        
    return xtrain, ytrain, xtest, ytest


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


def flatten_features(dta):
    tmplen = np.shape(dta)[0]
    return np.reshape(dta, [tmplen,-1])