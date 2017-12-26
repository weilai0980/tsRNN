#!/usr/bin/python

# data processing packages
import sys

import collections
import hashlib
import numbers

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import *

# local 
from utils_libs import *


w = tf.get_variable('w',   [4, 10, 1, 7], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
w1 = tf.get_variable('w1', [4, 1, 7, 7], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())


w2 = tf.tile(w1, [1,1, 6, 1])

res = tf.shape( tf.reduce_sum(w*w1, 2) )
#res = tf.tensordot(w, w1, [[2], [1]])

#res = tf.concat( [w, w2], 3 )

with tf.Session() as sess:
    
    sess.run( [tf.global_variables_initializer()] )
    
    print 'shape: ', sess.run( res )
    
    