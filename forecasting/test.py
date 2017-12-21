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


w = tf.get_variable('w', [4, 6, 1], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
w1 = tf.get_variable('w1', [4, 1, 9], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())


res = tf.tensordot(w, w1, [[2], [1]])


with tf.Session() as sess:
    
    sess.run( [tf.global_variables_initializer()] )
    
    print sess.run( tf.shape(res) )
    
    