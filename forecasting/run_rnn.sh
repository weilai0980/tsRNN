#!/bin/bash

python train_mv_rnn.py energy mv

python train_mv_rnn.py energy plain

python train_mv_rnn.py energy sep


python train_mv_rnn.py plant mv

python train_mv_rnn.py plant plain

python train_mv_rnn.py plant sep


python train_mv_rnn.py pm25 mv

python train_mv_rnn.py pm25 plain

python train_mv_rnn.py pm25 sep


python train_mv_rnn.py syn mv

python train_mv_rnn.py syn plain

python train_mv_rnn.py syn sep
