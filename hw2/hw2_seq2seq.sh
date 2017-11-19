#!/bin/bash
wget https://www.dropbox.com/s/luz2j03bhcni7wq/hw2_model_seq2seq.ckpt-500.data-00000-of-00001?dl=1 -O model_dir_seq2seq/hw2_model_seq2seq.ckpt-500.data-00000-of-00001
python3 model_seq2seq_test.py $1 $2 $3
python3 model_seq2seq_peer.py $1 $2 $3