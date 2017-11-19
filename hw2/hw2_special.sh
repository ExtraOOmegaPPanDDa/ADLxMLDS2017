#!/bin/bash
wget https://www.dropbox.com/s/c98ckpv0rxbfefm/hw2_model_special.ckpt-345.data-00000-of-00001?dl=1 -O model_dir_special/hw2_model_special.ckpt-345.data-00000-of-00001
python3 test_special.py $1 $2