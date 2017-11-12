#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:02:04 2017

@author: HSIN
"""

import sys
import os
import time
import pickle
import json
import string
import csv
import tensorflow as tf
import numpy as np
import hw2_model_special


data_path = sys.argv[1]
output_path = sys.argv[2]
test_id_path = 'testing_id_special.txt'
test_feat_path = os.path.join(data_path, 'testing_data/feat/')
model_dir = 'model_dir_special'



def clean_text(text):
    
	# remove \n
	text = text.replace('\n',' ')

	# remove abbreviated beV
	text = text.replace("'m "," am ")
	text = text.replace("'re "," are ")
	text = text.replace("'s "," is ")
	text = text.replace("'ll "," will ")
	text = text.replace("n't "," not ")
	text = text.replace("'d "," would ")

	temp_text = ''
	for ch in text:
		if ch not in string.punctuation and not ch.isdigit():
			temp_text += ch
		else:
			temp_text += ' '

	text = temp_text.lower()

	return text



	
with open('vocabs_dict_special', 'rb') as f:
    vocabs_dict = pickle.load(f)

    
inv_vocabs_dict = {v:k for k, v in vocabs_dict.items()}


test_ids = []
f = open(test_id_path, 'r')
for line in f:
    test_ids.append(line.replace('\n',''))

model = hw2_model_special.VCG_model(n_vocabs=len(vocabs_dict), for_testing = True)

with tf.Session() as sess:
    
    results = []
    
    for i in range(len(test_ids)):
        init = tf.global_variables_initializer()
        sess.run(init)
        
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            
        sess.run(tf.global_variables())
        
        test_id = test_ids[i]
        
        test_feat = np.load(os.path.join(test_feat_path, test_id + '.npy'))
        test_feat_mask = np.ones(test_feat.shape[0])
            
        gen_words_idx, probs = sess.run([model.gen_words_idx, model.probs], 
                                       feed_dict={
                                               model.feat: np.asarray([test_feat]),
                                               model.feat_mask: np.asarray([test_feat_mask])
                                               })

        gen_words = []
        for k in range(len(gen_words_idx)):
            gen_words.append(inv_vocabs_dict.get(gen_words_idx[k], '<unknown>'))
            
        gen_words = np.asarray(gen_words)
        
        punctuation = np.argmax(gen_words == '</s>') + 1
        
        if punctuation > 1:
            gen_words = gen_words[:punctuation]
        
        caption_words = []
        aword_count = 0
        isword_count = 0
        for word in gen_words:
            if word not in ['<s>', '</s>', '<pad>', '<unknown>']:
                if word == 'a':
                    aword_count += 1
                    if aword_count <= 2:
                        caption_words.append(word)
                elif word == 'is':
                    isword_count += 1
                    if isword_count <= 1:
                        caption_words.append(word)
                else:
                    caption_words.append(word)
        
        test_caption_len_set = 7
        if len(caption_words) > test_caption_len_set:
            caption_words = caption_words[:test_caption_len_set]
        
        while len(caption_words) < test_caption_len_set:
            caption_words.append('<pad>')
        
        
        caption = ' '.join(caption_words)
        res = []
        res.append(test_id)
        res.append(caption)
        
        print('['+str(test_id)+']')
        print(caption)
        print('\n')
        
        results.append(res)
    
    
    f = open(output_path, 'w')
    w = csv.writer(f)
    w.writerows(results)
    f.close()



