#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 12:08:24 2017

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
train_label_path = os.path.join(data_path, 'training_label.json')
test_label_path = os.path.join(data_path, 'testing_label.json')
train_feat_path = os.path.join(data_path, 'training_data/feat/')
test_feat_path = os.path.join(data_path, 'testing_data/feat/')
model_dir = 'model_dir_special'




def clean_text(text):
    
    
	text = text.replace('\n',' ')
    
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




with open(train_label_path) as f:
    train_label = json.load(f)


train_captions = []
train_ids = []

for i in range(len(train_label)):
    train_captions.append(train_label[i]['caption'])
    train_ids.append(train_label[i]['id'])
	
train_captions = np.asarray(train_captions)
train_ids = np.asarray(train_ids)

vocabs_dict = {}
counter = 0
for captions in train_captions:
    for caption in captions:
        caption = clean_text(caption)
    
        for word in caption.split(' '):
            vocabs_dict[word] = vocabs_dict.get(word, 0) + 1
    
del vocabs_dict[""]

vocabs_list = [vocab for vocab in vocabs_dict]

vocabs_list.append('<s>')
vocabs_list.append('</s>')
vocabs_list.append('<pad>')
vocabs_list.append('<unknown>')


idx = 0
for vocab in vocabs_list:
    vocabs_dict[vocab] = idx
    idx += 1

del vocabs_list

with open('vocabs_dict_special', 'wb') as f:
    pickle.dump(vocabs_dict, f)

with open('vocabs_dict_special', 'rb') as f:
    vocabs_dict = pickle.load(f)

    
inv_vocabs_dict = {v:k for k, v in vocabs_dict.items()}

model = hw2_model_special.VCG_model(n_vocabs=len(vocabs_dict))
max_sent_len = model.max_sent_len + 1

epochs = 350

with tf.Session() as sess:
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    saver = tf.train.Saver(tf.global_variables())
    
    for epoch in range(epochs):
        
        # np.random.seed(seed = 46)
        indices = np.random.permutation(train_ids.shape[0])
        
        ids = train_ids[indices]
        captions = train_captions[indices]
        
        stime = time.time()
        
        idx = 0
        start_idx = idx
        end_idx = start_idx + model.batch_size
        
        while(1):
            
            batch_ids = ids[start_idx:end_idx]
            batch_captions = captions[start_idx:end_idx]
            
            batch_feats = []
            batch_feat_masks = []
            for i in range(len(batch_ids)):
                feat = np.load(os.path.join(train_feat_path, batch_ids[i] + '.npy'))
                batch_feats.append(feat)
                batch_feat_masks.append(np.ones(feat.shape[0]))
            
            batch_feats = np.asarray(batch_feats)
            batch_feat_masks = np.asarray(batch_feat_masks)
    
            
            batch_selected_captions = []
            batch_selected_caption_masks = []
            for i in range(len(batch_captions)):
                # np.random.seed(seed = 46*i)
                caption_indices = np.random.permutation(len(batch_captions[i]))
                selected_idx = caption_indices[0]
                
                selected_caption = batch_captions[i][selected_idx]
                
                selected_caption = clean_text(selected_caption)
                selected_caption = selected_caption.split(' ')
                selected_caption = np.append(['<s>'],selected_caption)
                selected_caption = np.append(selected_caption,['<\s>'])
                
                caption_len = len(selected_caption)
                
                selected_caption_mask = list(np.ones(len(selected_caption)))
                
                if len(selected_caption) < max_sent_len:
                    while len(selected_caption) < max_sent_len:
                        selected_caption = np.append(selected_caption,['<pad>'])
                        selected_caption_mask = selected_caption_mask + [0]
                else:
                    selected_caption = selected_caption[:max_sent_len]
                    selected_caption_mask = selected_caption_mask[:max_sent_len]
                
                selected_caption = np.asarray(selected_caption)
                selected_caption_mask = np.asarray(selected_caption_mask)
            
                indexed_selected_caption = []
                for word in selected_caption:
                    if word not in vocabs_dict:
                        indexed_selected_caption.append(vocabs_dict['<unknown>'])
                    else:
                        indexed_selected_caption.append(vocabs_dict[word])	
            
                indexed_selected_caption = np.asarray(indexed_selected_caption)
            
            
                batch_selected_captions.append(indexed_selected_caption)
                batch_selected_caption_masks.append(selected_caption_mask)
            
            batch_selected_captions = np.asarray(batch_selected_captions)
            batch_selected_caption_masks = np.asarray(batch_selected_caption_masks)
            
            train_loss,_ = sess.run([model.loss, model.opt],
                                   feed_dict={
                                           model.feat: batch_feats,
                                           model.feat_mask : batch_feat_masks,
                                           model.caption: batch_selected_captions,
                                           model.caption_mask: batch_selected_caption_masks
                                           })
    
    
            start_idx += model.batch_size
            end_idx += model.batch_size
            
            
            if end_idx >= len(ids):
                break
            
    
        
            print("Epoch:", epoch, str(end_idx) + '/' + str(len(ids)), "Loss:", train_loss, 'Time:', time.time() - stime)
        
        print("\t\tEpoch:", epoch, "Loss:", train_loss, 'Time:', time.time() - stime)
        print('\n\n')

        if epoch % 5 == 0:
            checkpoint_path = os.path.join(model_dir, 'hw2_model_special.ckpt')
            saver.save(sess, checkpoint_path, global_step = epoch)
            print("Saver: Epoch-" + str(epoch), checkpoint_path)
            print('\n\n')
            
            
    