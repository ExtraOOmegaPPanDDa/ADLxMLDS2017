#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 12:08:24 2017

@author: HSIN
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import sys
import os
import time
import pickle
import json
import string
import csv
import tensorflow as tf
import numpy as np
import hw2_model


data_path = sys.argv[1]
#data_path = '../../ADLxMLDS2017_DATA/hw2/MLDS_hw2_data/'
train_label_path = os.path.join(data_path, 'training_label.json')
test_label_path = os.path.join(data_path, 'testing_label.json')
train_feat_path = os.path.join(data_path, 'training_data/feat/')
test_feat_path = os.path.join(data_path, 'testing_data/feat/')
model_dir = 'model_dir'


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

vocabs_list = []
word_count_threshold = 1

for vocab in vocabs_dict:
    if vocabs_dict[vocab] >= word_count_threshold:
        vocabs_list.append(vocab)

vocabs_list = sorted(vocabs_list)    

vocabs_list = ['<pad>', '<s>', '</s>', '<unknown>'] + vocabs_list


del vocabs_dict
vocabs_dict = {}

idx = 0
for vocab in vocabs_list:
    vocabs_dict[vocab] = idx
    idx += 1

del vocabs_list

with open('vocabs_dict', 'wb') as f:
    pickle.dump(vocabs_dict, f)

with open('vocabs_dict', 'rb') as f:
    vocabs_dict = pickle.load(f)


print('Vocabs:', len(vocabs_dict))
# print(vocabs_dict)

    
inv_vocabs_dict = {v:k for k, v in vocabs_dict.items()}


attention_bool = False
match_test_bool = False

model = hw2_model.VCG_model(n_vocabs=len(vocabs_dict), for_testing = False, attention = attention_bool, match_test = match_test_bool)
max_sent_len = model.max_sent_len

epochs = 351

loss_record = []
epoch_time_record = []

with tf.Session() as sess:
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    saver = tf.train.Saver(tf.global_variables(), max_to_keep = 2)

    # scheduled_sampling_ratio = 0

    for epoch in range(epochs):
        
        # np.random.seed(seed = 46)
        indices = np.random.permutation(train_ids.shape[0])
        
        ids = train_ids[indices]
        captions = train_captions[indices]
        
        stime = time.time()
        
        idx = 0
        start_idx = idx
        end_idx = start_idx + model.batch_size

        # scheduled_sampling_ratio = scheduled_sampling_ratio + 1/epochs
        
        while(1):

            # scheduled_sampling_point = np.random.binomial(1, scheduled_sampling_ratio)
            # print(scheduled_sampling_ratio)
            # print(scheduled_sampling_point)
            
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
                # np.random.seed(seed = 46 * i)
                caption_indices = np.random.permutation(len(batch_captions[i]))
                selected_idx = caption_indices[0]
                
                selected_caption = batch_captions[i][selected_idx]
                
                selected_caption = clean_text(selected_caption)
                selected_caption = selected_caption.split(' ')
                
                
                
                if '' in selected_caption:
                    selected_caption = list(filter(lambda a: a != '', selected_caption))
                if ' ' in selected_caption:
                    selected_caption = list(filter(lambda a: a != ' ', selected_caption))
                
                selected_caption = ['<s>'] + selected_caption
                
                if len(selected_caption) < max_sent_len:
                    selected_caption = selected_caption + ['</s>']
                else:
                    selected_caption = selected_caption[:max_sent_len - 1] + ['</s>']
                
                selected_caption_mask = list(np.ones(len(selected_caption)))
                
                pad_pattern = ['<pad>'] * len(selected_caption)
                
                if len(selected_caption) < max_sent_len:
                    while len(selected_caption) < max_sent_len: 
                        selected_caption = selected_caption + pad_pattern
                        selected_caption_mask = selected_caption_mask + [0] * len(pad_pattern)
                        
                if len(selected_caption) > max_sent_len:
                    selected_caption = selected_caption[:max_sent_len]
                    selected_caption_mask = selected_caption_mask[:max_sent_len]
                    
                selected_caption = np.asarray(selected_caption + ['<pad>'])
                selected_caption_mask = np.asarray(selected_caption_mask + [0])
                
            
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
                                           model.feat_mask: batch_feat_masks,
                                           model.caption: batch_selected_captions,
                                           model.caption_mask: batch_selected_caption_masks
                                           })

            print("Epoch:", epoch, str(end_idx) + '/' + str(len(ids)), "Loss:", train_loss, 'Time:', time.time() - stime)

            start_idx += model.batch_size
            end_idx += model.batch_size
            
            
            if end_idx >= len(ids):
                break

        print("\t\tEpoch:", str(epoch) + '/' + str(epochs - 1), "Loss:", train_loss, 'Time:', time.time() - stime)
        print('\n\n')

        
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(model_dir, 'hw2_model.ckpt')
            saver.save(sess, checkpoint_path, global_step = epoch)
            print("Saver: Epoch-" + str(epoch), checkpoint_path)
            print('\n\n')


        loss_record.append(train_loss)
        epoch_time_record.append(time.time() - stime)




with open('loss_record','wb') as fp:
    pickle.dump(loss_record, fp)

with open('epoch_time_record','wb') as fp:
    pickle.dump(epoch_time_record, fp)



with open('loss_record','rb') as fp:
    loss_record = pickle.load(fp)

with open('epoch_time_record','rb') as fp:
    epoch_time_record = pickle.load(fp)




plt.plot(list(range(len(loss_record))), loss_record)
plt.title('model loss')
plt.ylim((0, 100))
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.show()
plt.savefig('model_loss.png')
plt.clf()


plt.plot(list(range(len(epoch_time_record))), epoch_time_record)
plt.title('Epoch Time')
plt.ylim((0, 100))
plt.ylabel('Time')
plt.xlabel('epoch')
# plt.show()
plt.savefig('model_epoch_time.png')
plt.clf()


print('Final Loss:', loss_record[-1])
print('Average Epoch Time:', np.mean(epoch_time_record))
        
       
            
    