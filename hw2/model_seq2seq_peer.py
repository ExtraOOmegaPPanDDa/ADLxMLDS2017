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





class VCG_model():
    def __init__(self, n_vocabs, for_testing = False, attention = False, match_test = False):
        
        self.feat_dim = 4096
        self.frame_dim  = 80
        self.max_sent_len = 15

        self.batch_size = 128
        
        self.layer_dim = 768
        self.lr = 0.001
        
        self.n_vocabs = n_vocabs
        
        self.word_embed = tf.Variable(tf.random_uniform([self.n_vocabs, self.layer_dim], -1, 1))
        
        self.encode_w = tf.Variable(tf.random_uniform([self.feat_dim, self.layer_dim], -1, 1))
        self.encode_b = tf.Variable(tf.zeros([self.layer_dim]))
        
        # self.encoding_rnn = tf.contrib.rnn.BasicLSTMCell(self.layer_dim, state_is_tuple = False)
        # self.decoding_rnn = tf.contrib.rnn.BasicLSTMCell(self.layer_dim, state_is_tuple = False)

        self.encoding_rnn = tf.contrib.rnn.GRUCell(self.layer_dim)
        self.decoding_rnn = tf.contrib.rnn.GRUCell(self.layer_dim)
        
        self.embed_w  = tf.Variable(tf.random_uniform([self.layer_dim, self.n_vocabs], -1, 1))
        self.embed_b  = tf.Variable(tf.zeros([self.n_vocabs]))

        if attention:
            self.attention_w  = tf.Variable(tf.random_uniform([self.encoding_rnn.state_size, self.decoding_rnn.state_size], -1, 1))
            self.attention_z  = tf.Variable(tf.random_uniform([self.batch_size, self.decoding_rnn.state_size], -1, 1))

        if not for_testing: # train
            
            self.feat = tf.placeholder(tf.float32, [self.batch_size, self.frame_dim, self.feat_dim])
            self.feat_mask = tf.placeholder(tf.float32, [self.batch_size, self.frame_dim])
            
            self.caption = tf.placeholder(tf.int32, [self.batch_size, self.max_sent_len + 1])
            self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.max_sent_len + 1])
            
            feat_flatten = tf.reshape(self.feat, [-1, self.feat_dim]) 
            feat_embed = tf.nn.xw_plus_b(feat_flatten, self.encode_w, self.encode_b)
            feat_embed = tf.reshape(feat_embed, [self.batch_size, self.frame_dim, self.layer_dim])
            
            encoding_state = tf.zeros([self.batch_size, self.encoding_rnn.state_size])
            decoding_state = tf.zeros([self.batch_size, self.decoding_rnn.state_size])

            if not attention: # train no-attention

                for i in range(self.frame_dim):
                    
                    with tf.variable_scope("encoding_rnn", reuse = (i > 0)):
                        encoding_output, encoding_state = self.encoding_rnn(feat_embed[:, i, :], encoding_state)
                    with tf.variable_scope("decoding_rnn", reuse = (i > 0)):
                        padding = tf.zeros([self.batch_size, self.layer_dim])
                        decoding_output, decoding_state = self.decoding_rnn(tf.concat([padding, encoding_output], 1), decoding_state)                      
                

                loss = 0

                for i in range(self.max_sent_len):

                    if i == 0:
                        step_embed = tf.nn.embedding_lookup(self.word_embed, self.caption[:, i])

                    with tf.variable_scope("encoding_rnn", reuse = True):
                        encoding_output, encoding_state = self.encoding_rnn(padding, encoding_state)
                    with tf.variable_scope("decoding_rnn", reuse = True):
                        decoding_output, decoding_state = self.decoding_rnn(tf.concat([step_embed, encoding_output], 1), decoding_state)
                        
                    labels = tf.concat([tf.expand_dims(tf.range(self.batch_size), 1), tf.expand_dims(self.caption[:, i + 1], 1)], 1)
                    labels = tf.sparse_to_dense(labels, tf.stack([self.batch_size, self.n_vocabs]), 1.0, 0.0)
                        
                    logits = tf.nn.xw_plus_b(decoding_output, self.embed_w, self.embed_b)
                    prob = tf.nn.softmax(logits)

                    max_prob_index = tf.argmax(logits, 1)

                    cross_entropy = self.caption_mask[:, i] * tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits) 
                    
                    sub_loss = tf.reduce_mean(cross_entropy)
                    loss += sub_loss

                    if match_test:
                        step_embed = tf.nn.embedding_lookup(self.word_embed, max_prob_index)

                    else:
                        step_embed = tf.nn.embedding_lookup(self.word_embed, self.caption[:, i + 1])
                        
                        
                    
                self.loss = loss
                self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


            else: # train attention

                
                h_record = []

                for i in range(self.frame_dim):
                    with tf.variable_scope("encoding_rnn", reuse = (i > 0)):
                        encoding_output, encoding_state = self.encoding_rnn(feat_embed[:, i, :], encoding_state)
                        h_record.append(encoding_state)

                    with tf.variable_scope("decoding_rnn", reuse = (i > 0)):
                        padding = tf.zeros([self.batch_size, self.layer_dim])
                        context_padding = tf.zeros([self.batch_size, self.decoding_rnn.state_size])
                        decoding_output, decoding_state = self.decoding_rnn(tf.concat([padding, encoding_output, context_padding], 1), decoding_state)

                h_record = tf.stack(h_record, axis = 1) 


                loss = 0

                z = self.attention_z

                for i in range(self.max_sent_len):

                    if i == 0:
                        step_embed = tf.nn.embedding_lookup(self.word_embed, self.caption[:, i])

                    with tf.variable_scope("encoding_rnn", reuse = True):
                        encoding_output, encoding_state = self.encoding_rnn(padding, encoding_state)

                    attention_contexts = []

                    h_record_flatten = tf.reshape(h_record, [-1, self.encoding_rnn.state_size])
                    hw = tf.matmul(h_record_flatten, self.attention_w)
                    hw = tf.reshape(hw, [self.batch_size, self.frame_dim, self.decoding_rnn.state_size])
                    
                    for j in range(self.batch_size):
                        alpha = tf.reduce_sum(tf.multiply(hw[j, :, :], z[j, :]), axis = 1)
                        alpha = tf.nn.softmax(alpha)
                        alpha = tf.expand_dims(alpha, 1)
                        context = tf.reduce_sum(tf.multiply(alpha, h_record[j, :, :]), axis = 0)
                        attention_contexts.append(context)

                    attention_contexts = tf.stack(attention_contexts)

                    with tf.variable_scope("decoding_rnn", reuse = True):
                        decoding_output, decoding_state = self.decoding_rnn(tf.concat([step_embed, encoding_output, attention_contexts], 1), decoding_state)
                        z = decoding_state
                        
                        
                    labels = tf.concat([tf.expand_dims(tf.range(self.batch_size), 1), tf.expand_dims(self.caption[:, i + 1], 1)], 1)
                    labels = tf.sparse_to_dense(labels, tf.stack([self.batch_size, self.n_vocabs]), 1.0, 0.0)
                        
                    logits = tf.nn.xw_plus_b(decoding_output, self.embed_w, self.embed_b)
                    prob = tf.nn.softmax(logits)

                    max_prob_index = tf.argmax(logits, 1)
                    
                    cross_entropy = self.caption_mask[:, i] * tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)

                    sub_loss = tf.reduce_mean(cross_entropy)
                    loss += sub_loss

                    if match_test:
                        step_embed = tf.nn.embedding_lookup(self.word_embed, max_prob_index)

                    else:
                        step_embed = tf.nn.embedding_lookup(self.word_embed, self.caption[:, i + 1])
                        
                        

                self.loss = loss
                self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        else: # test

            self.feat = tf.placeholder(tf.float32, [1, self.frame_dim, self.feat_dim])
            self.feat_mask = tf.placeholder(tf.float32, [1, self.frame_dim])
            
            feat_flatten = tf.reshape(self.feat, [-1, self.feat_dim])
            feat_embed = tf.nn.xw_plus_b(feat_flatten, self.encode_w, self.encode_b)
            feat_embed = tf.reshape(feat_embed, [1, self.frame_dim, self.layer_dim])
            
            encoding_state = tf.zeros([1, self.encoding_rnn.state_size])
            decoding_state = tf.zeros([1, self.decoding_rnn.state_size])
            
            
            
            gen_words_idx = []
            probs = []


            if not attention: # test no-attention
            
                for i in range(self.frame_dim):
                    with tf.variable_scope("encoding_rnn", reuse = (i > 0)):
                        encoding_output, encoding_state = self.encoding_rnn(feat_embed[:, i, :], encoding_state)
                    with tf.variable_scope("decoding_rnn", reuse = (i > 0)):
                        padding = tf.zeros([1, self.layer_dim])
                        decoding_output, decoding_state = self.decoding_rnn(tf.concat([padding, encoding_output], 1), decoding_state)
                        
                for i in range(self.max_sent_len):

                    if i == 0:
                        step_embed = tf.nn.embedding_lookup(self.word_embed, tf.ones([1], dtype = tf.int64))
                    
                    with tf.variable_scope("encoding_rnn", reuse = True):
                        encoding_output, encoding_state = self.encoding_rnn(padding, encoding_state)    
                    with tf.variable_scope("decoding_rnn", reuse = True):
                        decoding_output, decoding_state = self.decoding_rnn(tf.concat([step_embed, encoding_output], 1), decoding_state)
                    
                    logits = tf.nn.xw_plus_b(decoding_output, self.embed_w, self.embed_b)
                    prob = tf.nn.softmax(logits)
                    probs.append(prob)

                    max_prob_index = tf.argmax(logits, 1)[0]
                    gen_words_idx.append(max_prob_index)
                    
                    
                    step_embed = tf.nn.embedding_lookup(self.word_embed, max_prob_index)
                    step_embed = tf.expand_dims(step_embed, 0)
                    
                self.gen_words_idx = gen_words_idx
                self.probs = probs

            else: # test attention

                h_record = []

                for i in range(self.frame_dim):
                    with tf.variable_scope("encoding_rnn", reuse = (i > 0)):
                        encoding_output, encoding_state = self.encoding_rnn(feat_embed[:, i, :], encoding_state)
                        h_record.append(encoding_state)
                    with tf.variable_scope("decoding_rnn", reuse = (i > 0)):
                        padding = tf.zeros([1, self.layer_dim])
                        context_padding = tf.zeros([1, self.decoding_rnn.state_size])
                        decoding_output, decoding_state = self.decoding_rnn(tf.concat([padding, encoding_output, context_padding], 1), decoding_state)              
                
                h_record = tf.stack(h_record, axis = 1)

                z = self.attention_z

                for i in range(self.max_sent_len):

                    if i == 0:
                        step_embed = tf.nn.embedding_lookup(self.word_embed, tf.ones([1], dtype = tf.int64))
                    
                    with tf.variable_scope("encoding_rnn", reuse = True):
                        encoding_output, encoding_state = self.encoding_rnn(padding, encoding_state)

                    attention_contexts = []

                    h_record_flatten = tf.reshape(h_record, [-1, self.encoding_rnn.state_size])
                    hw = tf.matmul(h_record_flatten, self.attention_w)
                    hw = tf.reshape(hw, [1, self.frame_dim, self.encoding_rnn.state_size])

                    for j in range(1):
                        alpha = tf.reduce_sum(tf.multiply(hw[j, :, :], z[j, :]), axis = 1)
                        alpha = tf.nn.softmax(alpha)
                        alpha = tf.expand_dims(alpha, 1)
                        context = tf.reduce_sum(tf.multiply(alpha, h_record[j, :, :]), axis = 0)
                        attention_contexts.append(context)

                    attention_contexts = tf.stack(attention_contexts)

                    with tf.variable_scope("decoding_rnn", reuse = True):
                        decoding_output, decoding_state = self.decoding_rnn(tf.concat([step_embed, encoding_output, attention_contexts], 1), decoding_state)
                        z = decoding_state

                    logits = tf.nn.xw_plus_b(decoding_output, self.embed_w, self.embed_b)
                    prob = tf.nn.softmax(logits)
                    probs.append(prob)

                    max_prob_index = tf.argmax(logits, 1)[0]
                    gen_words_idx.append(max_prob_index)
                
                
                    step_embed = tf.nn.embedding_lookup(self.word_embed, max_prob_index)
                    step_embed = tf.expand_dims(step_embed, 0)
                
                self.gen_words_idx = gen_words_idx
                self.probs = probs



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







data_path = sys.argv[1]
#data_path = '../../ADLxMLDS2017_DATA/hw2/MLDS_hw2_data/'
test_feat_path = os.path.join(data_path, 'peer_review/feat/')
model_dir = 'model_dir_seq2seq'

test_output_path = sys.argv[3]
test_id_path = os.path.join(data_path, 'peer_review_id.txt')




attention_bool = False
match_test_bool = False
epochs = 501


    
with open('vocabs_dict_seq2seq', 'rb') as f:
    vocabs_dict = pickle.load(f)

print('Vocabs:', len(vocabs_dict))
# print(vocabs_dict)

inv_vocabs_dict = {v:k for k, v in vocabs_dict.items()}


test_ids = []
f = open(test_id_path, 'r')
for line in f:
    test_ids.append(line.replace('\n',''))

model = VCG_model(n_vocabs=len(vocabs_dict), for_testing = True, attention = attention_bool, match_test = match_test_bool)

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
        
        
        caption_words = []
        
        for word in gen_words:
            if word == '</s>':
                break
            caption_words.append(word)
        
        
        caption = ' '.join(caption_words)
        res = []
        res.append(test_id)
        res.append(caption)
        
        print('['+str(test_id)+']')
        print(caption)
        print(gen_words)
        print('\n')
        
        results.append(res)
    
    
    f = open(test_output_path, 'w')
    w = csv.writer(f)
    w.writerows(results)
    f.close()

