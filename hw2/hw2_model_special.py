#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:02:10 2017

@author: HSIN
"""

import tensorflow as tf

class VCG_model():
    def __init__(self, n_vocabs, for_testing = False):
        
        # device_using = "/gpu:0"
        
        self.n_vocabs = n_vocabs
        self.feat_dim = 4096
        self.frame_dim = 80
        self.max_sent_len = 20
        self.layer_dim = 512
        self.batch_size = 64
        self.lr = 0.001
        
        
        
        self.word_embed = tf.Variable(tf.random_uniform([self.n_vocabs, self.layer_dim], -1, 1))
        
        self.encode_w = tf.Variable(tf.random_uniform([self.feat_dim, self.layer_dim], -1, 1))
        self.encode_b = tf.Variable(tf.zeros([self.layer_dim]))
        
        self.embed_w  = tf.Variable(tf.random_uniform([self.layer_dim,self.n_vocabs], -1, 1))
        self.embed_b  = tf.Variable(tf.zeros([self.n_vocabs]))
        
        self.encoding_lstm = tf.contrib.rnn.BasicLSTMCell(self.layer_dim, state_is_tuple = False)
        self.decoding_lstm = tf.contrib.rnn.BasicLSTMCell(self.layer_dim, state_is_tuple = False)
        
        
        
        
        
        if not for_testing:
            
            self.feat = tf.placeholder(tf.float32, [self.batch_size, self.frame_dim, self.feat_dim])
            self.feat_mask = tf.placeholder(tf.float32, [self.batch_size, self.frame_dim])
            
            self.caption = tf.placeholder(tf.int32, [self.batch_size, self.max_sent_len + 1])
            self.caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.max_sent_len + 1])
            
            
            
            feat_flatten = tf.reshape(self.feat, [-1, self.feat_dim])
            encoding_input = tf.nn.xw_plus_b(feat_flatten, self.encode_w, self.encode_b)
            encoding_input = tf.reshape(encoding_input, [self.batch_size, self.frame_dim, self.layer_dim])
            
            encoding_state = tf.zeros([self.batch_size, self.encoding_lstm.state_size])
            decoding_state = tf.zeros([self.batch_size, self.decoding_lstm.state_size])
            
            padding = tf.zeros([self.batch_size, self.layer_dim])
            
            loss = 0
            
            for i in range(self.frame_dim):
                with tf.variable_scope("encoding_lstm", reuse = (i != 0)):
                    encoding_output, encoding_state = self.encoding_lstm(encoding_input[:,i,:], encoding_state)
                with tf.variable_scope("decoding_lstm", reuse = (i != 0)):
                    decoding_output, decoding_state = self.decoding_lstm(tf.concat([padding, encoding_output],1), decoding_state)						
            
            
            for i in range(self.max_sent_len):
                
                if i == 0:
                    current_embed = tf.zeros([self.batch_size, self.layer_dim])
                else:
                    current_embed = tf.nn.embedding_lookup(self.word_embed, self.caption[:,i])
                    
                with tf.variable_scope("encoding_lstm", reuse = True):
                    encoding_output, encoding_state = self.encoding_lstm(padding, encoding_state)
                    
                with tf.variable_scope("decoding_lstm", reuse = True):
                    decoding_output, decoding_state = self.decoding_lstm(tf.concat([current_embed,encoding_output],1), decoding_state)
                    
                
                labels = tf.expand_dims(self.caption[:,i+1], 1)
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                labels = tf.sparse_to_dense(tf.concat([indices, labels],1), tf.stack([self.batch_size, self.n_vocabs]), 1, 0)
                    
                logits = tf.nn.xw_plus_b(decoding_output, self.embed_w, self.embed_b)
                
                cross_entropy = self.caption_mask[:,i] * tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
                
                sub_loss = tf.reduce_mean(cross_entropy)
                loss += sub_loss
                
                
            self.loss = loss
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        
        
        else:
        
            self.feat = tf.placeholder(tf.float32, [1, self.frame_dim, self.feat_dim])
            self.feat_mask = tf.placeholder(tf.float32, [1, self.frame_dim])
            
            feat_flatten = tf.reshape(self.feat, [-1, self.feat_dim])
            encoding_input = tf.nn.xw_plus_b(feat_flatten, self.encode_w, self.encode_b)
            encoding_input = tf.reshape(encoding_input, [1, self.frame_dim, self.layer_dim])
            
            encoding_state = tf.zeros([1, self.encoding_lstm.state_size])
            decoding_state = tf.zeros([1, self.decoding_lstm.state_size])
            
            padding = tf.zeros([1, self.layer_dim])
            
            gen_words_idx = []
            probs = []
            
            for i in range(self.frame_dim):
                with tf.variable_scope("encoding_lstm", reuse=(i != 0)):
                    encoding_output, encoding_state = self.encoding_lstm(encoding_input[:,i,:], encoding_state)
                with tf.variable_scope("decoding_lstm", reuse=(i != 0)):
                    decoding_output, decoding_state = self.decoding_lstm(tf.concat([padding, encoding_output],1), decoding_state)
            
            
            for i in range(self.max_sent_len):
                if i == 0:
                    current_embed = tf.nn.embedding_lookup(self.word_embed, tf.zeros([1], dtype=tf.int64))
    
                with tf.variable_scope("encoding_lstm", reuse = True):
                    encoding_output, encoding_state = self.encoding_lstm(padding, encoding_state)
                
                with tf.variable_scope("decoding_lstm", reuse = True):
                    decoding_output, decoding_state = self.decoding_lstm(tf.concat([current_embed,encoding_output],1), decoding_state)
                    logits = tf.nn.xw_plus_b(decoding_output, self.embed_w, self.embed_b)
                    prob = tf.nn.softmax(logits)
                    gen_words_idx.append(tf.argmax(logits, 1)[0])
                    probs.append(prob)
                
            self.gen_words_idx = gen_words_idx
            self.probs = probs