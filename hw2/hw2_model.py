import tensorflow as tf
import numpy as np

class VCG_model():
    def __init__(self, n_vocabs, for_testing = False, attention = False, match_test = False, scheduled_sampling = False):
        
        self.feat_dim = 4096
        self.frame_dim  = 80
        self.max_sent_len = 15

        self.batch_size = 48
        
        self.layer_dim = 512
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

            self.scheduled_sampling_mask1 = tf.placeholder(tf.float32, [self.batch_size, self.layer_dim])
            self.scheduled_sampling_mask2 = tf.placeholder(tf.float32, [self.batch_size, self.layer_dim])
            
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


                    if scheduled_sampling:
                        step_embed_from_model = tf.nn.embedding_lookup(self.word_embed, max_prob_index)
                        step_embed_reference = tf.nn.embedding_lookup(self.word_embed, self.caption[:, i + 1])
                        step_embed = self.scheduled_sampling_mask1 * step_embed_from_model + self.scheduled_sampling_mask2 * step_embed_reference
                        
                        
                    
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