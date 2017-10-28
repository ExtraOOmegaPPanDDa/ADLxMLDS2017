# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 23:15:55 2017

@author: HSIN
"""

import os
import sys
import csv
import time
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPooling1D, MaxPooling2D
from keras.layers import RepeatVector, TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers.advanced_activations import PReLU
from keras.models import Model
from keras.models import load_model
from keras.optimizers import SGD, Adadelta, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint


stime = time.time()

max_sent_len = 800


char_list = []
phone_48_list = []
phone_39_list = []


data_path = sys.argv[1]
output_path = sys.argv[2]


print('Get 48-Char Map', time.time()-stime)
f = open(os.path.join(data_path, '48phone_char.map'), 'r')
for line in f:
    data = line.replace('\n', '')
    data = data.split('\t')
    phone_48_list.append(data[0])
    char_list.append(data[2])
    phone_39_list.append(0)
f.close()

print('Get 48-39 Map', time.time()-stime)
f = open(os.path.join(data_path, 'phones/48_39.map'), 'r')
for line in f:
    data = line.replace('\n', '')
    data = data.split('\t')
    phone_39_list[phone_48_list.index(data[0])] = data[1]
f.close()



phone_39_set_list = sorted(list(set(phone_39_list)))


"""

train_ids = []
train_mfccs = []
train_fbanks = []
train_labels = []


print('Get Train MFCC Feature', time.time()-stime)
f = open(os.path.join(data_path, 'mfcc/train.ark'), 'r')
ids = []
count = 0
for line in f:
    count += 1
    if count  % 100000 == 0:
        print('Get Train MFCC', count)
    
    data = line.replace('\n', '')
    data = data.split(' ')
    ids.append(data[0])
    train_mfccs.append(([float(x) for x in data[1:]]))
f.close()


sorted_zip_list = list(sorted(zip(ids,train_mfccs)))
sorted_zip_list = list(zip(*sorted_zip_list))
ids = np.asarray(sorted_zip_list[0])
train_mfccs = np.asarray(sorted_zip_list[1])





print('Get Train FBANK Feature', time.time()-stime)
f = open(os.path.join(data_path, 'fbank/train.ark'), 'r')
ids = []
count = 0
for line in f:
    count += 1
    if count  % 100000 == 0:
        print('Get Train FBANK', count)
    
    data = line.replace('\n', '')
    data = data.split(' ')
    ids.append(data[0])
    train_fbanks.append(([float(x) for x in data[1:]]))
f.close()


sorted_zip_list = list(sorted(zip(ids,train_fbanks)))
sorted_zip_list = list(zip(*sorted_zip_list))
ids = np.asarray(sorted_zip_list[0])
train_fbanks = np.asarray(sorted_zip_list[1])


train_features = np.concatenate((train_mfccs,train_fbanks), axis=1)


print('Get Train Label', time.time()-stime)
f = open(os.path.join(data_path, 'label/train.lab'), 'r')
ids = []
count = 0
for line in f:
    count += 1
    if count  % 100000 == 0:
        print('Get Train Label', count)
    
    data = line.replace('\n', '')
    data = data.split(',')
    ids.append(data[0])
    lab = phone_39_list[phone_48_list.index(data[1])]
    train_labels.append(lab)
f.close()


sorted_zip_list = list(sorted(zip(ids,train_labels)))
sorted_zip_list = list(zip(*sorted_zip_list))
ids = np.asarray(sorted_zip_list[0])
train_labels = np.asarray(sorted_zip_list[1])


train_ids = ids
speaker_ids = list(set(ids))


for i in range(len(speaker_ids)):
    sids = speaker_ids[i]
    sids_split = sids.split('_')
    speaker_ids[i] = '_'.join(sids_split[:2])

speaker_ids = list(set(speaker_ids))

X = []
y = []

for i in range(len(speaker_ids)):
    if i % 1000 == 0:
        print('Train Padding Set', i)
    to_append = []
    for j in range(max_sent_len):
        to_append.append(np.zeros(train_features.shape[1]))
    X.append(to_append)
    
    
    to_append = []
    for j in range(max_sent_len):
        to_append.append(np.zeros(39))
    y.append(to_append)



id_max_record = np.zeros(len(speaker_ids))

for i in range(len(train_ids)):
    if i % 10000 == 0:
        print('Train ID Transform Step1', i)
        print(train_labels[i])
    tids = train_ids[i]
    tids_split = tids.split('_')
    speaker_id = '_'.join(tids_split[:2])
    context_id = int(tids_split[2])
     
    if context_id > id_max_record[speaker_ids.index(speaker_id)]:
        id_max_record[speaker_ids.index(speaker_id)] = context_id



for i in range(len(train_ids)):
    if i % 10000 == 0:
        print('Train ID Transform Step2', i)
        print(train_labels[i])
    tids = train_ids[i]
    tids_split = tids.split('_')
    speaker_id = '_'.join(tids_split[:2])
    context_id = int(tids_split[2])
    
    pad_time = 0
    while(1):
        if context_id-1 + pad_time*id_max_record[speaker_ids.index(speaker_id)] >= max_sent_len:
            break
        X[speaker_ids.index(speaker_id)][int(context_id-1 + pad_time*id_max_record[speaker_ids.index(speaker_id)])]  = train_features[i]
        pad_time += 1
    
    
    
    pad_time = 0
    while(1):
        if context_id-1 + pad_time*id_max_record[speaker_ids.index(speaker_id)] >= max_sent_len:
            break
        y[speaker_ids.index(speaker_id)][int(context_id-1 + pad_time*id_max_record[speaker_ids.index(speaker_id)])][phone_39_set_list.index(train_labels[i])] = 1
        pad_time += 1
        

print('Dump X', time.time()-stime)
with open('X', 'wb') as fp:
    pickle.dump(X, fp)

print('Dump y', time.time()-stime)
with open('y', 'wb') as fp:
    pickle.dump(y, fp)
"""





#########################################
############## STAGE 1 ##################
############### START ###################
#########################################


"""


print('load X', time.time()-stime)
with open ('X', 'rb') as fp:
    X = pickle.load(fp)

print('load y', time.time()-stime)    
with open ('y', 'rb') as fp:
    y = pickle.load(fp)


X = np.asarray(X)
y = np.asarray(y)

print('X_shape', X.shape, time.time()-stime)
print('y_shape', y.shape, time.time()-stime)

"""





def build_LSTM_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.5


    model = Sequential()

    model.add(Conv1D(128, 
                     padding = 'causal', 
                     kernel_size = 8,
                     input_shape=(max_sent_len, word_size)))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 8))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Bidirectional(LSTM(128,return_sequences=True),merge_mode='ave'))

    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model




def build_GRU_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.25


    model = Sequential()

    model.add(Conv1D(128, 
                     padding = 'causal', 
                     kernel_size = 10,
                     input_shape=(max_sent_len, word_size)))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(64, 
                     padding = 'causal',
                     kernel_size = 8))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(64, 
                     padding = 'causal',
                     kernel_size = 5))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Bidirectional(GRU(256,return_sequences=True),merge_mode='ave'))
    model.add(Bidirectional(GRU(128,return_sequences=True),merge_mode='ave'))
                            
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model





def build_SimpleRNN_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.25


    model = Sequential()

    model.add(Conv1D(128, 
                     padding = 'causal', 
                     kernel_size = 10,
                     input_shape=(max_sent_len, word_size)))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(64, 
                     padding = 'causal',
                     kernel_size = 8))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(64, 
                     padding = 'causal',
                     kernel_size = 5))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Bidirectional(SimpleRNN(256,return_sequences=True),merge_mode='ave'))
    model.add(Bidirectional(SimpleRNN(128,return_sequences=True),merge_mode='ave'))
                            
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model







def build_SimpleRNN2_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.25


    model = Sequential()

    model.add(Conv1D(128, 
                     padding = 'causal', 
                     kernel_size = 8,
                     input_shape=(max_sent_len, word_size)))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(64, 
                     padding = 'causal',
                     kernel_size = 5))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(64, 
                     padding = 'causal',
                     kernel_size = 3))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Bidirectional(SimpleRNN(512,return_sequences=True),merge_mode='ave'))
                            
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model




"""

nfold = 15
batch_size = 128


val_acc_box = []
train_mean_box = []
train_std_box = []


# Stage1 LSTM Model

np.random.seed(seed = 464646)

indices = list(np.random.permutation(X.shape[0]))
indices_split_box = np.array_split(indices, nfold)

skf_iteration = 0
for i in range(nfold):
    
   
    skf_iteration += 1
    print('SKF MEAN STD', skf_iteration)

    train_idx = []
    valid_idx = []

    for ifold in range(nfold):
        if ifold != i:
            train_idx = train_idx + list(indices_split_box[ifold])
        else:
            valid_idx = list(indices_split_box[ifold])

    train_idx = np.asarray(train_idx)
    valid_idx = np.asarray(valid_idx)

    X_train, X_valid = X[train_idx,:], X[valid_idx,:]
    y_train, y_valid = y[train_idx,:], y[valid_idx,:]


    # normalization
    X_train_mean = np.mean(X_train, axis = 0)
    X_train_std = np.std(X_train, axis = 0)

    train_mean_box.append(X_train_mean)
    train_std_box.append(X_train_std)




with open('train_mean_box','wb') as fp:
        pickle.dump(train_mean_box, fp)

with open('train_std_box','wb') as fp:
    pickle.dump(train_std_box, fp)






skf_iteration = 0
epochs = 100
patience = 5
for i in range(nfold):

    skf_iteration += 1
    print('SKF', skf_iteration)

    train_idx = []
    valid_idx = []

    for ifold in range(nfold):
        if ifold != i:
            train_idx = train_idx + list(indices_split_box[ifold])
        else:
            valid_idx = list(indices_split_box[ifold])

    train_idx = np.asarray(train_idx)
    valid_idx = np.asarray(valid_idx)

    X_train, X_valid = X[train_idx,:], X[valid_idx,:]
    y_train, y_valid = y[train_idx,:], y[valid_idx,:]


    # normalization
    X_train_mean = np.mean(X_train, axis = 0)
    X_train_std = np.std(X_train, axis = 0)

    train_mean_box.append(X_train_mean)
    train_std_box.append(X_train_std)


    X_train = X_train - np.tile(X_train_mean, (X_train.shape[0],1,1))
    X_train = X_train / np.tile(X_train_std, (X_train.shape[0],1,1))
    X_valid = X_valid - np.tile(X_train_mean, (X_valid.shape[0],1,1))
    X_valid = X_valid / np.tile(X_train_std, (X_valid.shape[0],1,1))
    
    
    model = build_LSTM_model(X.shape[1] , X.shape[2], y.shape[2])
    model.summary()


    earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
    checkpoint = ModelCheckpoint('hw1_lstm_model_'+str(skf_iteration)+'.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 mode='min')

    
    history = model.fit(X_train, y_train, 
                     epochs = epochs, 
                     batch_size = batch_size,
                     validation_data = (X_valid, y_valid),
                     callbacks=[earlystopping,checkpoint])




    del model
    

    model = build_LSTM_model(X.shape[1] , X.shape[2], y.shape[2])
    model.load_weights('hw1_lstm_model_'+str(skf_iteration)+'.hdf5')





# GRU Model

np.random.seed(seed = 46)


indices = list(np.random.permutation(X.shape[0]))
indices_split_box = np.array_split(indices, nfold)



skf_iteration = 0
for i in range(nfold):
    
   
    skf_iteration += 1
    print('SKF MEAN STD', skf_iteration)

    train_idx = []
    valid_idx = []

    for ifold in range(nfold):
        if ifold != i:
            train_idx = train_idx + list(indices_split_box[ifold])
        else:
            valid_idx = list(indices_split_box[ifold])

    train_idx = np.asarray(train_idx)
    valid_idx = np.asarray(valid_idx)

    X_train, X_valid = X[train_idx,:], X[valid_idx,:]
    y_train, y_valid = y[train_idx,:], y[valid_idx,:]


    # normalization
    X_train_mean = np.mean(X_train, axis = 0)
    X_train_std = np.std(X_train, axis = 0)

    train_mean_box.append(X_train_mean)
    train_std_box.append(X_train_std)




with open('train_mean_box','wb') as fp:
        pickle.dump(train_mean_box, fp)

with open('train_std_box','wb') as fp:
    pickle.dump(train_std_box, fp)






skf_iteration = 0
epochs = 100
patience = 4
for i in range(nfold):

    skf_iteration += 1
    print('SKF', skf_iteration)

    train_idx = []
    valid_idx = []

    for ifold in range(nfold):
        if ifold != i:
            train_idx = train_idx + list(indices_split_box[ifold])
        else:
            valid_idx = list(indices_split_box[ifold])

    train_idx = np.asarray(train_idx)
    valid_idx = np.asarray(valid_idx)

    X_train, X_valid = X[train_idx,:], X[valid_idx,:]
    y_train, y_valid = y[train_idx,:], y[valid_idx,:]


    # normalization
    X_train_mean = np.mean(X_train, axis = 0)
    X_train_std = np.std(X_train, axis = 0)

    train_mean_box.append(X_train_mean)
    train_std_box.append(X_train_std)


    X_train = X_train - np.tile(X_train_mean, (X_train.shape[0],1,1))
    X_train = X_train / np.tile(X_train_std, (X_train.shape[0],1,1))
    X_valid = X_valid - np.tile(X_train_mean, (X_valid.shape[0],1,1))
    X_valid = X_valid / np.tile(X_train_std, (X_valid.shape[0],1,1))

    
    
    model = build_GRU_model(X.shape[1] , X.shape[2], y.shape[2])
    model.summary()


    earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
    checkpoint = ModelCheckpoint('hw1_gru_model_'+str(skf_iteration)+'.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 mode='min')

    
    history = model.fit(X_train, y_train, 
                     epochs = epochs, 
                     batch_size = batch_size,
                     validation_data = (X_valid, y_valid),
                     callbacks=[earlystopping,checkpoint])




    del model
    

    model = build_GRU_model(X.shape[1] , X.shape[2], y.shape[2])
    model.load_weights('hw1_gru_model_'+str(skf_iteration)+'.hdf5')




# SimpleRNN Model

np.random.seed(seed = 4646)


indices = list(np.random.permutation(X.shape[0]))
indices_split_box = np.array_split(indices, nfold)



skf_iteration = 0
for i in range(nfold):
    
   
    skf_iteration += 1
    print('SKF MEAN STD', skf_iteration)

    train_idx = []
    valid_idx = []

    for ifold in range(nfold):
        if ifold != i:
            train_idx = train_idx + list(indices_split_box[ifold])
        else:
            valid_idx = list(indices_split_box[ifold])

    train_idx = np.asarray(train_idx)
    valid_idx = np.asarray(valid_idx)

    X_train, X_valid = X[train_idx,:], X[valid_idx,:]
    y_train, y_valid = y[train_idx,:], y[valid_idx,:]


    # normalization
    X_train_mean = np.mean(X_train, axis = 0)
    X_train_std = np.std(X_train, axis = 0)

    train_mean_box.append(X_train_mean)
    train_std_box.append(X_train_std)




with open('train_mean_box','wb') as fp:
        pickle.dump(train_mean_box, fp)

with open('train_std_box','wb') as fp:
    pickle.dump(train_std_box, fp)






skf_iteration = 0
epochs = 250
patience = 15
for i in range(nfold):

    skf_iteration += 1
    print('SKF', skf_iteration)

    train_idx = []
    valid_idx = []

    for ifold in range(nfold):
        if ifold != i:
            train_idx = train_idx + list(indices_split_box[ifold])
        else:
            valid_idx = list(indices_split_box[ifold])

    train_idx = np.asarray(train_idx)
    valid_idx = np.asarray(valid_idx)

    X_train, X_valid = X[train_idx,:], X[valid_idx,:]
    y_train, y_valid = y[train_idx,:], y[valid_idx,:]


    # normalization
    X_train_mean = np.mean(X_train, axis = 0)
    X_train_std = np.std(X_train, axis = 0)

    train_mean_box.append(X_train_mean)
    train_std_box.append(X_train_std)


    X_train = X_train - np.tile(X_train_mean, (X_train.shape[0],1,1))
    X_train = X_train / np.tile(X_train_std, (X_train.shape[0],1,1))
    X_valid = X_valid - np.tile(X_train_mean, (X_valid.shape[0],1,1))
    X_valid = X_valid / np.tile(X_train_std, (X_valid.shape[0],1,1))
    
    
    model = build_SimpleRNN_model(X.shape[1] , X.shape[2], y.shape[2])
    model.summary()


    earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
    checkpoint = ModelCheckpoint('hw1_simplernn_model_'+str(skf_iteration)+'.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 mode='min')

    
    history = model.fit(X_train, y_train, 
                     epochs = epochs, 
                     batch_size = batch_size,
                     validation_data = (X_valid, y_valid),
                     callbacks=[earlystopping,checkpoint])




    del model
    

    model = build_SimpleRNN_model(X.shape[1] , X.shape[2], y.shape[2])
    model.load_weights('hw1_simplernn_model_'+str(skf_iteration)+'.hdf5')




# SimpleRNN2 Model

np.random.seed(seed = 4466)


indices = list(np.random.permutation(X.shape[0]))
indices_split_box = np.array_split(indices, nfold)



skf_iteration = 0
for i in range(nfold):
    
   
    skf_iteration += 1
    print('SKF MEAN STD', skf_iteration)

    train_idx = []
    valid_idx = []

    for ifold in range(nfold):
        if ifold != i:
            train_idx = train_idx + list(indices_split_box[ifold])
        else:
            valid_idx = list(indices_split_box[ifold])

    train_idx = np.asarray(train_idx)
    valid_idx = np.asarray(valid_idx)

    X_train, X_valid = X[train_idx,:], X[valid_idx,:]
    y_train, y_valid = y[train_idx,:], y[valid_idx,:]


    # normalization
    X_train_mean = np.mean(X_train, axis = 0)
    X_train_std = np.std(X_train, axis = 0)

    train_mean_box.append(X_train_mean)
    train_std_box.append(X_train_std)




with open('train_mean_box','wb') as fp:
        pickle.dump(train_mean_box, fp)

with open('train_std_box','wb') as fp:
    pickle.dump(train_std_box, fp)




skf_iteration = 0
epochs = 250
patience = 15
for i in range(nfold):

    skf_iteration += 1
    print('SKF', skf_iteration)

    train_idx = []
    valid_idx = []

    for ifold in range(nfold):
        if ifold != i:
            train_idx = train_idx + list(indices_split_box[ifold])
        else:
            valid_idx = list(indices_split_box[ifold])

    train_idx = np.asarray(train_idx)
    valid_idx = np.asarray(valid_idx)

    X_train, X_valid = X[train_idx,:], X[valid_idx,:]
    y_train, y_valid = y[train_idx,:], y[valid_idx,:]


    # normalization
    X_train_mean = np.mean(X_train, axis = 0)
    X_train_std = np.std(X_train, axis = 0)

    train_mean_box.append(X_train_mean)
    train_std_box.append(X_train_std)


    X_train = X_train - np.tile(X_train_mean, (X_train.shape[0],1,1))
    X_train = X_train / np.tile(X_train_std, (X_train.shape[0],1,1))
    X_valid = X_valid - np.tile(X_train_mean, (X_valid.shape[0],1,1))
    X_valid = X_valid / np.tile(X_train_std, (X_valid.shape[0],1,1))
    
    model = build_SimpleRNN2_model(X.shape[1] , X.shape[2], y.shape[2])
    model.summary()


    earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
    checkpoint = ModelCheckpoint('hw1_simplernn2_model_'+str(skf_iteration)+'.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 mode='min')

    
    history = model.fit(X_train, y_train, 
                     epochs = epochs, 
                     batch_size = batch_size,
                     validation_data = (X_valid, y_valid),
                     callbacks=[earlystopping,checkpoint])




    del model
    

    model = build_SimpleRNN2_model(X.shape[1] , X.shape[2], y.shape[2])
    model.load_weights('hw1_simplernn2_model_'+str(skf_iteration)+'.hdf5')


with open('train_mean_box','rb') as fp:
    train_mean_box = pickle.load(fp)

with open('train_std_box','rb') as fp:
    train_std_box = pickle.load(fp)



"""


"""

with open('train_mean_box','rb') as fp:
    train_mean_box = pickle.load(fp)

with open('train_std_box','rb') as fp:
    train_std_box = pickle.load(fp)





nfold = 15


# LSTM Model

X_lstm_blend = 0
check = 0

skf_iteration = 0
to_add = 0
for i in range(nfold):

    skf_iteration += 1
    print('SKF', skf_iteration)

    
    X_norm = X - np.tile(train_mean_box[i+to_add], (X.shape[0],1,1))
    X_norm = X_norm / np.tile(train_std_box[i+to_add], (X.shape[0],1,1))

    model = build_LSTM_model(X.shape[1] , X.shape[2], y.shape[2])
    model.load_weights('hw1_lstm_model_'+str(skf_iteration)+'.hdf5')


    X_toStack = model.predict(X_norm, verbose = 1)

    if check == 0:
        X_lstm_blend = X_toStack
        check = 1
    else:
        X_lstm_blend = X_lstm_blend + X_toStack




# GRU Model

X_gru_blend = 0
check = 0

skf_iteration = 0
to_add = 15
for i in range(nfold):

    skf_iteration += 1
    print('SKF', skf_iteration)

    
    X_norm = X - np.tile(train_mean_box[i+to_add], (X.shape[0],1,1))
    X_norm = X_norm / np.tile(train_std_box[i+to_add], (X.shape[0],1,1))

    model = build_GRU_model(X.shape[1] , X.shape[2], y.shape[2])
    model.load_weights('hw1_gru_model_'+str(skf_iteration)+'.hdf5')


    X_toStack = model.predict(X_norm, verbose = 1)

    if check == 0:
        X_gru_blend = X_toStack
        check = 1
    else:
        X_gru_blend = X_gru_blend + X_toStack




# SimpleRNN Model

X_simplernn_blend = 0
check = 0

skf_iteration = 0
to_add = 30
for i in range(nfold):

    skf_iteration += 1
    print('SKF', skf_iteration)
    
    X_norm = X - np.tile(train_mean_box[i+to_add], (X.shape[0],1,1))
    X_norm = X_norm / np.tile(train_std_box[i+to_add], (X.shape[0],1,1))

    model = build_SimpleRNN_model(X.shape[1] , X.shape[2], y.shape[2])
    model.load_weights('hw1_simplernn_model_'+str(skf_iteration)+'.hdf5')


    X_toStack = model.predict(X_norm, verbose = 1)

    if check == 0:
        X_simplernn_blend = X_toStack
        check = 1
    else:
        X_simplernn_blend = X_simplernn_blend + X_toStack





# SimpleRNN2 Model

X_simplernn2_blend = 0
check = 0

skf_iteration = 0
to_add = 45
for i in range(nfold):

    skf_iteration += 1
    print('SKF', skf_iteration)
    
    X_norm = X - np.tile(train_mean_box[i+to_add], (X.shape[0],1,1))
    X_norm = X_norm / np.tile(train_std_box[i+to_add], (X.shape[0],1,1))

    model = build_SimpleRNN2_model(X.shape[1] , X.shape[2], y.shape[2])
    model.load_weights('hw1_simplernn2_model_'+str(skf_iteration)+'.hdf5')


    X_toStack = model.predict(X_norm, verbose = 1)

    if check == 0:
        X_simplernn2_blend = X_toStack
        check = 1
    else:
        X_simplernn2_blend = X_simplernn2_blend + X_toStack



X_stacks = np.concatenate(( X_lstm_blend,
                            X_gru_blend,
                            X_simplernn_blend,
                            X_simplernn2_blend
                            ), axis=2)


print(X_stacks.shape)



print('Dump X_stacks', time.time()-stime)
with open('X_stacks', 'wb') as fp:
    pickle.dump(X_stacks, fp)


"""

#########################################
############## STAGE 1 ##################
###############  END  ###################
#########################################




#########################################
############## STAGE 2 ##################
############### START ###################
#########################################



"""

print('load X_stacks', time.time()-stime)
with open ('X_stacks', 'rb') as fp:
    X_stacks = pickle.load(fp)

print('load y_stacks', time.time()-stime)    
with open ('y', 'rb') as fp:
    y_stacks = pickle.load(fp)

X_stacks = np.asarray(X_stacks)
y_stacks = np.asarray(y_stacks)


"""
def build_Stack_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.4


    model = Sequential()

    model.add(Conv1D(128, 
                     padding = 'causal', 
                     kernel_size = 8,
                     input_shape=(max_sent_len, word_size)))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 5))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))

    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 3))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Bidirectional(LSTM(256,return_sequences=True),merge_mode='ave'))

    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model





def build_Stack2_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.4


    model = Sequential()

    model.add(Conv1D(128, 
                     padding = 'causal', 
                     kernel_size = 10,
                     input_shape=(max_sent_len, word_size)))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 6))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))

    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 4))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Bidirectional(SimpleRNN(256,return_sequences=True),merge_mode='ave'))

    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model



def build_Stack3_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.4


    model = Sequential()

    model.add(Conv1D(128, 
                     padding = 'causal', 
                     kernel_size = 12,
                     input_shape=(max_sent_len, word_size)))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 8))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))

    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 5))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Bidirectional(GRU(256,return_sequences=True),merge_mode='ave'))

    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model



"""


# Stack model
np.random.seed(seed = 17461)

train_valid_ratio = 0.9
indices = np.random.permutation(X_stacks.shape[0])
train_idx, valid_idx = indices[:int(X_stacks.shape[0] * train_valid_ratio)], indices[int(X_stacks.shape[0] * train_valid_ratio):]
X_stacks_train, X_stacks_valid = X_stacks[train_idx,:], X_stacks[valid_idx,:]
y_stacks_train, y_stacks_valid = y_stacks[train_idx,:], y_stacks[valid_idx,:]




epochs = 250
patience = 10
batch_size = 128

model = build_Stack_model(X_stacks.shape[1] , X_stacks.shape[2], y_stacks.shape[2])
model.summary()


earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
checkpoint = ModelCheckpoint('hw1_stack_model.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             mode='min')


history = model.fit(X_stacks_train, y_stacks_train, 
                 epochs = epochs, 
                 batch_size = batch_size,
                 validation_data = (X_stacks_valid, y_stacks_valid),
                 callbacks=[earlystopping,checkpoint])




del model




# Stack model 2
np.random.seed(seed = 17462)

train_valid_ratio = 0.9
indices = np.random.permutation(X_stacks.shape[0])
train_idx, valid_idx = indices[:int(X_stacks.shape[0] * train_valid_ratio)], indices[int(X_stacks.shape[0] * train_valid_ratio):]
X_stacks_train, X_stacks_valid = X_stacks[train_idx,:], X_stacks[valid_idx,:]
y_stacks_train, y_stacks_valid = y_stacks[train_idx,:], y_stacks[valid_idx,:]




epochs = 250
patience = 10
batch_size = 128

model = build_Stack2_model(X_stacks.shape[1] , X_stacks.shape[2], y_stacks.shape[2])
model.summary()


earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
checkpoint = ModelCheckpoint('hw1_stack2_model.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             mode='min')


history = model.fit(X_stacks_train, y_stacks_train, 
                 epochs = epochs, 
                 batch_size = batch_size,
                 validation_data = (X_stacks_valid, y_stacks_valid),
                 callbacks=[earlystopping,checkpoint])




del model




# Stack model 3
np.random.seed(seed = 17463)

train_valid_ratio = 0.9
indices = np.random.permutation(X_stacks.shape[0])
train_idx, valid_idx = indices[:int(X_stacks.shape[0] * train_valid_ratio)], indices[int(X_stacks.shape[0] * train_valid_ratio):]
X_stacks_train, X_stacks_valid = X_stacks[train_idx,:], X_stacks[valid_idx,:]
y_stacks_train, y_stacks_valid = y_stacks[train_idx,:], y_stacks[valid_idx,:]




epochs = 250
patience = 10
batch_size = 128

model = build_Stack3_model(X_stacks.shape[1] , X_stacks.shape[2], y_stacks.shape[2])
model.summary()


earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
checkpoint = ModelCheckpoint('hw1_stack3_model.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             mode='min')


history = model.fit(X_stacks_train, y_stacks_train, 
                 epochs = epochs, 
                 batch_size = batch_size,
                 validation_data = (X_stacks_valid, y_stacks_valid),
                 callbacks=[earlystopping,checkpoint])




del model



model = build_Stack_model(X_stacks.shape[1] , X_stacks.shape[2], y_stacks.shape[2])
model.load_weights('hw1_stack_model.hdf5')
stack1_pred = model.predict(X_stacks)


model = build_Stack2_model(X_stacks.shape[1] , X_stacks.shape[2], y_stacks.shape[2])
model.load_weights('hw1_stack2_model.hdf5')
stack2_pred = model.predict(X_stacks)

model = build_Stack3_model(X_stacks.shape[1] , X_stacks.shape[2], y_stacks.shape[2])
model.load_weights('hw1_stack3_model.hdf5')
stack3_pred = model.predict(X_stacks)






X_last_stacks = np.concatenate((  stack1_pred,
                                  stack2_pred,
                                  stack3_pred
                                ), axis=2)

print(X_last_stacks.shape)



print('Dump X_last_stacks', time.time()-stime)
with open('X_last_stacks', 'wb') as fp:
    pickle.dump(X_last_stacks, fp)


"""

#########################################
############## STAGE 2 ##################
###############  END  ###################
#########################################



#########################################
############## STAGE 3 ##################
############### START ###################
#########################################



"""
print('load X_last_stacks', time.time()-stime)
with open ('X_last_stacks', 'rb') as fp:
    X_last_stacks = pickle.load(fp)

print('load y_last_stacks', time.time()-stime)    
with open ('y', 'rb') as fp:
    y_last_stacks = pickle.load(fp)

X_last_stacks = np.asarray(X_last_stacks)
y_last_stacks = np.asarray(y_last_stacks)


"""


def build_lastStack_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.4


    model = Sequential()

    model.add(Conv1D(128, 
                     padding = 'causal', 
                     kernel_size = 10,
                     input_shape=(max_sent_len, word_size)))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 8))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))

    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 5))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Bidirectional(LSTM(256,return_sequences=True),merge_mode='ave'))

    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model




def build_lastStack2_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.4


    model = Sequential()

    model.add(Conv1D(128, 
                     padding = 'causal', 
                     kernel_size = 12,
                     input_shape=(max_sent_len, word_size)))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 6))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))

    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 3))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Bidirectional(GRU(256,return_sequences=True),merge_mode='ave'))

    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model




def build_lastStack3_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.4


    model = Sequential()

    model.add(Conv1D(128, 
                     padding = 'causal', 
                     kernel_size = 12,
                     input_shape=(max_sent_len, word_size)))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 8))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))

    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 5))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Bidirectional(SimpleRNN(256,return_sequences=True),merge_mode='ave'))

    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model




"""


# last Stack
np.random.seed(seed = 17461746)

train_valid_ratio = 0.95
indices = np.random.permutation(X_last_stacks.shape[0])
train_idx, valid_idx = indices[:int(X_last_stacks.shape[0] * train_valid_ratio)], indices[int(X_last_stacks.shape[0] * train_valid_ratio):]
X_last_stacks_train, X_last_stacks_valid = X_last_stacks[train_idx,:], X_last_stacks[valid_idx,:]
y_last_stacks_train, y_last_stacks_valid = y_last_stacks[train_idx,:], y_last_stacks[valid_idx,:]




epochs = 250
patience = 10
batch_size = 128

model = build_lastStack_model(X_last_stacks.shape[1] , X_last_stacks.shape[2], y_last_stacks.shape[2])
model.summary()


earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
checkpoint = ModelCheckpoint('hw1_lastStack_model.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             mode='min')


history = model.fit(X_last_stacks_train, y_last_stacks_train, 
                 epochs = epochs, 
                 batch_size = batch_size,
                 validation_data = (X_last_stacks_valid, y_last_stacks_valid),
                 callbacks=[earlystopping,checkpoint])




del model







# last Stack 2
np.random.seed(seed = 46461717)

train_valid_ratio = 0.95
indices = np.random.permutation(X_last_stacks.shape[0])
train_idx, valid_idx = indices[:int(X_last_stacks.shape[0] * train_valid_ratio)], indices[int(X_last_stacks.shape[0] * train_valid_ratio):]
X_last_stacks_train, X_last_stacks_valid = X_last_stacks[train_idx,:], X_last_stacks[valid_idx,:]
y_last_stacks_train, y_last_stacks_valid = y_last_stacks[train_idx,:], y_last_stacks[valid_idx,:]




epochs = 250
patience = 10
batch_size = 128

model = build_lastStack2_model(X_last_stacks.shape[1] , X_last_stacks.shape[2], y_last_stacks.shape[2])
model.summary()


earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
checkpoint = ModelCheckpoint('hw1_lastStack2_model.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             mode='min')


history = model.fit(X_last_stacks_train, y_last_stacks_train, 
                 epochs = epochs, 
                 batch_size = batch_size,
                 validation_data = (X_last_stacks_valid, y_last_stacks_valid),
                 callbacks=[earlystopping,checkpoint])




del model



# last Stack 3
np.random.seed(seed = 46174617)

train_valid_ratio = 0.95
indices = np.random.permutation(X_last_stacks.shape[0])
train_idx, valid_idx = indices[:int(X_last_stacks.shape[0] * train_valid_ratio)], indices[int(X_last_stacks.shape[0] * train_valid_ratio):]
X_last_stacks_train, X_last_stacks_valid = X_last_stacks[train_idx,:], X_last_stacks[valid_idx,:]
y_last_stacks_train, y_last_stacks_valid = y_last_stacks[train_idx,:], y_last_stacks[valid_idx,:]




epochs = 250
patience = 10
batch_size = 128

model = build_lastStack3_model(X_last_stacks.shape[1] , X_last_stacks.shape[2], y_last_stacks.shape[2])
model.summary()


earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
checkpoint = ModelCheckpoint('hw1_lastStack3_model.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             mode='min')


history = model.fit(X_last_stacks_train, y_last_stacks_train, 
                 epochs = epochs, 
                 batch_size = batch_size,
                 validation_data = (X_last_stacks_valid, y_last_stacks_valid),
                 callbacks=[earlystopping,checkpoint])




del model










model = build_lastStack_model(X_last_stacks.shape[1] , X_last_stacks.shape[2], y_last_stacks.shape[2])
model.load_weights('hw1_lastStack_model.hdf5')
last1_pred = model.predict(X_last_stacks)


model = build_lastStack2_model(X_last_stacks.shape[1] , X_last_stacks.shape[2], y_last_stacks.shape[2])
model.load_weights('hw1_lastStack2_model.hdf5')
last2_pred = model.predict(X_last_stacks)


model = build_lastStack3_model(X_last_stacks.shape[1] , X_last_stacks.shape[2], y_last_stacks.shape[2])
model.load_weights('hw1_lastStack3_model.hdf5')
last3_pred = model.predict(X_last_stacks)







X_output_stacks = np.concatenate((  last1_pred,
                                    last2_pred,
                                    last3_pred
                                ), axis=2)


print(X_output_stacks.shape)



print('Dump X_output_stacks', time.time()-stime)
with open('X_output_stacks', 'wb') as fp:
    pickle.dump(X_output_stacks, fp)
"""

#########################################
############## STAGE 3 ##################
###############  END  ###################
#########################################


#########################################
############## STAGE 4 ##################
############### START ###################
#########################################


"""
print('load X_output_stacks', time.time()-stime)
with open ('X_output_stacks', 'rb') as fp:
    X_output_stacks = pickle.load(fp)

print('load y_output_stacks', time.time()-stime)    
with open ('y', 'rb') as fp:
    y_output_stacks = pickle.load(fp)

X_output_stacks = np.asarray(X_output_stacks)
y_output_stacks = np.asarray(y_output_stacks)

"""

def build_Output_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.5


    model = Sequential()

    model.add(Conv1D(128, 
                     padding = 'causal', 
                     kernel_size = 10,
                     input_shape=(max_sent_len, word_size)))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 7))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))

    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 3))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Bidirectional(LSTM(256,return_sequences=True),merge_mode='ave'))

    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model




def build_Output2_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.5


    model = Sequential()

    model.add(Conv1D(128, 
                     padding = 'causal', 
                     kernel_size = 9,
                     input_shape=(max_sent_len, word_size)))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 6))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))

    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 3))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Bidirectional(GRU(256,return_sequences=True),merge_mode='ave'))

    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model




"""
# output Stack
np.random.seed(seed = 17174646)

train_valid_ratio = 0.95
indices = np.random.permutation(X_output_stacks.shape[0])
train_idx, valid_idx = indices[:int(X_output_stacks.shape[0] * train_valid_ratio)], indices[int(X_output_stacks.shape[0] * train_valid_ratio):]
X_output_stacks_train, X_output_stacks_valid = X_output_stacks[train_idx,:], X_output_stacks[valid_idx,:]
y_output_stacks_train, y_output_stacks_valid = y_output_stacks[train_idx,:], y_output_stacks[valid_idx,:]




epochs = 250
patience = 10
batch_size = 128

model = build_Output_model(X_output_stacks.shape[1] , X_output_stacks.shape[2], y_output_stacks.shape[2])
model.summary()


earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
checkpoint = ModelCheckpoint('hw1_output_model.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             mode='min')


history = model.fit(X_output_stacks_train, y_output_stacks_train, 
                 epochs = epochs, 
                 batch_size = batch_size,
                 validation_data = (X_output_stacks_valid, y_output_stacks_valid),
                 callbacks=[earlystopping,checkpoint])




del model




# output Stack2
np.random.seed(seed = 48484646)

train_valid_ratio = 0.95
indices = np.random.permutation(X_output_stacks.shape[0])
train_idx, valid_idx = indices[:int(X_output_stacks.shape[0] * train_valid_ratio)], indices[int(X_output_stacks.shape[0] * train_valid_ratio):]
X_output_stacks_train, X_output_stacks_valid = X_output_stacks[train_idx,:], X_output_stacks[valid_idx,:]
y_output_stacks_train, y_output_stacks_valid = y_output_stacks[train_idx,:], y_output_stacks[valid_idx,:]




epochs = 250
patience = 10
batch_size = 128

model = build_Output2_model(X_output_stacks.shape[1] , X_output_stacks.shape[2], y_output_stacks.shape[2])
model.summary()


earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
checkpoint = ModelCheckpoint('hw1_output2_model.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             mode='min')


history = model.fit(X_output_stacks_train, y_output_stacks_train, 
                 epochs = epochs, 
                 batch_size = batch_size,
                 validation_data = (X_output_stacks_valid, y_output_stacks_valid),
                 callbacks=[earlystopping,checkpoint])


del model
"""


"""
batch_size = 128
model = build_Output_model(X_output_stacks.shape[1] , X_output_stacks.shape[2], y_output_stacks.shape[2])
model.load_weights('hw1_output_model.hdf5')
out1_pred = model.predict(X_output_stacks, batch_size = batch_size)

model = build_Output2_model(X_output_stacks.shape[1] , X_output_stacks.shape[2], y_output_stacks.shape[2])
model.load_weights('hw1_output2_model.hdf5')
out2_pred = model.predict(X_output_stacks, batch_size = batch_size)




X_result_stacks = np.concatenate((  out1_pred,
                                    out2_pred,
                                ), axis=2)


print(X_result_stacks.shape)



print('Dump X_result_stacks', time.time()-stime)
with open('X_result_stacks', 'wb') as fp:
    pickle.dump(X_result_stacks, fp)


"""

#########################################
############## STAGE 4 ##################
###############  END  ###################
#########################################





#########################################
############## STAGE 5 ##################
############### START ###################
#########################################


"""
print('load X_result_stacks', time.time()-stime)
with open ('X_result_stacks', 'rb') as fp:
    X_result_stacks = pickle.load(fp)

print('load y_result_stacks', time.time()-stime)    
with open ('y', 'rb') as fp:
    y_result_stacks = pickle.load(fp)

X_result_stacks = np.asarray(X_result_stacks)
y_result_stacks = np.asarray(y_result_stacks)
"""



def build_Result_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.5


    model = Sequential()

    model.add(Conv1D(128, 
                     padding = 'causal', 
                     kernel_size = 8,
                     input_shape=(max_sent_len, word_size)))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 6))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))

    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 4))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Bidirectional(LSTM(256,return_sequences=True),merge_mode='ave'))

    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model




def build_Result2_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.5


    model = Sequential()

    model.add(Conv1D(128, 
                     padding = 'causal', 
                     kernel_size = 7,
                     input_shape=(max_sent_len, word_size)))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 7))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))

    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 3))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Bidirectional(GRU(256,return_sequences=True),merge_mode='ave'))

    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model








"""
# result Stack
np.random.seed(seed = 4846)

train_valid_ratio = 0.95
indices = np.random.permutation(X_result_stacks.shape[0])
train_idx, valid_idx = indices[:int(X_result_stacks.shape[0] * train_valid_ratio)], indices[int(X_result_stacks.shape[0] * train_valid_ratio):]
X_result_stacks_train, X_result_stacks_valid = X_result_stacks[train_idx,:], X_result_stacks[valid_idx,:]
y_result_stacks_train, y_result_stacks_valid = y_result_stacks[train_idx,:], y_result_stacks[valid_idx,:]




epochs = 250
patience = 10
batch_size = 128

model = build_Result_model(X_result_stacks.shape[1] , X_result_stacks.shape[2], y_result_stacks.shape[2])
model.summary()


earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
checkpoint = ModelCheckpoint('hw1_result_model.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             mode='min')


history = model.fit(X_result_stacks_train, y_result_stacks_train, 
                 epochs = epochs, 
                 batch_size = batch_size,
                 validation_data = (X_result_stacks_valid, y_result_stacks_valid),
                 callbacks=[earlystopping,checkpoint])


del model


"""




"""
# result Stack2
np.random.seed(seed = 4648)

train_valid_ratio = 0.95
indices = np.random.permutation(X_result_stacks.shape[0])
train_idx, valid_idx = indices[:int(X_result_stacks.shape[0] * train_valid_ratio)], indices[int(X_result_stacks.shape[0] * train_valid_ratio):]
X_result_stacks_train, X_result_stacks_valid = X_result_stacks[train_idx,:], X_result_stacks[valid_idx,:]
y_result_stacks_train, y_result_stacks_valid = y_result_stacks[train_idx,:], y_result_stacks[valid_idx,:]




epochs = 250
patience = 10
batch_size = 128

model = build_Result2_model(X_result_stacks.shape[1] , X_result_stacks.shape[2], y_result_stacks.shape[2])
model.summary()


earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
checkpoint = ModelCheckpoint('hw1_result2_model.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             mode='min')


history = model.fit(X_result_stacks_train, y_result_stacks_train, 
                 epochs = epochs, 
                 batch_size = batch_size,
                 validation_data = (X_result_stacks_valid, y_result_stacks_valid),
                 callbacks=[earlystopping,checkpoint])


del model

"""





"""

batch_size = 128
model = build_Result_model(X_result_stacks.shape[1] , X_result_stacks.shape[2], y_result_stacks.shape[2])
model.load_weights('hw1_result_model.hdf5')
res1_pred = model.predict(X_result_stacks, batch_size = batch_size)

model = build_Result2_model(X_result_stacks.shape[1] , X_result_stacks.shape[2], y_result_stacks.shape[2])
model.load_weights('hw1_result2_model.hdf5')
res2_pred = model.predict(X_result_stacks, batch_size = batch_size)




X_final_stacks = np.concatenate((   res1_pred,
                                    res2_pred,
                                ), axis=2)


print(X_final_stacks.shape)



print('Dump X_final_stacks', time.time()-stime)
with open('X_final_stacks', 'wb') as fp:
    pickle.dump(X_final_stacks, fp)
"""


#########################################
############## STAGE 5 ##################
###############  END  ###################
#########################################


#########################################
############## STAGE 6 ##################
############### START ###################
#########################################

"""

print('load X_final_stacks', time.time()-stime)
with open ('X_final_stacks', 'rb') as fp:
    X_final_stacks = pickle.load(fp)

print('load y_final_stacks', time.time()-stime)    
with open ('y', 'rb') as fp:
    y_final_stacks = pickle.load(fp)

X_final_stacks = np.asarray(X_final_stacks)
y_final_stacks = np.asarray(y_final_stacks)
"""



def build_Final_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.5


    model = Sequential()

    model.add(Conv1D(128, 
                     padding = 'causal', 
                     kernel_size = 9,
                     input_shape=(max_sent_len, word_size)))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 5))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))

    model.add(Conv1D(128, 
                     padding = 'causal',
                     kernel_size = 3))
    model.add(Activation(PReLU()))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(Bidirectional(LSTM(256,return_sequences=True),merge_mode='ave'))

    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model





"""
# final Stack
np.random.seed(seed = 484466)

train_valid_ratio = 0.95
indices = np.random.permutation(X_final_stacks.shape[0])
train_idx, valid_idx = indices[:int(X_final_stacks.shape[0] * train_valid_ratio)], indices[int(X_final_stacks.shape[0] * train_valid_ratio):]
X_final_stacks_train, X_final_stacks_valid = X_final_stacks[train_idx,:], X_final_stacks[valid_idx,:]
y_final_stacks_train, y_final_stacks_valid = y_final_stacks[train_idx,:], y_final_stacks[valid_idx,:]




epochs = 250
patience = 10
batch_size = 128

model = build_Final_model(X_final_stacks.shape[1] , X_final_stacks.shape[2], y_final_stacks.shape[2])
model.summary()


earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
checkpoint = ModelCheckpoint('hw1_final_model.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             mode='min')


history = model.fit(X_final_stacks_train, y_final_stacks_train, 
                 epochs = epochs, 
                 batch_size = batch_size,
                 validation_data = (X_final_stacks_valid, y_final_stacks_valid),
                 callbacks=[earlystopping,checkpoint])


del model
"""

#########################################
############## STAGE 6 ##################
###############  END  ###################
#########################################




print('Time Taken:', time.time()-stime)


