# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 23:15:55 2017

@author: HSIN
"""

import time
import csv
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import LSTM, GRU
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



char_list = []
phone_48_list = []
phone_39_list = []



print('Get 48-Char Map', time.time()-stime)
f = open('./48phone_char.map', 'r')
for line in f:
    data = line.replace('\n', '')
    data = data.split('\t')
    phone_48_list.append(data[0])
    char_list.append(data[2])
    phone_39_list.append(0)
f.close()

print('Get 48-39 Map', time.time()-stime)
f = open('./phones/48_39.map', 'r')
for line in f:
    data = line.replace('\n', '')
    data = data.split('\t')
    phone_39_list[phone_48_list.index(data[0])] = data[1]
f.close()



phone_39_set_list = list(set(phone_39_list))




train_ids = []
train_mfccs = []
train_fbanks = []
train_labels = []




print('Get Train MFCC Feature', time.time()-stime)
f = open('./mfcc/train.ark', 'r')
ids = []
count = 0
for line in f:
    count += 1
    if count  % 100000 == 0:
        print(count)
    
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
f = open('./fbank/train.ark', 'r')
ids = []
count = 0
for line in f:
    count += 1
    if count  % 100000 == 0:
        print(count)
    
    data = line.replace('\n', '')
    data = data.split(' ')
    ids.append(data[0])
    train_fbanks.append(([float(x) for x in data[1:]]))
f.close()


sorted_zip_list = list(sorted(zip(ids,train_fbanks)))
sorted_zip_list = list(zip(*sorted_zip_list))
ids = np.asarray(sorted_zip_list[0])
train_fbanks = np.asarray(sorted_zip_list[1])




train_features = np.concatenate((train_mfccs,train_fbanks),axis=1)

print('Get Train Label', time.time()-stime)
f = open('./label/train.lab', 'r')
ids = []
count = 0
for line in f:
    count += 1
    if count  % 100000 == 0:
        print(count)
    
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



max_sent_len = 0
for i in range(len(speaker_ids)):
    sids = speaker_ids[i]
    sids_split = sids.split('_')
    speaker_ids[i] = '_'.join(sids_split[:2])
    if int(sids_split[2]) > max_sent_len:
        max_sent_len = int(sids_split[2])

speaker_ids = list(set(speaker_ids))

X = []
y = []

for i in range(len(speaker_ids)):
    if i % 1000 == 0:
        print(i)
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
        print(i)
        print(train_labels[i])
    tids = train_ids[i]
    tids_split = tids.split('_')
    speaker_id = '_'.join(tids_split[:2])
    context_id = int(tids_split[2])
     
    if context_id > id_max_record[speaker_ids.index(speaker_id)]:
        id_max_record[speaker_ids.index(speaker_id)] = context_id



for i in range(len(train_ids)):
    if i % 10000 == 0:
        print(i)
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

        # if pad_time > 0:
        #    break
    
    
    
    pad_time = 0
    while(1):
        if context_id-1 + pad_time*id_max_record[speaker_ids.index(speaker_id)] >= max_sent_len:
            break
        y[speaker_ids.index(speaker_id)][int(context_id-1 + pad_time*id_max_record[speaker_ids.index(speaker_id)])][phone_39_set_list.index(train_labels[i])] = 1
        pad_time += 1

        # if pad_time > 0:
        #    break
        
    




print('Dumnp X', time.time()-stime)
with open('X', 'wb') as fp:
    pickle.dump(X, fp)

print('Dumnp y', time.time()-stime)
with open('y', 'wb') as fp:
    pickle.dump(y, fp)




print('load X', time.time()-stime)
with open ('X', 'rb') as fp:
    X = pickle.load(fp)

print('load y', time.time()-stime)    
with open ('y', 'rb') as fp:
    y = pickle.load(fp)

X = np.asarray(X)
y = np.asarray(y)


X_train = np.asarray(X)
y_train = np.asarray(y)


"""
X_train_mean = np.mean(X_train, axis = 0)
X_train_std = np.std(X_train, axis = 0)
X_train = X_train - np.tile(X_train_mean, (X_train.shape[0],1,1))
X_train = X_train / np.tile(X_train_std, (X_train.shape[0],1,1))
"""




print('X_train_shape', X_train.shape, time.time()-stime)
print('y_train_shape', y_train.shape, time.time()-stime)

np.random.seed(seed = 464646)

train_valid_ratio = 0.9
indices = np.random.permutation(X_train.shape[0])
train_idx, valid_idx = indices[:int(X_train.shape[0] * train_valid_ratio)], indices[int(X_train.shape[0] * train_valid_ratio):]
X_train, X_valid = X_train[train_idx,:], X_train[valid_idx,:]
y_train, y_valid = y_train[train_idx,:], y_train[valid_idx,:]



epochs = 1000
batch_size = 128
drop_out_ratio = 0.25
patience = 30


def build_model(max_sent_len, word_size, output_size):
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
    
    model.add(Bidirectional(LSTM(256,return_sequences=True),merge_mode='ave'))
    model.add(Bidirectional(LSTM(128,return_sequences=True),merge_mode='ave'))
                            
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    # model.add(LSTM(128, input_shape = (max_sent_len, word_size), 
    #                stateful  =False,
    #                dropout = drop_out_ratio,
    #                return_sequences = True))
    
    
    # model.add(Bidirectional(LSTM(128, input_shape = (max_sent_len, word_size), 
    #                stateful  =False,
    #                dropout = drop_out_ratio,
    #                return_sequences = True)))
                   
    # model.add(TimeDistributed(Dense(output_size,
    #                                 activation='softmax')))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model




model = build_model(X_train.shape[1] , X_train.shape[2], y_train.shape[2])
model.summary()


earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
checkpoint = ModelCheckpoint('hw1_model.hdf5',
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







model = build_model(X_train.shape[1] , X_train.shape[2], y_train.shape[2])
model.load_weights('hw1_model.hdf5')



"""
model.save('hw1_complete_model.h5')
del model


model = load_model('hw1_complete_model.h5')
"""




val_pred = model.predict(X_valid)
val_pred = np.argmax(val_pred, axis = 2)
val_pred = val_pred.astype('<U5')
for i in range(val_pred.shape[0]):
    for j in range(val_pred.shape[1]):
        val_pred[i][j]=phone_39_set_list[int(val_pred[i][j])]




val_true = np.argmax(y_valid, axis = 2)
val_true = val_true.astype('<U5')
for i in range(val_true.shape[0]):
    for j in range(val_true.shape[1]):
        val_true[i][j]=phone_39_set_list[int(val_true[i][j])]


print('Val Pred 0')
print(val_pred[0])
print('Val True 0')
print(val_true[0])

print('Val Pred 20')
print(val_pred[20])
print('Val True 20')
print(val_true[20])

print('Val Pred 100')
print(val_pred[100])
print('Val True 100')
print(val_true[100])







test_ids = []
test_mfccs = []
test_fbanks = []



print('Get Test MFCC Feature', time.time()-stime)
f = open('./mfcc/test.ark', 'r')
ids = []
count = 0
for line in f:
    count += 1
    if count  % 100000 == 0:
        print(count)
    
    data = line.replace('\n', '')
    data = data.split(' ')
    ids.append(data[0])
    test_mfccs.append(([float(x) for x in data[1:]]))
f.close()

sorted_zip_list = list(sorted(zip(ids,test_mfccs)))
sorted_zip_list = list(zip(*sorted_zip_list))
ids = np.asarray(sorted_zip_list[0])
test_mfccs = np.asarray(sorted_zip_list[1])




print('Get Test FBANK Feature', time.time()-stime)
f = open('./fbank/test.ark', 'r')
ids = []
count = 0
for line in f:
    count += 1
    if count  % 100000 == 0:
        print(count)
    
    data = line.replace('\n', '')
    data = data.split(' ')
    ids.append(data[0])
    test_fbanks.append(([float(x) for x in data[1:]]))
f.close()


sorted_zip_list = list(sorted(zip(ids,test_fbanks)))
sorted_zip_list = list(zip(*sorted_zip_list))
ids = np.asarray(sorted_zip_list[0])
test_fbanks = np.asarray(sorted_zip_list[1])




test_features = np.concatenate((test_mfccs,test_fbanks),axis=1)


test_ids = ids
test_speaker_ids = list(set(ids))


for i in range(len(test_speaker_ids)):
    sids = test_speaker_ids[i]
    sids_split = sids.split('_')
    test_speaker_ids[i] = '_'.join(sids_split[:2])
    
test_speaker_ids = list(set(test_speaker_ids))

X_test = []

max_sent_len = X_train.shape[1]


for i in range(len(test_speaker_ids)):
    if i % 1000 == 0:
        print(i)
    to_append = []
    for j in range(max_sent_len):
        to_append.append(np.zeros(test_features.shape[1]))
    X_test.append(to_append)
    




test_id_max_record = np.zeros(len(test_speaker_ids))

for i in range(len(test_ids)):
    if i % 10000 == 0:
        print(i)
    tids = test_ids[i]
    tids_split = tids.split('_')
    test_speaker_id = '_'.join(tids_split[:2])
    context_id = int(tids_split[2])
 
    if context_id > test_id_max_record[test_speaker_ids.index(test_speaker_id)]:
        test_id_max_record[test_speaker_ids.index(test_speaker_id)] = context_id



for i in range(len(test_ids)):
    if i % 10000 == 0:
        print(i)
    tids = test_ids[i]
    tids_split = tids.split('_')
    test_speaker_id = '_'.join(tids_split[:2])
    context_id = int(tids_split[2])
    
    pad_time = 0
    while(1):
        if context_id-1 + pad_time*test_id_max_record[test_speaker_ids.index(test_speaker_id)] >= max_sent_len:
            break
        X_test[test_speaker_ids.index(test_speaker_id)][int(context_id-1 + pad_time*test_id_max_record[test_speaker_ids.index(test_speaker_id)])]  = test_features[i]
        pad_time += 1
        
        #if pad_time > 0:
        #    break
    

X_test = np.asarray(X_test)

res = model.predict(X_test)
res= np.argmax(res, axis = 2)
res = res.astype('<U5')
for i in range(res.shape[0]):
    for j in range(res.shape[1]):
        res[i][j]=phone_39_set_list[int(res[i][j])]



print(res[0])


res_signed = np.ones(res.shape)
res_signed = res_signed.astype('<U20')



for i in range(res.shape[0]):
    for j in range(res.shape[1]):
        if res[i][j] == 'sil':
            res_signed[i][j] = 'to_be_deleted'
        else:
            break
    
    
    for j in range(res.shape[1]):
        if res[i][res.shape[1]-1-j] == 'sil':
            res_signed[i][res.shape[1]-1-j] = 'to_be_deleted'
        else:
            break
    
    
    for j in range(res.shape[1]):
        if j == 0:
            continue
        elif res[i][j] == res[i][j-1]:
            res_signed[i][j] = 'to_be_deleted'
            
    for j in range(res.shape[1]):
        if j > test_id_max_record[i]-1:
            res_signed[i][j] = 'to_be_deleted'
        
    

result = []
for i in range(res.shape[0]):
    seq = []
    for j in range(res.shape[1]):
        if res_signed[i][j] == 'to_be_deleted':
            continue
        else:
            seq.append(res[i][j])
    
    result.append(seq)



order_result = []
for i in range(len(result)):
    data = []
    seq = ''
    for j in range(len(result[i])):
        seq = seq + char_list[phone_48_list.index(result[i][j])]
    
    data.append(test_speaker_ids[i])
    data.append(seq)
    
    order_result.append(data)




order_result = sorted(order_result)

order_result = [['id','phone_sequence']] + order_result

f = open('prediction.csv', 'w', newline='')
w = csv.writer(f)
w.writerows(order_result)
f.close()

print('Time Taken:', time.time()-stime)
