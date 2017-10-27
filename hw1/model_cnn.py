# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 23:15:55 2017

@author: HSIN
"""

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

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



def build_CNN_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.2


    model = Sequential()

    model.add(Conv1D(512, 
                     padding = 'causal', 
                     kernel_size = 6,
                     input_shape=(max_sent_len, word_size)))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))


    model.add(SimpleRNN(512,
                        dropout = drop_out_ratio,
                        return_sequences=True))

    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model







np.random.seed(seed = 174646)

train_valid_ratio = 0.9
indices = np.random.permutation(X.shape[0])
train_idx, valid_idx = indices[:int(X.shape[0] * train_valid_ratio)], indices[int(X.shape[0] * train_valid_ratio):]
X_train, X_valid = X[train_idx,:], X[valid_idx,:]
y_train, y_valid = y[train_idx,:], y[valid_idx,:]




epochs = 250
patience = 10
batch_size = 128

model = build_CNN_model(X.shape[1] , X.shape[2], y.shape[2])
model.summary()


earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')
checkpoint = ModelCheckpoint('hw1_cnn_model.hdf5',
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



"""
loss_list = list(history.history['loss'])
val_loss_list = list(history.history['val_loss'])

acc_list = list(history.history['acc'])
val_acc_list = list(history.history['val_acc'])




history_list = [loss_list, val_loss_list, acc_list, val_acc_list]


with open('cnn_history','wb') as fp:
    pickle.dump(history_list, fp)



with open('cnn_history','rb') as fp:
    history_list = pickle.load(fp)



plt.plot(history_list[0])
plt.plot(history_list[1])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xlim(0, 80)
plt.ylim(0, 3.5)
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('cnn_model_loss.png')
plt.clf()


plt.plot(history_list[2])
plt.plot(history_list[3])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.xlim(0, 80)
plt.ylim(0, 1)
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('cnn_model_acc.png')
plt.clf()
"""


"""
print('Train Loss')
print(history_list[0])

print('Valid Loss')
print(history_list[1])


print('Train Acc')
print(history_list[2])


print('Valid Acc')
print(history_list[3])

"""


"""
def build_LSTM_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.2


    model = Sequential()

    model.add(Conv1D(384, 
                     padding = 'causal', 
                     kernel_size = 6,
                     input_shape=(max_sent_len, word_size)))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))


    model.add(LSTM(256,
                        dropout = drop_out_ratio,
                        return_sequences=True))

    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model








epochs = 250
patience = 10
batch_size = 128

model = build_LSTM_model(X.shape[1] , X.shape[2], y.shape[2])
model.summary()


earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')


history = model.fit(X_train, y_train, 
                 epochs = epochs, 
                 batch_size = batch_size,
                 validation_data = (X_valid, y_valid),
                 callbacks=[earlystopping])


del model





def build_BLSTM_model(max_sent_len, word_size, output_size):
    
    drop_out_ratio = 0.2


    model = Sequential()

    model.add(Conv1D(128, 
                     padding = 'causal', 
                     kernel_size = 6,
                     input_shape=(max_sent_len, word_size)))
    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))


    model.add(Bidirectional(LSTM(256,
                        dropout = drop_out_ratio,
                        return_sequences=True), merge_mode = 'ave'))

    model.add(BatchNormalization())
    model.add(Dropout(drop_out_ratio))
    
    model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal')

    return model








epochs = 250
patience = 10
batch_size = 128

model = build_BLSTM_model(X.shape[1] , X.shape[2], y.shape[2])
model.summary()


earlystopping = EarlyStopping(monitor='val_loss', patience = patience, verbose=1, mode='min')


history = model.fit(X_train, y_train, 
                 epochs = epochs, 
                 batch_size = batch_size,
                 validation_data = (X_valid, y_valid),
                 callbacks=[earlystopping])


del model
"""




"""
test_ids = []
test_mfccs = []
test_fbanks = []



print('Get Test MFCC Feature', time.time()-stime)
f = open(os.path.join(data_path, 'mfcc/test.ark'), 'r')
ids = []
count = 0
for line in f:
    count += 1
    if count  % 100000 == 0:
        print('Get Test MFCC', count)
    
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
f = open(os.path.join(data_path, 'fbank/test.ark'), 'r')
ids = []
count = 0
for line in f:
    count += 1
    if count  % 100000 == 0:
        print('Get Test FBANK', count)
    
    data = line.replace('\n', '')
    data = data.split(' ')
    ids.append(data[0])
    test_fbanks.append(([float(x) for x in data[1:]]))
f.close()


sorted_zip_list = list(sorted(zip(ids,test_fbanks)))
sorted_zip_list = list(zip(*sorted_zip_list))
ids = np.asarray(sorted_zip_list[0])
test_fbanks = np.asarray(sorted_zip_list[1])




test_features = np.concatenate((test_mfccs,test_fbanks), axis=1)


test_ids = ids
test_speaker_ids = list(set(ids))


for i in range(len(test_speaker_ids)):
    sids = test_speaker_ids[i]
    sids_split = sids.split('_')
    test_speaker_ids[i] = '_'.join(sids_split[:2])
    
test_speaker_ids = list(set(test_speaker_ids))

X_test = []


for i in range(len(test_speaker_ids)):
    if i % 1000 == 0:
        print('Test Padding Set', i)
    to_append = []
    for j in range(max_sent_len):
        to_append.append(np.zeros(test_features.shape[1]))
    X_test.append(to_append)
    




test_id_max_record = np.zeros(len(test_speaker_ids))

for i in range(len(test_ids)):
    if i % 10000 == 0:
        print('Test ID Transform Step1', i)
    tids = test_ids[i]
    tids_split = tids.split('_')
    test_speaker_id = '_'.join(tids_split[:2])
    context_id = int(tids_split[2])
 
    if context_id > test_id_max_record[test_speaker_ids.index(test_speaker_id)]:
        test_id_max_record[test_speaker_ids.index(test_speaker_id)] = context_id



for i in range(len(test_ids)):
    if i % 10000 == 0:
        print('Test ID Transform Step2', i)
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
        

X_test = np.asarray(X_test)


model = build_CNN_model(X_test.shape[1] , X_test.shape[2], 39)
model.load_weights('hw1_cnn_model.hdf5')


batch_size = 128
res = model.predict(X_test, verbose = 1, batch_size = batch_size)
res = np.argmax(res, axis = 2)
res = res.astype('<U5')
for i in range(res.shape[0]):
    for j in range(res.shape[1]):
        res[i][j]=phone_39_set_list[int(res[i][j])]




smooth_range = 5

for i in range(res.shape[0]):
    for j in range(res.shape[1]):
        if j == 0:
            continue

        else:

            for k in range(smooth_range):
                if k == 0:
                    continue
                else:
                    if j < res.shape[1] - k:
                        if res[i][j-1] == res[i][j+k]:
                            for l in range(k):
                                res[i][j+l] = res[i][j-1]



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



char_result = []
for i in range(len(result)):
    data = []
    seq = ''
    for j in range(len(result[i])):
        seq = seq + char_list[phone_48_list.index(result[i][j])]
    
    data.append(test_speaker_ids[i])
    data.append(seq)
    
    char_result.append(data)




char_result = sorted(char_result)



def most_common(lst):
    return max(set(lst), key=lst.count)



ids = []
seqs = []
seq_s = []
seq_e = []
for data in char_result:
    ids.append(data[0])
    seqs.append(data[1])
    seq_s.append(data[1][0])
    seq_e.append(data[1][-1])


most_common_s = most_common(seq_s)
most_common_e = most_common(seq_e)

if seq_s.count(most_common_s)/len(seq_s) > 0.5:
    print('remove Head', most_common_s)
    print(seq_s.count(most_common_s)/len(seq_s))
    
    for i in range(len(seqs)):
        if seqs[i][0] == most_common_s:
            seqs[i] = seqs[i][1:]


if seq_e.count(most_common_e)/len(seq_e) > 0.5:
    print('remove Tail', most_common_e)
    print(seq_e.count(most_common_e)/len(seq_e))
    
    for i in range(len(seqs)):
        if seqs[i][-1] == most_common_e:
            seqs[i] = seqs[i][:-1]


revised_result = []
for i in range(len(seqs)):
    data = []
    data.append(ids[i])
    data.append(seqs[i])

    revised_result.append(data)


revised_result = sorted(revised_result)


revised_result = [['id','phone_sequence']] + revised_result

f = open(output_path, 'w', newline='')
w = csv.writer(f)
w.writerows(revised_result)
f.close()
"""

print('Time Taken:', time.time()-stime)