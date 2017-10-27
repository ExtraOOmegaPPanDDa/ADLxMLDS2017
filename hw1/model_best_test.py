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
batch_size = 128


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




#########################################
############## STAGE 1 ##################
############### START ###################
#########################################




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

    
    X_norm = X_test - np.tile(train_mean_box[i+to_add], (X_test.shape[0],1,1))
    X_norm = X_norm / np.tile(train_std_box[i+to_add], (X_test.shape[0],1,1))

    model = build_LSTM_model(X_test.shape[1] , X_test.shape[2], 39)
    model.load_weights('hw1_lstm_model_'+str(skf_iteration)+'.hdf5')


    X_toStack = model.predict(X_norm, verbose = 1, batch_size = batch_size)

    if check == 0:
        X_lstm_blend = X_toStack
        check = 1
    else:
        X_lstm_blend = X_lstm_blend + X_toStack

    # print(X_lstm_blend.shape)



# GRU Model

X_gru_blend = 0
check = 0

skf_iteration = 0
to_add = 15
for i in range(nfold):

    skf_iteration += 1
    print('SKF', skf_iteration)

    
    X_norm = X_test - np.tile(train_mean_box[i+to_add], (X_test.shape[0],1,1))
    X_norm = X_norm / np.tile(train_std_box[i+to_add], (X_test.shape[0],1,1))

    model = build_GRU_model(X_test.shape[1] , X_test.shape[2], 39)
    model.load_weights('hw1_gru_model_'+str(skf_iteration)+'.hdf5')


    X_toStack = model.predict(X_norm, verbose = 1, batch_size = batch_size)

    if check == 0:
        X_gru_blend = X_toStack
        check = 1
    else:
        X_gru_blend = X_gru_blend + X_toStack

    # print(X_gru_blend.shape)




# SimpleRNN Model

X_simplernn_blend = 0
check = 0

skf_iteration = 0
to_add = 30
for i in range(nfold):

    skf_iteration += 1
    print('SKF', skf_iteration)

    
    X_norm = X_test - np.tile(train_mean_box[i+to_add], (X_test.shape[0],1,1))
    X_norm = X_norm / np.tile(train_std_box[i+to_add], (X_test.shape[0],1,1))

    model = build_SimpleRNN_model(X_test.shape[1] , X_test.shape[2], 39)
    model.load_weights('hw1_simplernn_model_'+str(skf_iteration)+'.hdf5')


    X_toStack = model.predict(X_norm, verbose = 1, batch_size = batch_size)

    if check == 0:
        X_simplernn_blend = X_toStack
        check = 1
    else:
        X_simplernn_blend = X_simplernn_blend + X_toStack

    # print(X_simplernn_blend.shape)




# SimpleRNN2 Model
X_simplernn2_blend = 0
check = 0

skf_iteration = 0
to_add = 45
for i in range(nfold):

    skf_iteration += 1
    print('SKF', skf_iteration)

    
    X_norm = X_test - np.tile(train_mean_box[i+to_add], (X_test.shape[0],1,1))
    X_norm = X_norm / np.tile(train_std_box[i+to_add], (X_test.shape[0],1,1))

    model = build_SimpleRNN2_model(X_test.shape[1] , X_test.shape[2], 39)
    model.load_weights('hw1_simplernn2_model_'+str(skf_iteration)+'.hdf5')


    X_toStack = model.predict(X_norm, verbose = 1, batch_size = batch_size)

    if check == 0:
        X_simplernn2_blend = X_toStack
        check = 1
    else:
        X_simplernn2_blend = X_simplernn2_blend + X_toStack

    # print(X_simplernn2_blend.shape)



X_stacks = np.concatenate(( X_lstm_blend,
                            X_gru_blend,
                            X_simplernn_blend,
                            X_simplernn2_blend
                            ), axis=2)


print(X_stacks.shape)



#########################################
############## STAGE 1 ##################
###############  END  ###################
#########################################



#########################################
############## STAGE 2 ##################
############### START ###################
#########################################



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





print('Stacking')

model = build_Stack_model(X_stacks.shape[1] , X_stacks.shape[2], 39)
model.load_weights('hw1_stack_model.hdf5')
res_stack1 = model.predict(X_stacks, verbose = 1, batch_size = batch_size)

model = build_Stack2_model(X_stacks.shape[1] , X_stacks.shape[2], 39)
model.load_weights('hw1_stack2_model.hdf5')
res_stack2 = model.predict(X_stacks, verbose = 1, batch_size = batch_size)

model = build_Stack3_model(X_stacks.shape[1] , X_stacks.shape[2], 39)
model.load_weights('hw1_stack3_model.hdf5')
res_stack3 = model.predict(X_stacks, verbose = 1, batch_size = batch_size)



X_last_stacks = np.concatenate((    res_stack1,
                                    res_stack2,
                                    res_stack3
                            ), axis=2)

print(X_last_stacks.shape)



#########################################
############## STAGE 2 ##################
###############  END  ###################
#########################################



#########################################
############## STAGE 3 ##################
############### START ###################
#########################################



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






print('Last Stacking')
model = build_lastStack_model(X_last_stacks.shape[1] , X_last_stacks.shape[2], 39)
model.load_weights('hw1_lastStack_model.hdf5')
last1_pred = model.predict(X_last_stacks, verbose = 1, batch_size = batch_size)


model = build_lastStack2_model(X_last_stacks.shape[1] , X_last_stacks.shape[2], 39)
model.load_weights('hw1_lastStack2_model.hdf5')
last2_pred = model.predict(X_last_stacks, verbose = 1, batch_size = batch_size)

model = build_lastStack3_model(X_last_stacks.shape[1] , X_last_stacks.shape[2], 39)
model.load_weights('hw1_lastStack3_model.hdf5')
last3_pred = model.predict(X_last_stacks, verbose = 1, batch_size = batch_size)



X_output_stacks = np.concatenate((    last1_pred,
                                      last2_pred,
                                      last3_pred
                            ), axis=2)

print(X_output_stacks.shape)



#########################################
############## STAGE 3 ##################
###############  END  ###################
#########################################



#########################################
############## STAGE 4 ##################
############### START ###################
#########################################


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



print('Output Stacking')
model = build_Output_model(X_output_stacks.shape[1] , X_output_stacks.shape[2], 39)
model.load_weights('hw1_output_model.hdf5')
out1_pred = model.predict(X_output_stacks, verbose = 1, batch_size = batch_size)

model = build_Output2_model(X_output_stacks.shape[1] , X_output_stacks.shape[2], 39)
model.load_weights('hw1_output2_model.hdf5')
out2_pred = model.predict(X_output_stacks, verbose = 1, batch_size = batch_size)


X_result_stacks = np.concatenate((  out1_pred,
                                    out2_pred,
                                ), axis=2)


print(X_result_stacks.shape)



#########################################
############## STAGE 4 ##################
###############  END  ###################
#########################################





#########################################
############## STAGE 5 ##################
############### START ###################
#########################################




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



print('Result Stacking')
model = build_Result_model(X_result_stacks.shape[1] , X_result_stacks.shape[2], 39)
model.load_weights('hw1_result_model.hdf5')
res = model.predict(X_result_stacks, verbose = 1, batch_size = batch_size)


#########################################
############## STAGE 5 ##################
###############  END  ###################
#########################################



#########################################
############ BINARY2PHONE ###############
#########################################


res = np.argmax(res, axis = 2)
res = res.astype('<U5')
for i in range(res.shape[0]):
    for j in range(res.shape[1]):
        res[i][j]=phone_39_set_list[int(res[i][j])]




#########################################
############# SMOOTHING #################
#########################################

smooth_range = 3

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







#########################################
######### REMOVE DUPLICATED #############
#########################################

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




#########################################
############# PHONE2CHAR ################
#########################################


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

f = open('prediction.csv', 'w')
w = csv.writer(f)
w.writerows(revised_result)
f.close()


print('Time Taken:', time.time()-stime)




