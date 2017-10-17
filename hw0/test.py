import sys
import time
import csv
import numpy as np
from mnist_reader import load_mnist
from mnist_reader import load_mnist_im
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from keras.layers import Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.utils import np_utils
from keras.initializers import Constant
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

def get_acc(true, pred):
    count = 0
    match = 0
    for i, j in list(zip(true,pred)):
        count += 1
        if i == j:
            match += 1
    
    return match/count

nf = 20
batch_size = 128
epochs = 40
patience = 5

res_box = []

model_taken = ['CNN2']

# Load data
# Function load_minst is available in git.
#X_train, y_train = load_mnist('data/fashion', kind='train')
#X_test, y_test = load_mnist('data/fashion', kind='t10k')

X_data, y_data = load_mnist('data/given', kind='train')
X_test_data = load_mnist_im('data/given', kind='t10k')

np.random.seed(seed = 46)
indices = np.random.permutation(X_data.shape[0])

X_data = X_data[indices]
y_data = y_data[indices]

all_X_data = np.concatenate((X_data, X_test_data), axis=0)
all_X_data = scale(all_X_data)
X_data = all_X_data[:len(X_data)]
X_test_data = all_X_data[len(X_data):]

#X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.2, random_state=46)

skf = list(StratifiedKFold(y_data, nf))

for iter_time in range(nf):
    if 'CNN1' not in model_taken:
        break
    print('CNN1', iter_time)    
    train_index, valid_index = skf[iter_time]
    X_train = X_data[train_index]
    X_valid = X_data[valid_index]
    y_train = y_data[train_index]
    y_valid = y_data[valid_index]
    
    X_test = X_test_data


    # Prepare datasets
    # This step contains normalization and reshaping of input.
    # For output, it is important to change number to one-hot vector. 
    X_train = X_train.astype('float32') / 255
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_valid = X_valid.astype('float32') / 255
    X_valid = X_valid.reshape(X_valid.shape[0], 1, 28, 28)
    X_test = X_test.astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    y_train = np_utils.to_categorical(y_train, 10)
    y_valid = np_utils.to_categorical(y_valid, 10)
    
    
    # Create model in Keras
    # This model is linear stack of layers
    clf = Sequential()
    # This layer is used as an entry point into a graph. 
    # So, it is important to define input_shape.
    clf.add(
        InputLayer(input_shape=(1, 28, 28))
    )
    # Normalize the activations of the previous layer at each batch.
    clf.add(
        BatchNormalization()
    )
    # Next step is to add convolution layer to model.
    clf.add(
        Conv2D(
            32, (2, 2), 
            padding='same', 
            bias_initializer=Constant(0.01), 
            kernel_initializer='random_uniform'
        )
    )
    # Add max pooling layer for 2D data.
    clf.add(MaxPool2D(padding='same'))
    # Add this same two layers to model.
    clf.add(
        Conv2D(
            32, 
            (2, 2), 
            padding='same', 
            bias_initializer=Constant(0.01), 
            kernel_initializer='random_uniform', 
            input_shape=(1, 28, 28)
        )
    )
    
    clf.add(MaxPool2D(padding='same'))
    # It is necessary to flatten input data to a vector.
    clf.add(Flatten())
    # Last step is creation of fully-connected layers.
    clf.add(
        Dense(
            128,
            activation='relu',
            bias_initializer=Constant(0.01), 
            kernel_initializer='random_uniform',         
        )
    )
    # Add output layer, which contains ten numbers.
    # Each number represents cloth type.
    clf.add(Dense(10, activation='softmax'))
    # Last step in Keras is to compile model.
    clf.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )
    
    earlystopping = EarlyStopping(monitor='val_acc', patience = patience, verbose=1, mode='max')


    checkpoint = ModelCheckpoint('best_model'+str(iter_time)+'.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_acc',
                                 mode='max')
    
    clf.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_valid, y_valid),
        callbacks=[earlystopping,checkpoint]
    )
    
    # clf.evaluate(X_valid, y_valid)
    
    
    clf.load_weights('best_model'+str(iter_time)+'.hdf5')
    
    true = y_data[valid_index]
    pred = clf.predict_classes(X_valid, batch_size=batch_size)
    print('Valid Acc', get_acc(true, pred))
    
    res = clf.predict_classes(X_test, batch_size=batch_size)
    res = np_utils.to_categorical(res, 10)
    res_box.append(res)
    
    
    



for iter_time in range(nf):
    if 'CNN2' not in model_taken:
        break
    print('CNN2', iter_time)    
    train_index, valid_index = skf[iter_time]
    X_train = X_data[train_index]
    X_valid = X_data[valid_index]
    y_train = y_data[train_index]
    y_valid = y_data[valid_index]
    
    X_test = X_test_data


    # Prepare datasets
    # This step contains normalization and reshaping of input.
    # For output, it is important to change number to one-hot vector. 
    X_train = X_train.astype('float32') / 255
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_valid = X_valid.astype('float32') / 255
    X_valid = X_valid.reshape(X_valid.shape[0], 1, 28, 28)
    X_test = X_test.astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    y_train = np_utils.to_categorical(y_train, 10)
    y_valid = np_utils.to_categorical(y_valid, 10)
    
    
    clf = Sequential()
    clf.add(InputLayer(input_shape=(1, 28, 28)))
    clf.add(BatchNormalization())
    
#    clf.add(
#        Conv2D(
#            32, 
#            (5, 5), 
#            padding='same', 
#            bias_initializer=Constant(0.01), 
#            kernel_initializer='random_uniform'
#        )
#    )
#    
#    clf.add(PReLU())
#    clf.add(ZeroPadding2D(padding=(3, 3)))
#    clf.add(MaxPool2D(padding='same'))
#    clf.add(ZeroPadding2D(padding=(2, 2)))
    
    clf.add(
        Conv2D(
            16, 
            (5, 5), 
            padding='same', 
            bias_initializer=Constant(0.01), 
            kernel_initializer='random_uniform'
        )
    )
    
    clf.add(PReLU())
    clf.add(ZeroPadding2D(padding=(2, 2)))
    clf.add(MaxPool2D(padding='same'))
    clf.add(ZeroPadding2D(padding=(1, 1)))  
    
    
    
    
    clf.add(
        Conv2D(
            32, 
            (3, 3), 
            padding='same', 
            bias_initializer=Constant(0.01), 
            kernel_initializer='random_uniform', 
            input_shape=(1, 28, 28)
        )
    )
    
    clf.add(PReLU())
    clf.add(ZeroPadding2D(padding=(1, 1)))
    clf.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
    clf.add(ZeroPadding2D(padding=(1, 1)))
    
    clf.add(Flatten())
    clf.add(Dense(512))
    clf.add(PReLU())
    clf.add(Dropout(0.5))
    clf.add(Dense(256))
    clf.add(PReLU())
    clf.add(Dropout(0.5))
    
    clf.add(Dense(10, activation='softmax'))
    clf.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )
    
    earlystopping = EarlyStopping(monitor='val_acc', patience = patience, verbose=1, mode='max')


    checkpoint = ModelCheckpoint('best_model'+str(iter_time)+'.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_acc',
                                 mode='max')
    
    clf.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_valid, y_valid),
        callbacks=[earlystopping,checkpoint]
    )
    
    # clf.evaluate(X_valid, y_valid)
    
    
    clf.load_weights('best_model'+str(iter_time)+'.hdf5')
    
    true = y_data[valid_index]
    pred = clf.predict_classes(X_valid, batch_size=batch_size)
    print('Valid Acc', get_acc(true, pred))
    
    res = clf.predict_classes(X_test, batch_size=batch_size)
    res = np_utils.to_categorical(res, 10)
    res_box.append(res)






for iter_time in range(nf):
    
    if 'EXT' not in model_taken:
        break
    
    print('EXT', iter_time)
    
    train_index, valid_index = skf[iter_time]
    X_train = X_data[train_index]
    X_valid = X_data[valid_index]
    y_train = y_data[train_index]
    y_valid = y_data[valid_index]
    
    X_test = X_test_data


    # Prepare datasets
    # This step contains normalization and reshaping of input.
    # For output, it is important to change number to one-hot vector. 
    X_train = X_train.astype('float32') / 255
    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_valid = X_valid.astype('float32') / 255
    X_valid = X_valid.reshape(X_valid.shape[0], 28*28)
    X_test = X_test.astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], 28*28)
    # y_train = np_utils.to_categorical(y_train, 10)
    # y_valid = np_utils.to_categorical(y_valid, 10)
    
    
    clf = ExtraTreesClassifier(
                            n_estimators=300,
                            max_depth=30,
                            verbose=1
                            )
    clf.fit(X_train, y_train)
    
    
    true = y_data[valid_index]
    pred = clf.predict(X_valid)
    print('Valid Acc', get_acc(true, pred))    
    
    res = clf.predict(X_test)
    res = np_utils.to_categorical(res, 10)
    res_box.append(res)
    




for iter_time in range(nf):
    
    if 'RF' not in model_taken:
        break
    
    print('RF', iter_time)
    
    train_index, valid_index = skf[iter_time]
    X_train = X_data[train_index]
    X_valid = X_data[valid_index]
    y_train = y_data[train_index]
    y_valid = y_data[valid_index]
    
    X_test = X_test_data


    # Prepare datasets
    # This step contains normalization and reshaping of input.
    # For output, it is important to change number to one-hot vector. 
    X_train = X_train.astype('float32') / 255
    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_valid = X_valid.astype('float32') / 255
    X_valid = X_valid.reshape(X_valid.shape[0], 28*28)
    X_test = X_test.astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], 28*28)
    # y_train = np_utils.to_categorical(y_train, 10)
    # y_valid = np_utils.to_categorical(y_valid, 10)
    
    
    clf = RandomForestClassifier(
                            n_estimators=500, 
                            max_depth=30,
                            verbose=1
                            )
                            
    clf.fit(X_train, y_train)
    
    
    true = y_data[valid_index]
    pred = clf.predict(X_valid)
    print('Valid Acc', get_acc(true, pred))    
    
    res = clf.predict(X_test)
    res = np_utils.to_categorical(res, 10)
    res_box.append(res)


reses = res_box[0]
for res in res_box[1:]:
    reses += res


res = np.argmax(reses, axis=1)

result = [['id','label']]
for i, j in enumerate(res):
  line = []
  line.append(i)
  line.append(j)
  result.append(line)

f = open('prediction.csv', 'w')
w = csv.writer(f)
w.writerows(result)
f.close()