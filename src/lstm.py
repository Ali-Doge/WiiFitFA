import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,10) # Make the figures a bit bigger
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Softmax
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
from random import shuffle

BATCH_SIZE = 4
TIME_STEPS = 100
NUM_FEATURES = 8 + 3 + 3    # 8 EMG channels + 3 Acc Channels + 3 Gyro Channels
NUM_CLASSES = 4


def parse_file(filepath: str):
    return np.loadtxt(filepath, delimiter=',', dtype=np.int32, skiprows=1, usecols=(0,1,2,3,4,5,6,7,12,13,14,15,16,17))

def get_dataset(participants : list, classes : list, num_features : int, batch_size=5, shuffle=False):
    num_data_points = 0
    for participant in participants:
        dir = os.path.join('./data', participant)
        num_data_points += len(os.listdir(dir))
    dataset = np.zeros((num_data_points, TIME_STEPS, num_features))
    ground_truth = np.zeros((num_data_points, len(classes)), dtype=np.int32)
    data_point_index = 0
    
    for participant in participants:
        dir = os.path.join('./data', participant)
        files = os.listdir(dir)
        for file in files:
            for i, cls in enumerate(classes):
                if cls in file:
                    dataset[data_point_index] = parse_file(os.path.join(dir, file))
                    ground_truth[data_point_index][i] = 1
                    data_point_index += 1
                    break

    if shuffle:
        indices = np.arange(num_data_points)
        np.random.shuffle(indices)
        dataset = dataset[indices]
        ground_truth = ground_truth[indices]

    remainder = num_data_points % batch_size
    dataset = dataset[:dataset.shape[0]-remainder]
    ground_truth = ground_truth[:ground_truth.shape[0]-remainder]
    
    return train_test_split(dataset, ground_truth, test_size=0.2)


def create_model():
    lstm_model = Sequential()
    lstm_model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, 
                                                NUM_FEATURES), 
                                                dropout=0.0, 
                                                recurrent_dropout=0.0))
    lstm_model.add(Dense(100,activation='relu'))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(50,activation='relu'))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(15,activation='relu'))
    lstm_model.add(Dense(NUM_CLASSES,activation='linear'))
    lstm_model.add(Softmax())
    lstm_model.compile(loss='mean_squared_error', 
                    optimizer='adam',
                    metrics=['mae'])
    return lstm_model

if __name__ == '__main__':
    participants = ['Tyler', 'Ali', 'Issac']
    classes = ['idle', 'kick', 'pass', 'walk']
    x_train, x_val, y_train, y_val = get_dataset(participants, classes, NUM_FEATURES, batch_size=BATCH_SIZE, shuffle=True)
    print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
    lstm_model = create_model()

    history = lstm_model.fit(x_train, y_train,
          batch_size=BATCH_SIZE, epochs=50, verbose=1,
          validation_data=(x_val, y_val))
    
    score = lstm_model.evaluate(x_train, y_train, verbose=1, batch_size=BATCH_SIZE)
    print('Test score:', score[0])
    print('Test accuracy:', score[1] )
    
    output = lstm_model.predict(x_val, batch_size=BATCH_SIZE)
    print(output.shape)
    predicted = np.argmax(output, axis=1)
    gt = np.argmax(y_val, axis=1)
    print(confusion_matrix(gt, predicted))

    
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

