# Evaluate the model on unseen data
# Author: Tyler Talarico

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,10) # Make the figures a bit bigger
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Softmax
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
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
    
    return dataset, ground_truth



if __name__ == '__main__':
    participants = ['Geoffrey']
    classes = ['idle', 'kick', 'pass', 'walk']
    x_test, y_test = get_dataset(participants, classes, NUM_FEATURES, batch_size=1, shuffle=False)
	
    # load the model
    path = os.path.join('./src', 'lstm_model.keras')
    lstm_model = tf.keras.models.load_model(path)

    acc = 0
    for x,y in zip(x_test, y_test):
        x = x.reshape(1, 100, 14)
        y = np.argmax(y)
        pred = np.argmax(lstm_model.predict(x, verbose=0))
        if y == pred:
            acc += 1
    print(y_test.shape)
    acc = acc / len(y_test)
    print('Test Accuracy: ', acc)

