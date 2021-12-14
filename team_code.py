#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of the required functions, remove non-required functions, and add your own functions.

################################################################################
#
# Imported functions and variables
#
################################################################################

# Import functions. These functions are not required. You can change or remove them.
from helper_code import *
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat
import pandas as pd
from scipy import signal
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import tensorflow as tf
import math

# Define the Challenge lead sets. These variables are not required. You can change or remove them.
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)

other_diag = ["bundle branch block","bradycardia","1st degree av block", "incomplete right bundle branch block", "left axis deviation", "left anterior fascicular block", "left bundle branch block", "low qrs voltages",
        "nonspecific intraventricular conduction disorder", "poor R wave Progression", "prolonged pr interval", "prolonged qt interval", "qwave abnormal", "right axis deviation", "right bundle branch block", "t wave abnormal", "t wave inversion"]


################################################################################
#
# Training model function
#
################################################################################

# Train your model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory) # Find header and recording files.
    num_recordings = len(recording_files) # Number of recordings.

    # If no recordings are found, exit the program.
    if not num_recordings: 
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract the classes from the dataset.
    print('Extracting classes...')

    print(f"Shape of rec files: {len(recording_files)}")

    classes = set() # Set of classes.
    ecg_len = [] # List of ECG lengths.
    for header_file in header_files:
        header = load_header(header_file) # Load the header data from the header file.
        ecg_len.append(int(get_num_samples(header))/int(get_frequency(header))) # Add the ECG length to the list (in seconds).
        classes |= set(get_labels(header)) # Add the labels to the set of classes.
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically if not numbers.
    num_classes_all = len(classes) # Number of classes.
    ecg_len = np.asarray(ecg_len) # Convert to numpy array.

    recording_files = only_ten_sec(ecg_len, np.asarray(recording_files)) # Only use recordings with length equal to 10 seconds.
    num_recordings = len(recording_files) # Update number of recordings.

    print(f"Shape of rec files: {num_recordings}")

    with open('classes.txt', 'w') as f:
        for class_ in classes:
            f.write("%s\n" % class_)
    f.close()

    print('Number of classes (all) = ', num_classes_all)

    SNOMED_scored=pd.read_csv("./dx_mapping_scored.csv", sep=",") # Read in the SNOMED scored data.
    lab_arr = np.asarray(SNOMED_scored['SNOMEDCTCode'], dtype="str") # Convert to array.
    scored_classes = [] # Empty list for scored classes.
    for i in classes:
        for j in lab_arr:
            if i == '':
                continue
            if i == j:
                scored_classes.append(i) # Add the class to the scored classes.
    scored_classes = sorted(scored_classes) # Sort classes alphanumerically if not numbers.
    num_classes = len(scored_classes) # Number of classes.

    print('Number of scored classes = ', num_classes)

    labels = np.zeros((num_recordings, num_classes), dtype=np.bool) # One-hot encoding of classes

    for i in range(len(recording_files)):
        current_labels = get_labels(load_header(recording_files[i].replace('.mat','.hea'))) # Get the labels from the header file.
        for label in current_labels:
            if label in scored_classes:
                j = scored_classes.index(label) # Get the index of the class.
                labels[i, j] = 1 # Set the one-hot encoding to 1.
    labels = labels * 1 # Convert to int.

    abbr = abbreviation(scored_classes) # Abbreviate the classes.

    print(f"Shape of rec files: {len(recording_files)}")
    print(f"Shape of label files: {len(labels)}")

    # Train a model for each lead set.
    for leads in lead_sets:
        print('Training model for {}-lead set: {}...'.format(len(leads), ', '.join(leads)))
        num_leads = len(leads) # Number of leads.
        batchsize = 30 # Set batch size.
        epochs = 15 # Set number of epochs.
        print(f"num labels = {labels.shape[1]}")

        if num_leads == 2:
            lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler_2,verbose=1) # Set learning rate scheduler.
        elif num_leads == 3:
            lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler_3,verbose=1) # Set learning rate scheduler.
        elif num_leads == 4:
            lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler_4,verbose=1) # Set learning rate scheduler.
        elif num_leads == 6:
            lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler_6,verbose=1) # Set learning rate scheduler.
        elif num_leads == 12:
            lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler_12,verbose=1) # Set learning rate scheduler.


        model = inception_model_f1(num_leads,labels.shape[1]) # Create the model.

        # Train the model.
        model.fit(x=batch_generator(batch_size=batchsize, gen_x=generate_X(recording_files, num_leads), 
                                    gen_y=generate_y(labels), num_leads=num_leads, num_classes=labels.shape[1]), 
          epochs=epochs, steps_per_epoch=(labels.shape[0]/batchsize), 
          callbacks=[lr_schedule], verbose=1) 

        # Save model.
        print('Saving model...')
        #model.save("model.h5")
        name = str(num_leads) + "_lead_model.h5"
        model_name = model_directory + "/" + name
        model.save_weights(model_name)



################################################################################
#
# Running trained model function
#
################################################################################

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def run_model(model, header, recording):
    num_leads = model.input_shape[2]

    temp_classes = []
    with open('classes.txt', 'r') as f:
        for line in f: # Read in the classes.
            temp_classes.append(line) # Add the class to the scored classes.
        

    classes=[s.strip('\n') for s in temp_classes] # Remove newline characters from the end of each string.

    probabilities = np.zeros(len(classes)) # Make empty array for probabilities of classes.

    preprocessed_ecg = preprocess_ecg(recording, header, num_leads) # Preprocess the ECG recording.

    probabilities = model.predict(np.expand_dims(preprocessed_ecg,0)) # Predict probabilities for each class.


    threshold = np.ones(len(classes))*0.5 # Threshold for the model.
    binary_prediction = probabilities > threshold # Binary prediction.
    binary_prediction = binary_prediction * 1 # Convert to int.
    binary_prediction = np.asarray(binary_prediction, dtype=np.int).ravel() # Flatten.
    probabilities = np.asarray(probabilities, dtype=np.float32).ravel() # Flatten.
    return classes, binary_prediction, probabilities # Return the classes, binary prediction, and probabilities.

################################################################################
#
# File I/O functions
#
################################################################################

# Load a trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def load_model(model_directory, leads):
    temp_classes = []
    with open('classes.txt', 'r') as f:
        for line in f: # Read in the classes.
            temp_classes.append(line) # Add the class to the scored classes.
    num_classes = len(temp_classes)

    model = inception_model_f1(len(leads),num_classes) # Create the model.
    model_name = str(len(leads)) + "_lead_model.h5"
    #model = tf.keras.models.load_model(os.path.join(model_directory, model_name))
    model.load_weights(os.path.join(model_directory, model_name))
    return model

# Define the filename(s) for the trained models. This function is not required. You can change or remove it.
def get_model_filename(leads):
    sorted_leads = sort_leads(leads)
    return 'model_{}_leads.h5'.format(len(sorted_leads))

################################################################################
#
# Feature extraction function
#
################################################################################

# Extract features from the header and recording. This function is not required. You can change or remove it.

#-----------------------------------------------------------#
#                                                           #
#                    My functions                           #
#                                                           #
#-----------------------------------------------------------#

def abbreviation(snomed_classes):
    SNOMED_scored = pd.read_csv("./dx_mapping_scored.csv", sep=",")
    snomed_abbr = []
    for j in range(len(snomed_classes)):
        for i in range(len(SNOMED_scored.iloc[:,1])):
            if (str(SNOMED_scored.iloc[:,1][i]) == snomed_classes[j]):
                snomed_abbr.append(SNOMED_scored.iloc[:,0][i])
                
    snomed_abbr = np.asarray(snomed_abbr)
    return snomed_abbr

def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data



def finn_diagnoser(labels, abbr, navn):
    arr = np.zeros((labels.shape[0],1))
    idx = np.where(labels[:,np.where(abbr == navn)[0]] == 1)[0]
    arr[idx] = 1
    return arr


def preprocess_ecg(data, header, num_leads):
    samp_freq = 75
    time = 10
    if num_leads == 12:
        data = data[[0,1,2,3,4,5,6,7,8,9,10,11]]
    elif num_leads == 6:
        data = data[[0,1,2,3,4,5]]
    elif num_leads == 4:
        data = data[[0,1,2,7]]
    elif num_leads == 3:
        data = data[[0,1,7]]
    elif num_leads == 2:
        data = data[[0,1]]

    if int(header.split(" ")[2]) != samp_freq:
        data_new = np.ones([num_leads,int((int(header.split(" ")[3])/int(header.split(" ")[2]))*samp_freq)])
        for i,j in enumerate(data):
            data_new[i] = signal.resample(j, int((int(header.split(" ")[3])/int(header.split(" ")[2]))*samp_freq))
        data = data_new
    data = pad_sequences(data, maxlen=samp_freq*time, truncating='post',padding="post")
        
    data = np.moveaxis(data, 0, -1)
    return data



def batch_generator(batch_size, gen_x, gen_y, num_leads, num_classes): 
    samp_freq = 75
    time = 10
    batch_features = np.zeros((batch_size,samp_freq*time, num_leads))
    batch_labels = np.zeros((batch_size,num_classes))
    while True:
        for i in range(batch_size):

            batch_features[i] = next(gen_x)
            batch_labels[i] = next(gen_y)
            
        yield batch_features, batch_labels

def generate_y(y_train):
    while True:
        for i in y_train:
            y_gen = i
            yield y_gen


def generate_X(X_train_file, num_leads):
    samp_freq = 75
    time = 10
    while True:
        for i in range(len(X_train_file)):
            data, header_data = load_challenge_data(X_train_file[i])
            if num_leads == 12:
                ecg = data[[0,1,2,3,4,5,6,7,8,9,10,11]]
            elif num_leads == 6:
                ecg = data[[0,1,2,3,4,5]]
            elif num_leads == 4:
                ecg = data[[0,1,2,7]]
            elif num_leads == 3:
                ecg = data[[0,1,7]]
            elif num_leads == 2:
                ecg = data[[0,1]]

            if int(header_data[0].split(" ")[2]) != samp_freq:                        # sampling frequency
                data_new = np.ones([num_leads,                                    # leads
                                  int((int(header_data[0].split(" ")[3])      # samples
                                  /int(header_data[0].split(" ")[2]))*samp_freq)])  # sampling frequency ratio
                for i,j in enumerate(ecg):
                    data_new[i] = signal.resample(j, int((int(header_data[0].split(" ")[3])/int(header_data[0].split(" ")[2]))*samp_freq))
                ecg = data_new

            ecg = pad_sequences(ecg, maxlen=samp_freq*time, truncating='post',padding="post")
            #ecg = ecg.reshape(samp_freq*time,num_leads)
            noise = np.random.randint(0,1)
            baseline_w = np.random.randint(0,1)
            if noise == 1:
                ecg = ecg + np.random.rand(samp_freq*time)* random.uniform(50,100) 
            if baseline_w == 1:
                ecg = ecg + np.cos(np.linspace(0, 2*math.pi, samp_freq*time)+random.uniform(0,2*math.pi))*random.uniform(0,300)
            ecg = np.moveaxis(ecg, 0, -1)
            yield ecg


#https://towardsdatascience.com/multi-label-image-classification-in-tensorflow-2-0-7d4cf8a4bc72
def macro_double_soft_f1(y_true, y_pred):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y_true, tf.float32)
    y_hat = tf.cast(y_pred, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
    soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost


def inception_block(prev_layer):
    
    conv1=tf.keras.layers.Conv1D(filters = 64, kernel_size = 1, padding = 'same')(prev_layer)
    conv1=tf.keras.layers.BatchNormalization()(conv1)
    conv1=tf.keras.layers.Activation('relu')(conv1)
    
    conv3=tf.keras.layers.Conv1D(filters = 64, kernel_size = 1, padding = 'same')(prev_layer)
    conv3=tf.keras.layers.BatchNormalization()(conv3)
    conv3=tf.keras.layers.Activation('relu')(conv3)
    conv3=tf.keras.layers.Conv1D(filters = 64, kernel_size = 3, padding = 'same')(conv3)
    conv3=tf.keras.layers.BatchNormalization()(conv3)
    conv3=tf.keras.layers.Activation('relu')(conv3)
    
    conv5=tf.keras.layers.Conv1D(filters = 64, kernel_size = 1, padding = 'same')(prev_layer)
    conv5=tf.keras.layers.BatchNormalization()(conv5)
    conv5=tf.keras.layers.Activation('relu')(conv5)
    conv5=tf.keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'same')(conv5)
    conv5=tf.keras.layers.BatchNormalization()(conv5)
    conv5=tf.keras.layers.Activation('relu')(conv5)
    
    pool= tf.keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(prev_layer)
    convmax=tf.keras.layers.Conv1D(filters = 64, kernel_size = 1, padding = 'same')(pool)
    convmax=tf.keras.layers.BatchNormalization()(convmax)
    convmax=tf.keras.layers.Activation('relu')(convmax)
    
    layer_out = tf.keras.layers.concatenate([conv1, conv3, conv5, convmax], axis=1)
    
    return layer_out

def inception_model_f1(num_leads, num_labels):
    X_input=tf.keras.layers.Input(shape=(750,num_leads)) 
    
    X = tf.keras.layers.ZeroPadding1D(3)(X_input)
    
    X = tf.keras.layers.Conv1D(filters = 64, kernel_size = 7, padding = 'same')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')(X)
    
    X = tf.keras.layers.Conv1D(filters = 64, kernel_size = 1, padding = 'same')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = inception_block(X)
    X = inception_block(X)
    
    X = tf.keras.layers.MaxPool1D(pool_size=7, strides=2, padding='same')(X)
    
    X = tf.keras.layers.GlobalAveragePooling1D()(X)
    
    
    output_layer = tf.keras.layers.Dense(units=num_labels,activation='sigmoid')(X)

    model = tf.keras.Model(inputs=X_input, outputs=output_layer)

    model.compile(loss=macro_double_soft_f1, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.BinaryAccuracy(
    name='accuracy', dtype=None, threshold=0.5)])
    return model

def scheduler_2(epoch, lr):
    if epoch == 3:
        return lr * 0.1
    elif epoch == 6:
        return lr * 0.1
    elif epoch == 8:
        return lr * 0.1
    elif epoch == 10:
        return lr * 0.1
    elif epoch == 12:
        return lr * 0.1
    elif epoch == 14:
        return lr * 0.1
    else:
        return lr

def scheduler_3(epoch, lr):
    if epoch == 9:
        return lr * 0.1
    elif epoch == 14:
        return lr * 0.1
    else:
        return lr

def scheduler_4(epoch, lr):
    if epoch == 6:
        return lr * 0.1
    elif epoch == 12:
        return lr * 0.1
    elif epoch == 14:
        return lr * 0.1
    else:
        return lr

def scheduler_6(epoch, lr):
    if epoch == 5:
        return lr * 0.1
    elif epoch == 8:
        return lr * 0.1
    elif epoch == 10:
        return lr * 0.1
    elif epoch == 12:
        return lr * 0.1
    elif epoch == 14:
        return lr * 0.1
    else:
        return lr

def scheduler_12(epoch, lr):
    if epoch == 5:
        return lr * 0.1
    elif epoch == 9:
        return lr * 0.1
    elif epoch == 11:
        return lr * 0.1
    elif epoch == 14:
        return lr * 0.1
    else:
        return lr

def only_ten_sec(ecg_len, filename):
    idx = np.where(ecg_len == 10)[0]
    filename = filename[idx]
    return filename