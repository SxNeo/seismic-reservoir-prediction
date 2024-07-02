import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score  # Import accuracy_score to evaluate the model
from sklearn.metrics import f1_score, roc_curve, auc
import tensorflow
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import cycle
from multi_AdaBoost_CNN import AdaBoostClassifier as Ada_CNN
import sys
from sklearn.model_selection import StratifiedKFold

# Define experiment number variables
sample_num = 119
channel_num = 300
algorithm = 'SAMME.R'  # 'SAMME.R' or 'SAMME'
input_path = r"C:\01 CodeofPython\AdaBoost_CNN\AdaBoost_CNN-master\code"
output_path = rf"C:\01 CodeofPython\AdaBoost_CNN\AdaBoost_CNN-master\code\sample length"
# Set global font size and font family
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})

# Set random seed for reproducibility
seed = 50
np.random.seed(seed)
tensorflow.random.set_seed(seed)

# Load image data
def load_images(image_folder):
    images = []
    # Get all filenames ending with .bmp
    bmp_files = [f for f in os.listdir(image_folder) if f.endswith('.bmp')]
    # Sort filenames alphabetically
    sorted_files = sorted(bmp_files)

    for filename in sorted_files:
        print(f"Reading file: {filename}")  # Print filename
        image_path = os.path.join(image_folder, filename)
        img = load_img(image_path)  # Load color image, default is color mode
        img_array = img_to_array(img) / 255.0  # Normalize
        images.append(img_array)
    return np.array(images)

# Load label data
def load_labels(labels_file):
    # Read data with header=None to avoid skipping the first row
    labels_df = pd.read_excel(labels_file, header=None)
    # Assuming labels are in the first column, use iloc[:, 0] to get all rows of the first column
    labels = labels_df.iloc[:, 0].values
    return labels

# Define a function to build the baseline CNN model
def baseline_model(img_height, img_width, img_channels, n_classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="same", input_shape=(img_height, img_width, img_channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    adam_optimizer = Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.25, patience=4, min_lr=0.0001)
    early_stopping = EarlyStopping(monitor='accuracy', patience=4, verbose=1)

    return model, [reduce_lr, early_stopping]

def baseline_model_False(img_height, img_width, img_channels, n_classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="same", input_shape=(img_height, img_width, img_channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    adam_optimizer = Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.25, patience=4, min_lr=0.0001)

    return model, reduce_lr

# Define function to train and evaluate AdaBoost + CNN model
def train_adaboost_cnn_with_cross_validation(X, y, factors):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    CNN_all_avg_f1_train = []
    CNN_all_avg_f1_test = []

    Ada_CNN_all_avg_f1_train = []
    Ada_CNN_all_avg_f1_test = []

    for factor in factors:
        print(f"Starting factor: {factor}")
        CNN_all_f1_train = []
        CNN_all_f1_test = []

        Ada_CNN_all_f1_train = []
        Ada_CNN_all_f1_test = []

        # Iterate through each fold
        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(f"Starting fold {fold + 1}")
            X_train, X_test = X[train_index], X[test_index]  # Ensure each fold's test and train sets have the same class proportion
            y_train, y_test = y[train_index], y[test_index]  # Ensure each fold's test and train sets have the same class proportion

            img_height, img_width = X_train.shape[1], X_train.shape[2]
            img_channels = 3  # Images are colored
            n_classes = y_train.shape[1] if y_train.ndim > 1 else len(np.unique(y_train))

            # One-hot encode the combined labels
            encoder = OneHotEncoder(sparse=False)
            encoder.fit(y.reshape(-1, 1))  # Fit using combined labels
            y_train_encoded = encoder.transform(y_train.reshape(-1, 1))
            # y_val_encoded = encoder.transform(y_val.reshape(-1, 1))
            y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

            model, callbacks = baseline_model(img_height, img_width, img_channels, n_classes)

            model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, callbacks=callbacks)

            CNN_pred_train = model.predict(X_train)
            CNN_pred_test = model.predict(X_test)

            # Decode prediction results to original labels
            CNN_train_pred_decoded = np.argmax(CNN_pred_train, axis=1) if CNN_pred_train.ndim > 1 else CNN_pred_train
            CNN_test_pred_decoded = np.argmax(CNN_pred_test, axis=1) if CNN_pred_test.ndim > 1 else CNN_pred_test

            CNN_f1_train = f1_score(y_train, CNN_train_pred_decoded, average='macro')
            CNN_f1_test = f1_score(y_test, CNN_test_pred_decoded, average='macro')

            CNN_all_f1_train.append(CNN_f1_train)
            CNN_all_f1_test.append(CNN_f1_test)

            Ada_CNN_model = Ada_CNN(base_estimator=model, n_estimators=10, learning_rate=0.01,
                                    epochs=10, batch_size=32, algorithm=algorithm, callbacks=callbacks)
            Ada_CNN_model.fit(X_train, y_train)

            Ada_CNN_pred_train = Ada_CNN_model.predict(X_train)
            Ada_CNN_pred_test = Ada_CNN_model.predict(X_test)

            Ada_CNN_f1_train = f1_score(y_train, Ada_CNN_pred_train, average='macro')
            Ada_CNN_f1_test = f1_score(y_test, Ada_CNN_pred_test, average='macro')

            Ada_CNN_all_f1_train.append(Ada_CNN_f1_train)
            Ada_CNN_all_f1_test.append(Ada_CNN_f1_test)

        CNN_all_avg_f1_train.append(np.mean(CNN_all_f1_train))
        CNN_all_avg_f1_test.append(np.mean(CNN_all_f1_test))

        Ada_CNN_all_avg_f1_train.append(np.mean(Ada_CNN_all_f1_train))
        Ada_CNN_all_avg_f1_test.append(np.mean(Ada_CNN_all_f1_test))

    return CNN_all_avg_f1_train, CNN_all_avg_f1_test, Ada_CNN_all_avg_f1_train, Ada_CNN_all_avg_f1_test

def save_all_results_to_txt(all_results, output_path):
    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define file path
    file_path = os.path.join(output_path, 'all_results.txt')

    # Open file and write
    with open(file_path, 'w') as file:
        for way, results in all_results.items():
            file.write(f"Method: {way}\n")
            for key, value in results.items():
                file.write(f"  {key}:\n")
                for metric, scores in value.items():
                    # Assume scores is a list, need to convert to string
                    scores_str = ', '.join([f"{score:.2f}" for score in scores])
                    file.write(f"    {metric}: {scores_str}\n")
            file.write("\n")  # Add a blank line after each method to increase readability

    print(f"All results have been saved to {file_path}")

# Usage example
all_results = {}
name1 = '3'
# ways = ['STFT', 'CWT', 'ISD']
ways = ['CWT', 'ISD']
factors = [10]
sample_lengths = [5,10,15,20,25,30]  # [5,10,15,20,25,30] [3,7,11,15,19,23,27]

# Main program
for way in ways:
    results = {}  # To store F1 scores for different sample lengths
    labels_file1 = rf'{input_path}\WELL1_labels_{sample_num}_time_3class_new4.xlsx'
    labels_file2 = rf'{input_path}\WELl2_labels_{sample_num}_time_3class_new4.xlsx'

    for sample_length in sample_lengths:
        image_folder_63 = rf'{input_path}\training and testing set\{way}\{way}_063_{sample_length}'
        image_folder_64 = rf'{input_path}\training and testing set\{way}\{way}_064_{sample_length}'
        image_folder_65 = rf'{input_path}\training and testing set\{way}\{way}_065_{sample_length}'
        image_folder_240 = rf'{input_path}\training and testing set\{way}\{way}_240_{sample_length}'
        image_folder_241 = rf'{input_path}\training and testing set\{way}\{way}_241_{sample_length}'
        image_folder_242 = rf'{input_path}\training and testing set\{way}\{way}_242_{sample_length}'

        # Load image and label data
        X_63 = load_images(image_folder_63)
        X_64 = load_images(image_folder_64)
        X_65 = load_images(image_folder_65)
        X_240 = load_images(image_folder_240)
        X_241 = load_images(image_folder_241)
        X_242 = load_images(image_folder_242)
        y1 = load_labels(labels_file1)
        y2 = load_labels(labels_file2)

        # Combine image datasets and label datasets
        X = np.concatenate((X_64,X_65,X_241, X_242), axis=0)
        y = np.concatenate((y1, y1, y2,y2), axis=0)

        # Call AdaBoost + CNN training and evaluation function
        CNN_training, CNN_validation, Ada_CNN_training, Ada_CNN_validation = train_adaboost_cnn_with_cross_validation(
            X, y, factors)

        # Save results
        key = f"Sample_Length_{sample_length}"
        results[key] = {
            'CNN_training': CNN_training,
            'CNN_validation': CNN_validation,
            'Ada_CNN_training': Ada_CNN_training,
            'Ada_CNN_validation': Ada_CNN_validation,
        }

    all_results[way] = results

# Call function to save data to text file
save_all_results_to_txt(all_results, output_path)
