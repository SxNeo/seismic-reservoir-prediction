import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
import matplotlib.pyplot as plt
from tag_multi_AdaBoost_CNN import AdaBoostClassifier as Ada_CNN
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split

# Set experiment parameters
sample_num = 119
channel_num = 300
algorithm = 'SAMME.R'  # 'SAMME.R' or 'SAMME'
input_path = r"C:\01 CodeofPython\AdaBoost_CNN\AdaBoost_CNN-master\code"
output_path = rf"C:\01 CodeofPython\AdaBoost_CNN\AdaBoost_CNN-master\code\sample length"
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})

# Set random seed
seed = 50
np.random.seed(seed)
tf.random.set_seed(seed)

def load_images(image_folder):
    images = []
    bmp_files = [f for f in os.listdir(image_folder) if f.endswith('.bmp')]
    sorted_files = sorted(bmp_files)

    for filename in sorted_files:
        print(f"Reading file: {filename}")
        image_path = os.path.join(image_folder, filename)
        img = load_img(image_path)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
    return np.array(images)

def load_labels(labels_file):
    labels_df = pd.read_excel(labels_file, header=None)
    labels = labels_df.iloc[:, 0].values
    return labels

class PrintLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = K.get_value(self.model.optimizer.lr)
        print(f"\nEpoch {epoch + 1}: Current learning rate = {lr}")

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

def train_adaboost_cnn(X_train, y_train, X_test, y_test, factors):
    img_height, img_width = X_train.shape[1], X_train.shape[2]
    img_channels = 3
    n_classes = y_train.shape[1] if y_train.ndim > 1 else len(np.unique(y_train))

    encoder = OneHotEncoder(sparse=False)
    encoder.fit(y_train.reshape(-1, 1))
    y_train_encoded = encoder.transform(y_train.reshape(-1, 1))

    CNN_f1_train = []
    CNN_f1_test = []
    Ada_CNN_f1_train = []
    Ada_CNN_f1_test = []

    model_False, callbacks_False = baseline_model_False(img_height, img_width, img_channels, n_classes)

    model_False.fit(X_train, y_train_encoded, epochs=10, batch_size=32, callbacks=callbacks_False)

    CNN_pred_train = model_False.predict(X_train)
    CNN_pred_test = model_False.predict(X_test)

    CNN_train_pred_decoded = np.argmax(CNN_pred_train, axis=1) if CNN_pred_train.ndim > 1 else CNN_pred_train
    CNN_test_pred_decoded = np.argmax(CNN_pred_test, axis=1) if CNN_pred_test.ndim > 1 else CNN_pred_test

    CNN_f1_train.append(f1_score(y_train, CNN_train_pred_decoded, average='macro'))
    CNN_f1_test.append(f1_score(y_test, CNN_test_pred_decoded, average='macro'))

    model, callbacks = baseline_model(img_height, img_width, img_channels, n_classes)

    Ada_CNN_model = Ada_CNN(base_estimator=model, n_estimators=10, learning_rate=0.01,
                            epochs=10, batch_size=32, algorithm=algorithm, callbacks=callbacks)
    Ada_CNN_model.fit(X_train, y_train)

    Ada_CNN_pred_train = Ada_CNN_model.predict(X_train)
    Ada_CNN_pred_test = Ada_CNN_model.predict(X_test)

    Ada_CNN_f1_train.append(f1_score(y_train, Ada_CNN_pred_train, average='macro'))
    Ada_CNN_f1_test.append(f1_score(y_test, Ada_CNN_pred_test, average='macro'))

    return CNN_f1_train, CNN_f1_test, Ada_CNN_f1_train, Ada_CNN_f1_test

def save_all_results_to_txt(all_results, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_path = os.path.join(output_path, 'all_results.txt')
    with open(file_path, 'w') as file:
        for way, results in all_results.items():
            file.write(f"Method: {way}\n")
            for key, value in results.items():
                file.write(f"  {key}:\n")
                for metric, scores in value.items():
                    scores_str = ', '.join([f"{score:.2f}" for score in scores])
                    file.write(f"    {metric}: {scores_str}\n")
            file.write("\n")
    print(f"All results have been saved to {file_path}")

all_results = {}
ways = ['CWT', 'ISD']
factors = [10]
sample_lengths = [5, 10, 15, 20, 25]

for way in ways:
    results = {}
    labels_file1 = rf'{input_path}\WELL1_labels_{sample_num}_time_new3_3class.xlsx'
    labels_file2 = rf'{input_path}\WELl2_labels_{sample_num}_time_new3_3class.xlsx'

    for sample_length in sample_lengths:
        image_folder_63 = rf'{input_path}\training and testing set\{way}\{way}_063_{sample_length}'
        image_folder_64 = rf'{input_path}\training and testing set\{way}\{way}_064_{sample_length}'
        image_folder_65 = rf'{input_path}\training and testing set\{way}\{way}_065_{sample_length}'
        image_folder_240 = rf'{input_path}\training and testing set\{way}\{way}_240_{sample_length}'
        image_folder_241 = rf'{input_path}\training and testing set\{way}\{way}_241_{sample_length}'
        image_folder_242 = rf'{input_path}\training and testing set\{way}\{way}_242_{sample_length}'

        X_63 = load_images(image_folder_63)
        X_64 = load_images(image_folder_64)
        X_65 = load_images(image_folder_65)
        X_240 = load_images(image_folder_240)
        X_241 = load_images(image_folder_241)
        X_242 = load_images(image_folder_242)
        y1 = load_labels(labels_file1)
        y2 = load_labels(labels_file2)

        X_train = np.concatenate((X_64, X_65, X_241, X_242), axis=0)
        y_train = np.concatenate((y1, y1, y2, y2), axis=0)
        X_test = np.concatenate((X_63, X_240), axis=0)
        y_test = np.concatenate((y1, y2), axis=0)

        CNN_training, CNN_testing, Ada_CNN_training, Ada_CNN_testing = train_adaboost_cnn(X_train, y_train, X_test, y_test, factors)

        key = f"Sample_Length_{sample_length}"
        results[key] = {
            'CNN_training': CNN_training,
            'CNN_testing': CNN_testing,
            'Ada_CNN_training': Ada_CNN_training,
            'Ada_CNN_testing': Ada_CNN_testing,
        }

    all_results[way] = results

save_all_results_to_txt(all_results, output_path)
