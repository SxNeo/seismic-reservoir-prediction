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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_curve, auc
import tensorflow
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import sys
import seaborn as sns
import matplotlib.patches as patches
import time
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

# Define experiment variables
sample_num = 119
channel_num = 300
way = 'ISD'  # CWT ISD
algorithm = 'SAMME.R'  # 'SAMME.R' or 'SAMME'
input_path = r"C:\01 CodeofPython\AdaBoost_CNN\AdaBoost_CNN-master\code"
output_path = rf"C:\01 CodeofPython\AdaBoost_CNN\AdaBoost_CNN-master\code\{way}"
plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})

# Set random seed to ensure result reproducibility
seed = 50
np.random.seed(seed)
tensorflow.random.set_seed(seed)

# Load image data
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

# Modified load_images function with batch processing
def load_images_in_batches(image_folder, batch_size):
    bmp_files = [f for f in os.listdir(image_folder) if f.endswith('.bmp')]
    sorted_files = sorted(bmp_files)
    for i in range(0, len(sorted_files), batch_size):
        batch_files = sorted_files[i:i + batch_size]
        images = []
        for filename in batch_files:
            print(f"Reading file: {filename}")
            image_path = os.path.join(image_folder, filename)
            img = load_img(image_path)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
        yield np.array(images), batch_files

# Read label data
def load_labels(labels_file):
    labels_df = pd.read_excel(labels_file, header=None)
    labels = labels_df.iloc[:, 0].values
    return labels

class PrintLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = K.get_value(self.model.optimizer.lr)
        print(f"\nEpoch {epoch + 1}: Current learning rate = {lr}")

# Define function to build the baseline CNN model
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

def plot_and_save_confusion_matrix(y_true, y_pred, classes, output_path, file_name="confusion_matrix.tif"):
    cm = confusion_matrix(y_true, y_pred)
    class_labels = [classes[i] for i in sorted(classes)]

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(os.path.join(output_path, file_name), bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()

def save_operating_data(way):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_path = os.path.join(output_path, f"{way['algorithm']}_Operating_data.txt")
    content = f"""
    {way['algorithm']}_train_accuracy_: {way['acc_train']}
    {way['algorithm']}_test_accuracy_: {way['acc_test']}
    {way['algorithm']}_training_Precision: {way['pre_train']}
    {way['algorithm']}_test_Precision: {way['pre_test']}
    {way['algorithm']}_training_Recall: {way['rec_train']}
    {way['algorithm']}_test_Recall: {way['rec_test']}
    {way['algorithm']}_training_F1-score: {way['f1_train']}
    {way['algorithm']}_test_F1-score: {way['f1_test']}
    """

    with open(file_path, "w") as file:
        file.write(content)

    print(f"Operating data saved to: {file_path}")

def model_test(model, X_train, y_train, X_test, y_test, classes, algorithm):
    if algorithm == 'CNN':
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        y_train_pred = np.argmax(y_train_pred, axis=1)
        y_test_pred = np.argmax(y_test_pred, axis=1)

        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)
    else:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_train = precision_score(y_train, y_train_pred, average='macro')
    precision_test = precision_score(y_test, y_test_pred, average='macro')
    recall_train = recall_score(y_train, y_train_pred, average='macro')
    recall_test = recall_score(y_test, y_test_pred, average='macro')
    f1_score_train = f1_score(y_train, y_train_pred, average='macro')
    f1_score_test = f1_score(y_test, y_test_pred, average='macro')

    plot_and_save_confusion_matrix(y_train, y_train_pred, classes, output_path,
                                   rf"{algorithm}_confusion_matrix_train.png")
    plot_and_save_confusion_matrix(y_test, y_test_pred, classes, output_path,
                                   rf"{algorithm}_confusion_matrix_test.png")

    way = {
        'algorithm': algorithm,
        'acc_train': accuracy_train,
        'acc_test': accuracy_test,
        'pre_train': precision_train,
        'pre_test': precision_test,
        'rec_train': recall_train,
        'rec_test': recall_test,
        'f1_train': f1_score_train,
        'f1_test': f1_score_test,
    }

    save_operating_data(way)

def plot_and_save_lithology(labels, output_path, colors, file_name="lithology.png"):
    lithology_image = np.zeros((len(labels), 1, 3))

    for i, label in enumerate(labels):
        if isinstance(label, np.ndarray):
            label = label[0]
        lithology_image[i, 0, :] = mcolors.to_rgb(colors[int(label)])

    fig, ax = plt.subplots(figsize=(2, 10))
    ax.imshow(lithology_image, aspect='auto')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, file_name), bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()

def plot_and_save_probability_lithology(probabilities, output_path, colors, file_name="probability_distribution.png"):
    fig, ax = plt.subplots(figsize=(2, 10))

    for index, prob in enumerate(probabilities):
        start = 0
        for class_index, class_prob in enumerate(prob):
            ax.add_patch(
                patches.Rectangle(
                    (start, index),
                    class_prob,
                    1,
                    facecolor=colors[class_index]
                )
            )
            start += class_prob

    ax.set_ylim(0, len(probabilities))
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, file_name), bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()

def calculate_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    percentages = (counts / len(y)) * 100
    return unique, counts, percentages

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{p:.1f}%  ({v:d})'.format(p=pct, v=val)
    return my_format

def plot_and_save_distribution(unique, counts, percentages, classes, colors, title, output_path):
    pie_colors = [colors[i] for i in unique]
    labels = [classes[i] for i in unique]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=140, colors=pie_colors)
    plt.title(f'{title} Class Distribution')

    plt.subplot(1, 2, 2)
    plt.bar(labels, counts, color=pie_colors)
    plt.title(f'{title} Class Distribution')
    plt.ylabel('Counts')
    for i, count in enumerate(counts):
        plt.text(i, count + 0.2, str(count), ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{title}_Distribution.png'), bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()

sample_length = '20'

image_folder_63 = rf'{input_path}\training and testing set\{way}\{way}_063_{sample_length}'
image_folder_64 = rf'{input_path}\training and testing set\{way}\{way}_064_{sample_length}'
image_folder_65 = rf'{input_path}\training and testing set\{way}\{way}_065_{sample_length}'
image_folder_240 = rf'{input_path}\training and testing set\{way}\{way}_240_{sample_length}'
image_folder_241 = rf'{input_path}\training and testing set\{way}\{way}_241_{sample_length}'
image_folder_242 = rf'{input_path}\training and testing set\{way}\{way}_242_{sample_length}'
labels_file1 = rf'{input_path}\WELL1_labels_{sample_num}_time_new3_3class.xlsx'
labels_file2 = rf'{input_path}\WELl2_labels_{sample_num}_time_new3_3class.xlsx'
pred_file = rf'{input_path}\predicting set\{way}_block_{sample_length}'

X1 = load_images(image_folder_63)
X2 = load_images(image_folder_64)
X3 = load_images(image_folder_65)
X4 = load_images(image_folder_240)
X5 = load_images(image_folder_241)
X6 = load_images(image_folder_242)
y1 = load_labels(labels_file1)
y2 = load_labels(labels_file2)

X = np.concatenate((X2, X3, X5, X6), axis=0)
y = np.concatenate((y1, y1, y2, y2), axis=0)
X_test1 = X1
X_test2 = X4
X_train = X
y_train = y
X_test = np.concatenate((X1, X4), axis=0)
y_test = np.concatenate((y1, y2), axis=0)

n_classes = len(np.unique(y1))
classes = {
    0: '''Gas
sandstone''',
    1: '''Non-gas
sandstone''',
    2: '''Mudstone'''
}
colors = {0: 'red', 1: 'yellow', 2: 'gray'}

unique1, counts1, percentages1 = calculate_distribution(y1)
unique2, counts2, percentages2 = calculate_distribution(y2)
unique3, counts3, percentages3 = calculate_distribution(y)

plot_and_save_distribution(unique1, counts1, percentages1, classes, colors, 'Well 1', output_path)
plot_and_save_distribution(unique2, counts2, percentages2, classes, colors, 'Well 2', output_path)
plot_and_save_distribution(unique3, counts3, percentages3, classes, colors, 'ALL', output_path)

plot_and_save_lithology(y1, output_path, colors, file_name="WELL1_real_lithology.png")
plot_and_save_lithology(y2, output_path, colors, file_name="WELL2_real_lithology.png")

encoder = OneHotEncoder(sparse=False)
encoder.fit(y.reshape(-1, 1))
y_train_encoded = encoder.transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

batch_size = 32
n_estimators = 20
epochs = 10
img_height, img_width = X.shape[1], X.shape[2]
img_channels = 3
learning_rate = 0.01

from tag_multi_AdaBoost_CNN import AdaBoostClassifier as Ada_CNN

model, callbacks = baseline_model(img_height, img_width, img_channels, n_classes)
learning_rate_printer = PrintLearningRate()
callbacks.append(learning_rate_printer)

model.fit(X_train, y_train_encoded, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

Ada_CNN_model = Ada_CNN(base_estimator=model, n_estimators=n_estimators, learning_rate=learning_rate,
                        epochs=epochs, batch_size=batch_size, algorithm=algorithm, callbacks=callbacks)

model_False, callbacks_False = baseline_model_False(img_height, img_width, img_channels, n_classes)
learning_rate_printer = PrintLearningRate()

Ada_CNN_model_False = Ada_CNN(base_estimator=model_False, n_estimators=n_estimators, learning_rate=learning_rate,
                              epochs=epochs, batch_size=batch_size, algorithm=algorithm, callbacks=callbacks_False, copy_previous_estimator=False)

print(f"Training with copy_previous_estimator=False")
start_time_False = time.time()
Ada_CNN_model_False.fit(X_train, y_train)
end_time_False = time.time()
print(f"Training with copy_previous_estimator=True")
start_time_True = time.time()
Ada_CNN_model.fit(X_train, y_train)
end_time_True = time.time()
print(f"Training with copy_previous_estimator=False took {end_time_False - start_time_False:.2f} seconds")
print(f"Training with copy_previous_estimator=True took {end_time_True - start_time_True:.2f} seconds")

model_test(model, X_train, y_train_encoded, X_test, y_test_encoded, classes, algorithm='CNN')
model_test(Ada_CNN_model_False, X_train, y_train, X_test, y_test, classes, algorithm='Ada_CNN_False')
model_test(Ada_CNN_model, X_train, y_train, X_test, y_test, classes, algorithm='Ada_CNN')

CNN_y_proba1 = model.predict(X_test1)
CNN_y_proba2 = model.predict(X_test2)
Ada_CNN_y_pred1 = Ada_CNN_model.predict(X_test1)
Ada_CNN_y_pred2 = Ada_CNN_model.predict(X_test2)

plot_and_save_lithology(np.argmax(CNN_y_proba1, axis=1), output_path, colors, file_name="CNN_WELL1_pred_lithology.png")
plot_and_save_lithology(np.argmax(CNN_y_proba2, axis=1), output_path, colors, file_name="CNN_WELL2_pred_lithology.png")
plot_and_save_lithology(Ada_CNN_y_pred1, output_path, colors, file_name="Ada_CNN_WELL1_pred_lithology.png")
plot_and_save_lithology(Ada_CNN_y_pred2, output_path, colors, file_name="Ada_CNN_WELL2_pred_lithology.png")

if algorithm == 'SAMME.R':
    Ada_CNN_y_proba1 = Ada_CNN_model.predict_proba(X_test1)
    Ada_CNN_y_proba2 = Ada_CNN_model.predict_proba(X_test2)

    plot_and_save_probability_lithology(CNN_y_proba1, output_path, colors, file_name="CNN_WELL1_pred_proba_lithology.png")
    plot_and_save_probability_lithology(CNN_y_proba2, output_path, colors, file_name="CNN_WELL2_pred_proba_lithology.png")
    plot_and_save_probability_lithology(Ada_CNN_y_proba1, output_path, colors, file_name="Ada_CNN_WELL1_pred_proba_lithology.png")
    plot_and_save_probability_lithology(Ada_CNN_y_proba2, output_path, colors, file_name="Ada_CNN_WELL2_pred_proba_lithology.png")

batch_size = sample_num
print(f"Predicting profile")
# Ada_CNN_model_all = Ada_CNN(base_estimator=model, n_estimators=n_estimators, learning_rate=learning_rate,
#                             epochs=epochs, batch_size=batch_size, algorithm=algorithm, callbacks=callbacks)
# Ada_CNN_model_all.fit(X_train, y_train)
# Ada_CNN_model_all = Ada_CNN_model

if algorithm == 'SAMME':
    all_CNN_predictions = []

    for i, (X_batch, batch_files) in enumerate(load_images_in_batches(pred_file, batch_size)):
        print(f"Processing batch {i + 1}/{len(range(0, len(os.listdir(pred_file)), batch_size))}...")
        y_pred_prob_batch = model.predict(X_batch)
        y_pred_batch = np.argmax(y_pred_prob_batch, axis=1)
        all_CNN_predictions.append(y_pred_batch)

    all_CNN_predictions_array = np.column_stack(all_CNN_predictions)

    predictions_CNN_df = pd.DataFrame(all_CNN_predictions_array)
    predictions_CNN_df.to_excel(rf"{output_path}\{way}_{algorithm}_CNN_predictions_{n_classes}.xlsx", index=False)

    colors = {0: 'red', 1: 'yellow', 2: 'gray'}
    rgb_image = np.zeros((*predictions_CNN_df.shape, 3))

    for value, color in colors.items():
        mask = predictions_CNN_df == value
        rgb_image[mask.values] = mcolors.to_rgb(color)

    labels_df1 = pd.read_excel(labels_file1, header=None)
    labels_df2 = pd.read_excel(labels_file2, header=None)

    white_color = [1, 1, 1]

    well1_rows = min(sample_num, len(labels_df1))
    rgb_image[:, 64, :] = white_color

    well2_rows = min(sample_num, len(labels_df2))
    rgb_image[14:14 + well2_rows, 241, :] = white_color

    fig, ax = plt.subplots()
    ax.imshow(rgb_image, interpolation='nearest')
    ax.axis('off')

    fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
    ax.set_position([0, 0, 1, 1])

    plt.tight_layout()
    plt.savefig(rf'{output_path}\{way}_{algorithm}_CNN_predictions_image_{n_classes}.png', format='png', bbox_inches='tight', pad_inches=0,
                dpi=600)

    all_predictions = []

    for i, (X_batch, batch_files) in enumerate(load_images_in_batches(pred_file, batch_size)):
        print(f"Processing batch {i + 1}/{len(range(0, len(os.listdir(pred_file)), batch_size))}...")
        y_pred_batch = Ada_CNN_model.predict(X_batch)
        all_predictions.append(y_pred_batch)

    all_predictions_array = np.column_stack(all_predictions)

    predictions_df = pd.DataFrame(all_predictions_array)
    predictions_df.to_excel(rf"{output_path}\{way}_{algorithm}_predictions_{n_classes}.xlsx", index=False)

    colors = {0: 'red', 1: 'yellow', 2: 'gray'}
    rgb_image = np.zeros((*predictions_df.shape, 3))

    for value, color in colors.items():
        mask = predictions_df == value
        rgb_image[mask.values] = mcolors.to_rgb(color)

    labels_df1 = pd.read_excel(labels_file1, header=None)
    labels_df2 = pd.read_excel(labels_file2, header=None)

    white_color = [1, 1, 1]

    well1_rows = min(sample_num, len(labels_df1))
    rgb_image[:, 64, :] = white_color

    well2_rows = min(sample_num, len(labels_df2))
    rgb_image[14:14 + well2_rows, 241, :] = white_color

    fig, ax = plt.subplots()
    ax.imshow(rgb_image, interpolation='nearest')
    ax.axis('off')

    fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
    ax.set_position([0, 0, 1, 1])

    plt.tight_layout()
    plt.savefig(rf'{output_path}\{way}_{algorithm}_predictions_image_{n_classes}.png', format='png', bbox_inches='tight', pad_inches=0,
                dpi=600)

else:
    all_CNN_proba_predictions = []

    for i, (X_batch, batch_files) in enumerate(load_images_in_batches(pred_file, batch_size)):
        print(f"Processing batch {i + 1}/{len(range(0, len(os.listdir(pred_file)), batch_size))}...")
        y_proba_batch = model.predict(X_batch)
        all_CNN_proba_predictions.extend(y_proba_batch)

    all_CNN_proba_predictions_array = np.array(all_CNN_proba_predictions)

    for class_index in range(n_classes):
        class_proba = all_CNN_proba_predictions_array[:, class_index]
        class_image_data = class_proba.reshape(-1, batch_size).T

        class_proba_df = pd.DataFrame(class_image_data)
        class_proba_df.to_excel(
            rf"{output_path}\{way}_{algorithm}_CNN_sorted_proba_class_{n_classes}_{class_index}.xlsx",
            index=False)

    colors = {0: 'red', 1: 'yellow', 2: 'gray'}
    gray_Green_yellow_cmap = LinearSegmentedColormap.from_list(
        "GrayGreenYellow",
        ['gray', 'green', 'yellow', 'red']
    )
    cmap = gray_Green_yellow_cmap

    labels_df1 = pd.read_excel(labels_file1, header=None)
    labels_df2 = pd.read_excel(labels_file2, header=None)

    for class_index in range(n_classes):
        sorted_proba_df = pd.read_excel(
            rf"{output_path}\{way}_{algorithm}_CNN_sorted_proba_class_{n_classes}_{class_index}.xlsx")
        sorted_class_proba = sorted_proba_df.values

        class_image = cmap(sorted_class_proba)

        white_color = [1, 1, 1, 1]

        well1_rows = min(sample_num, len(labels_df1))
        class_image[:, 64, :] = white_color

        well2_rows = min(sample_num, len(labels_df2))
        class_image[14:14 + well2_rows, 241, :] = white_color

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(class_image, aspect='auto')
        ax.axis('off')
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
        ax.set_position([0, 0, 1, 1])

        plt.tight_layout()
        plt.savefig(
            rf'{output_path}\{way}_{algorithm}_CNN_sorted_proba_predictions_image_class_{n_classes}_{class_index}.png',
            format='png',
            bbox_inches='tight', pad_inches=0, dpi=600)

    all_proba_predictions = []

    for i, (X_batch, batch_files) in enumerate(load_images_in_batches(pred_file, batch_size)):
        print(f"Processing batch {i + 1}/{len(range(0, len(os.listdir(pred_file)), batch_size))}...")
        y_proba_batch = Ada_CNN_model.predict_proba(X_batch)
        all_proba_predictions.extend(y_proba_batch)

    all_proba_predictions_array = np.array(all_proba_predictions)

    for class_index in range(n_classes):
        class_proba = all_proba_predictions_array[:, class_index]
        class_image_data = class_proba.reshape(-1, batch_size).T

        class_proba_df = pd.DataFrame(class_image_data)
        class_proba_df.to_excel(rf"{output_path}\{way}_{algorithm}_sorted_proba_class_{n_classes}_{class_index}.xlsx", index=False)

    colors = {0: 'red', 1: 'yellow', 2: 'gray'}
    gray_Green_yellow_cmap = LinearSegmentedColormap.from_list(
        "GrayGreenYellow",
        ['gray', 'green', 'yellow', 'red']
    )
    cmap = gray_Green_yellow_cmap

    labels_df1 = pd.read_excel(labels_file1, header=None)
    labels_df2 = pd.read_excel(labels_file2, header=None)

    for class_index in range(n_classes):
        sorted_proba_df = pd.read_excel(rf"{output_path}\{way}_{algorithm}_sorted_proba_class_{n_classes}_{class_index}.xlsx")
        sorted_class_proba = sorted_proba_df.values

        class_image = cmap(sorted_class_proba)

        white_color = [1, 1, 1, 1]

        well1_rows = min(sample_num, len(labels_df1))
        class_image[:, 64, :] = white_color

        well2_rows = min(sample_num, len(labels_df2))
        class_image[14:14 + well2_rows, 241, :] = white_color

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(class_image, aspect='auto')
        ax.axis('off')
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0)
        ax.set_position([0, 0, 1, 1])

        plt.tight_layout()
        plt.savefig(rf'{output_path}\{way}_{algorithm}_sorted_proba_predictions_image_class_{n_classes}_{class_index}.png', format='png',
                    bbox_inches='tight', pad_inches=0, dpi=600)
