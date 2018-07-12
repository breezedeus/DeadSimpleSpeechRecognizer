import os
from collections import Counter
import numpy as np
import librosa.display
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

from preprocess import wav2mfcc


LABEL_MAPPING = {
    'W': 'anger',
    'L': 'boredom',
    'E': 'disgust',
    'A': 'anxiety/fear',
    'F': 'happiness',
    'T': 'sadness',
    'N': 'neutral'
}


def read_data_to_array(path, max_len=21, rescale=False):
    X = []
    y = []
    for wavfile in os.listdir(path):
        if wavfile.startswith('.') or not wavfile.endswith('.wav'):
            continue
        y.append(LABEL_MAPPING[wavfile[5]])
        X.append(wav2mfcc(os.path.join(path, wavfile), max_len, rescale))

    y_counter = Counter(y)
    print(y_counter.most_common())
    label_to_int_dict = {l: idx for idx, (l, _) in enumerate(y_counter.most_common())}
    y = [label_to_int_dict[l] for l in y]
    X, y = np.stack(X), np.stack(y)
    y = to_categorical(y)

    return X, y, label_to_int_dict


def print_sample(sample):
    import matplotlib.pyplot as plt
    import librosa.display
    plt.figure(figsize=(10, 4))
    print(sample.shape)  # 11帧，每帧20个特征
    librosa.display.specshow(sample, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()


def create_model(num_classes, feature_dim_1, feature_dim_2, channel):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))
    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


# Predicts one sample
def predict(sample, model, labels):
    feature_dim_1, feature_dim_2 = sample.shape[0], sample.shape[1]
    channel = 1
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    probs = model.predict(sample_reshaped)[0]
    print('labels: %s' % str(labels))
    print('probs: %s' % probs)
    return labels[np.argmax(probs)]