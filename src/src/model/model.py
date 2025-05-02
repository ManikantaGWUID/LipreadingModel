import os
import glob
import string
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, callbacks

# ------------------------------
# Configuration
# ------------------------------

VOCAB = list(string.ascii_lowercase + " ")
DATA_PATH = "../Processed_data1"
ALIGN_PATH = "../Processed_data1"
BATCH_SIZE = 10
EPOCHS = 100

# ------------------------------
# Vocabulary Encoding
# ------------------------------

char_to_num = tf.keras.layers.StringLookup(vocabulary=VOCAB, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# ------------------------------
# Data Loading Functions
# ------------------------------

def load_alignment(path: str):
    with open(path, "r") as f:
        lines = f.readlines()
    tokens = [line.split()[2] for line in lines if line.split()[2] != 'sil']
    chars = list(" ".join(tokens))
    return char_to_num(chars)

def load_data(path: str):
    path = bytes.decode(path.numpy())
    speaker, video = path.split("\\")[-2], path.split("\\")[-1].split(".")[0]
    frames = np.load(os.path.join(DATA_PATH, speaker, f"{video}.npy"))
    alignments = load_alignment(os.path.join(ALIGN_PATH, speaker, 'align', f"{video}.align"))
    return frames, alignments

def mappable_function(path: str):
    return tf.py_function(load_data, [path], (tf.float32, tf.int64))

# ------------------------------
# Dataset Preparation
# ------------------------------

def prepare_datasets():
    all_videos = glob.glob(os.path.join(DATA_PATH, "*/*.npy"))
    train, test = train_test_split(all_videos, test_size=0.1, random_state=42)
    train_new, val_new = train_test_split(train, test_size=0.1, random_state=42)

    def create_dataset(paths):
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.shuffle(500)
        ds = ds.map(mappable_function, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.padded_batch(BATCH_SIZE, padded_shapes=([75, 50, 150, 3], [40]))
        ds = ds.prefetch(tf.data.AUTOTUNE).cache()
        return ds

    return create_dataset(train_new), create_dataset(val_new), create_dataset(test)

# ------------------------------
# Model Definition
# ------------------------------

def build_model():
    model = models.Sequential([
        # Conv Block 1
        layers.Conv3D(128, kernel_size=(3, 3, 3), padding='same', input_shape=(75, 50, 150, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool3D(pool_size=(1, 2, 2)),

        # Conv Block 2
        layers.Conv3D(256, kernel_size=(3, 3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool3D(pool_size=(1, 2, 2)),

        # Conv Block 3
        layers.Conv3D(256, kernel_size=(3, 3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool3D(pool_size=(1, 2, 2)),

        layers.SpatialDropout3D(0.3),

        # Flatten temporal dimension (T, H, W, C) → (T, features)
        layers.TimeDistributed(layers.Flatten()),

        # Recurrent Layers
        layers.Bidirectional(layers.LSTM(512, return_sequences=True, kernel_initializer='orthogonal')),
        layers.Dropout(0.5),

        layers.Bidirectional(layers.LSTM(512, return_sequences=True, kernel_initializer='orthogonal')),
        layers.Dropout(0.5),
        # Final output layer
        layers.Dense(char_to_num.vocabulary_size() + 1, activation='softmax', kernel_initializer='he_normal')
    ])
    return model

# ------------------------------
# Loss Function
# ------------------------------

def CTCLoss(y_true, y_pred):
    batch_size = tf.cast(tf.shape(y_true)[0], tf.int64)
    input_len = tf.cast(tf.shape(y_pred)[1], tf.int64)
    label_len = tf.cast(tf.shape(y_true)[1], tf.int64)
    input_len = input_len * tf.ones(shape=(batch_size, 1), dtype=tf.int64)
    label_len = label_len * tf.ones(shape=(batch_size, 1), dtype=tf.int64)
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_len, label_len)

# ------------------------------
# Callbacks
# ------------------------------

def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

class ProduceExample(callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset.as_numpy_iterator()

    def on_epoch_end(self, epoch, logs=None):
        data = self.dataset.next()
        yhat = self.model.predict(np.array([data[0][0]]))
        decoded = tf.keras.backend.ctc_decode(yhat, [75], greedy=False)[0][0].numpy()

        print('Original:', tf.strings.reduce_join(num_to_char(data[1][0])).numpy().decode('utf-8'))
        print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[0])).numpy().decode('utf-8'))

# ------------------------------
# Training Pipeline
# ------------------------------

def train():
    train_ds, val_ds, test_ds = prepare_datasets()
    model = build_model()
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=CTCLoss)

    checkpoint_cb = callbacks.ModelCheckpoint('../models/checkpoint.weights.h5', monitor='loss', save_weights_only=True, save_best_only=True)
    schedule_cb = callbacks.LearningRateScheduler(scheduler)
    example_cb = ProduceExample(test_ds)

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[checkpoint_cb, schedule_cb, example_cb])

# ------------------------------
# Entry Point
# ------------------------------

if __name__ == "__main__":
    train()


