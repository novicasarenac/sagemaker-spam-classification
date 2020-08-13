import argparse
import os
from typing import Text, Tuple

import numpy as np
import tensorflow as tf

NUM_WORDS: int = 3000
MAX_SEQ_LEN: int = 100


class Trainer:
    def __init__(
        self,
        train_path: Text,
        test_path: Text,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        model_dir: Text,
    ) -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_dir = model_dir

    def __call__(self) -> None:
        X_train, y_train = self._read_train_data()
        X_test, y_test = self._read_test_data()
        print(f"Train shape: {X_train.shape}")
        print(f"Test shape: {X_test.shape}")
        model = self._get_model()
        model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_test, y_test),
        )
        _, accuracy = model.evaluate(X_test, y_test, self.batch_size)
        print(f"Model accuracy: {accuracy}")
        model.save(self.model_dir + "/1")

    def _read_train_data(self) -> Tuple[np.array, np.array]:
        X_train = np.load(os.path.join(self.train_path, "X_train.npy"))
        y_train = np.load(os.path.join(self.train_path, "y_train.npy"))
        return X_train, y_train

    def _read_test_data(self) -> Tuple[np.array, np.array]:
        X_test = np.load(os.path.join(self.test_path, "X_test.npy"))
        y_test = np.load(os.path.join(self.test_path, "y_test.npy"))
        return X_test, y_test

    def _get_model(self) -> tf.keras.Model:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(name="inputs", shape=[MAX_SEQ_LEN]),
                tf.keras.layers.Embedding(
                    NUM_WORDS, 50, input_length=MAX_SEQ_LEN
                ),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="Number of epochs.")
    parser.add_argument("--batch_size", help="Batch size for training.")
    parser.add_argument("--learning_rate", help="Learning rate for training.")
    parser.add_argument("--model_dir", help="Path to the model directory.")
    args = parser.parse_args()
    train_path = os.environ.get("SM_CHANNEL_TRAIN")
    test_path = os.environ.get("SM_CHANNEL_TEST")
    trainer = Trainer(
        train_path,
        test_path,
        int(args.epochs),
        int(args.batch_size),
        float(args.learning_rate),
        args.model_dir,
    )
    trainer()
