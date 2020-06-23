import os
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from typing import Text, Tuple, Dict, List


FILE_NAME: Text = "SMSSpamCollection"
NUM_WORDS: int = 3000
MAX_SEQ_LEN: int = 100


class DataProcessor:
    def __init__(self, data_path: Text, output_path: Text) -> None:
        self.data_path = os.path.join(data_path, FILE_NAME)
        self.train_path = os.path.join(output_path, "train")
        self.test_path = os.path.join(output_path, "test")
        self.vocabulary_path = os.path.join(output_path, "vocabulary.pkl")
        self._make_dirs()

    def __call__(self) -> None:
        data = self._read_raw_data()
        self._encode_labels(data)
        vectorizer = CountVectorizer(max_features=NUM_WORDS)
        vectorizer.fit(data.sms)
        X = self._encode(data.sms, vectorizer)
        y = data.label.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self._save_data(vectorizer, X_train, X_test, y_train, y_test)

    def _encode(self, texts: List, vectorizer: CountVectorizer) -> np.array:
        encoded = []
        tokenizer = vectorizer.build_tokenizer()
        vocabulary = vectorizer.vocabulary_
        for sms in texts:
            sms = sms.lower()
            tokens = tokenizer(sms)
            encoded_sms = np.array(
                [vocabulary[token] for token in tokens if token in vocabulary.keys()]
            )
            if MAX_SEQ_LEN > len(encoded_sms):
                padding_size = MAX_SEQ_LEN - len(encoded_sms)
            else:
                padding_size = 0
                encoded_sms = encoded_sms[:MAX_SEQ_LEN]
            encoded_sms = np.pad(
                encoded_sms, (0, padding_size), mode="constant", constant_values=0
            )
            encoded.append(encoded_sms)
        return np.vstack(encoded)

    def _encode_labels(self, dataframe: pd.DataFrame) -> None:
        encoder = LabelEncoder()
        dataframe["label"] = encoder.fit_transform(dataframe["label"])

    def _read_raw_data(self) -> pd.DataFrame:
        raw_data = pd.read_csv(
            self.data_path, names=["label", "sms"], header=None, sep="\t"
        )
        return raw_data

    def _make_dirs(self) -> None:
        if not os.path.exists(self.train_path):
            os.makedirs(self.train_path)
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)

    def _save_data(
        self,
        vectorizer: CountVectorizer,
        X_train: np.array,
        X_test: np.array,
        y_train: np.array,
        y_test: np.array,
    ) -> None:
        vocabulary = vectorizer.vocabulary_
        pickle.dump(vocabulary, open(self.vocabulary_path, "wb"))
        X_train_path = os.path.join(self.train_path, "X_train.npy")
        np.save(X_train_path, X_train)
        X_test_path = os.path.join(self.test_path, "X_test.npy")
        np.save(X_test_path, X_test)
        y_train_path = os.path.join(self.train_path, "y_train.npy")
        np.save(y_train_path, y_train)
        y_test_path = os.path.join(self.test_path, "y_test.npy")
        np.save(y_test_path, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--data_path", help="Path to the raw data.")
    parser.add_argument("-op", "--output_path", help="Path to the output data.")
    args = parser.parse_args()
    data_processor = DataProcessor(args.data_path, args.output_path)
    data_processor()
