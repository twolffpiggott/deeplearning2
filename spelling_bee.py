import matplotlib
import re

import numpy as np

from dataclasses import dataclass, InitVar, field
from keras.layers import Bidirectional, Dense, Embedding, Input, LSTM
from keras.layers import TimeDistributed, RepeatVector
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from utils2 import limit_mem
from log import setup_custom_logger
from tqdm import tqdm

logger = setup_custom_logger('root')

PATH = 'data/'


@dataclass
class PhonemeData:
    """
    Class wrapping phoneme data, and mapping between pronunciation and
    phonemes, as well as from letters and phonemes to indices.

    :attribute lines: list of lines from data file to parse for words and
                      phonemes
    :attribute letters: str representing dictionary of characters across all
                        words
    :attribute phonemes: set of all possible phonemes
    :attribute p2i: dict mapping phonemes to indices
    :attribute l2i: dict mapping letters to indices
    :attribute minlen: minimum word length to consider
    :attribute maxlen: maximum word length to consider
    :attribute pronounce_dict: dict mapping words to lists of phoneme indices
    """
    lines: InitVar[list]
    letters: str = "_abcdefghijklmnopqrstuvwxyz*"
    phonemes: list = field(init=False)
    p2i: dict = field(init=False)
    l2i: dict = field(init=False)
    minlen: int = 5
    maxlen: int = 15
    pronounce_dict: dict = field(init=False)
    maxlen_p: int = field(init=False)

    def __post_init__(self, lines):

        parsed_lines = [l.strip().split("  ") for l in lines
                        if re.match('^[A-Z]', l)]
        tuples = [(word, phoneme.split()) for word, phoneme in parsed_lines]

        self.phonemes = ["_"] + sorted(set(p for w, ps in tuples for p in ps))
        logger.info('Phonemes identified: %d', len(self.phonemes))

        self.p2i = dict((v, k) for k, v in enumerate(self.phonemes))
        logger.info('Created size %d p2i dict mapping phonemes'
                    ' to indices', len(self.p2i))

        self.l2i = dict((v, k) for k, v in enumerate(self.letters))
        logger.info('Created size %d l2i dict mapping letters'
                    ' to indices', len(self.l2i))

        self.pronounce_dict = {w.lower(): [self.p2i[p] for p in ps]
                               for w, ps in tuples
                               if (self.minlen <= len(w) <= self.maxlen)
                               and re.match("^[A-Z]+$", w)}
        logger.info('Created size %d dict mapping words to lists of phoneme'
                    ' indices', len(self.pronounce_dict))

        self.maxlen_p = max([len(v) for k, v in self.pronounce_dict.items()])
        logger.info('Largest number of phonemes for any word %d',
                    self.maxlen_p)

    def prepare_training_data(self) -> tuple:
        """
        Convert words and phonemes into input and labels for a deep
        learning model.

        Returns: (input_, labels_, dec_input_)

        input_: arrays of phoneme indices for words
        labels_: arrays of letter indices for words
        dec_input_: decoder input for teaching forcing: go charac index
                    followed by previous correct word index
        """

        # effectively shuffle data
        pairs = np.random.permutation(list(self.pronounce_dict.keys()))
        n = len(pairs)
        input_ = np.zeros((n, self.maxlen_p), np.int32)
        labels_ = np.zeros((n, self.maxlen), np.int32)

        # zero pad and structure data for training
        # the '_' padding token for phonemes and indices is the 0th index
        for i, word in tqdm(enumerate(pairs), total=len(pairs)):
            for j, phoneme_index in enumerate(self.pronounce_dict[word]):
                input_[i, j] = phoneme_index
            for j, letter in enumerate(word):
                labels_[i, j] = self.l2i[letter]
        go_token_index = self.l2i['*']
        # teacher forcing: feed the decoder the previous correct char
        dec_input_ = np.concatenate([np.ones((n, 1)) * go_token_index,
                                    labels_[:, :-1]], axis=1)
        logger.info('Prepared training data: %d observations, %d longest'
                    ' phoneme', n, self.maxlen_p)
        return input_, labels_, dec_input_


def lstm_layer(return_sequences=True):
    """
    Standard LSTM layer, keeping dropout constant throughout.
    """
    return LSTM(240, dropout= 0.1, recurrent_dropout= 0.1,
                return_sequences=return_sequences)


def make_model(phoneme_data):
    """
    Create Keras seq2seq model predicting spelling from phonemes.

    :param phoneme_data: instance of PhonemeData
    """
    inp = Input((phoneme_data.maxlen_p,))
    x = Embedding(len(phoneme_data.phonemes), 120)(inp)

    x = Bidirectional(lstm_layer())(x)
    x = lstm_layer(return_sequences=False)(x)

    x = RepeatVector(phoneme_data.maxlen)(x)
    x = lstm_layer()(x)
    x = lstm_layer()(x)
    x = TimeDistributed(Dense(len(phoneme_data.letters),
                              activation='softmax'))(x)

    model = Model(inp, x)
    return model.compile(Adam(), 'sparse_categorical_crossentropy',
                         metrics=['acc'])


if __name__ == "__main__":
    with open(PATH+"cmudict-0.7b.txt", encoding='latin1') as stream:
        lines = stream.read().split('\n')
    data = PhonemeData(lines=lines)
    input_, labels_, dec_input_ = data.prepare_training_data()
    (input_train, input_test,
     labels_train, labels_test,
     dec_input_train, dec_input_test) = train_test_split(input_, labels_,
                                                         dec_input_,
                                                         test_size=0.1)
    logger.info('Input shape: %s Labels shape: %s', input_train.shape,
                labels_train.shape)
    model = make_model(phoneme_data=data)

