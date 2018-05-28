#!/usr/bin/env python3
# coding: utf-8
# File: lstm_train.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-5-23

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF

class LSTMNER:
    def __init__(self):
        self.train_path = 'data/ner_train.txt'
        self.vocab_path = 'model/vocab.txt'
        self.labels = ['ni-B', 'ni-E', 'nh-E', 'ns-E', 'ni-M', 'nt-B', 'nt-E', 'nh-B', 'nt-M', 'ns-B', 'O', 'nh-M', 'ns-M']
        self.vocabs, self.train_data, self.maxlen = self.load_data()
        self.wordindex_dict = {char: index for index, char in enumerate(self.vocabs)}
        self.labelindex_dict = {label: index for index, label in enumerate(self.labels)}
        self.EMBED_DIM = 200
        self.BiRNN_UNITS = 200
        self.EPOCHS = 4
        self.BATCH_SIZE = 50

    '''加载数据'''
    def load_data(self):
        words = []
        train_data = []
        maxlen = 0
        for line in open(self.train_path):
            line = line.strip().split()
            chars = [word.split('_')[0] for word in line]
            labels = [word.split('_')[1] for word in line]
            train_data.append([chars, labels])
            if len(chars) > maxlen:
                maxlen = len(chars)
            words += [word.split('_')[0] for word in line]

        vocabs = sorted(set(list(words)))
        with open(self.vocab_path, 'w+') as f:
            f.write('@'.join([word for word in vocabs]))
            f.write('@' + str(maxlen))
        f.close()

        return vocabs, train_data, maxlen

    '''构造输入'''
    def build_input(self):
        x_train = [[self.wordindex_dict[char] for char in data[0]] for data in self.train_data]
        y_train = [[self.labelindex_dict[label] for label in data[1]] for data in self.train_data]
        x_train = pad_sequences(x_train, self.maxlen)
        y = pad_sequences(y_train, self.maxlen)
        y_train = np.expand_dims(y, 2)

        return x_train, y_train

    '''构造模型'''
    def build_model(self):
        model = Sequential()
        model.add(Embedding(len(self.vocabs), self.EMBED_DIM, mask_zero=False))
        model.add(Bidirectional(LSTM(self.BiRNN_UNITS, return_sequences=True)))
        crf = CRF(len(self.labels), sparse_target=True)
        model.add(crf)
        model.summary()
        model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
        return model

    '''训练模型'''
    def train_model(self):
        x_train, y_train = self.build_input()
        model = self.build_model()
        model.fit(x_train[:], y_train[:], batch_size=self.BATCH_SIZE, epochs=self.EPOCHS)
        model.save('model/bilstm_crf.h5')
        return model

ner = LSTMNER()
ner.train_model()