#!/usr/bin/env python3
# coding: utf-8
# File: lstm_predict.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-5-23

from keras.models import load_model
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF

class LSTMNER:
    def __init__(self):
        self.weight_path = 'model/bilstm_crf.h5'
        self.EMBED_DIM = 200
        self.BiRNN_UNITS = 200
        self.vocab_path = 'model/vocab.txt'
        self.labels = ['ni-B', 'ni-E', 'nh-E', 'ns-E', 'ni-M', 'nt-B', 'nt-E', 'nh-B', 'nt-M', 'ns-B', 'O', 'nh-M', 'ns-M']
        self.vocabs, self.maxlen = self.load_vocabs()
        self.wordindex_dict = {char: index for index, char in enumerate(self.vocabs)}
        self.labelindex_dict = {index: label for index, label in enumerate(self.labels)}
        self.model = self.load_model()
        self.model.load_weights(self.weight_path)

    def load_vocabs(self):
        vocabs = open(self.vocab_path).read().split('@')[:-1]
        maxlen = int(open(self.vocab_path).read().split('@')[-1])
        return vocabs, maxlen

    def build_input(self, text):
        x = [self.wordindex_dict[char] for char in text]
        x = pad_sequences([x], self.maxlen)
        return x, len(text)

    def ner(self, text):
        str, length = self.build_input(text)
        print(length)
        print(self.model.predict(str))
        raw = self.model.predict(str)[0][-length:]
        result = [np.argmax(row) for row in raw]
        print(result)
        print(len(result))
        result_tags = [self.labelindex_dict[i] for i in result]
        print(text)
        print(result_tags)
        entity_result = self.label2word(result_tags, text)

        return entity_result


    def label2word(self, labels, sent):
        per = []
        loc = []
        org = []
        tim = []
        tim_tmp = []
        loc_tmp = []
        per_tmp = []
        org_tmp = []
        ner_dict = {}
        for index in range(len(labels)):
            label = labels[index]
            word = sent[index]
            pair = [word, label]

            if label == 'ns-B':
                if loc_tmp and 'ns-E' in loc_tmp:
                    loc.append(loc_tmp)
                loc_tmp = []
                loc_tmp.extend(pair)

            elif label == 'ns-M':
                loc_tmp.extend(pair)
            elif label == 'ns-E':
                loc_tmp.extend(pair)
                if 'ns-B' in loc_tmp:
                    loc.append(loc_tmp)
                loc_tmp = []

            if label == 'nh-B':
                if per_tmp and 'nh-E' in per_tmp:
                    per.append(per_tmp)
                per_tmp = []
                per_tmp.extend(pair)
            elif label == 'nh-M':
                per_tmp.extend(pair)
            elif label == 'nh-E':
                per_tmp.extend(pair)
                if 'nh-B' in per_tmp:
                    per.append(per_tmp)
                per_tmp = []

            if label == 'ni-B':
                if org_tmp and 'ni-E' in org_tmp:
                    org.append(org_tmp)
                org_tmp = []
                org_tmp.extend(pair)
            elif label == 'ni-M':
                org_tmp.extend(pair)
            elif label == 'ni-E':
                org_tmp.extend(pair)
                if 'ni-B' in org_tmp:
                    org.append(org_tmp)
                org_tmp = []

            if label == 'nt-B':
                if tim_tmp and 'nt-E' in tim_tmp:
                    tim.append(tim_tmp)
                tim_tmp = []
                tim_tmp.extend(pair)
            elif label == 'nt-M':
                tim_tmp.extend(pair)
            elif label == 'nt-E':
                tim_tmp.extend(pair)
                if 'nt-B' in tim_tmp:
                    tim.append(tim_tmp)
                tim_tmp = []

        LOC = [''.join([loc_ for loc_ in [sub_item for sub_item in item if item.index(sub_item) % 2 == 0]]) for item in
               loc
               if item]
        PER = [''.join([per_ for per_ in [sub_item for sub_item in item if item.index(sub_item) % 2 == 0]]) for item in
               per
               if item]
        ORG = [''.join([org_ for org_ in [sub_item for sub_item in item if item.index(sub_item) % 2 == 0]]) for item in
               org
               if item]
        TIM = [''.join([org_ for org_ in [sub_item for sub_item in item if item.index(sub_item) % 2 == 0]]) for item in
               tim
               if item]

        ner_dict['LOC'] = list(set(LOC))
        ner_dict['ORG'] = list(set(ORG))
        ner_dict['PER'] = list(set(PER))
        ner_dict['TIM'] = list(set(TIM))
        return ner_dict

    def load_model(self):
        model = Sequential()
        model.add(Embedding(len(self.vocabs), self.EMBED_DIM, mask_zero=True))  # Random embedding
        model.add(Bidirectional(LSTM(self.BiRNN_UNITS, return_sequences=True)))
        crf = CRF(len(self.labels), sparse_target=True)
        model.add(crf)
        return model



text = '中国是我的祖国，我想回江西'
text = '刘焕勇硕士毕业于北京语言大学'
Ner = LSTMNER()
entity = Ner.ner(text)
print(entity)