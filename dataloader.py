import os
import logging
import numpy as np
import json
import torch
from torch.utils.data import Dataset

from utils import read_entity, get_tokenizer


class TrainDataset(Dataset):
    def __init__(self, path, lm='bert', max_length=256, skip=True, add_token=True):
        self.path = path
        self.tokenizer = get_tokenizer(lm)
        tableA = os.path.join(path, 'tableA.csv')
        tableB = os.path.join(path, 'tableB.csv')
        self.entityA = read_entity(tableA, skip=skip, add_token=add_token)
        self.entityB = read_entity(tableB, skip=skip, add_token=add_token)
        self.lenA = len(self.entityA)
        self.lenB = len(self.entityB)
        self.max_len = max_length

        self.seeds, self.y = self.read_seeds_file()
        self.len = len(self.seeds)

    def read_seeds_file(self):
        path = os.path.join(self.path, 'seeds.csv')
        x = []
        y = []
        with open(path) as seeds_file:
            for i, line in enumerate(seeds_file.readlines()):
                if i == 0:
                    continue
                values = line.strip().split(',')
                a = int(values[0])
                b = int(values[1])
                x.append((a, b))
                y.append(int(values[2]))
        logging.info('Num seeds: {}'.format(len(x)))
        return x, y

    def pair2sentence(self, sample):
        res = []
        resA = []
        resB = []
        for x in sample:
            a_ = self.entityA[x[0]]
            b_ = self.entityB[x[1]]
            x = self.tokenizer.encode(text=a_, text_pair=b_, add_special_tokens=True,
                                      truncation='longest_first', max_length=self.max_len)
            res.append(x)
            x = self.tokenizer.encode(text=a_, add_special_tokens=True, truncation='longest_first',
                                      max_length=self.max_len)
            resA.append(x)
            x = self.tokenizer.encode(text=b_, add_special_tokens=True, truncation='longest_first',
                                      max_length=self.max_len)
            resB.append(x)
        return res, resA, resB

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        sample = [self.seeds[index]]
        sentence, sentencesA, sentencesB = self.pair2sentence(sample)
        label = [self.y[index]]
        seqlen = len(sentence[0])
        seqlenA = len(sentencesA[0])
        seqlenB = len(sentencesB[0])

        return sentence, label, seqlen, sample, sentencesA, sentencesB, seqlenA, seqlenB

    @staticmethod
    def pad(batch):
        """
        Pads to the longest sample.
        """
        f = lambda x: [sample[x] for sample in batch]

        seqlens = f(2)
        max_len = np.array(seqlens).max()

        sentences = f(0)
        label = f(1)
        sample = f(3)
        sentencesA = f(4)
        sentencesB = f(5)
        seqlensA = f(6)
        seqlensB = f(7)
        for sentence in sentences:
            sentence[0] += [0] * (max_len - len(sentence[0]))
        max_len = np.array(seqlensA).max()
        for sentence in sentencesA:
            sentence[0] += [0] * (max_len - len(sentence[0]))
        max_len = np.array(seqlensB).max()
        for sentence in sentencesB:
            sentence[0] += [0] * (max_len - len(sentence[0]))

        return sentences, label, seqlens, sample, sentencesA, sentencesB


class TestDataset(Dataset):
    def __init__(self, path, test=True, lm='bert', max_length=512, skip=False, add_token=True):
        self.path = path
        self.tokenizer = get_tokenizer(lm)
        tableA = os.path.join(path, 'tableA.csv')
        tableB = os.path.join(path, 'tableB.csv')
        self.entityA = read_entity(tableA, skip=skip, add_token=add_token)
        self.entityB = read_entity(tableB, skip=skip, add_token=add_token)
        self.lenA = len(self.entityA)
        self.lenB = len(self.entityB)
        self.max_len = max_length

        self.x, self.y = self.read_evaluate_file(test)
        self.len = len(self.x)

    def read_evaluate_file(self, test=True):
        if test:
            paths = [os.path.join(self.path, 'test.csv')]
        else:
            paths = [os.path.join(self.path, 'train.csv'),
                     os.path.join(self.path, 'valid.csv'),
                     os.path.join(self.path, 'test.csv')]
        x = []
        y = []
        for path in paths:
            with open(path) as evaluate_file:
                for i, line in enumerate(evaluate_file.readlines()):
                    if i == 0:
                        continue
                    values = line.strip().split(',')
                    x.append((int(values[0]), int(values[1])))
                    y.append(int(values[2]))
        return x, y

    def pair2sentence(self, sample):
        res = []
        for x in sample:
            a_ = self.entityA[x[0]]
            b_ = self.entityB[x[1]]
            x = self.tokenizer.encode(text=a_, text_pair=b_, add_special_tokens=True,
                                      truncation='longest_first', max_length=self.max_len)
            res.append(x)
        return res

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sample = [self.x[index]]
        sentence = self.pair2sentence(sample)
        label = [self.y[index]]

        return sentence, label, len(sentence[0]), sample

    @staticmethod
    def test_pad(batch):
        """
        Pads to the longest sample.
        """
        f = lambda x: [sample[x] for sample in batch]

        seqlens = f(2)
        max_len = np.array(seqlens).max()

        sentences = f(0)
        label = f(1)
        sample = f(3)
        for sentence in sentences:
            sentence[0] += [0] * (max_len - len(sentence[0]))

        return sentences, label, seqlens, sample
