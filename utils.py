import csv
import os
import logging
import torch
import random
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def worker_init(worker_init):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def read_entity(table_paths, skip=True, add_token=True):
    """
    Read entities from tables.
    """
    entity_list = []
    if type(table_paths) is list:
        for table_path in table_paths:
            lines = list(csv.reader(open(table_path, 'r')))
            att = []
            for i, line in enumerate(lines):
                sentence = ''
                for j in range(1, len(line)):
                    if i == 0:
                        att.append(line[j])
                    elif skip and (line[j] == ''):
                        continue
                    elif add_token:
                        sentence += 'COL ' + att[j - 1] + ' VAL ' + line[j] + ' '
                    else:
                        sentence += att[j - 1] + ' ' + line[j] + ' '
                if i != 0:
                    entity_list.append(sentence.strip())
    else:
        lines = list(csv.reader(open(table_paths, 'r')))
        att = []
        for i, line in enumerate(lines):
            sentence = ''
            for j in range(1, len(line)):
                if i == 0:
                    att.append(line[j])
                elif skip and (line[j] == ''):
                    continue
                elif add_token:
                    sentence += 'COL ' + att[j - 1] + ' VAL ' + line[j] + ' '
                else:
                    sentence += att[j - 1] + ' ' + line[j] + ' '
            if i != 0:
                entity_list.append(sentence.strip())

    return entity_list


def evaluate(y_truth, y_pred):
    """
    Evaluate model.
    """
    precision = precision_score(y_truth, y_pred)
    recall = recall_score(y_truth, y_pred)
    f1 = f1_score(y_truth, y_pred)
    return precision, recall, f1


def set_logger(name):
    """
    Write logs to checkpoint and console.
    """

    log_file = os.path.join('./log', name)

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def get_tokenizer(lm='bert'):
    """Return the tokenizer. Initialize it if not initialized.

    Args:
        lm (string): the name of the language model (bert, albert, or distilbert)
    Returns:
        BertTokenizer or DistilBertTokenizer or AlbertTokenizer
    """
    tokenizer = None
    if lm == 'bert':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif lm == 'distilbert':
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    elif lm == 'albert':
        from transformers import AlbertTokenizer
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    elif lm == 'roberta':
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif lm == 'xlnet':
        from transformers import XLNetTokenizer
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    elif lm == 'longformer':
        from transformers import LongformerTokenizer
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    return tokenizer
