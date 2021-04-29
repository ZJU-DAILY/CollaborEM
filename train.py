import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import AdamW
from apex import amp
from dataloader import TrainDataset, TestDataset
from model import LMNet
from utils import *


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Models.',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--vis_device', type=str, default='0')

    parser.add_argument('--data_name', type=str, default='Structured/iTunes-Amazon')
    parser.add_argument('--save_model', action='store_true')

    parser.add_argument('--scheduler', default=False)
    parser.add_argument('--fp16', default=True)

    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--n_epoch', type=int, default=40)

    parser.add_argument('--literal', action='store_true')
    parser.add_argument('--digital', action='store_true')
    parser.add_argument('--structure', action='store_true')
    parser.add_argument('--name', action='store_true')

    parser.add_argument('--skip', default=True)
    parser.add_argument('--add_token', default=True)

    parser.add_argument('--seed', default=2021, type=int)

    args = parser.parse_args(args)

    return args


def train(model, train_set, optimizer, scheduler=None, fp16=True):
    """
    Perform one epoch of the training process.

    Args:
    model: the current model
    train_set: the training dataset
    optimizer: the optimizer for training (e.g., Adam)
    scheduler: the scheduler for training
    batch_size (int, optional): the batch size
    fp16 (boolean): whether to use fp16

    Returns:
        None
    """
    iterator = DataLoader(dataset=train_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=8,
                          worker_init_fn=worker_init,
                          collate_fn=TrainDataset.pad)
    criterion = nn.CrossEntropyLoss()
    dist_criterion = nn.CosineEmbeddingLoss(margin=0.5)
    model.train()
    for batch in iterator:
        x, y, seqlen, sample, sentencesA, sentencesB = batch
        y = torch.tensor(y).cuda()
        x = torch.tensor(x).view(y.size(0), -1).cuda()
        sentencesA = torch.tensor(sentencesA).view(y.size(0), -1).cuda()
        sentencesB = torch.tensor(sentencesB).view(y.size(0), -1).cuda()
        sample = torch.LongTensor(sample).view(y.size(0), -1)

        # forward
        optimizer.zero_grad()
        logits, _, eA, eB = model(x, sample, sentencesA, sentencesB)

        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)

        bce_loss = criterion(logits, y)

        y_ = 2 * y
        y_ -= 1
        dist_loss = dist_criterion(eA, eB, y_)

        loss = bce_loss + 0.2 * dist_loss

        # back propagation
        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()


def eval_model(model, dataset):
    iterator = DataLoader(dataset=dataset,
                          batch_size=256,
                          collate_fn=TestDataset.test_pad)
    model.eval()
    y_truth = []
    y_pre = []
    for batch in iterator:
        x, y, seqlens, sample = batch
        x = torch.tensor(x).squeeze().cuda()
        sample = torch.LongTensor(sample).squeeze()
        with torch.no_grad():
            logits, _ = model(x, sample)
            for item in y:
                y_truth.append(item[0])
            for item in _.cpu().numpy().tolist():
                y_pre.append(item)

    precision, recall, F1 = evaluate(y_truth, y_pre)
    return precision, recall, F1


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.vis_device

    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    configs = json.load(open('configs.json'))
    configs = {conf['name']: conf for conf in configs}
    config = configs[args.data_name]
    max_length = int(config['max_length'])
    batch_size = int(config['batch_size'])
    n_epoch = args.n_epoch
    model_name = config['fine_tune_model']
    data_name = args.data_name
    path = config['path']

    cur_time = '_' + datetime.now().strftime('%F %T')
    logger_name = data_name.replace('/', '')

    if args.literal:
        logger_name += '_literal'
    if args.digital:
        logger_name += '_digital'
    if args.structure:
        logger_name += '_structure'
    if args.name:
        logger_name += '_name'
    logger_name += '_fine-tune_'
    logger_name += cur_time
    set_logger(logger_name)
    logging.info(args)
    logging.info(config)

    logging.info('Seed: {}'.format(torch.initial_seed()))

    train_set = TrainDataset(path, model_name, max_length, skip=args.skip, add_token=args.add_token)

    test_set = TestDataset(path, True, model_name, max_length, skip=args.skip, add_token=args.add_token)
    all_set = TestDataset(path, False, model_name, max_length, skip=args.skip, add_token=args.add_token)

    model = LMNet(
        finetuning=True,
        lm=model_name,
        data_path=path,
        use_literal_gnn=args.literal,
        use_digital_gnn=args.digital,
        use_structure_gnn=args.structure,
        use_name_gnn=args.name,
    )
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    logging.info(model)

    test_precision, test_recall, test_F1 = 0.0, 0.0, 0.0
    all_precision, all_recall, all_F1 = 0.0, 0.0, 0.0

    for epoch in range(1, n_epoch + 1):
        logging.info('epoch: {}'.format(epoch))

        torch.cuda.empty_cache()
        train(model,
              train_set,
              optimizer,
              scheduler=None)

        torch.cuda.empty_cache()

        test_precision, test_recall, test_F1 = eval_model(model, test_set)
        logging.info(
            '[Test]  precision: {:.4f}  recall: {:.4f}  F1: {:.4f}'.format(test_precision, test_recall, test_F1))
        all_precision, all_recall, all_F1 = eval_model(model, all_set)
        logging.info(
            '[All]  precision: {:.4f}  recall: {:.4f}  F1: {:.4f}'.format(all_precision, all_recall, all_F1))

        if args.save_model:
            torch.save(model.state_dict(),
                       os.path.join('./checkpoint', data_name.replace('/', '') + '_' + str(epoch) + '_.pt'))

    logging.info('Finish training!')
    logging.info('[Result]')
    logging.info(
        '[Test]  precision: {:.4f}  recall: {:.4f}  F1: {:.4f}'.format(test_precision, test_recall, test_F1))

    logging.info(
        '[All]  precision: {:.4f}  recall: {:.4f}  F1: {:.4f}'.format(all_precision, all_recall, all_F1))

