import argparse
import json
import random
import time
import numpy as np
import torch.cuda
from sentence_transformers import SentenceTransformer, util
from utils import *
from datetime import datetime


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_device', type=str, default='0')

    parser.add_argument('--data_name', type=str, default='Structured/Fodors-Zagats')

    parser.add_argument('--hard_sample', action='store_true')

    parser.add_argument('--left', type=int, default=2)
    parser.add_argument('--right', type=int, default=10)

    parser.add_argument('--seed', default=2021, type=int)

    args = parser.parse_args(args)

    return args


def sample(topkA, topkB, sim_score, hard_sample=True, threshold=0.03):
    pos = set()
    lenA = topkA.shape[0]
    for e1 in range(lenA):
        e2 = topkA[e1][0].item()
        if e1 == topkB[e2][0].item():
            e2_ = topkA[e1][1]
            e1_ = topkB[e2][1]
            score1 = (sim_score[e1][e2] - sim_score[e1][e2_]).item()
            score2 = (sim_score[e1][e2] - sim_score[e1_][e2]).item()

            if score1 >= threshold and score2 >= threshold:
                pos.add((e1, e2, 1))

    neg = negative_sample(pos, topkA, topkB, hard_sample)
    return pos, neg


def negative_sample(pos, topkA, topkB, hard_sample=True):
    neg = set()
    lenA = topkA.shape[0]
    lenB = topkB.shape[0]
    for seed in pos:
        e1, e2, label = seed
        if hard_sample:
            # (2, 10)
            for i in range(left, right):
                if (e1, topkA[e1][i].item(), 1) not in pos:
                    neg.add((e1, topkA[e1][i].item(), 0))
                if (topkB[e2][i].item(), e2, 1) not in pos:
                    neg.add((topkB[e2][i].item(), e2, 0))
        else:
            for i in range(8):
                e = np.random.randint(lenB)
                while e == e2 or (e1, e, 1) in pos:
                    e = np.random.randint(lenB)
                neg.add((e1, e, 0))
                e = np.random.randint(lenA)
                while e == e1 or (e, e2, 1) in pos:
                    e = np.random.randint(lenA)
                neg.add((e, e2, 0))
    return neg


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

    if not os.path.exists('./log'):
        os.mkdir('./log')

    if not os.path.exists('./checkpoint'):
        os.mkdir('./checkpoint')

    configs = json.load(open('configs.json'))
    configs = {conf['name']: conf for conf in configs}
    config = configs[args.data_name]
    model_name = config['model']
    max_length = int(config['max_length'])
    data_name = args.data_name
    left = args.left
    right = args.right
    model = SentenceTransformer(model_name)

    cur_time = '_' + datetime.now().strftime('%F %T')

    set_logger(data_name.replace('/', '') + '_seeds_' + cur_time)

    logging.info(args)
    if args.hard_sample:
        logging.info('Hard sample.')

    logging.info('Seed: {}'.format(torch.initial_seed()))

    logging.info('Left: {}, Right: {}'.format(left, right))

    path = config['path']
    vecA = os.path.join(path, 'vecA.npy')
    vecB = os.path.join(path, 'vecB.npy')

    logging.info('Compute embedding...')
    tableA = os.path.join(path, 'tableA.csv')
    tableB = os.path.join(path, 'tableB.csv')
    entityA = read_entity(tableA, skip=True, add_token=True)
    entityB = read_entity(tableB, skip=True, add_token=True)

    # Encode
    embeddingA = model.encode(entityA, batch_size=512)
    embeddingB = model.encode(entityB, batch_size=512)

    # Norm
    embeddingA = [v / np.linalg.norm(v) for v in embeddingA]
    embeddingB = [v / np.linalg.norm(v) for v in embeddingB]

    np.save(vecA, embeddingA)
    np.save(vecB, embeddingB)

    t1 = time.time()

    embeddingA = torch.tensor(embeddingA).cuda()
    embeddingB = torch.tensor(embeddingB).cuda()
    sim_score = util.pytorch_cos_sim(embeddingA, embeddingB)
    distA, topkA = torch.topk(sim_score, k=30, dim=1)
    distB, topkB = torch.topk(sim_score, k=30, dim=0)
    topkB = topkB.t()

    logging.info('Time: {:.4f}.'.format(time.time() - t1))

    pos_seeds, neg_seeds = sample(topkA, topkB, sim_score, args.hard_sample)

    logging.info('Num positive seeds: {}.'.format(len(pos_seeds)))
    logging.info('Num negative seeds: {}.'.format(len(neg_seeds)))

    seeds = pos_seeds | neg_seeds
    seeds = list(seeds)

    logging.info('Num seeds: {}'.format(len(seeds)))

    seeds_path = os.path.join(path, 'seeds.csv')
    seeds_writer = csv.writer(open(seeds_path, 'w'))
    seeds_writer.writerow(['ltable_id', 'rtable_id', 'label'])
    seeds_writer.writerows(seeds)
