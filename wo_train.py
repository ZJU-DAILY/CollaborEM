import argparse
import json
import time
from sentence_transformers import SentenceTransformer, util
from utils import *
from datetime import datetime


def read_ground_truth(file_path, only_test_set=True):
    train_path = os.path.join(file_path, 'train.csv')
    valid_path = os.path.join(file_path, 'valid.csv')
    test_path = os.path.join(file_path, 'test.csv')
    x = []
    y_truth = []
    with open(train_path) as train_file, open(valid_path) as valid_file, open(test_path) as test_file:
        if only_test_set:
            files = [test_file.readlines()]
        else:
            files = [train_file.readlines(), valid_file.readlines(), test_file.readlines()]
        for lines in files:
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                values = line.strip().split(',')
                x.append((int(values[0]), int(values[1])))
                y_truth.append(int(values[2]))

    return x, y_truth


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_device', type=str, default='0')

    parser.add_argument('--data_name', type=str, default='Structured/Fodors-Zagats')

    parser.add_argument('--seed', default=2021, type=int)

    args = parser.parse_args(args)

    return args


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
    model = SentenceTransformer(model_name)

    cur_time = '_' + datetime.now().strftime('%F %T')

    set_logger(data_name.replace('/', '') + '_seeds_' + cur_time)

    logging.info(args)

    logging.info('Seed: {}'.format(torch.initial_seed()))

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

    pairs, y_truth = read_ground_truth(path, only_test_set=True)


    distA, topkA = torch.topk(sim_score, k=1, dim=1)
    distB, topkB = torch.topk(sim_score, k=1, dim=0)
    topkB = topkB.t()

    y_pred = []
    for pair in pairs:
        a, b = pair
        if b in topkA[a] and a in topkB[b]:
            y_pred.append(1)
        else:
            y_pred.append(0)

    precision, recall, F1 = evaluate(y_truth, y_pred)

    logging.info('precision: {:.4f} recall: {:.4f} F1: {:.4f}'.format(precision, recall, F1))


