import torch
import os
import json
import random
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from load_data import LoadData
from gnn_model import MultiLayerGCN, AttSeq
from torch.optim import Adagrad
from utils import print_time_info


def cosine_similarity_nbyn(a, b):
    """
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    """
    a = a / torch.clamp(a.norm(dim=-1, keepdim=True, p=2), min=1e-10)
    b = b / torch.clamp(b.norm(dim=-1, keepdim=True, p=2), min=1e-10)
    if b.shape[0] * b.shape[1] > 20000 * 128:
        return cosine_similarity_nbyn_batched(a, b)
    return torch.mm(a, b.t())


def cosine_similarity_nbyn_batched(a, b):
    """
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    """
    batch_size = 512
    data_num = b.shape[0]
    b = b.t()
    sim_matrix = []
    for i in range(0, data_num, batch_size):
        sim_matrix.append(torch.mm(a, b[:, i:i + batch_size]).cpu())
    sim_matrix = torch.cat(sim_matrix, dim=1)
    return sim_matrix


def torch_l2distance(a, b):
    # shape a = (num_ent1, embed_dim)
    # shape b = (num_ent2, embed_dim)
    assert len(a.size()) == len(b.size()) == 2
    assert a.shape[1] == b.shape[1]
    x1 = torch.sum(torch.pow(a, 2), dim=-1).view(-1, 1)  # shape = (num_ent1, 1)
    x2 = torch.sum(torch.pow(b, 2), dim=-1).view(-1, 1)  # shape = (num_ent2, 1)
    if b.shape[0] < 20000:
        x3 = -2 * torch.mm(a, b.t())  # shape = (num_ent1, num_ent2)
    else:
        x3 = -2 * torch_mm_batched(a, b.t())
    is_cuda = x3.is_cuda
    if not is_cuda:
        x1 = x1.cpu()
        x2 = x2.cpu()

    sim = x3 + x1 + x2.t()
    return sim.pow(0.5)


def torch_mm_batched(a, b):
    """
    a shape: [dim1, dim2]
    b shape: [dim2, dim3]
    return sim_matrix: [dim1, dim3]
    """
    batch_size = 512
    cols_num = b.shape[-1]
    output = []
    for i in range(0, cols_num, batch_size):
        output.append(torch.mm(a, b[:, i:i + batch_size]).cpu())
    output = torch.cat(output, dim=1)
    return output


def get_nearest_neighbor(sim, nega_sample_num=25):
    # Sim do not have to be a square matrix
    # Let us assume sim is a numpy array
    ranks = torch.argsort(sim, dim=1)
    ranks = ranks[:, 1:nega_sample_num + 1]
    return ranks


class AlignLoss(nn.Module):
    def __init__(self, margin, p=2, reduction='mean'):
        super(AlignLoss, self).__init__()
        self.p = p
        self.criterion = nn.TripletMarginLoss(margin, p=p, reduction=reduction)

    def forward(self, repre_sr, repre_tg):
        """
        score shape: [batch_size, 2, embedding_dim]
        """
        # distance = torch.abs(score).sum(dim=-1) * self.re_scale
        sr_true = repre_sr[:, 0, :]
        sr_nega = repre_sr[:, 1, :]
        tg_true = repre_tg[:, 0, :]
        tg_nega = repre_tg[:, 1, :]

        loss = self.criterion(torch.cat((sr_true, tg_true), dim=0), torch.cat((tg_true, sr_true), dim=0),
                              torch.cat((tg_nega, sr_nega), dim=0))
        return loss


def sort_and_keep_indices(matrix, device):
    batch_size = 512
    data_len = matrix.shape[0]
    sim_matrix = []
    indice_list = []
    for i in range(0, data_len, batch_size):
        batch = matrix[i:i + batch_size]
        batch = torch.from_numpy(batch).to(device)
        sorted_batch, indices = torch.sort(batch, dim=-1)
        sorted_batch = sorted_batch[:, :500].cpu()
        indices = indices[:, :500].cpu()
        sim_matrix.append(sorted_batch)
        indice_list.append(indices)
    sim_matrix = torch.cat(sim_matrix, dim=0).numpy()
    indice_array = torch.cat(indice_list, dim=0).numpy()
    sim = np.concatenate([np.expand_dims(sim_matrix, 0), np.expand_dims(indice_array, 0)], axis=0)
    return sim


class GNNChannel(nn.Module):

    def __init__(self, ent_num_sr, ent_num_tg, dim, layer_num, drop_out, channels):
        super(GNNChannel, self).__init__()
        assert len(channels) == 1
        if 'structure' in channels:
            self.gnn = StruGNN(ent_num_sr, ent_num_tg, dim, layer_num, drop_out, **channels['structure'])
        if 'attribute' in channels:
            self.gnn = AttSeq(layer_num, ent_num_sr, ent_num_tg, dim, drop_out, residual=True, **channels['attribute'])
        if 'name' in channels:
            self.gnn = NameGCN(dim, layer_num, drop_out, **channels['name'])

    def forward(self, sr_ent_seeds, tg_ent_seeds):
        sr_seed_hid, tg_seed_hid, sr_ent_hid, tg_ent_hid = self.gnn.forward(sr_ent_seeds, tg_ent_seeds)
        return sr_seed_hid, tg_seed_hid, sr_ent_hid, tg_ent_hid

    def predict(self, sr_ent_seeds, tg_ent_seeds):
        with torch.no_grad():
            sr_seed_hid, tg_seed_hid, _, _ = self.forward(sr_ent_seeds, tg_ent_seeds)
            if isinstance(self.gnn, NameGCN):
                sim = torch_l2distance(sr_seed_hid, tg_seed_hid)
            else:
                sim = - cosine_similarity_nbyn(sr_seed_hid, tg_seed_hid)
        return sim

    def negative_sample(self, sr_ent_seeds, tg_ent_seeds):
        with torch.no_grad():
            sr_seed_hid, tg_seed_hid, sr_ent_hid, tg_ent_hid = self.forward(sr_ent_seeds, tg_ent_seeds)
            if isinstance(self.gnn, NameGCN):
                sim_sr = torch_l2distance(sr_seed_hid, sr_ent_hid)
                sim_tg = torch_l2distance(tg_seed_hid, tg_ent_hid)
            else:
                sim_sr = - cosine_similarity_nbyn(sr_seed_hid, sr_ent_hid)
                sim_tg = - cosine_similarity_nbyn(tg_seed_hid, tg_ent_hid)
        return sim_sr, sim_tg


class NameGCN(nn.Module):
    def __init__(self, dim, layer_num, drop_out, sr_ent_embed, tg_ent_embed, edges_sr, edges_tg):
        super(NameGCN, self).__init__()
        self.embedding_sr = nn.Parameter(sr_ent_embed, requires_grad=False)
        self.embedding_tg = nn.Parameter(tg_ent_embed, requires_grad=False)
        self.edges_sr = nn.Parameter(edges_sr, requires_grad=False)
        self.edges_tg = nn.Parameter(edges_tg, requires_grad=False)
        in_dim = sr_ent_embed.shape[1]
        self.gcn = MultiLayerGCN(in_dim, dim, layer_num, drop_out, featureless=False, residual=True)

    def forward(self, sr_ent_seeds, tg_ent_seeds):
        sr_ent_hid = self.gcn(self.edges_sr, self.embedding_sr)
        tg_ent_hid = self.gcn(self.edges_tg, self.embedding_tg)
        sr_seed_hid = sr_ent_hid[sr_ent_seeds]
        tg_seed_hid = tg_ent_hid[tg_ent_seeds]
        return sr_seed_hid, tg_seed_hid, sr_ent_hid, tg_ent_hid


class StruGNN(nn.Module):
    def __init__(self, ent_num_sr, ent_num_tg, dim, layer_num, drop_out, edges_sr, edges_tg):
        super(StruGNN, self).__init__()
        # self.feats_sr = nn.Parameter(self.prepare_entity_feats(ent_num_sr, edges_sr), requires_grad=False)
        # self.feats_tg = nn.Parameter(self.prepare_entity_feats(ent_num_tg, edges_tg), requires_grad=False)
        embedding_weight = torch.zeros((ent_num_sr + ent_num_tg, dim), dtype=torch.float)
        nn.init.xavier_uniform_(embedding_weight)
        self.feats_sr = nn.Parameter(embedding_weight[:ent_num_sr], requires_grad=True)
        self.feats_tg = nn.Parameter(embedding_weight[ent_num_sr:], requires_grad=True)
        self.edges_sr = nn.Parameter(edges_sr, requires_grad=False)
        self.edges_tg = nn.Parameter(edges_tg, requires_grad=False)
        assert len(self.feats_sr) == ent_num_sr
        assert len(self.feats_tg) == ent_num_tg
        self.gcn = MultiLayerGCN(self.feats_sr.shape[-1], dim, layer_num, drop_out, featureless=True, residual=False)

    def forward(self, sr_ent_seeds, tg_ent_seeds):
        sr_ent_hid = self.gcn(self.edges_sr, self.feats_sr)
        tg_ent_hid = self.gcn(self.edges_tg, self.feats_tg)
        sr_ent_hid = F.normalize(sr_ent_hid, p=2, dim=-1)
        tg_ent_hid = F.normalize(tg_ent_hid, p=2, dim=-1)
        sr_seed_hid = sr_ent_hid[sr_ent_seeds]
        tg_seed_hid = tg_ent_hid[tg_ent_seeds]
        return sr_seed_hid, tg_seed_hid, sr_ent_hid, tg_ent_hid


class AttConf(object):

    def __init__(self):
        self.train_seeds_ratio = 1.0
        self.dim = 128
        self.drop_out = 0.0
        self.layer_num = 1
        self.epoch_num = 100
        self.nega_sample_freq = 5
        self.nega_sample_num = 25

        self.learning_rate = 0.001
        self.l2_regularization = 1e-2
        self.margin_gamma = 1.0

        self.log_comment = "comment"

        self.structure_channel = False
        self.name_channel = False
        self.attribute_value_channel = False
        self.literal_attribute_channel = False
        self.digit_attribute_channel = False

    def set_channel(self, channel_name):
        if channel_name == 'Literal':
            self.set_literal_attribute_channel(True)
        elif channel_name == 'Digital':
            self.set_digit_attribute_channel(True)
        elif channel_name == 'Attribute':
            self.set_attribute_value_channel(True)
        elif channel_name == 'Structure':
            self.set_structure_channel(True)
        elif channel_name == 'Name':
            self.set_name_channel(True)
        else:
            raise Exception()

    def set_epoch_num(self, epoch_num):
        self.epoch_num = epoch_num

    def set_nega_sample_num(self, nega_sample_num):
        self.nega_sample_num = nega_sample_num

    def set_log_comment(self, log_comment):
        self.log_comment = log_comment

    def set_name_channel(self, use_name_channel):
        self.name_channel = use_name_channel

    def set_digit_attribute_channel(self, use_digit_attribute_channel):
        self.digit_attribute_channel = use_digit_attribute_channel

    def set_literal_attribute_channel(self, use_literal_attribute_channel):
        self.literal_attribute_channel = use_literal_attribute_channel

    def set_attribute_value_channel(self, use_attribute_value_channel):
        self.attribute_value_channel = use_attribute_value_channel

    def set_structure_channel(self, use_structure_channel):
        self.structure_channel = use_structure_channel

    def set_drop_out(self, drop_out):
        self.drop_out = drop_out

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_l2_regularization(self, l2_regularization):
        self.l2_regularization = l2_regularization

    def init(self, directory, device):
        self.directory = Path(directory)
        self.loaded_data = LoadData(self.train_seeds_ratio, self.directory, self.nega_sample_num,
                                    name_channel=self.name_channel, attribute_channel=self.attribute_value_channel,
                                    digit_literal_channel=self.digit_attribute_channel or self.literal_attribute_channel,
                                    device=device)
        self.sr_ent_num = self.loaded_data.sr_ent_num
        self.tg_ent_num = self.loaded_data.tg_ent_num
        self.att_num = self.loaded_data.att_num

        # Init graph adjacent matrix
        print_time_info('Begin preprocessing adjacent matrix')
        self.channels = {}

        edges_sr = torch.tensor(self.loaded_data.triples_sr)[:, :2]
        edges_tg = torch.tensor(self.loaded_data.triples_tg)[:, :2]
        edges_sr = torch.unique(edges_sr, dim=0)
        edges_tg = torch.unique(edges_tg, dim=0)

        if self.name_channel:
            self.channels['name'] = {'edges_sr': edges_sr, 'edges_tg': edges_tg,
                                     'sr_ent_embed': self.loaded_data.sr_embed,
                                     'tg_ent_embed': self.loaded_data.tg_embed, }
        if self.structure_channel:
            self.channels['structure'] = {'edges_sr': edges_sr, 'edges_tg': edges_tg}
        if self.attribute_value_channel:
            self.channels['attribute'] = {'edges_sr': edges_sr, 'edges_tg': edges_tg,
                                          'att_num': self.loaded_data.att_num,
                                          'attribute_triples_sr': self.loaded_data.attribute_triples_sr,
                                          'attribute_triples_tg': self.loaded_data.attribute_triples_tg,
                                          'value_embedding': self.loaded_data.value_embedding}
        if self.literal_attribute_channel:
            self.channels['attribute'] = {'edges_sr': edges_sr, 'edges_tg': edges_tg,
                                          'att_num': self.loaded_data.literal_att_num,
                                          'attribute_triples_sr': self.loaded_data.literal_triples_sr,
                                          'attribute_triples_tg': self.loaded_data.literal_triples_tg,
                                          'value_embedding': self.loaded_data.literal_value_embedding}
        if self.digit_attribute_channel:
            self.channels['attribute'] = {'edges_sr': edges_sr, 'edges_tg': edges_tg,
                                          'att_num': self.loaded_data.digit_att_num,
                                          'attribute_triples_sr': self.loaded_data.digital_triples_sr,
                                          'attribute_triples_tg': self.loaded_data.digital_triples_tg,
                                          'value_embedding': self.loaded_data.digit_value_embedding}
        print_time_info('Finished preprocesssing adjacent matrix')

    def train(self, device):
        self.loaded_data.negative_sample()
        # Compose Graph NN
        gnn_channel = GNNChannel(self.sr_ent_num, self.tg_ent_num, self.dim, self.layer_num, self.drop_out,
                                 self.channels)
        self.gnn_channel = gnn_channel
        gnn_channel.to(device)
        gnn_channel.train()

        # Prepare optimizer
        optimizer = Adagrad(filter(lambda p: p.requires_grad, gnn_channel.parameters()), lr=self.learning_rate,
                            weight_decay=self.l2_regularization)
        criterion = AlignLoss(self.margin_gamma)

        for epoch_num in range(1, self.epoch_num + 1):
            gnn_channel.train()
            optimizer.zero_grad()
            sr_seed_hid, tg_seed_hid, _, _ = gnn_channel.forward(self.loaded_data.train_sr_ent_seeds,
                                                                 self.loaded_data.train_tg_ent_seeds)
            loss = criterion(sr_seed_hid, tg_seed_hid)
            loss.backward()
            optimizer.step()
            if epoch_num % self.nega_sample_freq == 0:
                self.negative_sample()

    def negative_sample(self, ):
        sim_sr, sim_tg = self.gnn_channel.negative_sample(self.loaded_data.train_sr_ent_seeds_ori,
                                                          self.loaded_data.train_tg_ent_seeds_ori)
        sr_nns = get_nearest_neighbor(sim_sr, self.nega_sample_num)
        tg_nns = get_nearest_neighbor(sim_tg, self.nega_sample_num)
        self.loaded_data.update_negative_sample(sr_nns, tg_nns)

    def save_embedding(self, data_path, name, lenA, lenB):
        print('Save embedding...')
        self.gnn_channel.eval()
        sr_ent_seeds = []
        tg_ent_seeds = []
        for i in range(lenA):
            sr_ent_seeds.append(i)
        for i in range(lenB):
            tg_ent_seeds.append(i)
        sr_ent_seeds = torch.tensor(sr_ent_seeds).cuda()
        tg_ent_seeds = torch.tensor(tg_ent_seeds).cuda()
        sr_ent_seeds, tg_ent_seeds, _, __ = self.gnn_channel(sr_ent_seeds, tg_ent_seeds)
        sr_ent_seeds = sr_ent_seeds.cpu().detach().numpy()
        tg_ent_seeds = tg_ent_seeds.cpu().detach().numpy()

        pathA = os.path.join(data_path, name + '_' + 'embeddingA.npy')
        pathB = os.path.join(data_path, name + '_' + 'embeddingB.npy')
        np.save(pathA, sr_ent_seeds)
        np.save(pathB, tg_ent_seeds)


def grid_search(log_comment, data_set, layer_num, device):
    att_conf = AttConf()
    att_conf.set_channel(log_comment)
    att_conf.set_epoch_num(100)
    att_conf.set_nega_sample_num(25)
    att_conf.layer_num = layer_num
    att_conf.set_log_comment(log_comment)
    data_path = './data/ER-Magellan/' + data_set
    data_name = data_set
    att_conf.init(data_path, device)

    best_parameter = (1e-4, 1e-3)
    if not os.path.exists('./cache_log'):
        os.mkdir('./cache_log')


    att_conf.set_learning_rate(best_parameter[0])
    att_conf.set_l2_regularization(best_parameter[1])

    att_conf.train(device)

    configs = json.load(open('./configs.json'))
    configs = {conf['name']: conf for conf in configs}
    config = configs[data_name]
    lenA = config['lenA']
    lenB = config['lenB']
    att_conf.save_embedding(data_path, log_comment, lenA, lenB)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', type=str, default='all')
    parser.add_argument('--dataset', type=str, default='Structured/Fodors-Zagats')
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('--seed', default=2021, type=int)
    args = parser.parse_args()

    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    print('Seed: {}'.format(torch.initial_seed()))

    device = 'cuda:0'
    if args.channel == 'all':
        grid_search('Literal', args.dataset, args.layer_num, device)
        grid_search('Digital', args.dataset, args.layer_num, device)
        grid_search('Structure', args.dataset, args.layer_num, device)
        grid_search('Name', args.dataset, args.layer_num, device)
    else:
        grid_search(args.channel, args.dataset, args.layer_num, device)
