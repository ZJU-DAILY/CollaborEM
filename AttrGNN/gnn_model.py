import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, featureless=False, act_func=F.relu, residual=True):
        super(GraphConvolution, self).__init__()
        self.act_func = act_func
        self.residual = residual
        self.featureless = featureless
        if self.residual and input_dim != output_dim:
            self.root = nn.Linear(input_dim, output_dim, False)
            nn.init.xavier_uniform_(self.root.weight)
        if not self.featureless:
            self.linear = nn.Linear(input_dim, output_dim)
            nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, adj, feats):
        to_feats = torch.sparse.mm(adj, feats)
        degree = torch.sparse.sum(adj, dim=1).to_dense().reshape(-1, 1)
        to_feats = to_feats / degree
        if not self.featureless:
            to_feats = self.linear(to_feats)
        to_feats = self.act_func(to_feats)
        if self.residual:
            if feats.shape[-1] != to_feats.shape[-1]:
                to_feats = self.root(feats) + to_feats
            else:
                to_feats = feats + to_feats
        return to_feats


class MultiLayerGCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layer, dropout_rate=0.5, featureless=True, residual=False):
        super(MultiLayerGCN, self).__init__()
        self.residual = residual
        self.dropout = nn.Dropout(dropout_rate)
        self.gcn_list = nn.ModuleList()
        assert num_layer >= 1
        dim = in_dim
        for i in range(num_layer - 1):
            if i == 0:
                self.gcn_list.append(GraphConvolution(dim, out_dim, featureless, residual=residual))
                dim = out_dim
            else:
                self.gcn_list.append(GraphConvolution(out_dim, out_dim, False, residual=residual))
        self.gcn_list.append(
            GraphConvolution(dim, out_dim, False, act_func=lambda x: x, residual=residual))

    def preprocess_adj(self, edges):
        device = next(self.parameters()).device
        edges = torch.cat((edges, edges.flip(dims=[1, ])), dim=0)  # shape=[E * 2, 2]
        adj = torch.sparse.FloatTensor(edges.transpose(0, 1), torch.ones(edges.shape[0], device=device))
        M, N = adj.shape
        assert M == N
        # add self_loop
        self_loop = torch.arange(N, device=device).reshape(-1, 1).repeat(1, 2)  # shape = [N, 2]
        self_loop = torch.sparse.FloatTensor(self_loop.transpose(0, 1),
                                             torch.ones(self_loop.shape[0], device=device))
        adj = adj + self_loop
        adj = adj.coalesce()
        torch.clamp_max_(adj._values(), 1)
        return adj

    def forward(self, edges, graph_embedding):
        adj = self.preprocess_adj(edges)
        for gcn in self.gcn_list:
            graph_embedding = self.dropout(graph_embedding)
            graph_embedding = gcn(adj, graph_embedding)
        return graph_embedding


class AttributedEncoder(nn.Module):
    def __init__(self, key_dim, val_dim):
        super(AttributedEncoder, self).__init__()
        self.a = nn.Linear(key_dim * 2, 1)
        nn.init.xavier_uniform_(self.a.weight)
        self.W = nn.Parameter(torch.zeros(key_dim + val_dim, key_dim))
        nn.init.xavier_uniform_(self.W)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)  # For attention scores

    def forward(self, attribute_triples, att_feats, val_feats, ent_feats):
        # fixme: consider not use norisy attribute if all the attribute are noisy
        # fixme: consider share the attribute importance to all nodes
        N = ent_feats.shape[0]
        E = attribute_triples.shape[0]
        device = ent_feats.device
        h, val, att = attribute_triples.transpose(0, 1)  # shape=[E]

        attention_score = self.a(torch.cat((ent_feats[h], att_feats[att]), dim=-1))
        attention_score = attention_score.squeeze(-1)  # shape = [E,]
        attention_score = torch.exp(self.leaky_relu(attention_score))
        edges = torch.stack((h, torch.arange(E, device=device)), dim=0)
        incidence_matrix = torch.sparse.FloatTensor(edges, torch.ones(edges.shape[1], device=device),
                                                    size=(N, E))  # shape = [N, E]
        row_sum = torch.sparse.mm(incidence_matrix, attention_score.reshape(-1, 1)).squeeze(-1)  # shape = [N,]
        attention_p = attention_score / row_sum[h]  # shape = [E]
        att_vals = torch.cat((att_feats[att], val_feats[val]), dim=1)  # shape [E, dim1 + dim2]
        att_vals = att_vals @ self.W  # shape = [E, dim]
        # att_vals = self.W(att_vals)

        att_vals = att_vals * attention_p.reshape(-1, 1)  # shape = [E, dim]
        to_feats = torch.sparse.mm(incidence_matrix, att_vals)  # shape = [N, dim]
        to_feats = to_feats + ent_feats
        to_feats = F.elu(to_feats)
        return to_feats


class AttSeq(nn.Module):
    def __init__(self, layer_num, sr_ent_num, tg_ent_num, dim, drop_out, att_num, attribute_triples_sr,
                 attribute_triples_tg, value_embedding, edges_sr, edges_tg, residual=True):
        super(AttSeq, self).__init__()
        self.residual = residual
        # KG Feature Loading
        self.edges_sr = nn.Parameter(edges_sr, requires_grad=False)
        self.edges_tg = nn.Parameter(edges_tg, requires_grad=False)
        self.attribute_triples_sr = nn.Parameter(attribute_triples_sr, requires_grad=False)  # shape = [E1, 3]
        self.attribute_triples_tg = nn.Parameter(attribute_triples_tg, requires_grad=False)  # shape = [E2, 3]
        self.val_feats = nn.Parameter(torch.from_numpy(value_embedding), requires_grad=False)

        att_num += 1  # + 1 for unrecognized attribute
        # Initialize trainable embeddings
        embedding_weight = torch.zeros((att_num + sr_ent_num + tg_ent_num, dim), dtype=torch.float,
                                       requires_grad=False)
        nn.init.xavier_uniform_(embedding_weight)
        self.att_feats = nn.Parameter(embedding_weight[:att_num], requires_grad=True)
        self.ent_feats_sr = nn.Parameter(embedding_weight[att_num: att_num + sr_ent_num],
                                         requires_grad=True)
        self.ent_feats_tg = nn.Parameter(embedding_weight[att_num + sr_ent_num:], requires_grad=True)

        # initialize networks
        self.value_encoder = AttributedEncoder(dim, value_embedding.shape[1])
        self.gnns = nn.ModuleList()
        assert layer_num >= 1
        layer_num -= 1
        for i in range(layer_num):
            if i == layer_num - 1:
                self.gnns.append(GraphConvolution(dim, dim, featureless=False, residual=residual, act_func=lambda x: x))
            else:
                self.gnns.append(GraphConvolution(dim, dim, featureless=False, residual=residual))

    def preprocess_adj(self, edges):
        device = next(self.parameters()).device
        edges = torch.cat((edges, edges.flip(dims=[1, ])), dim=0)  # shape=[E * 2, 2]
        adj = torch.sparse.FloatTensor(edges.transpose(0, 1), torch.ones(edges.shape[0], device=device))
        M, N = adj.shape
        assert M == N

        self_loop = torch.arange(N, device=device).reshape(-1, 1).repeat(1, 2)  # shape = [N, 2]
        self_loop = torch.sparse.FloatTensor(self_loop.transpose(0, 1),
                                             torch.ones(self_loop.shape[0], device=device))
        adj = adj + self_loop

        adj = adj.coalesce()
        torch.clamp_max_(adj._values(), 1)
        return adj

    def forward(self, ent_seed_sr, ent_seed_tg):
        ent_feats_sr = self.value_encoder(self.attribute_triples_sr, self.att_feats, self.val_feats, self.ent_feats_sr)
        ent_feats_tg = self.value_encoder(self.attribute_triples_tg, self.att_feats, self.val_feats, self.ent_feats_tg)

        if 'adj_sr' not in self.__dict__:
            self.adj_tg = self.preprocess_adj(self.edges_tg)
            self.adj_sr = self.preprocess_adj(self.edges_sr)

        for gnn in self.gnns:
            ent_feats_sr = gnn(self.adj_sr, ent_feats_sr)
            ent_feats_tg = gnn(self.adj_tg, ent_feats_tg)

        ent_feats_sr = F.normalize(ent_feats_sr, p=2, dim=-1)
        ent_feats_tg = F.normalize(ent_feats_tg, p=2, dim=-1)
        sr_seed_feats = ent_feats_sr[ent_seed_sr]
        tg_seed_feats = ent_feats_tg[ent_seed_tg]
        return sr_seed_feats, tg_seed_feats, ent_feats_sr, ent_feats_tg
