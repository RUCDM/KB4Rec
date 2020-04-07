import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse


class KGGraphConvolutionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, match_matrix, devidce):
        super(KGGraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.original_map = torch.nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_normal_(self.original_map.weight, gain=1.414)
        self.match_matrix = match_matrix
        self.device = devidce

    def get_weight(self):
        return self.original_map.weight

    def forward(self, input, tail_rep, rel_Adj, activate=True):
        emb_all = self.original_map(input)
        if rel_Adj is not None:
            for rel in rel_Adj:
                edge_list, weight_list, head_indice = rel_Adj[rel]
                ent_indice = torch.LongTensor(head_indice).to(self.device)
                rel_update = self.rel_update(len(head_indice), tail_rep, rel, edge_list, weight_list)
                # emb_all[ent_indice] += rel_update
                emb_all[ent_indice] = rel_update + emb_all[ent_indice]
        if activate:
            return F.relu(emb_all)
        return emb_all

    def rel_update(self, B, tail_rep, rel, edge_list, weight_list):
        assert not torch.isnan(tail_rep).any()
        N = tail_rep.size(0)

        rel_ind = torch.LongTensor([rel]).to(self.device)
        rel_matrix = self.match_matrix(rel_ind).view(self.in_features, self.out_features)

        all_rep_h = torch.mm(tail_rep, rel_matrix)
        assert not torch.isnan(all_rep_h).any()

        edge = torch.LongTensor(edge_list).to(self.device)
        edge_weight = torch.FloatTensor(weight_list).to(self.device)

        sp_edge_e = torch.sparse_coo_tensor(edge, edge_weight, size=(B, N))  # No grad

        new_rep = torch.sparse.mm(sp_edge_e, all_rep_h)

        #new_rep = F.relu(new_rep)

        return new_rep

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class KGGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, match_matrix, devidce, args):
        super(KGGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.match_matrix = match_matrix
        self.alpha = args.alpha
        self.device = devidce

        self.attn = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))  # 2F'
        nn.init.xavier_normal_(self.attn, gain=1.414)

        self.dropout = nn.Dropout(args.input_dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def calc_score(self, head_rep, tail_rep, edge_list):
        B = head_rep.size(0)
        N = tail_rep.size(0)

        ent_indices_head = torch.LongTensor(edge_list[0]).to(self.device)
        ent_indices_tail = torch.LongTensor(edge_list[1]).to(self.device)
        # rel_indices = torch.LongTensor(rel_list).to(self.device)
        edge = torch.LongTensor(edge_list).to(self.device)

        tail_feature_ent = tail_rep[ent_indices_tail]
        head_feature_ent = head_rep[ent_indices_head]
        edge_feature = torch.cat([head_feature_ent, tail_feature_ent], dim=1)
        atten_vec_batch = self.attn
        edge_score = torch.sum(atten_vec_batch * edge_feature, dim=1)
        # edge_socre = self.dropout(edge_socre)
        edge_score = torch.clamp(edge_score, -10.0, 10.0)

        edge_e = torch.exp(self.leakyrelu(edge_score))
        return edge_e

    def calc_attn_kg(self, head_rep, tail_rep, edge_list, requires_grad=False):
        score_rel = {}
        for rel in edge_list:
            rel_edge_list = edge_list[rel]
            rel_matrix_now = self.match_matrix.weight[rel].view(self.in_features, self.out_features)
            rh_rep = torch.mm(head_rep, rel_matrix_now)
            rt_rep = torch.mm(tail_rep, rel_matrix_now)
            score_rel[rel] = self.calc_score(rh_rep, rt_rep, rel_edge_list)
        return score_rel

    def calc_attn_rs(self, head_rep, tail_rep, edge_list, rs_rel):
        rel_now = self.match_matrix.weight[rs_rel].view(self.in_features, self.out_features)
        rh_rep = torch.mm(head_rep, rel_now)
        rt_rep = torch.mm(tail_rep, rel_now)
        score_rel = self.calc_score(rh_rep, rt_rep, edge_list)
        return score_rel

    def forward(self, head_rep, tail_rep, tail_val, edge_list, rel_list, requires_grad=False):
        assert not torch.isnan(head_rep).any()
        assert not torch.isnan(tail_rep).any()
        assert not torch.isnan(tail_val).any()
        B = head_rep.size(0)
        N = tail_rep.size(0)

        ent_indices_head = torch.LongTensor(edge_list[0]).to(self.device)
        ent_indices_tail = torch.LongTensor(edge_list[1]).to(self.device)
        #rel_indices = torch.LongTensor(rel_list).to(self.device)
        edge = torch.LongTensor(edge_list).to(self.device)

        tail_feature_ent = tail_rep[ent_indices_tail]
        head_feature_ent = head_rep[ent_indices_head]
        edge_feature = torch.cat([head_feature_ent, tail_feature_ent], dim=1)
        atten_vec_batch = self.attn
        edge_score = torch.sum(atten_vec_batch * edge_feature, dim=1)
        #edge_socre = self.dropout(edge_socre)
        edge_score = torch.clamp(edge_score, -10.0, 10.0)

        edge_e = torch.exp(self.leakyrelu(edge_score))

        assert not torch.isnan(edge_e).any()

        sp_edge_e = torch.sparse_coo_tensor(edge, edge_e, size=(B, N), requires_grad=requires_grad)
        e_rowsum = torch.sparse.mm(sp_edge_e, torch.ones(size=(N, 1)).to(self.device))

        h_prime = torch.sparse.mm(sp_edge_e, tail_val)  # entity_rep_all
        assert not torch.isnan(h_prime).any()
        return e_rowsum, h_prime

    def forward_kg(self, head_rep, tail_rep, edge_list, requires_grad=False):
        tp_rep = None
        e_rowsum = None
        for rel in edge_list:
            rel_edge_list = edge_list[rel]
            rel_list = [rel] * len(rel_edge_list[0])
            rel_matrix_now = self.match_matrix.weight[rel].view(self.in_features, self.out_features)
            rh_rep = torch.mm(head_rep, rel_matrix_now)
            rt_rep = torch.mm(tail_rep, rel_matrix_now)
            tail_val = tail_rep
            e_rowsum_tp, h_prime_tp = self.forward(rh_rep, rt_rep, tail_val,
                                                   rel_edge_list, rel_list, requires_grad=requires_grad)
            if e_rowsum is None:
                e_rowsum = e_rowsum_tp
            else:
                # e_rowsum += e_rowsum_tp
                e_rowsum = e_rowsum + e_rowsum_tp
            if tp_rep is None:
                tp_rep = h_prime_tp
            else:
                # tp_rep += h_prime_tp
                tp_rep = tp_rep + h_prime_tp
        h_prime = tp_rep.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        return h_prime

    def forward_rs(self, head_rep, tail_rep, edge_list, rs_rel, requires_grad=False):
        rel_list = [rs_rel] * len(edge_list[0])
        rel_now = self.match_matrix.weight[rs_rel].view(self.in_features, self.out_features)
        rh_rep = torch.mm(head_rep, rel_now)
        rt_rep = torch.mm(tail_rep, rel_now)
        tail_val = tail_rep
        e_rowsum_tp, h_prime_tp = self.forward(rh_rep, rt_rep, tail_val,
                                               edge_list, rel_list, requires_grad=requires_grad)
        h_prime = h_prime_tp.div(e_rowsum_tp)
        assert not torch.isnan(h_prime).any()
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'