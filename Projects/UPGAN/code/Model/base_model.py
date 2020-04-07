from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def set_edges(self, edge_list_kg, edge_list_rs, remapped_entity_list, unreach_ents, rgcn_kenel):
        self.edge_list_kg = edge_list_kg
        self.edge_list_rs = edge_list_rs
        self.hop = max(self.edge_list_kg.keys()) + 1
        self.order = torch.LongTensor(remapped_entity_list).to(self.device)
        self.unreach = torch.LongTensor(unreach_ents).to(self.device)
        self.rgcn_kernel = rgcn_kenel

    def encode_kg(self, ent_emb, rel_emb):
        pass

    def forward(self, heads, rels, tails, e1_embedded_user):
        pass

    def evaluate(self, heads, rels, batch_rel_cand):
        pass

    @staticmethod
    def l2_loss(vector):
        return torch.sum(vector ** 2) / 2

    @staticmethod
    def l3_loss(vector):
        norm_ = torch.norm(vector, p=3) ** 3
        return norm_

    def get_norm(self):
        pass

    @staticmethod
    def normLoss(embeddings, dim=1):
        norm = torch.sum(embeddings ** 2, dim=dim, keepdim=True)
        return torch.sum(torch.max(norm - torch.ones_like(norm), torch.zeros_like(norm)))

    def form_query(self, heads, rels, e1_embedded_user):
        pass

    def query_judge(self, query_now, tails):
        tail_emb = self.ent_embeddings(tails)
        if len(tail_emb.size()) == 3:
            query_now = query_now.unsqueeze(1)#(B, 1, emb)
        score = torch.sum(query_now * tail_emb, dim=-1)
        norm_tail = self.normLoss(tail_emb)
        return score, norm_tail

    def get_user_all(self, rgcn_kernel, pretrain=False):
        ent_embedded_all = self.get_ent_all()
        if pretrain:
            # user_all = self.bn_init(self.user_embeddings.weight)
            user_all = self.user_embeddings.weight
            return user_all, ent_embedded_all
        final_indices = torch.LongTensor(rgcn_kernel[self.hop]).to(self.device)
        emb_hop_list = [ent_embedded_all[final_indices]]
        for i in range(self.hop - 1, 0, -1):
            rel_edge_list, ent_now = rgcn_kernel[i]
            now_indice = torch.LongTensor(ent_now).to(self.device)
            input_rep = ent_embedded_all[now_indice]
            tail_rep = emb_hop_list[-1]
            current_rep = self.gnn_layer_1(input_rep, tail_rep, rel_edge_list)
            emb_hop_list.append(current_rep)
        rel_edge_list = rgcn_kernel[0]
        input_rep = self.user_embeddings.weight
        # input_rep = self.bn_init(self.user_embeddings.weight)
        tail_rep = emb_hop_list[-1]
        current_rep = self.gnn_layer_1(input_rep, tail_rep, rel_edge_list, activate=False)
        ent_emb_list = emb_hop_list[::-1]
        ent_user_all = torch.cat(ent_emb_list, dim=0)
        if ent_user_all.size(0) < self.entity_total:
            unreachable = self.gnn_layer_1(ent_embedded_all[self.unreach], None, None)
            ent_user_all = torch.cat([ent_user_all, unreachable], dim=0)
        return current_rep, ent_user_all[self.order]

    def get_candidates(self, pretrain=False):
        return self.fetch_user_for_ent_new(pretrain=pretrain)

    def get_ent_all(self):
        return self.ent_embeddings.weight

    def norm_emb(self):
        with torch.no_grad():
            emb = self.ent_embeddings
            # norms = torch.norm(emb.weight, p=2, dim=1, keepdim=True).data
            emb.weight.data = F.normalize(emb.weight, p=2, dim=1, eps=1.0)#Max norm 1.0
            # emb.weight.data = emb.weight.data.div(norms)

    def norm_user(self):
        with torch.no_grad():
            emb = self.user_embeddings
            # norms = torch.norm(emb.weight, p=2, dim=1, keepdim=True).data
            emb.weight.data = F.normalize(emb.weight, p=2, dim=1, eps=1.0)#Max norm 1.0
            # emb.weight.data = emb.weight.data.div(norms)

    def norm_one(self):
        with torch.no_grad():
            emb = self.ent_embeddings
            norms = torch.norm(emb.weight, p=2, dim=1, keepdim=True).data
            emb.weight.data = emb.weight.data.div(norms)

    def fetch_user_batch(self, rgcn_kernel, rs_graph, kg_graph, tp_hop, unreach_ids, batch_order, pretrain=False):
        user_all, ent_embedded_all = self.get_user_all(rgcn_kernel, pretrain)
        temp_user_list = []
        for edge_list_map, edge_list_ori, tp_ents in rs_graph:
            tp_indice = torch.LongTensor(tp_ents).to(self.device)
            tp_rep = ent_embedded_all[tp_indice]
            if torch.isnan(tp_rep).any():
                print(len(tp_ents))
                print(torch.sum(torch.isnan(tp_rep)))
            assert not torch.isnan(tp_rep).any()
            rs_rep = self.query_layer.forward_rs(tp_rep, user_all, edge_list_map,
                                                  self.relation_total * 2, requires_grad=False)
            temp_user_list.append(rs_rep)
        emb_hop_list = [torch.cat(temp_user_list, dim=0)]
        for hop in range(1, tp_hop):
            tail_rep = emb_hop_list[hop - 1]
            edge_list_hop, tp_ents = kg_graph[hop]
            tp_indice = torch.LongTensor(tp_ents).to(self.device)
            tp_rep = ent_embedded_all[tp_indice]
            hop_rep = self.query_layer.forward_kg(tp_rep, tail_rep, edge_list_hop, requires_grad=False)
            emb_hop_list.append(hop_rep)
        ent_user_all = torch.cat(emb_hop_list, dim=0)
        if len(unreach_ids) > 0:
            unreach = torch.LongTensor(unreach_ids).to(self.device)
            unreachable = ent_embedded_all[unreach].view(-1, self.embedding_size)
            ent_user_all = torch.cat([ent_user_all, unreachable], dim=0)
        batch_rep = ent_user_all[batch_order]
        return batch_rep

    def fetch_user_for_ent_new(self, rgcn_kernel=None, rs_graph=None, kg_graph=None, pretrain=False):
        if rs_graph is None:
            rs_graph = self.edge_list_rs
        if kg_graph is None:
            kg_graph = self.edge_list_kg
        if rgcn_kernel is None:
            rgcn_kernel = self.rgcn_kernel
        user_all, ent_embedded_all = self.get_user_all(rgcn_kernel, pretrain)
        temp_user_list = []
        for edge_list_map, edge_list_ori, tp_ents in rs_graph:
            tp_indice = torch.LongTensor(tp_ents).to(self.device)
            tp_rep = ent_embedded_all[tp_indice]
            rs_rep = self.query_layer.forward_rs(tp_rep, user_all, edge_list_map,
                                                  self.relation_total * 2, requires_grad=False)
            temp_user_list.append(rs_rep)
        emb_hop_list = [torch.cat(temp_user_list, dim=0)]
        for hop in range(1, self.hop):
            tail_rep = emb_hop_list[hop - 1]
            edge_list_hop, tp_ents = kg_graph[hop]
            tp_indice = torch.LongTensor(tp_ents).to(self.device)
            tp_rep = ent_embedded_all[tp_indice]
            hop_rep = self.query_layer.forward_kg(tp_rep, tail_rep, edge_list_hop, requires_grad=False)
            emb_hop_list.append(hop_rep)
        ent_user_all = torch.cat(emb_hop_list, dim=0)
        if ent_user_all.size(0) < self.entity_total:
            unreachable = ent_embedded_all[self.unreach]
            #unreachable = torch.zeros(self.entity_total - ent_user_all.size(0), self.embedding_size).to(self.device)
            ent_user_all = torch.cat([ent_user_all, unreachable], dim=0)
        return ent_user_all[self.order]
