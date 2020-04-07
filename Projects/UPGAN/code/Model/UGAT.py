from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
import math
from Model.base_model import Model
from Model.layers import KGGraphAttentionLayer, KGGraphConvolutionLayer

def build_model(args, user_total, item_total, entity_total, relation_total, i_map=None, e_map=None, new_map=None, share_total=0):
    model_cls = UGAT
    return model_cls(user_total,
                     item_total,
                     entity_total,
                     relation_total,
                     args)


class UGAT(Model):
    def __init__(self, user_total, linked_total, entity_total, relation_total, args):
        super(Model, self).__init__()
        self.args = args
        self.user_total = user_total
        self.linked_total = linked_total
        self.entity_total = entity_total
        self.relation_total = relation_total

        self.embedding_size = args.embedding_size
        self.lambda_rs = args.lambda_rs
        self.device = torch.device('cuda' if args.use_cuda else 'cpu')
        self.emb_def()

        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.module_def(args)
        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.bn_init = torch.nn.BatchNorm1d(self.embedding_size)
        self.bn_ent = torch.nn.BatchNorm1d(self.embedding_size)
        self = self.to(self.device)

    def module_def(self, args):
        self.match_matrix = torch.nn.Embedding(self.relation_total * 2 + 1, self.embedding_size * self.embedding_size)
        xavier_normal_(self.match_matrix.weight.data)

        self.query_layer = KGGraphAttentionLayer(in_features=self.embedding_size,
                                                 out_features=self.embedding_size,
                                                 match_matrix=self.match_matrix,
                                                 devidce=self.device,
                                                 args=args)
        self.concat_map = torch.nn.Linear(self.embedding_size * 2, self.embedding_size)
        xavier_normal_(self.concat_map.weight)
        # self.ent_map = torch.nn.Linear(self.embedding_size, self.embedding_size)
        # xavier_normal_(self.ent_map.weight)
        self.gnn_layer_1 = KGGraphConvolutionLayer(in_features=self.embedding_size,
                                                   out_features=self.embedding_size,
                                                   match_matrix=self.match_matrix,
                                                   devidce=self.device)

    def emb_def(self):
        self.ent_embeddings = torch.nn.Embedding(self.entity_total, self.embedding_size)
        xavier_normal_(self.ent_embeddings.weight.data)
        self.rel_embeddings_rs = torch.nn.Embedding(2 * self.relation_total, self.embedding_size)
        xavier_normal_(self.rel_embeddings_rs.weight.data)
        self.user_embeddings = torch.nn.Embedding(self.user_total, self.embedding_size)
        xavier_normal_(self.user_embeddings.weight.data)
        self.rel_embeddings = torch.nn.Embedding(2 * self.relation_total, self.embedding_size)
        xavier_normal_(self.rel_embeddings.weight.data)

    def encode_kg(self, ent_emb, rel_emb):
        return ent_emb * rel_emb

    def forward(self, heads, rels, tails, e1_embedded_user, pretrain=False):
        ent_embedded_all = self.get_ent_all()
        # tail_emb = self.ent_map(ent_embedded_all[tails])#(B, N, emb)
        tail_emb = ent_embedded_all[tails]
        e1_embedded_user = self.inp_drop(e1_embedded_user)
        rel_emb_rs = self.inp_drop(self.rel_embeddings_rs(rels))
        query_rs = self.encode_kg(e1_embedded_user, rel_emb_rs)
        if pretrain:
            query_now = query_rs
        else:
            e1_embedded_ent = self.inp_drop(ent_embedded_all[heads])
            rel_emb = self.inp_drop(self.rel_embeddings(rels))
            query_kg = self.encode_kg(e1_embedded_ent, rel_emb)
            concat_query = torch.cat([query_rs, query_kg], dim=1)
            if self.args.use_activation:
                query_now = torch.tanh(self.concat_map(concat_query))
            else:
                query_now = self.concat_map(concat_query)

        scores = torch.sum(query_now.unsqueeze(1) * tail_emb, dim=-1)#(B, N)
        return scores

    def get_candidates(self, pretrain=False):
        return self.fetch_user_for_ent_new(pretrain=pretrain)

    def get_ent_all(self):
        return self.ent_embeddings.weight

    def evaluate(self, heads, rels, batch_rel_cand, pretrain=False):
        ent_embedded_all = self.get_ent_all()
        ent_user_all = batch_rel_cand
        e1_embedded_user = ent_user_all[heads]
        rel_emb_rs = self.rel_embeddings_rs(rels)
        query_rs = self.encode_kg(e1_embedded_user, rel_emb_rs)
        # query_rs = self.encode_kg(e1_embedded_user, rel_emb)
        if pretrain:
            query_now = query_rs
        else:
            e1_embedded_ent = ent_embedded_all[heads]
            rel_emb = self.rel_embeddings(rels)
            query_kg = self.encode_kg(e1_embedded_ent, rel_emb)
            concat_query = torch.cat([query_rs, query_kg], dim=1)#(B, dim)
            if self.args.use_activation:
                query_now = torch.tanh(self.concat_map(concat_query))
            else:
                query_now = self.concat_map(concat_query)
        # answer_emb = self.ent_map(ent_embedded_all)
        answer_emb = ent_embedded_all
        pred = torch.mm(query_now, answer_emb.transpose(1, 0))
        return pred

    def form_query(self, heads, rels, e1_embedded_user, pretrain=False):
        ent_embedded_all = self.get_ent_all()
        e1_embedded_user = self.inp_drop(e1_embedded_user)
        rel_emb_rs = self.inp_drop(self.rel_embeddings_rs(rels))
        query_rs = self.encode_kg(e1_embedded_user, rel_emb_rs)
        # rel_emb = self.inp_drop(self.rel_embeddings(rels))
        # query_rs = self.encode_kg(e1_embedded_user, rel_emb)
        tp_loss_1 = self.normLoss(e1_embedded_user) + self.normLoss(rel_emb_rs)
        if pretrain:
            return query_rs, tp_loss_1
        e1_embedded_ent = self.inp_drop(ent_embedded_all[heads])
        rel_emb = self.inp_drop(self.rel_embeddings(rels))
        query_kg = self.encode_kg(e1_embedded_ent, rel_emb)
        concat_query = torch.cat([query_rs, query_kg], dim=1)
        if self.args.use_activation:
            query_now = torch.tanh(self.concat_map(concat_query))
        else:
            query_now = self.concat_map(concat_query)
        tp_loss = self.normLoss(e1_embedded_ent) #+ self.normLoss(rel_emb)
        # tp_loss_2 = self.normLoss(query_now)
        # weight_loss = self.l2_loss(self.concat_map.weight)
        norm_q = tp_loss + tp_loss_1#+ weight_loss
        return query_now, norm_q

    def prediction(self, query_now):
        ent_embedded_all = self.get_ent_all()
        pred = torch.mm(query_now, ent_embedded_all.transpose(1, 0))
        return pred

    def query_judge(self, query_now, tails):
        ent_embedded_all = self.get_ent_all()
        # tail_emb = self.ent_map(ent_embedded_all[tails])
        tail_emb = ent_embedded_all[tails]
        # tail_emb = self.ent_embeddings(tails)
        if len(tail_emb.size()) == 3:
            query_now = query_now.unsqueeze(1)#(B, 1, emb)
        score = torch.sum(query_now * tail_emb, dim=-1)
        norm_tail = self.normLoss(tail_emb)
        return score, norm_tail

    def encode_user(self, rels, e1_embedded_user):
        rel_emb_rs = self.rel_embeddings_rs(rels)
        # rel_emb_rs = self.rel_embeddings(rels)
        query_rs = self.encode_kg(e1_embedded_user, rel_emb_rs)
        return query_rs

    def freeze_part(self):
        tp_names = ["ent_embeddings.weight", "rel_embeddings.weight"]
        for name, param in self.named_parameters():
            if name in tp_names:
                param.requires_grad = False

    def unfreeze_part(self):
        tp_names = ["ent_embeddings.weight", "rel_embeddings.weight"]
        for name, param in self.named_parameters():
            if name in tp_names:
                param.requires_grad = True