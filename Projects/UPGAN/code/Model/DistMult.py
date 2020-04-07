from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
import math

def build_model(args, user_total, item_total, entity_total, relation_total, i_map=None, e_map=None, new_map=None, share_total=0):
    model_cls = DistMult
    return model_cls(entity_total,
                     relation_total,
                     args)


class DistMult(torch.nn.Module):
    def __init__(self, entity_total, relation_total, args):
        super(DistMult, self).__init__()
        self.entity_total = entity_total
        self.relation_total = relation_total

        self.embedding_size = args.embedding_size
        self.beta = 1.0

        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.emb_def()
        self.device = torch.device('cuda' if args.use_cuda else 'cpu')
        self = self.to(self.device)


    def emb_def(self):
        self.ent_embeddings = torch.nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = torch.nn.Embedding(2 *self.relation_total, self.embedding_size)
        xavier_normal_(self.ent_embeddings.weight.data)
        xavier_normal_(self.rel_embeddings.weight.data)

    def encode_kg(self, ent_emb, rel_emb):
        return ent_emb * rel_emb

    def forward(self, e1, rel):
        e1_embedded_all = self.get_emb_all()
        ent_emb = e1_embedded_all[e1]
        rel_emb = self.rel_embeddings(rel)
        ent_emb = self.inp_drop(ent_emb)
        rel_emb = self.inp_drop(rel_emb)

        hr_encoded = self.encode_kg(ent_emb, rel_emb)
        x = torch.mm(hr_encoded, e1_embedded_all.transpose(1, 0))
        #pred = torch.sigmoid(x)
        return x

    def get_candidates(self):
        return self.ent_embeddings.weight

    def evaluate(self, e1, rel, e1_embedded_all):
        #e1_embedded_all = self.get_emb_all()
        ent_emb = e1_embedded_all[e1]
        rel_emb = self.rel_embeddings(rel)

        hr_encoded = self.encode_kg(ent_emb, rel_emb)
        x = torch.mm(hr_encoded, e1_embedded_all.transpose(1, 0))
        # pred = torch.sigmoid(x)
        return x

    def forward_triple(self, heads, rels, tails):
        '''

        :param heads: (B, )
        :param rels:  (B, )
        :param tails: (B, N)
        :return:
        '''
        batch_size = heads.size(0)
        num_sample = tails.size(1)
        e1_embedded_all = self.get_emb_all()
        head_emb = e1_embedded_all[heads]
        rel_emb = self.rel_embeddings(rels)
        tail_emb = e1_embedded_all[tails].view(batch_size, num_sample, self.embedding_size)
        hr_encoded = self.encode_kg(head_emb, rel_emb).view(batch_size, self.embedding_size, 1)
        scores = torch.bmm(tail_emb, hr_encoded).squeeze()#(B, N, emb) (B, emb, 1) - > (B, N, 1)
        return scores

    def forward_batch(self, heads, rels, tails, user_rep=None):
        '''

        :param heads: (B,)
        :param rels: (B,)
        :param tails: (B.)
        :param e1_embedded_user:(B, emb)
        :return:
        '''
        batch_size = heads.size(0)
        if len(tails.size()) == 1:
            num_sample = 1
        else:
            num_sample = tails.size(1)
        ent_embedded_all = self.ent_embeddings.weight
        rel_emb = self.rel_embeddings(rels)
        head_emb = ent_embedded_all[heads]
        tail_emb = ent_embedded_all[tails].view(batch_size, num_sample, self.embedding_size)

        head_emb = self.inp_drop(head_emb)
        rel_emb = self.inp_drop(rel_emb)

        query_rs = self.encode_kg(head_emb, rel_emb).view(batch_size, self.embedding_size, 1)
        pred = torch.bmm(tail_emb, query_rs).view(batch_size, num_sample)#(B, N, e) (B, e, 1)
        return pred


    def get_emb_all(self):
        return self.ent_embeddings.weight

    def forward_kg(self, e1, rel, all_e_ids=None):
        return self.forward(e1, rel)