from torch.utils.data import Dataset
import torch
import time
import numpy as np
import random

def change_head(triple, entityTotal, relationTotal, train_hr_t):
    head, tail, rel = triple
    reverse_rel = rel + relationTotal
    answer_set = set()
    if tail in train_hr_t:
        if reverse_rel in train_hr_t[tail]:
            answer_set = train_hr_t[tail][reverse_rel]
    newHead = random.randrange(entityTotal)
    while newHead in answer_set:
        newHead = random.randrange(entityTotal)
    return newHead

def change_tail(triple, entityTotal, relationTotal, train_hr_t):
    head, tail, rel = triple
    #reverse_rel = rel + relationTotal
    answer_set = set()
    if head in train_hr_t:
        if rel in train_hr_t[head]:
            answer_set = train_hr_t[head][rel]
    newtail = random.randrange(entityTotal)
    while newtail in answer_set:
        newtail = random.randrange(entityTotal)
    return newtail

def calc_bern(triple_list):
    rel_ratio = {}
    rel_left = {}
    rel_right = {}
    rel_ct = {}
    for triple in triple_list:
        head, tail, rel = triple
        rel_left.setdefault(rel, set())
        rel_left[rel].add(head)
        rel_right.setdefault(rel, set())
        rel_right[rel].add(tail)
        rel_ct.setdefault(rel, 0.0)
        rel_ct[rel] += 1.0
    for rel in rel_ct:
        left_mean = rel_ct[rel] / len(rel_left[rel])
        right_mean = rel_ct[rel] / len(rel_right[rel])
        rel_ratio[rel] = right_mean / (left_mean + right_mean)
    return rel_ratio

def create_hr_t(triple_list, relation_num):
    hrt = {}
    for head, rel, tail in triple_list:
        hrt.setdefault(head, {})
        hrt.setdefault(tail, {})
        hrt[head].setdefault(rel, set())
        hrt[head][rel].add(tail)
        hrt[tail].setdefault(rel+relation_num, set())
        hrt[tail][rel+relation_num].add(head)
    return hrt

def count_pop(triple_list, entity_num):
    pop_list = [1.0] * entity_num
    for head, rel, tail in triple_list:
        pop_list[head] += 1
        pop_list[tail] += 1
    pop_list = np.log10(pop_list)
    exp_list = np.exp(pop_list)
    exp_sum = np.sum(np.exp(pop_list), axis=0)
    return exp_list / exp_sum
    # total_ct_soft = sum(pop_list)
    # for i in range(entity_num):
    #     pop_list[i] = pop_list[i] / total_ct_soft
    # return pop_list


class triple_dataset_unif(Dataset):  ###Create dataset which return triples interaction as batch
    def __init__(self, triple_list, entity_num, relation_num, args):
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.triple_list = triple_list
        self.num_sample = args.n_sample
        self.train_hrt = create_hr_t(triple_list, relation_num)
        self.pop_list = count_pop(triple_list, entity_num)
        self.importance_sample = args.importance_sample
        self.query_weight = args.query_weight
        self.num_pop = args.number_pop
        if args.sample_pop and self.num_pop < self.num_sample and self.num_pop > 0:
            self.sample_pop = True
        else:
            self.sample_pop = False
        std_label = [0] * self.num_sample + [1]
        self.std_label_ = torch.LongTensor(std_label)

    def __len__(self):
        return len(self.triple_list) * 2

    def create_neg_pop(self, head, rel, num_sample=200):
        # st = time.time()
        answer_set = set()
        if head in self.train_hrt:
            if rel in self.train_hrt[head]:
                answer_set = self.train_hrt[head][rel]
        neg_list = []
        while len(neg_list) < num_sample:
            negative_sample = np.random.choice(self.entity_num, size=num_sample * 2, p=self.pop_list)
            # negative_sample = np.random.randint(self.entity_num, size=self.num_sample * 2)
            for ent in negative_sample:
                if ent in answer_set:# or ent in neg_list
                    continue
                neg_list.append(ent)
        # print("pop", time.time() - st)
        return neg_list[:num_sample]

    def create_neg(self, head, rel, num_sample=200):
        # st = time.time()
        answer_set = set()
        if head in self.train_hrt:
            if rel in self.train_hrt[head]:
                answer_set = self.train_hrt[head][rel]
        neg_list = []
        while len(neg_list) < num_sample:
            negative_sample = np.random.randint(self.entity_num, size=num_sample * 2)
            for ent in negative_sample:
                if ent in answer_set:# or ent in neg_list
                    continue
                neg_list.append(ent)
        # print("normal", time.time() - st)
        return neg_list[:num_sample]

    def __getitem__(self, idx):
        true_idx = idx % len(self.triple_list)
        head, rel, tail = self.triple_list[true_idx]
        if self.importance_sample:
            # head_ct = len(self.train_hrt[head][rel]) + 4
            # tail_ct = len(self.train_hrt[tail][rel + self.relation_num]) + 4
            # sample_weight = head_ct + tail_ct
            # sample_weight = torch.sqrt(1 / torch.Tensor([sample_weight]))
            if true_idx == idx:
                # sample_weight = 1.0 / len(self.train_hrt[head][rel])
                sample_weight = len(self.train_hrt[head][rel]) + 4
            else:
                # sample_weight = 1.0 / len(self.train_hrt[tail][rel + self.relation_num])
                sample_weight = len(self.train_hrt[tail][rel + self.relation_num]) + 4
            # sample_weight = torch.Tensor([sample_weight])
            sample_weight = torch.sqrt(1.0 / torch.Tensor([sample_weight]))
        elif self.query_weight:
            if true_idx == idx:
                # sample_weight = 1.0 / len(self.train_hrt[head][rel])
                sample_weight = len(self.train_hrt[head][rel])
            else:
                # sample_weight = 1.0 / len(self.train_hrt[tail][rel + self.relation_num])
                sample_weight = len(self.train_hrt[tail][rel + self.relation_num])
            # sample_weight = torch.Tensor([sample_weight])
            sample_weight = torch.sqrt(1.0 / torch.Tensor([sample_weight]))
        else:
            if true_idx == idx:
                sample_weight = 1.0
            else:
                sample_weight = 1.0
            sample_weight = torch.Tensor([sample_weight])
        if true_idx == idx:
            # sample_weight = 1.0 / len(self.train_hrt[head][rel])
            if self.sample_pop:
                neg_list_pop = self.create_neg_pop(head, rel, self.num_pop)
                neg_list_random = self.create_neg(head, rel, self.num_sample - self.num_pop)
                neg_list = neg_list_pop + neg_list_random
            else:
                neg_list = self.create_neg(head, rel, self.num_sample)
            # neg_list.append(tail)
            # return head, rel, tail, torch.LongTensor(neg_list), self.std_label_
            return head, rel, tail, torch.LongTensor(neg_list), sample_weight
        else:
            # sample_weight = 1.0 / len(self.train_hrt[tail][rel+self.relation_num])
            if self.sample_pop:
                neg_list_pop = self.create_neg_pop(tail, rel + self.relation_num, self.num_pop)
                neg_list_random = self.create_neg(tail, rel + self.relation_num, self.num_sample - self.num_pop)
                neg_list = neg_list_pop + neg_list_random
            else:
                neg_list = self.create_neg(tail, rel + self.relation_num, self.num_sample)
            # neg_list.append(head)
            # return tail, rel+self.relation_num, torch.LongTensor(neg_list), self.std_label_
            return tail, rel + self.relation_num, head, torch.LongTensor(neg_list), sample_weight