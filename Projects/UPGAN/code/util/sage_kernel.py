import math
import numpy as np

def make_RGCN_kernel_batch(reserved_ents, path_reserve, entity_tot, relation_tot):
    triple_list = []
    for ent in reserved_ents:
        if ent in path_reserve:
            if ent > entity_tot:
                for entity in path_reserve[ent]:
                    triple_list.append((ent, relation_tot * 2, entity))
            else:
                for rel, tail in path_reserve[ent]:
                    triple_list.append((ent, rel, tail))
    print("Number triple:%d" % (len(triple_list)))
    return RGCN_kernal(triple_list)

def make_RGCN_kernel_sample(user_dict, entity_dict, entity_tot, relation_tot, num_sample=10, sample_flag=True):
    triple_list = universe_sample(user_dict, entity_dict, entity_tot, relation_tot,
                                  num_sample=num_sample, sample_flag=sample_flag)
    print("Number triple:%d"%(len(triple_list)))
    return RGCN_kernal(triple_list)

def universe_sample(user_dict, entity_dict, entity_tot, relation_tot, num_sample=10, sample_flag=True):
    triple_list = []
    for user in user_dict:
        if sample_flag and len(user_dict[user]) > num_sample:
            link_ent_list = np.random.choice(user_dict[user], num_sample, replace=False)
        else:
            link_ent_list = user_dict[user]
        for ent in link_ent_list:
            triple_list.append((user + entity_tot, relation_tot * 2, ent))
    for entity in entity_dict:
        if sample_flag and len(entity_dict[entity]) > num_sample:
            indice = np.random.choice(len(entity_dict[entity]), num_sample, replace=False)
            rt_list = [entity_dict[entity][index] for index in indice]
        else:
            rt_list = entity_dict[entity]
        for rel, tail in rt_list:
            if rel == 2 * relation_tot + 1:#Reverse edges to RS, if exist
                triple_list.append((entity, rel, tail + entity_tot))
            else:
                triple_list.append((entity, rel, tail))
    return triple_list

def RGCN_kernal_new(train_triple_list):
    rel_triple_list = {}
    head_ct = {}
    for triple in train_triple_list:
        head, rel, tail = triple
        rel_triple_list.setdefault(rel, [])
        rel_triple_list[rel].append((head, tail))
        head_ct.setdefault(head, 0)
        head_ct[head] += 1
    rel_edge_list = {}
    for rel in rel_triple_list:
        head_map = {}
        head_map_r = {}
        weight_list = []
        head_list = []
        tail_list = []
        num_head = 0
        for pair in rel_triple_list[rel]:
            head, tail = pair
            if head not in head_map:
                head_map[head] = num_head
                head_map_r[num_head] = head
                num_head += 1
            edge_weight = 1.0 / head_ct[head]
            head_list.append(head_map[head])
            tail_list.append(tail)
            weight_list.append(edge_weight)
        head_indice = [head_map_r[i] for i in range(num_head)]
        rel_edge_list[rel] = [[head_list, tail_list], weight_list, head_indice]
    return rel_edge_list