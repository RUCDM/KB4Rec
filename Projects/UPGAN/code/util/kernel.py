import math
import numpy as np

def make_rs_graph_split(ent_dict, node_tot, interaction, num_split=1, sample_flag=False, num_sample=5, logger=None):
    edge_list_rs = []
    interval = math.ceil(interaction / float(num_split))
    node_now = 0
    node_pre = 0
    head_list = []
    head_list_ori = []
    tail_list = []
    weight_list = []
    max_batch = 0
    while node_now < node_tot:
        assert node_now in ent_dict
        if sample_flag and len(ent_dict[node_now]) > num_sample:
            user_list = np.random.choice(ent_dict[node_now], num_sample, replace=False)
        else:
            user_list = ent_dict[node_now]
        for user in user_list:
            head_list.append(node_now - node_pre)
            head_list_ori.append(node_now)
            tail_list.append(user)
            edge_weight = 1.0 / len(user_list)
            weight_list.append(edge_weight)
        node_now += 1
        if len(head_list) >= interval:
            if len(head_list) > max_batch:
                max_batch = len(head_list)
            edge_list_map = [head_list, tail_list]
            edge_list_ori = [head_list_ori, tail_list]
            #edge_list_rs.append((edge_list_ori, edge_list_map, weight_list, range(node_pre, node_now)))
            edge_list_rs.append((edge_list_map, edge_list_ori, range(node_pre, node_now)))
            head_list = []
            head_list_ori = []
            tail_list = []
            weight_list = []
            node_pre = node_now
    if len(head_list) > 0:
        if len(head_list) > max_batch:
            max_batch = len(head_list)
        edge_list_map = [head_list, tail_list]
        edge_list_ori = [head_list_ori, tail_list]
        # edge_list_rs.append((edge_list_ori, edge_list_map, weight_list, range(node_pre, node_now)))
        edge_list_rs.append((edge_list_map, edge_list_ori, range(node_pre, node_tot)))
    if logger is not None:
        logger.info("RS edges splited, max number of edges: {}".format(max_batch))
    return edge_list_rs

def make_kg_graph(head_dict, max_hop, ent_hop, hop_map, sample_flag=False, num_sample=5):
    kg_graph = {}
    for i in range(1, max_hop):
        kg_list = {}
        tp_ent = []
        entity_map_pre = hop_map[i - 1]
        entity_map_now = hop_map[i]
        assert len(entity_map_now) == len(ent_hop[i])
        r_ht = {}
        tp_head_dict = {}
        for ent in ent_hop[i]:
            tp_ent.append(ent)
            if sample_flag and len(head_dict[ent]) > num_sample:
                indice = np.random.choice(len(head_dict[ent]), num_sample, replace=False)
                rt_list = [head_dict[ent][index] for index in indice]
            else:
                rt_list = head_dict[ent]
            tp_head_dict[ent] = rt_list
            for rt in rt_list:
                rel, tail = rt
                r_ht.setdefault(rel, [])
                r_ht[rel].append((ent, tail))
        for rel in r_ht:
            head_list = []
            tail_list = []
            for ht in r_ht[rel]:
                head, tail = ht
                head_list.append(entity_map_now[head])
                tail_list.append(entity_map_pre[tail])
            kg_list[rel] = [head_list, tail_list]
        kg_graph[i] = (kg_list, tp_ent)
    return kg_graph