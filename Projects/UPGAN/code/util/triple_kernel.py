import math
import numpy as np

def make_kernel_batch(path_list, max_hop, ent_hop, hop_map):
    rs_graph = make_triple_graph_rs_batch(path_list, ent_hop[0], hop_map[0], interval=100000)
    kg_graph = make_kg_graph_batch(path_list, max_hop, ent_hop, hop_map)
    #kg_graph = make_query_graph_hop_batch(path_list, max_hop, ent_hop, hop_map, interval=100000)
    return rs_graph, kg_graph

def make_triple_graph_rs_batch(path_list, batch_ents, ent_map, interval=100000):
    list_head_ori = []
    list_head_map = []
    list_tail_ori = []
    batch_ents_order = sorted(batch_ents, key=lambda x: ent_map[x])
    start = 0
    edge_list_rs = []
    tp_ents = []
    for head in batch_ents_order:
        assert head in path_list
        user_list = path_list[head]
        now_id = ent_map[head]
        tp_ents.append(now_id)
        for user in user_list:
            list_head_ori.append(head)
            list_head_map.append(now_id - start)
            list_tail_ori.append(user)
        if len(list_head_map) > interval:
            edge_list_map = [list_head_map, list_tail_ori]
            edge_list_ori = [list_head_ori, list_tail_ori]
            edge_list_rs.append((edge_list_map, edge_list_ori, tp_ents))
            tp_ents = []
            list_head_ori = []
            list_head_map = []
            list_tail_ori = []
            start = now_id + 1
    if len(list_head_map) > 0:
        edge_list_map = [list_head_map, list_tail_ori]
        edge_list_ori = [list_head_ori, list_tail_ori]
        edge_list_rs.append((edge_list_map, edge_list_ori, tp_ents))
    return edge_list_rs

def make_kg_graph_batch(path_list, max_hop, ent_hop, hop_map):
    kg_graph = {}
    for i in range(1, max_hop):
        kg_list = {}
        tp_ent = []
        entity_map_pre = hop_map[i - 1]
        entity_map_now = hop_map[i]
        assert len(entity_map_now) == len(ent_hop[i])
        r_ht = {}
        for ent in ent_hop[i]:
            tp_ent.append(ent)
            rt_list = path_list[ent]
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

def make_triple_graph_kg_single_hop(path_list, batch_ents, ent_map_now, ent_map_pre, interval=100000):
    head_list_ori = []
    tail_list_ori = []
    head_list_map = []
    tail_list_map = []
    rel_list = []
    start = 0
    batch_ents_order = sorted(batch_ents, key=lambda x: ent_map_now[x])
    tp_ents = []
    edge_list_hop = []
    for head in batch_ents_order:
        assert head in path_list
        rt_list = path_list[head]
        now_id = ent_map_now[head]
        for rel, tail in rt_list:
            head_list_ori.append(head)
            rel_list.append(rel)
            tail_list_ori.append(tail)
            head_list_map.append(now_id - start)
            tail_list_map.append(ent_map_pre[tail])
        if len(head_list_map) > interval:
            edge_list_map = [head_list_map, tail_list_map]
            edge_list_ori = [head_list_ori, tail_list_ori]
            edge_list_hop.append((edge_list_map, edge_list_ori, tp_ents, rel_list))
            tp_ents = []
            start = now_id + 1
            head_list_ori = []
            tail_list_ori = []
            head_list_map = []
            tail_list_map = []
            rel_list = []
    if len(head_list_map) > 0:
        edge_list_map = [head_list_map, tail_list_map]
        edge_list_ori = [head_list_ori, tail_list_ori]
        edge_list_hop.append((edge_list_map, edge_list_ori, tp_ents, rel_list))
    return edge_list_hop

def make_query_graph_hop_batch(path_list, max_hop, ent_hop, hop_map, interval=100000):
    kg_graph = {}
    for i in range(1, max_hop):
        batch_ents = ent_hop[i]
        ent_map_now = hop_map[i]
        ent_map_pre = hop_map[i - 1]
        kg_graph[i] = make_query_graph_single_batch(path_list, batch_ents, ent_map_now, ent_map_pre, interval)
    return kg_graph