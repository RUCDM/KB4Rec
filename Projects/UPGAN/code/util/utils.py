import sys
import torch
import numpy as np
import logging
import os
from functools import reduce

def create_logger(args):
    log_file = os.path.join(args.checkpoint_dir, args.experiment_name + ".log")
    logger = logging.getLogger()
    log_level = logging.DEBUG if args.log_level == "debug" else logging.INFO
    logger.setLevel(level=log_level)
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # FileHandler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("PARAMETER" + "-" * 10)
    for attr, value in sorted(args.__dict__.items()):
        logger.info("{}={}".format(attr.upper(), value))
    logger.info("---------" + "-" * 10)

    return logger

def load_entity_dict(train_triple_list, relation_num):
    entity_dict = {}
    for triple in train_triple_list:
        head, rel, tail = triple
        entity_dict.setdefault(tail, [])
        entity_dict[tail].append((rel + relation_num, head))
        entity_dict.setdefault(head, [])
        entity_dict[head].append((rel, tail))
    return entity_dict


def remap_entity(max_hop, linked_num, entity_total, ent_hop):
    hop_map = {}
    entity_remap = {}
    for hop in range(max_hop):
        hop_map[hop] = {}
        if hop == 0:
            for i in range(linked_num):
                hop_map[hop][i] = i
                entity_remap[i] = i
        else:
            hop_start = len(entity_remap)
            for j, ent in enumerate(ent_hop[hop]):
                hop_map[hop][ent] = j
                entity_remap[ent] = j + hop_start
    index = len(entity_remap)
    print(index)
    unreached_ents = []
    for k in range(entity_total):
        if k not in entity_remap:
            unreached_ents.append(k)
            entity_remap[k] = index
            index += 1
    return hop_map, entity_remap, unreached_ents


def load_head_dict(train_triple_list, ent2hop, relation_num):
    head_dict = {}
    for triple in train_triple_list:
        head, rel, tail = triple
        if head in ent2hop and tail in ent2hop:
            minus = ent2hop[head] - ent2hop[tail]
            if minus < 0:
                head_dict.setdefault(tail, [])
                head_dict[tail].append((rel, head))
            elif minus > 0:
                head_dict.setdefault(head, [])
                head_dict[head].append((rel + relation_num, tail))
            else:
                continue
    return head_dict

def load_tail_dict(train_triple_list, ent2hop, relation_num):
    tail_dict = {}
    for triple in train_triple_list:
        head, rel, tail = triple
        if head in ent2hop and tail in ent2hop:
            minus = ent2hop[head] - ent2hop[tail]
            if minus < 0:
                tail_dict.setdefault(head, [])
                tail_dict[head].append((rel, tail))
            elif minus > 0:
                tail_dict.setdefault(tail, [])
                tail_dict[tail].append((rel + relation_num, head))
            else:
                continue
    return tail_dict


def write_res(output, res_dict, relation_num):
    f = open(output, "w")
    for hr in res_dict:
        head, rel = hr
        if rel >= relation_num:
            continue
        for tail in res_dict[hr]:
            r_r, r_r_f = res_dict[hr][tail][0], res_dict[hr][tail][1]
            tp_dict = res_dict[(tail, rel+ relation_num)][head]
            l_r, l_r_f = tp_dict[0], tp_dict[1]
            f.write("%s %s %s\n%d %d %d %d\n"%(head, rel, tail, l_r - 1, l_r_f - 1, r_r - 1, r_r_f -1))
    f.close()

def write_res_inductive(output, res_dict):
    f = open(output, "w")
    for hr in res_dict:
        head, rel = hr
        for tail in res_dict[hr]:
            r_r, r_r_f = res_dict[hr][tail][0], res_dict[hr][tail][1]
            f.write("%s %s %s\n%d %d\n" % (head, rel, tail, r_r - 1, r_r_f - 1))
    f.close()


def eva_rank_list(rank_list):
    rank_list_np = np.array(rank_list) * 1.0
    mean_rank = np.mean(rank_list_np)
    mrr_rank = np.mean(1.0 / rank_list_np)
    hits_10 = np.mean(rank_list_np <= 10)
    return mean_rank, hits_10, mrr_rank


def print_rank_list(raw_list, fil_list, temp_str, logger):
    assert isinstance(raw_list, list)
    mr_raw, hits_10_raw, mrr_raw = eva_rank_list(raw_list)
    mr_fil, hits_10_fil, mrr_fil = eva_rank_list(fil_list)
    logger.info(temp_str)
    logger.info( "Raw mrr:{:.4f}, hit:{:.4f}, mr:{:.4f}, topn:{}.".format(mrr_raw, hits_10_raw, mr_raw, 10))
    logger.info("Fil mrr:{:.4f}, hit:{:.4f}, mr:{:.4f}, topn:{}.".format(mrr_fil, hits_10_fil, mr_fil, 10))


def read_map(filename):
    f = open(filename, encoding="utf-8")
    org2id = {}
    for line in f:
        line = line.strip().split("\t")
        org = line[1]
        id = int(line[0])
        org2id[org] = id
    return org2id


def load_i2kg_map(filename, i_map, e_map):
    f = open(filename, encoding="utf-8")
    item2kg_map = {}
    for line in f:
        line = line.strip().split("\t")
        item_org = line[0]
        entity_org = line[-1]
        if item_org in i_map and entity_org in e_map:
            entity_id = e_map[entity_org]
            item2kg_map[item_org] = entity_id
    return item2kg_map


def load_rating(filename, u_map, i_map):
    f = open(filename)
    user_dict = {}
    ent_dict = {}
    for line in f:
        line = line.strip().split("\t")
        user = line[0]
        item = line[1]
        if item not in i_map:
            continue
        item_ent_id = i_map[item]
        user_id = u_map[user]
        user_dict.setdefault(user_id, [])
        user_dict[user_id].append(item_ent_id)
        ent_dict.setdefault(item_ent_id, [])
        ent_dict[item_ent_id].append(user_id)
    return user_dict, ent_dict


def load_triple(filename):
    f = open(filename)
    triple_list = []
    for line in f:
        line = line.strip().split("\t")
        head, rel, tail = line
        triple_list.append((eval(head), eval(rel), eval(tail)))
    f.close()
    return triple_list


def triple_list2hrt(triple_list, relation_num):
    r_ht = {}
    for triple in triple_list:
        head, rel, tail = triple
        r_ht.setdefault(rel, {})
        r_ht[rel].setdefault(head, set())
        r_ht[rel][head].add(tail)
        r_ht.setdefault(rel + relation_num, {})
        r_ht[rel + relation_num].setdefault(tail, set())
        r_ht[rel + relation_num][tail].add(head)
    return r_ht

def MakeEvalIterator(r_ht, r_ht_all, batch_size):
    data_iter = {}
    for rel in r_ht:
        data_iter[rel] = []
        heads = []
        rels = []
        tails = []
        fil_ids = []
        id = 0
        for ent in r_ht[rel]:
            heads.append(ent)
            rels.append(rel)
            tails.append(list(r_ht[rel][ent]))
            fil_ids.append(list(r_ht_all[rel][ent]))
            id += 1
            if id % batch_size == 0:
                data_iter[rel].append((heads, rels, tails, fil_ids))
                id = 0
                heads = []
                rels = []
                tails = []
                fil_ids = []
        if id > 0:
            data_iter[rel].append((heads, rels, tails, fil_ids))
    return data_iter