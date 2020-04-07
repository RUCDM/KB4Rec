import sys, os
import torch
import numpy as np
from util.dataset import triple_dataset_unif
from util.utils import *
from torch.utils.data import DataLoader


def create_train(triple_list, entity_num, relation_num, args):
    train_dataset = triple_dataset_unif(triple_list, entity_num, relation_num, args)
    return train_dataset


def create_eval(triple_list, all_hrt, relation_num, batch_size = 128):
    eval_hrt = triple_list2hrt(triple_list, relation_num)
    return MakeEvalIterator(eval_hrt, all_hrt, batch_size)

def load_dataset(args, logger=None):
    data_path = os.path.join(args.data_folder, args.dataset)
    train_batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    num_workers = args.num_workers
    u_map_file = os.path.join(data_path, "rs/u_map.dat")
    i_map_file = os.path.join(data_path, "rs/i_map.dat")
    i2kg_map_file = os.path.join(data_path, "rs/i2kg_map.tsv")
    rating_file = os.path.join(data_path, "rs/ratings.txt")
    e_map_file = os.path.join(data_path, "kg/e_map.dat")
    r_map_file = os.path.join(data_path, "kg/r_map.dat")
    train_file = os.path.join(data_path, "kg/train.dat")
    valid_file = os.path.join(data_path, "kg/valid.dat")
    test_file = os.path.join(data_path, "kg/test.dat")

    u_map = read_map(u_map_file)
    i_map = read_map(i_map_file)
    e_map = read_map(e_map_file)
    r_map = read_map(r_map_file)

    item2kg_map = load_i2kg_map(i2kg_map_file, i_map, e_map)
    user_dict, ent_dict = load_rating(rating_file, u_map, item2kg_map)
    user_dict_all, item_dict_all = load_rating(rating_file, u_map, i_map)
    total_interaction_linked = np.sum([len(user_dict[user]) for user in user_dict])
    total_interaction_rs = np.sum([len(user_dict_all[user]) for user in user_dict_all])
    rs_dataset = {
        "u_map": u_map,
        "i_map": i_map,
        "i_kg_map": item2kg_map,
        "user_dict": user_dict,
        "ent_dict": ent_dict,
        "item_dict_all": item_dict_all,
        "user_all": user_dict_all,
        "total_int_linked": total_interaction_linked,
        "total_int_rs": total_interaction_rs
    }
    if logger is not None:
        logger.info("RS Construction Done! There are {} users and {} items available,"
                " {} interactions linked, and {} interactions in rs.".format(len(u_map), len(item2kg_map),
                total_interaction_linked, total_interaction_rs))

    train_triple_list = load_triple(train_file)
    valid_triple_list = load_triple(valid_file)
    test_triple_list = load_triple(test_file)
    all_triple_list = []
    all_triple_list.extend(train_triple_list)
    all_triple_list.extend(valid_triple_list)
    all_triple_list.extend(test_triple_list)
    all_hrt = triple_list2hrt(all_triple_list, len(r_map))

    train_dataset = create_train(train_triple_list, len(e_map), len(r_map), args)
    train_eval_dataset = create_eval(train_triple_list, all_hrt, len(r_map), batch_size=eval_batch_size)
    eval_dataset = create_eval(valid_triple_list, all_hrt, len(r_map), batch_size=eval_batch_size)
    test_dataset = create_eval(test_triple_list, all_hrt, len(r_map), batch_size=eval_batch_size)

    if logger is not None:
        logger.info("Eval dataset created! {} and {} batches with batch_size {}".format(len(eval_dataset),
                                                                                    len(test_dataset),
                                                                                    eval_batch_size))

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

    kg_dataset = {
        "e_map": e_map,
        "r_map": r_map,
        "train_triple_list": train_triple_list,
        "train_eval": train_eval_dataset,
        "train": train_loader,
        "valid": eval_dataset,
        "test": test_dataset
    }
    if logger is not None:
        logger.info("KG Construction Done! There are {} entities and {} relations available,"
                " {} triples at train, {} triples at valid, {} triples at test.".format(len(e_map),
                len(r_map), len(train_triple_list), len(valid_triple_list), len(test_triple_list)))

    return kg_dataset, rs_dataset