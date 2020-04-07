import numpy as np
import torch
from tqdm import tqdm

tqdm.monitor_iterval = 0


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
    logger.info("Raw mrr:{:.4f}, hit:{:.4f}, mr:{:.4f}, topn:{}.".format(mrr_raw, hits_10_raw, mr_raw, 10))
    logger.info("Fil mrr:{:.4f}, hit:{:.4f}, mr:{:.4f}, topn:{}.".format(mrr_fil, hits_10_fil, mr_fil, 10))
    # print(temp_str)
    # print("Raw mrr:{:.4f}, hit:{:.4f}, mr:{:.4f}, topn:{}.".format(mrr_raw, hits_10_raw, mr_raw, 10))
    # print("Fil mrr:{:.4f}, hit:{:.4f}, mr:{:.4f}, topn:{}.".format(mrr_fil, hits_10_fil, mr_fil, 10))

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
            f.write("%s %s %s\n%d %d\n"%(head, rel, tail, r_r - 1, r_r_f -1))
    f.close()


class Evaluator_kg(object):
    def __init__(self, model, use_cuda, middle_file, logger, relation_num, k=10, e_map=None):
        self.model = model
        self.topk = k
        self.logger = logger
        self.middle_file = middle_file
        self.relation_total = relation_num
        self.batch_size = 64
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def get_top(self, eval_loader, prefix="./", topk=100):
        self.model.eval()  ###eval_loader take data as batch
        f = {}
        for name in ["raw", "fil-train", "fil-all"]:
            filename = prefix + "%s_top100.txt" % (name)
            f[name] = open(filename, "w")
        total_batches = len(eval_loader)
        pbar = tqdm(total=total_batches)
        pbar.set_description("Run Eval")
        with torch.no_grad():
            candidates = self.model.get_candidates()
            for batch_rel in eval_loader:
                for heads, rels, tail_lists, right_alls in eval_loader[batch_rel]:
                    heads_t = torch.LongTensor(heads).to(self.device)
                    rels_t = torch.LongTensor(rels).to(self.device)
                    preds = self.model.evaluate(heads_t, rels_t, candidates)
                    for i in range(0, heads_t.size(0)):
                        pred = preds[i]
                        tp_res = {}
                        values = {}
                        # max_values, argsort1 = torch.sort(pred, descending=True)
                        max_values, argsort1 = torch.topk(pred, topk)
                        # argsort1 = argsort1
                        # max_values = max_values
                        tp_res["raw"] = argsort1.cpu().numpy()
                        values["raw"] = max_values.cpu().numpy()

                        fil_ids = torch.LongTensor(tail_lists[i]).to(self.device)
                        # cache = pred[fil_ids]
                        pred[fil_ids] = -100000.0
                        max_values, argsort1 = torch.topk(pred, topk)
                        # max_values, argsort1 = torch.sort(pred, descending=True)
                        #
                        # argsort1 = argsort1
                        # max_values = max_values
                        tp_res["fil-train"] = argsort1.cpu().numpy()
                        values["fil-train"] = max_values.cpu().numpy()
                        # preds[fil_ids] = cache

                        fil_ids = torch.LongTensor(right_alls[i]).to(self.device)
                        pred[fil_ids] = -100000.0
                        max_values, argsort1 = torch.sort(pred, descending=True)
                        argsort1 = argsort1
                        max_values = max_values
                        tp_res["fil-all"] = argsort1[:topk].cpu().numpy()
                        values["fil-all"] = max_values[:topk].cpu().numpy()
                        for name in ["raw", "fil-train", "fil-all"]:
                            raw_top10 = tp_res[name]
                            val_top10 = values[name]
                            # str_top10 = [str(item) for item in raw_top10]
                            # val_str_top10 = ["%.3f" % item for item in val_top10]
                            tp_str = []
                            for j in range(len(raw_top10)):
                                temp_str = "%s:%.3f"%(raw_top10[j], val_top10[j])
                                tp_str.append(temp_str)
                            f[name].write("%d\t%d\t%s\n" % (heads[i], rels[i],
                                                                "\t".join(tp_str)))
                pbar.update(1)
            pbar.close()
        for name in ["raw", "fil-train", "fil-all"]:
            f[name].close()

    def evaluate(self, eval_loader, name):
        self.model.eval()  ###eval_loader take data as batch
        ranks_raw = {}
        ranks_filter = {}
        for type in ["head", "tail", "all"]:
            ranks_raw[type] = []
            ranks_filter[type] = []
        total_batches = len(eval_loader)
        pbar = tqdm(total=total_batches)
        pbar.set_description("Run Eval")
        res_dict = {}
        with torch.no_grad():
            candidates = self.model.get_candidates()
            for batch_rel in eval_loader:
                for heads, rels, tail_lists, right_alls in eval_loader[batch_rel]:
                    heads_t = torch.LongTensor(heads)
                    heads_t = heads_t.to(self.device)
                    rels_t = torch.LongTensor(rels).to(self.device)
                    preds = self.model.evaluate(heads_t, rels_t, candidates)
                    for i in range(heads_t.size(0)):
                        pred = preds[i]
                        hr = (heads[i], rels[i])
                        res_dict[hr] = {}
                        fil_ids = torch.LongTensor(right_alls[i]).to(self.device)  # [0]
                        cache = pred[fil_ids]
                        max_values, raw_argsort = torch.sort(pred, descending=True)
                        raw_argsort = raw_argsort.cpu().numpy()  # (B, E)
                        for j in tail_lists[i]:
                            rank_raw = np.where(raw_argsort == j)[0][0] + 1
                            ranks_raw["all"].append(rank_raw)
                            if rels[i] >= self.relation_total:
                                ranks_raw["head"].append(rank_raw)
                            else:
                                ranks_raw["tail"].append(rank_raw)
                            res_dict[hr][j] = [rank_raw]

                        for j in tail_lists[
                            i]:  # For every answer, conduct evaluation with sort, I think it's stupid
                            answer = pred[j].item()
                            # answer = pred[j]
                            pred[fil_ids] = -100000.0
                            pred[j] = answer
                            max_values, argsort1 = torch.sort(pred, descending=True)
                            argsort1 = argsort1.cpu().numpy()  # (B, E)
                            rank1 = np.where(argsort1 == j)[0][0] + 1  # id start with 0
                            res_dict[hr][j].append(rank1)
                            ranks_filter["all"].append(rank1)
                            if rels[i] >= self.relation_total:
                                ranks_filter["head"].append(rank1)
                            else:
                                ranks_filter["tail"].append(rank1)
                            pred[fil_ids] = cache
                pbar.update(1)
            pbar.close()
        if name == "TEST":
            write_res(self.middle_file, res_dict, self.relation_total)
            print(len(ranks_raw["head"]), len(ranks_raw["tail"]), len(ranks_raw["all"]))
        print(len(ranks_raw))
        print_rank_list(ranks_raw["tail"], ranks_filter["tail"], "Tail-sort", self.logger)
        print_rank_list(ranks_raw["head"], ranks_filter["head"], "Head-sort", self.logger)

        mr_raw, hits_10_raw, mrr_raw = eva_rank_list(ranks_raw["all"])
        mr_fil, hits_10_fil, mrr_fil = eva_rank_list(ranks_filter["all"])
        self.logger.info(name)
        self.logger.info(
            "Raw mrr:{:.4f}, hit:{:.4f}, mr:{:.4f}, topn:{}.".format(mrr_raw, hits_10_raw, mr_raw,
                                                                     self.topk))
        self.logger.info(
            "Fil mrr:{:.4f}, hit:{:.4f}, mr:{:.4f}, topn:{}.".format(mrr_fil, hits_10_fil, mr_fil,
                                                                     self.topk))
        return mr_raw, hits_10_raw, mrr_raw, mr_fil, hits_10_fil, mrr_fil