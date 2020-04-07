import torch
import time
import numpy as np
import os, math
from train.load_data import load_dataset
from train.evaluation import Evaluator_kg

from util.utils import remap_entity, load_head_dict, load_entity_dict, load_tail_dict
from util.kernel import make_rs_graph_split, make_kg_graph
from util.sage_kernel import make_RGCN_kernel_sample, make_RGCN_kernel_batch, RGCN_kernal_new

from tqdm import tqdm
import torch.nn.functional as F
tqdm.monitor_iterval = 0


class Trainer(object):

    def data_load_kg_rs(self):
        kg_dataset, rs_dataset = load_dataset(self.args,
                                              logger=self.logger)
        self.u_map = rs_dataset['u_map']
        self.i_map = rs_dataset['i_map']
        self.i_kg_map = rs_dataset["i_kg_map"]
        self.e_map = kg_dataset['e_map']
        self.r_map = kg_dataset['r_map']
        self.model_def()
        self.loss_def()
        self.optim_def(self.learning_rate)

        self.kg_train_loader = kg_dataset["train"]
        self.train_eval_loader = kg_dataset["train_eval"]
        self.kg_eval_loader = kg_dataset["valid"]
        self.kg_test_loader = kg_dataset["test"]

        self.train_triple_list = kg_dataset["train_triple_list"]
        self.user_dict_linked = rs_dataset["user_dict"]
        self.user_dict_all = rs_dataset["user_all"]
        self.ent_dict = rs_dataset["ent_dict"]
        self.item_dict_all = rs_dataset["item_dict_all"]
        self.entity_dict = load_entity_dict(self.train_triple_list, self.relation_total)

        self.middle_file = os.path.join(self.args.checkpoint_dir, "{}.middle".format(self.args.experiment_name))


        self.max_hop, self.ent2hop, self.ent_hop = self.label_triples(self.args.hop_limit)
        self.hop_map, self.entity_remap, unreached_ents = remap_entity(self.max_hop,
                                                       self.share_total,
                                                       self.entity_total,
                                                       self.ent_hop)
        remapped_ent_list = [self.entity_remap[k] for k in range(self.entity_total)]

        self.head_dict = load_head_dict(self.train_triple_list, self.ent2hop, self.relation_total)
        self.tail_dict = load_tail_dict(self.train_triple_list, self.ent2hop, self.relation_total)
        #self.edge_list_kg = self.make_kg_graph(interval=10000, num_sample=self.args.kg_sample)
        self.edge_list_kg = make_kg_graph(head_dict=self.head_dict,
                                          max_hop=self.max_hop,
                                          ent_hop=self.ent_hop,
                                          hop_map=self.hop_map,
                                          sample_flag=False)
        self.interaction_linked = rs_dataset["total_int_linked"]
        self.interaction_all = rs_dataset["total_int_rs"]
        self.edge_list_rs = make_rs_graph_split(self.ent_dict,
                                                len(self.i_kg_map),
                                                interaction=self.interaction_linked,
                                                num_split=self.args.gat_split,
                                                sample_flag=False,
                                                logger=self.logger)
        self.evaluator_kg = Evaluator_kg(self.model,
                                         relation_num=len(self.r_map),
                                         use_cuda=self.args.use_cuda,
                                         middle_file=self.middle_file,
                                         logger=self.logger)
        self.path_reverse_all = self.sample_all_reverse()
        self.rgcn_kernel = self.get_rgcn_kenel(path_reserve=self.path_reverse_all,
                                               user_set=set(range(self.user_total)))
        # self.rgcn_kernel = make_RGCN_kernel_sample(self.user_dict_linked, self.entity_dict, self.entity_total,
        #                                           self.relation_total, num_sample=10, sample_flag=False)
        self.model.set_edges(self.edge_list_kg, self.edge_list_rs, remapped_ent_list, unreached_ents, self.rgcn_kernel)

    def label_triples(self, hop_limit):
        train_triple_list = self.train_triple_list
        linked_num = self.share_total
        ent2hop = {}
        known_ent = set()
        for i in range(linked_num):
            known_ent.add(i)
            ent2hop[i] = 0
        print(0, len(ent2hop))
        ent_hop = {}
        ent_hop[0] = known_ent.copy()
        num_hop = 1
        while True:
            new_ent = set()
            for triple in train_triple_list:
                head, rel, tail = triple
                if head not in known_ent and tail in known_ent:
                    ent2hop[head] = num_hop
                    new_ent.add(head)
                if tail not in known_ent and head in known_ent:
                    ent2hop[tail] = num_hop
                    new_ent.add(tail)
            if len(new_ent) == 0:
                break
            known_ent |= new_ent
            ent_hop[num_hop] = new_ent
            print("Hop k:%d, %d entities new, %d entities known"%(num_hop, len(new_ent), len(known_ent)))
            num_hop += 1
            if hop_limit > 0 and num_hop >= hop_limit:
                break
        self.logger.info("KG edges labeled, max number of hops: {}, number"
                         " of user-reachable entities: {}".format(num_hop, len(ent2hop)))
        return num_hop, ent2hop, ent_hop

    def eval_sample_popularity(self):
        recall_list = [1, 5, 10, 20]
        recall_dict = {}
        for j in recall_list:
            recall_dict[j] = []
        st = time.time()
        eval_topk = 20
        pop_list = [0.0] * self.entity_total
        for head, rel, tail in self.train_triple_list:
            pop_list[head] += 1
            pop_list[tail] += 1
        pop_vec = torch.Tensor(pop_list).to(self.device)
        with torch.no_grad():
            candidates = self.model.get_candidates()
            for heads, rels, tails, neg_tails, sample_weight in self.kg_train_loader:
                heads = torch.LongTensor(heads).to(self.device)
                rels = torch.LongTensor(rels).to(self.device)
                tails = torch.LongTensor(tails).to(self.device)
                neg_tails = torch.LongTensor(neg_tails).to(self.device)
                # sample_weight = sample_weight.float().to(self.device).squeeze()
                tails_gen = self.gen_sample(heads, rels, neg_tails,
                                            n_sample=self.args.n_sample_gen,
                                            temperature=1.0,
                                            lambda_smooth=self.args.lambda_smooth)
                n = heads.size(0)
                row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, eval_topk)
                user_rep = candidates[heads]
                query_now, _ = self.model.form_query(heads, rels, user_rep)
                # preds_gen, _ = self.model.query_judge(query_now, tails_gen)
                # print(preds_gen)
                # preds_neg, _ = self.model.query_judge(query_now, neg_tails)
                pop_neg = pop_vec[neg_tails]
                _, topk_indice = torch.topk(pop_neg, k=eval_topk, dim=1)
                topk_tails = neg_tails[row_idx, topk_indice]
                self.find_common(tails_gen, topk_tails, recall_list, recall_dict)
        for j in recall_list:
            print("Recall@{}: {:.4f}".format(j, np.mean(recall_dict[j])))
        print("time: {}".format(time.time() - st))

    def eval_sample_quality(self):
        recall_list = [1, 5, 10, 20]
        recall_dict = {}
        for j in recall_list:
            recall_dict[j] = []
        st = time.time()
        eval_topk = 20
        with torch.no_grad():
            candidates = self.model.get_candidates()
            for heads, rels, tails, neg_tails, sample_weight in self.kg_train_loader:
                heads = torch.LongTensor(heads).to(self.device)
                rels = torch.LongTensor(rels).to(self.device)
                tails = torch.LongTensor(tails).to(self.device)
                neg_tails = torch.LongTensor(neg_tails).to(self.device)
                # sample_weight = sample_weight.float().to(self.device).squeeze()
                tails_gen = self.gen_sample(heads, rels, neg_tails,
                                            n_sample=self.args.n_sample_gen,
                                            temperature=1.0,
                                            lambda_smooth=self.args.lambda_smooth)
                n = heads.size(0)
                row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, eval_topk)
                user_rep = candidates[heads]
                query_now, _ = self.model.form_query(heads, rels, user_rep)
                # preds_gen, _ = self.model.query_judge(query_now, tails_gen)
                # print(preds_gen)
                preds_neg, _ = self.model.query_judge(query_now, neg_tails)
                _, topk_indice = torch.topk(preds_neg, k=eval_topk, dim=1)
                topk_tails = neg_tails[row_idx, topk_indice]
                self.find_common(tails_gen, topk_tails, recall_list, recall_dict)
        for j in recall_list:
            print("Recall@{}: {:.4f}".format(j, np.mean(recall_dict[j])))
        print("time: {}".format(time.time() - st))

    def eval_sample_diversity(self, epoch=0):
        st = time.time()
        # filename = os.path.join(self.args.checkpoint_dir, "{}-gen_epoch-{}.txt".format(self.args.experiment_name, epoch))
        # f = open(filename, "w")
        step = 0
        hr_dict = {}
        with torch.no_grad():
            # candidates = self.model.get_candidates()
            for heads, rels, tails, neg_tails, sample_weight in self.kg_train_loader:
                step += 1
                # if step % 10 == 0:
                #     print("Step: {}, time :{:.4f}".format(step, time.time() - st))
                heads = torch.LongTensor(heads).to(self.device)
                rels = torch.LongTensor(rels).to(self.device)
                tails = torch.LongTensor(tails).to(self.device)
                neg_tails = torch.LongTensor(neg_tails).to(self.device)
                # sample_weight = sample_weight.float().to(self.device).squeeze()
                tails_gen = self.gen_sample(heads, rels, neg_tails,
                                            n_sample=self.args.n_sample_gen,
                                            temperature=1.0,
                                            lambda_smooth=self.args.lambda_smooth)
                heads_np = heads.cpu().numpy()
                rels_np = rels.cpu().numpy()
                tail_np = tails.cpu().numpy()
                tails_gen = tails_gen.cpu().numpy()
                for i in range(heads.size(0)):
                    query = (heads_np[i], rels_np[i])
                    hr_dict.setdefault(query, set())
                    hr_dict[query] |= set(tails_gen[i])
                    # gen_str = "\t".join([str(item) for item in tails_gen[i]])
                    # f.write("%d\t%d\t%d\t%s\n"%(heads_np[i], rels_np[i], tail_np[i], gen_str))
            # f.close()
        total_diverse = sum([len(hr_dict[query]) for query in hr_dict])
        print("Epoch: {}, diversity: {:.4f}, time: {:.4f},".format(epoch, total_diverse, time.time() - st))

    def show_norm(self):
        with torch.no_grad():
            norm_emb = torch.norm(self.model.ent_embeddings.weight, dim=1)
            max_norm = torch.max(norm_emb).item()
            min_norm = torch.min(norm_emb).item()
            mean_norm = torch.mean(norm_emb).item()
            print("Ent Min: {:.4f}, Max: {:.4f}, Mean: {:.4f}".format(min_norm, max_norm, mean_norm))
            norm_emb = torch.norm(self.model.user_embeddings.weight, dim=1)
            max_norm = torch.max(norm_emb).item()
            min_norm = torch.min(norm_emb).item()
            mean_norm = torch.mean(norm_emb).item()
            print("User Min: {:.4f}, Max: {:.4f}, Mean: {:.4f}".format(min_norm, max_norm, mean_norm))

    def dump_emb(self):
        with torch.no_grad():
            emb = self.model.ent_embeddings.weight
            model_name = os.path.join(self.args.checkpoint_dir, "{}.emb".format(self.args.experiment_name))
            torch.save(emb, model_name)
            print("Save model as %s" % model_name)

    def evaluate_single(self, filename):
        self.load_ckpt(filename)
        self.show_norm()
        self.evaluator_kg.evaluate(self.kg_test_loader, "TEST")
        # self.dump_emb()
        # data_prefix = os.path.join(self.args.checkpoint_dir, "{}-".format(self.args.experiment_name))
        # self.evaluator_kg.get_top(self.kg_test_loader, prefix = data_prefix, topk = 100)
        # self.evaluator_kg.get_top(self.train_eval_loader, prefix = data_prefix, topk = 100)

        # self.model.eval()
        # attn_file = os.path.join(self.args.checkpoint_dir, "{}.middle".format(self.args.experiment_name))
        # with torch.no_grad():
        #     self.model.calc_attn(output=attn_file)

    def evaluate_best(self):
        filename = os.path.join(self.args.checkpoint_dir, "{}.ckpt".format(self.args.experiment_name))
        self.load_ckpt(filename)
        #self.evaluator_kg.evaluate(self.kg_test_loader, "TEST")
        self.evaluator_kg.evaluate(self.kg_test_loader, "TEST")

    def save_ckpt(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'G_state_dict': self.G.state_dict(),
            'best_dev_performance': self.best_dev_performance
        }
        model_name = os.path.join(self.args.checkpoint_dir, "{}.ckpt".format(self.args.experiment_name))
        torch.save(checkpoint, model_name)
        print("Save model as %s" % model_name)

    def load_ckpt(self, filename):
        checkpoint = torch.load(filename)
        model_state_dict = checkpoint['model_state_dict']
        G_state_dict = checkpoint['G_state_dict']
        self.model.load_state_dict(model_state_dict, strict=False)
        self.G.load_state_dict(G_state_dict, strict=False)
        self.best_dev_performance = checkpoint['best_dev_performance']

    def load_G_pretrain(self):
        assert self.args.load_ckpt_G is not None
        load_ckpt_files = self.args.load_ckpt_G.split(':')
        for filename in load_ckpt_files:
            self.loadEmbedding(os.path.join(self.args.checkpoint_dir, filename), self.G, alternative='G_state_dict')

    def load_pretrain(self, args):
        if args.load_ckpt_G is not None:
            # if self.args.need_pretrain:
            load_ckpt_files = args.load_ckpt_G.split(':')
            for filename in load_ckpt_files:
                self.loadEmbedding(os.path.join(args.checkpoint_dir, filename), self.G, alternative='G_state_dict')
        if args.load_ckpt_D is not None:
            # if self.args.need_pretrain:
            load_ckpt_files = args.load_ckpt_D.split(':')
            for filename in load_ckpt_files:
                self.loadEmbedding(os.path.join(args.checkpoint_dir, filename), self.D, alternative='D_state_dict')
            # else:
            #     filename = os.path.join(args.checkpoint_dir, args.load_ckpt_G)
            #     self.load_G(filename)
        if args.load_ckpt_file is not None:
            load_ckpt_files = args.load_ckpt_file.split(':')
            for filename in load_ckpt_files:
                self.loadEmbedding(os.path.join(args.checkpoint_dir, filename), self.model, alternative='D_state_dict')
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            print("Loading pre trained model from {}".format(ckpt_path))
            self.load_ckpt(ckpt_path)

    def loadEmbedding(self, filename, model, cpu=False, alternative="model_state_dict"):
        assert os.path.isfile(filename), "Checkpoint file not found!"
        self.logger.info("Found checkpoint, restoring pre-trained embeddings.")

        if cpu:
            # Load GPU-based checkpoints on CPU
            checkpoint = torch.load(
                filename, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(filename)
        if 'model_state_dict' in checkpoint:
            old_model_state_dict = checkpoint['model_state_dict']
        else:
            assert alternative in checkpoint
            old_model_state_dict = checkpoint[alternative]

        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in old_model_state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        model.load_state_dict(model_dict, strict=False)

        self.logger.info("Load Embeddings of {} from {}.".format(", ".join(list(pretrained_dict.keys())), filename))

    def sample_all_eges(self):
        path_list = {}
        max_hop = self.max_hop
        ent_hop = self.ent_hop
        ent_dict = self.ent_dict
        head_dict = self.head_dict
        for j in range(self.share_total):
            path_list[j] = ent_dict[j]
        for i in range(1, max_hop):
            for j, ent in enumerate(ent_hop[i]):
                path_list[ent] = head_dict[ent]
        return path_list

    def sample_epoch_edges(self):
        #st = time.time()
        path_list = {}
        max_hop = self.max_hop
        ent_hop = self.ent_hop
        ent_dict = self.ent_dict
        head_dict = self.head_dict
        for j in range(self.share_total):
            if self.args.rs_sample_flag and len(ent_dict[j]) > self.args.rs_sample:
                user_list = np.random.choice(ent_dict[j], self.args.rs_sample, replace=False)
            else:
                user_list = ent_dict[j]
            path_list[j] = user_list
        for i in range(1, max_hop):
            for j, ent in enumerate(ent_hop[i]):
                if self.args.kg_sample_flag and len(head_dict[ent]) > self.args.kg_sample:
                    indice = np.random.choice(len(head_dict[ent]), self.args.kg_sample, replace=False)
                    rt_list = [head_dict[ent][index] for index in indice]
                else:
                    rt_list = head_dict[ent]
                path_list[ent] = rt_list
        #print("Sample all Done! cost time {}".format(time.time() - st))
        return path_list

    def sample_epoch_reverse(self):
        path_list = {}
        max_hop = self.max_hop
        ent_hop = self.ent_hop
        user_dict = self.user_dict_linked
        tail_dict = self.tail_dict
        for user in range(self.user_total):
            if user in user_dict:
                if len(user_dict[user]) > self.args.rs_sample:
                    ent_list = np.random.choice(user_dict[user], self.args.rs_sample, replace=False)
                else:
                    ent_list = user_dict[user]
                path_list[user + self.entity_total] = ent_list
        for i in range(0, max_hop - 1):
            for j, ent in enumerate(ent_hop[i]):
                if ent in tail_dict:
                    if len(tail_dict[ent]) > self.args.kg_sample:
                        indice = np.random.choice(len(tail_dict[ent]), self.args.kg_sample, replace=False)
                        rt_list = [tail_dict[ent][index] for index in indice]
                    else:
                        rt_list = tail_dict[ent]
                    path_list[ent] = rt_list
        return path_list

    def sample_all_reverse(self):
        path_list = {}
        max_hop = self.max_hop
        ent_hop = self.ent_hop
        user_dict = self.user_dict_linked
        tail_dict = self.tail_dict
        for user in range(self.user_total):
            if user in user_dict:
                path_list[user + self.entity_total] = user_dict[user]
        for i in range(0, max_hop - 1):
            for j, ent in enumerate(ent_hop[i]):
                if ent in tail_dict:
                    path_list[ent] = tail_dict[ent]
        return path_list

    def get_related_ents(self, ent_now, path_list):
        assert ent_now in path_list
        related_ents = set()
        related_ents.add(ent_now)
        if ent_now < self.share_total:#Mistakes here before, mark
            return related_ents
        rt_list = path_list[ent_now]
        for rt in rt_list:
            rel, tail = rt
            related_ents |= self.get_related_ents(tail, path_list)
        return related_ents

    def remap_entity_batch(self, max_hop, ent_hop):
        hop_map = {}
        entity_remap = {}
        for hop in range(max_hop):
            hop_map[hop] = {}
            hop_start = len(entity_remap)
            for i, ent in enumerate(ent_hop[hop]):
                hop_map[hop][ent] = i
                entity_remap[ent] = i + hop_start
        return hop_map, entity_remap

    def batch_reserve_ents(self, heads, path_list):
        head_list = list(heads.numpy())
        reserved_ents = set()
        for head in head_list:
            if head in self.ent2hop:
                tp_ents = self.get_related_ents(head, path_list)
                reserved_ents |= tp_ents
        hop_ents = {}
        for ent in reserved_ents:
            hop = self.ent2hop[ent]
            hop_ents.setdefault(hop, [])
            hop_ents[hop].append(ent)
        max_hop = max(hop_ents.keys()) + 1
        hop_map, entity_remap = self.remap_entity_batch(max_hop, hop_ents)
        unreach_ents = []
        new_order = []
        for head in head_list:
            if head not in entity_remap:
                entity_remap[head] = len(entity_remap)
                unreach_ents.append(head)
            place = entity_remap[head]
            new_order.append(place)
        return max_hop, hop_ents, hop_map, unreach_ents, new_order

    def get_rgcn_kenel(self, path_reserve, user_set):
        ent_hop = self.ent_hop
        rgcn_kernel = {}
        num_triple = 0
        hop_map_now = self.hop_map[self.max_hop - 1]
        rgcn_kernel[self.max_hop] = sorted([ent for ent in hop_map_now], key=lambda x: hop_map_now[x])
        for i in range(0, self.max_hop - 1):
            triple_list = []
            hop_map_now = self.hop_map[i]
            hop_map_next = self.hop_map[i + 1]
            for j, ent in enumerate(ent_hop[i]):
                if ent in path_reserve:
                    rt_list = path_reserve[ent]
                    for rel, tail in rt_list:
                        triple_list.append((hop_map_now[ent], rel, hop_map_next[tail]))
            rel_edge_list = RGCN_kernal_new(triple_list)
            num_triple += len(triple_list)
            ent_now = sorted([ent for ent in hop_map_now], key=lambda x: hop_map_now[x])
            # ent_next = sorted([ent for ent in hop_map_next], key=lambda x: hop_map_next[x])
            # rgcn_kernel[i + 1] = (rel_edge_list, ent_now, ent_next)
            rgcn_kernel[i + 1] = (rel_edge_list, ent_now)
        triple_list = []
        for user in range(self.user_total):
            if user in user_set and user + self.entity_total in path_reserve:
                current_items = path_reserve[user + self.entity_total]
                for item in current_items:
                    triple_list.append((user, self.relation_total * 2, item))
        rel_edge_list = RGCN_kernal_new(triple_list)
        num_triple += len(triple_list)
        # ent_next = range(self.share_total)
        rgcn_kernel[0] = rel_edge_list
        # print("Number triple:%d"%(num_triple))
        return rgcn_kernel
