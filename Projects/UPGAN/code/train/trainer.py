import torch
import time
import numpy as np
import os, math
from util.Regularization import Regularization
from util.triple_kernel import make_kernel_batch
from train.init import init_model
from train.base_trainer import Trainer
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
tqdm.monitor_iterval = 0


class Trainer_new(Trainer):
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.best_dev_performance = 0.0
        self.device = torch.device('cuda' if args.use_cuda else 'cpu')
        self.learning_rate = self.args.lr
        self.decay_rate = self.args.decay_rate
        self.reset_time = 0
        self.data_load_kg_rs()
        self.load_pretrain(args)
        self.pop_prior_def()
        # self.model.freeze_part()

    def optim_def(self, lr):
        self.learning_rate = lr
        self.optim = optim.Adam(self.model.parameters(), lr=lr)
        self.optim_G = optim.Adam(self.G.parameters(), lr=self.args.lr_g)
        if self.decay_rate:#decay_rate > 0
            self.scheduler = ExponentialLR(self.optim, self.decay_rate)
        self.reset_time += 1

    def model_def(self):
        self.entity_total = len(self.e_map)
        self.relation_total = len(self.r_map)
        self.user_total = len(self.u_map)
        self.item_total = len(self.i_map)
        self.share_total = len(self.i_kg_map)
        self.model, self.G = init_model(self.args, self.user_total, self.item_total,
                                        self.entity_total, self.relation_total, self.logger,
                                        None, None, None, share_total=self.share_total)

    def loss_def(self):
        self.loss_func_kg = torch.nn.BCEWithLogitsLoss(reduction="none")#
        self.loss_reg = Regularization(self.model, weight_decay=self.args.l2_lambda, p=2)
        self.loss_reg_G = Regularization(self.G, weight_decay=self.args.l2_lambda_g, p=2)

    def get_rewards(self, scores, baseline):
        reward_type = self.args.reward_type
        # mean_scores = torch.mean(scores, dim=1, keepdim=True)
        if reward_type == "sigmoid":
            rewards = torch.sigmoid(scores)# * 2 - 1.0
        elif reward_type == "softmax":
            rewards = F.softmax(scores, dim=1)
        elif reward_type == "tanh":
            rewards = torch.tanh(scores)
        elif reward_type == "baseline-tanh":
            rewards = torch.tanh(scores) - torch.tanh(baseline)
        elif reward_type == "baseline-sigmoid":
            rewards = torch.sigmoid(scores) - torch.sigmoid(baseline)
        elif reward_type == "baseline-softmax":
            rewards = F.softmax(scores, dim=1) - 1.0 / self.args.n_sample
        else:
            raise NotImplementedError
        return rewards

    def pop_prior_def(self):
        pop_list = [1.0] * self.entity_total#soft_label
        for head, rel, tail in self.train_triple_list:
            pop_list[head] += 1.0
            pop_list[tail] += 1.0
        self.pop_vec = torch.log10(torch.Tensor(pop_list)).to(self.device)

    def find_common(self, gen_tails, right_tails, recall_list, recall_dict):
        batch_size = gen_tails.size(0)
        num_sample = float(right_tails.size(1))
        assert recall_list[-1] <= num_sample
        g_t = gen_tails.cpu().numpy()
        r_t = right_tails.cpu().numpy()
        for i in range(batch_size):
            set_gen = set(g_t[i])
            for j in recall_list:
                set_real = set(r_t[i][:j])
                recall = len(set_gen & set_real) / j
                recall_dict[j].append(recall)

    def pretrain_G(self):
        num_epoch = self.args.pretrain_epoch
        for i in range(num_epoch):
            st = time.time()
            train_loss, pos_loss, neg_loss = self.pretrain_epoch_G()
            print("Pretrain Epoch: {}, loss: {:.4f}, pos: {:.4f},"
                  " neg: {:.4f}, time: {}".format(i + 1, train_loss, pos_loss,
                                                  neg_loss, time.time() - st))
        print("G pretrain Done!")

    def show_G_norm(self):
        with torch.no_grad():
            norm_emb = torch.norm(self.G.ent_embeddings.weight, dim=1)
            max_norm = torch.max(norm_emb).item()
            min_norm = torch.min(norm_emb).item()
            mean_norm = torch.mean(norm_emb).item()
            print("G Min: {:.4f}, Max: {:.4f}, Mean: {:.4f}".format(min_norm, max_norm, mean_norm))

    def train(self, start_epoch, end_epoch):
        eval_every = self.args.eval_every
        if self.args.need_pretrain:
            self.pretrain_G()
        print("Strat Training------------------")
        self.show_norm()
        self.show_G_norm()
        if self.args.norm_one:
            self.model.norm_one()
        elif self.args.norm_emb:
            self.model.norm_emb()
        if self.args.norm_user:
            self.model.norm_user()
        #self.evaluator_kg.evaluate(self.kg_eval_loader, "EVAL")
        #self.evaluator_kg.evaluate(self.kg_test_loader, "TEST")
        # self.load_G_pretrain()
        # self.eval_sample_diversity(0)
        # exit(-1)
        # self.eval_sample_popularity()
        # self.eval_sample_quality()
        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            self.losses_g = []
            self.reg_G = []
            self.entropy_G = []
            train_loss, pos_loss, neg_loss = self.train_epoch_kg()
            print("Entropy: {:.4f}".format(np.sum(self.entropy_G) / len(self.train_triple_list)))
            print("Epoch: {}, loss: {:.4f}, pos: {:.4f}, neg: {:.4f},"
                  " G: {:.4f}, G Reg: {:.4f},time: {}".format(epoch + 1, train_loss, pos_loss, neg_loss,
                                                              np.mean(self.losses_g), np.mean(self.reg_G),
                                                              time.time() - st))
            self.show_norm()
            self.show_G_norm()

            if self.args.show_quality:
                self.eval_sample_quality()
            # self.eval_sample_popularity()
            if self.args.show_diversity:
                self.eval_sample_diversity(epoch + 1)
            if (epoch + 1) % eval_every == 0 and epoch + 1 > 0:
                # self.eval_sample_quality()
                mr_raw, hits_10_raw, mrr_raw,\
                mr_fil, hits_10_fil, mrr_fil = self.evaluator_kg.evaluate(self.kg_eval_loader, "EVAL")
                #mr_raw, hits_10_raw, mrr_raw, mr_fil, hits_10_fil, mrr_fil = self.evaluate(self.kg_eval_loader, "EVAL")
                # self.load_G_pretrain()#Reset G parameters
                if mrr_fil > self.best_dev_performance:
                    self.best_dev_performance = mrr_fil
                    self.save_ckpt()
                    self.reset_time = 0
                # To guarantee the correctness of evaluation
                else:
                    self.logger.info('No improvement after one evaluation iter.')
                    self.reset_time += 1
                if self.reset_time >= 5:
                    self.logger.info('No improvement after 5 evaluation. Early Stopping.')
                    break
        self.logger.info('Train Done! Evaluate on testset with saved model')
        print("End Training------------------")
        self.evaluate_best()

    def calc_entropy(self, probs):
        entropy = torch.sum(- probs * torch.log(probs))
        return entropy.item()

    def gen_step(self, heads, rels, neg_tails, n_sample=1, temperature=1.0, lambda_smooth=0.5):
        n = heads.size(0)
        logits = self.G.forward_triple(heads, rels, neg_tails) / temperature
        probs = F.softmax(logits, dim=1)
        prior_distribution = self.get_prior(neg_tails)
        probs = (1 - lambda_smooth) * probs + lambda_smooth * prior_distribution
        # probs = (1 - lambda_smooth) * probs + lambda_smooth / probs.size(1)
        row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)
        sample_idx = torch.multinomial(probs, n_sample, replacement=False)
        sample_tails = neg_tails[row_idx, sample_idx]
        rewards = yield sample_tails
        self.optim_G.zero_grad()
        log_probs = F.log_softmax(logits, dim=1)
        # reinforce_loss = -torch.sum(rewards * log_probs[row_idx, sample_idx])
        reinforce_loss = -torch.mean(torch.sum(rewards * log_probs[row_idx, sample_idx], dim=1))
        reg_loss = self.loss_reg_G(self.G)
        loss = reinforce_loss + reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([param for name, param in self.G.named_parameters()],
                                       self.args.clipping_max_value)
        self.optim_G.step()
        self.entropy_G.append(self.calc_entropy(probs))
        self.losses_g.append(reinforce_loss.item())
        self.reg_G.append(reg_loss.item())
        yield None

    def pop_prior(self, neg_tails):
        log_pops = self.pop_vec[neg_tails]
        distributin = F.softmax(log_pops, dim=1)
        return distributin

    def random_prior(self, neg_tails):
        return 1.0 / neg_tails.size(1)

    def get_prior(self, nege_tails):
        if self.args.pop_prior:
            return self.pop_prior(nege_tails)
        return self.random_prior(nege_tails)

    def gen_sample(self, heads, rels, neg_tails, n_sample=1, temperature=1.0, lambda_smooth=0.5):
        with torch.no_grad():
            n = heads.size(0)
            logits = self.G.forward_triple(heads, rels, neg_tails) / temperature
            probs = F.softmax(logits, dim=1)
            prior_distribution = self.get_prior(neg_tails)
            probs = (1 - lambda_smooth) * probs + lambda_smooth * prior_distribution
            row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)
            # print(logits)
            # print(probs)
            sample_idx = torch.multinomial(probs, n_sample, replacement=False)
            sample_tails = neg_tails[row_idx, sample_idx]
        return sample_tails

    def random_sample(self, heads, rels, neg_tails, n_sample=1, temperature=1.0):
        with torch.no_grad():
            n = heads.size(0)
            logits = torch.ones_like(neg_tails).float()
            probs = F.softmax(logits, dim=1)
            row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)
            sample_idx = torch.multinomial(probs, n_sample, replacement=False)
            sample_tails = neg_tails[row_idx, sample_idx]
        return sample_tails

    def calc_scores(self, heads, rels, sample_tails, all_tails, user_rep):
        with torch.no_grad():
            query_now, _ = self.model.form_query(heads, rels, user_rep)
            preds, _ = self.model.query_judge(query_now, sample_tails)
            preds_all, _ = self.model.query_judge(query_now, all_tails)
            baseline = torch.mean(preds_all, dim=1, keepdim=True)
            return preds.data, baseline.data

    def dis_step(self, heads, rels, tails, neg_tails, sample_weight, user_rep):
        self.optim.zero_grad()
        query_now, norm_q = self.model.form_query(heads, rels, user_rep)
        preds, norm_p = self.model.query_judge(query_now, tails)
        preds_neg, norm_n = self.model.query_judge(query_now, neg_tails)

        random_ratio = self.args.label_smoothing_epsilon  / 1024.0#/ self.args.n_sample
        answers_true = torch.ones_like(preds) * (1.0 - self.args.label_smoothing_epsilon)
        answers_false = torch.zeros_like(preds_neg) + random_ratio

        loss_pos = self.loss_func_kg(preds, answers_true)
        loss_pos = (loss_pos * sample_weight).sum()  # / sample_weight.sum()

        loss_neg = torch.sum(self.loss_func_kg(preds_neg, answers_false), dim=1)
        loss_neg = (loss_neg * sample_weight).sum()  # / sample_weight.sum()

        # loss_reg = self.loss_reg(self.model)

        loss = loss_pos + loss_neg

        # losses_reg.append(loss_reg.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_([param for name, param in self.model.named_parameters()],
                                       self.args.clipping_max_value)
        self.optim.step()
        if self.args.norm_one:
            self.model.norm_one()
        elif self.args.norm_emb:
            self.model.norm_emb()
        if self.args.norm_user:
            self.model.norm_user()
        return loss.item(), loss_pos.item(), loss_neg.item()

    def pretrain_epoch_G(self):
        self.G.train()
        losses = []
        losses_pos = []
        losses_neg = []
        for heads, rels, tails, neg_tails, sample_weight in self.kg_train_loader:
            self.optim_G.zero_grad()
            heads = torch.LongTensor(heads).to(self.device)
            rels = torch.LongTensor(rels).to(self.device)
            tails = torch.LongTensor(tails).to(self.device)
            neg_tails = torch.LongTensor(neg_tails).to(self.device)
            sample_weight = sample_weight.float().to(self.device).squeeze()
            query_now = self.G.form_query(heads, rels)
            preds = self.G.query_judge(query_now, tails)
            preds_neg = self.G.query_judge(query_now, neg_tails)

            random_ratio = self.args.label_smoothing_epsilon / 1024.0
            answers_true = torch.ones_like(preds) * (1.0 - self.args.label_smoothing_epsilon)
            answers_false = torch.zeros_like(preds_neg) + random_ratio

            loss_pos = self.loss_func_kg(preds, answers_true)
            loss_pos = (loss_pos * sample_weight).sum()  # / sample_weight.sum()

            loss_neg = torch.sum(self.loss_func_kg(preds_neg, answers_false), dim=1)
            loss_neg = (loss_neg * sample_weight).sum()  # / sample_weight.sum()

            # loss_reg = self.loss_reg(self.model)

            loss = loss_pos + loss_neg
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for name, param in self.G.named_parameters()],
                                           self.args.clipping_max_value)
            self.optim_G.step()
            losses.append(loss.item())
            losses_pos.append(loss_pos.item())
            losses_neg.append(loss_neg.item())
        if self.decay_rate:
            self.scheduler.step()
        mean_losses = np.mean(losses)
        pos_loss = np.mean(losses_pos)
        neg_loss = np.mean(losses_neg)
        return mean_losses, pos_loss, neg_loss

    def train_epoch_kg(self):
        path_list = self.sample_epoch_edges()
        self.model.train()
        self.G.train()
        losses = []
        losses_pos = []
        losses_neg = []
        step = 0
        path_reverse = self.sample_epoch_reverse()
        rgcn_kernel = self.get_rgcn_kenel(path_reserve=path_reverse, user_set=set(range(self.user_total)))
        # st = time.time()
        hr_dict = {}
        for heads, rels, tails, neg_tails, sample_weight in self.kg_train_loader:
            step += 1
            # if step % 5 == 0:
            #     print(step, time.time() - st)
            self.optim.zero_grad()
            max_hop, ent_hop, hop_map, unreach_ents, new_order = self.batch_reserve_ents(heads, path_list)
            rs_graph, kg_graph = make_kernel_batch(path_list, max_hop, ent_hop, hop_map)
            batch_order = torch.LongTensor(new_order).to(self.device)
            user_rep = self.model.fetch_user_batch(rgcn_kernel=rgcn_kernel,
                                                   rs_graph=rs_graph,
                                                   kg_graph=kg_graph,
                                                   tp_hop=max_hop,
                                                   unreach_ids=unreach_ents,
                                                   batch_order=batch_order,
                                                   pretrain=False)
            heads = torch.LongTensor(heads).to(self.device)
            rels = torch.LongTensor(rels).to(self.device)
            tails = torch.LongTensor(tails).to(self.device)
            neg_tails = torch.LongTensor(neg_tails).to(self.device)
            sample_weight = sample_weight.float().to(self.device).squeeze()

            #Train G one step
            gen_step = self.gen_step(heads, rels, neg_tails,
                                     n_sample=self.args.n_sample_gen,
                                     temperature=1.0,
                                     lambda_smooth=self.args.lambda_smooth)
            tails_gen = next(gen_step)
            scores, baseline = self.calc_scores(heads, rels, tails_gen, neg_tails, user_rep)
            rewards = self.get_rewards(scores, baseline)
            gen_step.send(rewards)
            #Train D one step
            loss, loss_pos, loss_neg = self.dis_step(heads, rels, tails, tails_gen, sample_weight, user_rep)
            losses.append(loss)
            losses_pos.append(loss_pos)
            losses_neg.append(loss_neg)

            with torch.no_grad():
                heads_np = heads.cpu().numpy()
                rels_np = rels.cpu().numpy()
                tails_gen = tails_gen.cpu().numpy()
                for i in range(heads.size(0)):
                    query = (heads_np[i], rels_np[i])
                    hr_dict.setdefault(query, set())
                    hr_dict[query] |= set(tails_gen[i])
        if self.decay_rate:
            self.scheduler.step()
        total_diverse = sum([len(hr_dict[query]) for query in hr_dict])
        print("Diversity: {}".format(total_diverse))
        mean_losses = np.mean(losses)
        pos_loss = np.mean(losses_pos)
        neg_loss = np.mean(losses_neg)
        return mean_losses, pos_loss, neg_loss
