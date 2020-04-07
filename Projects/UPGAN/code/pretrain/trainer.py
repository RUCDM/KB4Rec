import torch
import time
import numpy as np
import os, math
from util.Regularization import Regularization
from util.triple_kernel import make_kernel_batch
from pretrain.init import init_model
from pretrain.base_trainer import Trainer
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
        # self.model.freeze_part()

    def optim_def(self, lr):
        self.learning_rate = lr
        self.optim = optim.Adam(self.model.parameters(), lr=lr)
        if self.decay_rate:#decay_rate > 0
            self.scheduler = ExponentialLR(self.optim, self.decay_rate)
        self.reset_time += 1

    def model_def(self):
        self.entity_total = len(self.e_map)
        self.relation_total = len(self.r_map)
        self.user_total = len(self.u_map)
        self.item_total = len(self.i_map)
        self.share_total = len(self.i_kg_map)
        self.model= init_model(self.args, self.user_total, self.item_total,
                               self.entity_total, self.relation_total, self.logger,
                               None, None, None, share_total=self.share_total)

    def loss_def(self):
        self.loss_func_kg = torch.nn.BCEWithLogitsLoss(reduction="none")#
        self.loss_reg = Regularization(self.model, weight_decay=self.args.l2_lambda, p=2)

    # @override
    def save_ckpt(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'best_dev_performance': self.best_dev_performance
        }
        model_name = os.path.join(self.args.checkpoint_dir, "{}.ckpt".format(self.args.experiment_name))
        torch.save(checkpoint, model_name)
        print("Save model as %s" % model_name)

    def train(self, start_epoch, end_epoch):
        eval_every = self.args.eval_every
        print("Strat Training------------------")
        self.show_norm()
        if self.args.norm_one:
            self.model.norm_one()
        elif self.args.norm_emb:
            self.model.norm_emb()
        if self.args.norm_user:
            self.model.norm_user()
        #self.evaluator_kg.evaluate(self.kg_eval_loader, "EVAL")
        #self.evaluator_kg.evaluate(self.kg_test_loader, "TEST")
        # if self.args.need_pretrain:
        #     self.pretrain(start_epoch, end_epoch)
        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            train_loss, pos_loss, neg_loss = self.train_epoch_kg()
            self.show_norm()
            print("Epoch: {}, loss: {:.4f}, pos: {:.4f},"
                  " neg: {:.4f}, time: {}".format(epoch + 1, train_loss, pos_loss,
                                                  neg_loss, time.time() - st))
            if (epoch + 1) % eval_every == 0 and epoch > 0:
                mr_raw, hits_10_raw, mrr_raw,\
                mr_fil, hits_10_fil, mrr_fil = self.evaluator_kg.evaluate(self.kg_eval_loader, "EVAL")
                #mr_raw, hits_10_raw, mrr_raw, mr_fil, hits_10_fil, mrr_fil = self.evaluate(self.kg_eval_loader, "EVAL")
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

    def dis_step(self, heads, rels, tails, neg_tails, sample_weight, user_rep):
        self.optim.zero_grad()
        query_now, norm_q = self.model.form_query(heads, rels, user_rep)
        preds, norm_p = self.model.query_judge(query_now, tails)
        preds_neg, norm_n = self.model.query_judge(query_now, neg_tails)

        random_ratio = self.args.label_smoothing_epsilon / self.args.n_sample
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

    def train_epoch_kg(self):
        path_list = self.sample_epoch_edges()
        self.model.train()
        losses = []
        losses_pos = []
        losses_neg = []
        step = 0
        path_reverse = self.sample_epoch_reverse()
        rgcn_kernel = self.get_rgcn_kenel(path_reserve=path_reverse, user_set=set(range(self.user_total)))
        for heads, rels, tails, neg_tails, sample_weight in self.kg_train_loader:
            step += 1
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

            loss, loss_pos, loss_neg = self.dis_step(heads, rels, tails, neg_tails, sample_weight, user_rep)
            losses.append(loss)
            losses_pos.append(loss_pos)
            losses_neg.append(loss_neg)
        if self.decay_rate:
            self.scheduler.step()
        mean_losses = np.mean(losses)
        pos_loss = np.mean(losses_pos)
        neg_loss = np.mean(losses_neg)
        return mean_losses, pos_loss, neg_loss
