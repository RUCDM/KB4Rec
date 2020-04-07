import argparse
import sys
from pretrain.trainer import Trainer_new
from util.utils import create_logger
import time
import torch
import numpy as np
import os
import datetime
import copy

parser = argparse.ArgumentParser()
# parse the optimizer arguments
parser.add_argument('--optimizer_type', default='Adam', type=str)
parser.add_argument('--l2_lambda', default=1e-6, type=float)
parser.add_argument('--l2_lambda_d', default=1e-6, type=float)
parser.add_argument('--l2_lambda_g', default=1e-6, type=float)
parser.add_argument('--momentum', default=0.9, type=float, help="The momentum of the optimizer.")
parser.add_argument('--eps', default=1e-6, type=float)
parser.add_argument("--seed", type=int, default=42, help="Seed for random initialization")
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--lr_d', default=2e-4, type=float)
parser.add_argument('--lr_g', default=2e-4, type=float)
parser.add_argument("--clipping_max_value", default=5.0, type=float)
parser.add_argument("--learning_rate_decay_when_no_progress", default=0.5, type=float)
parser.add_argument("--decay_rate", default=0.0, type=float)
# Hyper-parameters
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--eval_batch_size', default=512, type=int)
parser.add_argument('--n_epochs', default=500, type=int)
parser.add_argument("--embedding_size", type=int, default=100, help="using embedding")
parser.add_argument("--noise_size", type=int, default=100, help="using noise")

# Dataset-parameters
parser.add_argument('--dataset', default='music', type=str)
parser.add_argument('--save_dir', default='models', type=str)
parser.add_argument('--data_folder', default='/home/gaole_he/data/new_exp', type=str)
parser.add_argument('--n_sample', type=int, default=1)
parser.add_argument('--n_sample_gen', type=int, default=1)
parser.add_argument('--gen_topk', type=int, default=1)
parser.add_argument("--importance_sample", action='store_true')
parser.add_argument("--query_weight", action='store_true')
parser.add_argument('--number_pop', type=int, default=1)
parser.add_argument("--sample_pop", action='store_true')

#parameters for evaluation only mode
parser.add_argument("--is_eval", action='store_true')
parser.add_argument("--load_experiment", default=None, type=str)
parser.add_argument("--eval_every", default=3, type=int)

#parameters for training loss
parser.add_argument("--margin", default=1.0, type=float)
parser.add_argument("--bpr_target", default=1.0, type=float)
parser.add_argument("--wgan", action='store_true')
parser.add_argument("--wgan_gp", action='store_true')
parser.add_argument("--leak_user", action='store_true')
parser.add_argument('--gan_type', default='lsgan', type=str)

#parameters for pretrained models
parser.add_argument('--load_ckpt_file', default=None,  type=str)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/movie')
parser.add_argument('--log_level', type=str, default='info')
parser.add_argument('--experiment_name', type=str, default=None)
parser.add_argument("--need_pretrain", action='store_true')
parser.add_argument("--pretrain_epoch", default=10, type=int)

parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--num_processes", default=4, type=int)
parser.add_argument("--max_queue", default=10, type=int)

#Dropout
parser.add_argument("--input_dropout", default=0.2, type=float)
parser.add_argument("--rs_dropout", default=0.2, type=float)
parser.add_argument("--label_smoothing_epsilon", default=0.1, type=float)
parser.add_argument("--boundary_smoothing_epsilon", default=0.5, type=float)
parser.add_argument("--dropout_rate", default=0.25, type=float)

#UGAT parameters
parser.add_argument('--model_name', default='UGAT', type=str)
parser.add_argument("--lambda_rs", default=0.5, type=float)
parser.add_argument("--gat_split", default=10, type=int)
parser.add_argument("--rs_gnn", default='GCN', type=str)
parser.add_argument("--num_head", default=1, type=int)
parser.add_argument("--rs_sample", default=10, type=int)
parser.add_argument("--kg_sample", default=10, type=int)
parser.add_argument("--rs_sample_flag", action='store_true')
parser.add_argument("--kg_sample_flag", action='store_true')
parser.add_argument("--use_activation", action='store_true')
parser.add_argument("--hop_limit", default=-1, type=int)
parser.add_argument("--alpha", default=0.01, type=float)

# GAN parameters
parser.add_argument('--D_name', default='DistMult', type=str)
parser.add_argument('--G_name', default='Generator', type=str)
parser.add_argument('--load_ckpt_D', default=None,  type=str)
parser.add_argument('--load_ckpt_G', default=None,  type=str)
parser.add_argument("--g_steps", default=1, type=int)
parser.add_argument("--d_steps", default=1, type=int)
parser.add_argument("--g_use_concat", action='store_true')
parser.add_argument("--share_emb", action='store_true')
parser.add_argument("--g_layers", default=2, type=int)
parser.add_argument("--d_layers", default=2, type=int)
parser.add_argument("--lambda_gp", default=10.0, type=float)
parser.add_argument("--l2_coef", default=1.0, type=float)
parser.add_argument("--l1_coef", default=1.0, type=float)
parser.add_argument("--sigma", default=1.0, type=float)
parser.add_argument("--mix_step", default=0.02, type=float)
parser.add_argument("--fix_G", action='store_true')
parser.add_argument("--fix_D", action='store_true')
parser.add_argument("--norm_emb", action='store_true')
parser.add_argument("--norm_user", action='store_true')
parser.add_argument("--norm_one", action='store_true')
parser.add_argument("--noisy_labels", action='store_true')
parser.add_argument("--leak_info", action='store_true')
parser.add_argument("--add_noise", action='store_true')
parser.add_argument("--mask_pos", action='store_true')
parser.add_argument('--reward_type', default='softmax-mean', type=str)
# parser.add_argument("--lambda_g", default=1.0, type=float, help="The ratio of score loss in G.")
# parser.add_argument("--lambda_d", default=1.0, type=float, help="The ratio of entity loss in D.")

# Get the arguments
args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.experiment_name == None:
    timestamp = str(int(time.time()))
    args.experiment_name = "{}-{}-{}".format(
        args.dataset,
        args.model_name,
        timestamp,
    )


def main():
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    logger = create_logger(args)
    trainer = Trainer_new(args=args, logger=logger)
    if not args.is_eval:###Only create logger when training mode
        trainer.train(0, args.n_epochs - 1)
    else:
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            print("Loading pre trained model from {}".format(ckpt_path))
            trainer.evaluate_single(ckpt_path)
        else:
            print("Pre trained model is None!")

if __name__ == '__main__':
    main()
