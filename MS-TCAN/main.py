import os
import torch
from train import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--max_len', default=200, type=int)
parser.add_argument('--dim', default=64, type=int)
parser.add_argument('--num_blocks', default=5, type=int)
parser.add_argument('--num_stack', default=1, type=int)
parser.add_argument('--dilation', default=[1,2], type=list, help='len(dilation) must equal to num_blocks')
parser.add_argument('--kernal_size', default=7, type=int)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--early_stop', default=20, type=int)
parser.add_argument('--ks', default=[10, 20], type=list)
parser.add_argument('--num_evaluate', default=999, type=int)
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
args = parser.parse_args([])
args.dilation = [int(2**i) for i in range(args.num_blocks)]

data_file = './dataset'
train(data_file, args)