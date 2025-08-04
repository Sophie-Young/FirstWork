import argparse
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import os
import os.path as osp
from yaml import SafeLoader
from Data.get_data import get_data
from run import run
from run_batch import run_batch
from Data.data_utils1 import load_fixed_splits
import random
import warnings
from Data.SIR_seq_generate import *
from Data.threshold_SIR_seq import *
from graphwavefile import graphwave

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--gcn_wd', type=float, default=5e-4)
    parser.add_argument('--gpu_id', type=int, default=6)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--gcn_use_bn', action='store_true', help='gcn use batch norm')
    parser.add_argument('--use_patch_attn', action='store_true', help='transformer use patch attention')
    parser.add_argument('--show_details', type=bool, default=True)
    parser.add_argument('--gcn_type', type=int, default=1)
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--n_patch', type=int, default=112)
    parser.add_argument('--batch_size', type=int, default=100000)
    parser.add_argument('--rand_split', action='store_true', help='random split dataset')
    parser.add_argument('--rand_split_class', action='store_true', help='random split dataset by class')
    parser.add_argument('--protocol', type=str, default='semi')
    parser.add_argument('--label_num_per_class', type=int, default=20)
    parser.add_argument('--train_prop', type=float, default=.6)
    parser.add_argument('--valid_prop', type=float, default=.2)
    parser.add_argument('--alpha', type=float, default=.8)
    parser.add_argument('--tau', type=float, default=.3)

    args = parser.parse_args()
    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    fix_seed(config['seed'])
    path = osp.join(osp.expanduser('~/projects/rec1'), 'datasets', args.dataset)
    results = dict()
    n_patch = args.n_patch # 后续可能不要
    alpha = args.alpha
    tau = args.tau
    load_path = None
    if args.dataset in ['ogbn-products']:
        load_path = f'Data/partition/{args.dataset}_partition_{n_patch}.pt'

    postfix = f'{n_patch}'
    postfix = "test"
    runs = 5
    # print("n_patch: ", n_patch)
    #flag 标志着是否有序列需要使用虚拟节点填充  为 1 表示需要
    data = get_data(path, args.dataset) #data 是一个NCDataset类

    # graphwave:
    graphwave_emb=graphwave.extra_node_emb(data)
    # data.graph["node_feat"]=torch.cat((data.graph["node_feat"],graphwave_node_emb.to(torch.float32)),dim=1)

    # print(data.graph["num_nodes"])
    # print(data.graph["edge_index"].shape)
    # print(data.graph["node_feat"].shape)
    # print(len(data.label))

    subgraph_size=24
    max_hop=2
    #patch,flag = data.partition_patch(n_patch, load_path, subgraph_size, max_hop) #随机游走


    #patch=generate_sequence(24,data.graph,data.name) #SIR
    patch=generate_sequence(data.graph,data.name,3) #SIR
    flag=1
    print(patch)
    
    # 创建采样器
    #eg. sampler = EfficientKHopSampler(data, target_subgraph_size, first_hop_sample_size)
    #flag=0


    # get the splits for all runs
    if args.rand_split:
        split_idx_lst = [data.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                         for _ in range(runs)]
    elif args.rand_split_class:
        split_idx_lst = [data.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                         for _ in range(runs)]
    else:
        split_idx_lst = load_fixed_splits(path, data, name=args.dataset, protocol=args.protocol,flag=flag)

    batch_size = args.batch_size

    #results = [[]]
    results = [[], []]
    for r in range(runs):
        if args.dataset in ['Cora', 'CiteSeer', 'PubMed', 'ogbn-arxiv', 'ogbn-products'] and args.protocol == 'semi':
            split_idx = split_idx_lst[0]
        else:
            split_idx = split_idx_lst[r]

        if args.dataset in ['ogbn-products']:
            res_gnn, res_trans = run_batch(args, config, device, data, patch, batch_size, split_idx, alpha, tau,
                                           postfix)
        else:
            #res = run(args, config, device, data, patch, split_idx, alpha, tau, postfix)
            print(patch.shape)
            res_gnn, res_trans = run(graphwave_emb, args, config, device, data, patch, split_idx, alpha, tau, postfix)
        results[0].append(res_gnn)
        results[1].append(res_trans)
    print(f"==== Final GNN====")
    result = torch.tensor(results[0]) * 100.
    print(result)
    print(f"max: {torch.max(result, dim=0)[0]}")
    print(f"min: {torch.min(result, dim=0)[0]}")
    print(f"mean: {torch.mean(result, dim=0)}")
    print(f"std: {torch.std(result, dim=0)}")

    print(f'GNN Micro: {torch.mean(result, dim=0)[1]:.2f} ± {torch.std(result, dim=0)[1]:.2f}')
    print(f'GNN Macro: {torch.mean(result, dim=0)[3]:.2f} ± {torch.std(result, dim=0)[3]:.2f}')

    print(f"==== Final Trans====")
    result = torch.tensor(results[1]) * 100.
    print(result)
    print(f"max: {torch.max(result, dim=0)[0]}")
    print(f"min: {torch.min(result, dim=0)[0]}")
    print(f"mean: {torch.mean(result, dim=0)}")
    print(f"std: {torch.std(result, dim=0)}")

    print(f'Trans Micro: {torch.mean(result, dim=0)[1]:.2f} ± {torch.std(result, dim=0)[1]:.2f}')
    print(f'Trans Macro: {torch.mean(result, dim=0)[3]:.2f} ± {torch.std(result, dim=0)[3]:.2f}')


