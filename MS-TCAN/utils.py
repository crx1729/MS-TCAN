import sys
import torch
import random
import copy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class SeqData(Dataset):
    def __init__(self, data_file, max_len):
        super(SeqData, self).__init__()
        self.data_file = data_file
        self.max_len = max_len
        self.load()

    def load(self):
        df = pd.read_csv(self.data_file)
        self.dict_data = defaultdict(list)
        self.data_ts = defaultdict(list)
        df['timestamp'] = df['timestamp'] - df['timestamp'].min()
        for _, u, i, _, ts in df.itertuples():
            if u in self.dict_data.keys():
                self.dict_data[u].append(i)
                self.data_ts[u].append(ts)
            else:
                self.dict_data[u] = [i]
                self.data_ts[u] = [ts]
        self.user_num = len(df['userid'].unique())
        self.item_num = len(df['itemid'].unique())
        self.ts_len = df['timestamp'].max() + 1
        self.df = []
        for u in self.dict_data:
            for l in self.dict_data[u][:-2]:
                self.df.append([u, l])
#         self.df = np.array(df)
        print('Dataset built')

    def __len__(self):
        return len(self.dict_data)

    def __getitem__(self, idx):
        items = self.dict_data[idx][:-1][-(self.max_len+1):]
        ts = self.data_ts[idx][:-1][-(self.max_len+1):]
        l = len(items)
        user = torch.tensor(idx, dtype=torch.long)
        seq = torch.zeros(self.max_len, dtype=torch.long)
        ts_seq = torch.zeros(self.max_len, dtype=torch.long)
        pos = torch.zeros(self.max_len, dtype=torch.long)
        neg = torch.zeros(self.max_len, dtype=torch.long)
        seq[-(l-1):] = torch.tensor(items[:-1], dtype=torch.long)
        ts_seq[-(l-1):] = torch.tensor(ts[:-1], dtype=torch.long)
        pos[-(l-1):] = torch.tensor(items[1:], dtype=torch.long)
        neg[-(l-1):] = torch.tensor(self.neg_sample(idx, l-1), dtype=torch.long)
        return user, seq, pos, neg, ts_seq

    def neg_sample(self, user, n):
        neg_items = []
        while len(neg_items) < n:
            this_item = np.random.randint(1, self.item_num + 1)
            while this_item in self.dict_data[user]:
                this_item = np.random.randint(1, self.item_num + 1)
            neg_items.append(this_item)
        return neg_items

def evaluate(model, dict_data, dict_ts, args, user_num, item_num, ks, type_='valid', multi_class=False):
    test_idx = 2 if type_ == 'valid' else 1
    all_MRR = []
    all_NDCG = []
    all_HT = []
    int_num = []
    valid_user = 0.0

    if user_num > 10000:
        users = random.sample(range(user_num), 10000)
    else:
        users = range(user_num)
    for u in users:
        l = len(dict_data[u])
        if l < 5: continue
        int_num.append(l)
        seq = torch.zeros([args.max_len], dtype=torch.long)
        temp_seq = dict_data[u][-(args.max_len+test_idx):-test_idx]
        seq[-len(temp_seq):] = torch.tensor(temp_seq, dtype=torch.long)
        ts_seq = torch.zeros([args.max_len], dtype=torch.long)
        temp_ts = dict_ts[u][-(args.max_len+test_idx):-test_idx]
        ts_seq[-len(temp_ts):] = torch.tensor(temp_ts, dtype=torch.long)
        rated = set(dict_data[u])
        rated.add(0)
        item_idx = [dict_data[u][-test_idx]]
        if args.num_evaluate == 1000000:
            item_idx = list(range(1, item_num + 1))
        else:
            for _ in range(args.num_evaluate):
                t = np.random.randint(1, item_num + 1)
                while t in rated: t = np.random.randint(1, item_num + 1)
                item_idx.append(t)
        item_idx = torch.tensor(item_idx, dtype=torch.long, device=model.device)
        predictions = -model.predict(torch.tensor([u], dtype=torch.long, device=model.device),
                                     seq.unsqueeze(dim=0).to(model.device), item_idx)
        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1
        MRR = [0] * len(ks)
        NDCG = [0] * len(ks)
        HT = [0] * len(ks)
        for i, k in enumerate(ks):
            if rank < k:
                MRR[i] += 1 / (rank + 1)
                NDCG[i] += 1 / np.log2(rank + 2)
                HT[i] += 1
        all_MRR.append(MRR)
        all_NDCG.append(NDCG)
        all_HT.append(HT)
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    all_MRR = np.array(all_MRR)
    all_NDCG = np.array(all_NDCG)
    all_HT = np.array(all_HT)
    if multi_class:
        cl = ['int_num']+['HT@{}'.format(k) for k in ks]+['NDCG@{}'.format(k) for k in ks]+['MRR@{}'.format(k) for k in ks]
        metric = pd.DataFrame(np.c_[int_num, all_HT, all_NDCG, all_MRR], columns=cl)
        metric['user_type'] = np.clip((np.log2(metric['int_num'])-2).astype(int), 1, 5)
        cl.remove('int_num')
        result = metric.groupby('user_type').mean()[cl]
        try:
            result = result._append(pd.DataFrame([np.r_[all_HT.mean(axis=0), all_NDCG.mean(axis=0), all_MRR.mean(axis=0)]], columns=cl))
        except:
            result = result.append(pd.DataFrame([np.r_[all_HT.mean(axis=0), all_NDCG.mean(axis=0), all_MRR.mean(axis=0)]], columns=cl))
        result.index = ['max_seq{}'.format(2 ** (i + 3)) for i in result.index.values[:-1]] + ['average']
        return all_HT.mean(axis=0), all_NDCG.mean(axis=0), all_MRR.mean(axis=0), result
    else:
        return all_HT.mean(axis=0), all_NDCG.mean(axis=0), all_MRR.mean(axis=0)