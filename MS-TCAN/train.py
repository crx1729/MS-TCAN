import os
import time
from utils import *
from model import *


def train(data_file, args):
    random.seed(20240101)
    np.random.seed(20240101)
    torch.manual_seed(20240101)
    torch.cuda.manual_seed_all(20240101)

    train_data = SeqData(data_file=os.path.join(data_file, args.dataset + '.csv'), max_len=args.max_len)
    train_load = DataLoader(train_data, args.batch_size)
    model = Mamba4Rec(train_data.user_num, train_data.item_num, args,
                      args.device, args.dropout_rate).to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    loss_f = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    best_metric = 0.0
    best_epoch = 0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        average_loss = []
        t0 = time.time()
        for step, (u, seq, pos, neg, ts) in enumerate(train_load):
            u, seq, pos, neg, ts = u.to(args.device), seq.to(args.device), pos.to(args.device), neg.to(
                args.device), ts.to(args.device)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)
            optimizer.zero_grad()
            indices = torch.where(pos != 0)
            loss_f.pos_weight = torch.from_numpy(np.linspace(0.1, 1, args.max_len)).to(args.device).repeat(len(u), 1)[
                indices]
            loss = loss_f(pos_logits[indices], pos_labels[indices])
            loss += loss_f(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()
            average_loss.append(loss.item())
        average_loss = sum(average_loss) / len(average_loss)
        print("Epoch {}: loss={:.4f}, time_cost={:.4f}".format(epoch, average_loss, time.time() - t0))

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time()
            print('Evaluating', end='')
            HR_valid, NDCG_valid, MRR_valid, multi_metric = evaluate(model, train_data.dict_data, train_data.data_ts,
                                                                     args,
                                                                     train_data.user_num, train_data.item_num,
                                                                     ks=args.ks, type_='valid', multi_class=True)
            print('evaluation time cost: {:.4f}'.format(time.time() - t1))
            print(multi_metric)
            if HR_valid[0] > best_metric:
                best_metric = HR_valid[0]
                best_epoch = epoch

        if epoch - best_epoch >= 20 or epoch == args.num_epochs:
            print('Early Stop at Epoch{}'.format(epoch))
            HR_test, NDCG_test, MRR_test, multi_metric = evaluate(model, train_data.dict_data, train_data.data_ts, args,
                                                                  train_data.user_num, train_data.item_num,
                                                                  ks=args.ks, type_='test', multi_class=True)
            print('Test Set')
            print(multi_metric)
            break
        if epoch == 100:
            print('Metric at Epoch{}'.format(epoch))
            HR_test, NDCG_test, MRR_test, multi_metric = evaluate(model, train_data.dict_data, train_data.data_ts, args,
                                                                  train_data.user_num, train_data.item_num,
                                                                  ks=args.ks, type_='test', multi_class=True)
            print('Test Set')
            # print(test_metrics)
            print(multi_metric)
    return multi_metric