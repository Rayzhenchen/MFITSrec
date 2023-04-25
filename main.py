import os
import time

import numpy as np
import torch
import pickle
import argparse
import matplotlib.pyplot as plt


from model import TiSASRec
from tqdm import tqdm
from utils import *

torch.set_printoptions(threshold=np.inf)

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-1m_item')
parser.add_argument('--train_dir', default='10')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=150, type=int)
parser.add_argument('--item_hidden_units', default=50, type=int)
parser.add_argument('--d_c', default=50, type=int)
parser.add_argument('--d_b', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--time_span', default=256, type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, item_meta, usernum, itemnum,timenum, bnum, cnum] = dataset
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('user_sum:', usernum)
    print('item_num:', itemnum)
    print('average sequence length: %.2f' % (cc / len(user_train)))
    print('num_batch', num_batch)

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

    try:
        relation_matrix = pickle.load(open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.time_span),'rb'))
    except:
        relation_matrix = Relation(user_train, usernum, args.maxlen, args.time_span)
        pickle.dump(relation_matrix, open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.time_span),'wb'))

    sampler = WarpSampler(user_train, usernum, itemnum, relation_matrix, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = TiSASRec(usernum, itemnum, timenum, bnum, cnum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass # just ignore those failed init layers

    model.train() # enable model training

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, time_seq, time_matrix, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            time_seq, time_matrix = np.array(time_seq), np.array(time_matrix)
            pos_logits, neg_logits, att_w, attn_w_id, attn_w_b, attn_w_c = model(u, seq, time_matrix, pos, neg,item_meta)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            '''
            for param in model.abs_pos_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.abs_pos_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            '''
            for param in model.time_matrix_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.time_matrix_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()

            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

        if epoch % 1 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('\n')
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            # print('att_w: ', att_w)
            # print('att_w_id', attn_w_id)
            # print('att_w_b', attn_w_b)
            # print('att_w_c', attn_w_c)

            # attn_w_b = attn_w_b[0]
            # attn_w_c = attn_w_c[0]
            # attn_w_id = attn_w_id[0]
            # att = torch.cat([attn_w_id, attn_w_b, attn_w_c], dim=1)
            # print(att.shape, att)
            # folder = args.dataset + '_' + args.train_dir
            # txt = open(folder + '/att_w.txt', 'w')
            # attn_w_b = attn_w_b.cpu()
            # attn_w_b = attn_w_b.detach().numpy()
            # attn_w_c = attn_w_c.cpu()
            # attn_w_c = attn_w_c.detach().numpy()
            # attn_w_id = attn_w_id.cpu()
            # attn_w_id = attn_w_id.detach().numpy()
            #
            # att = att.cpu()
            # att = att.detach().numpy()
            # # np.savetxt(r'Test.txt', attn_w_b, fmt='%d', delimiter=',')  # 存储矩阵Test
            # # np.savetxt(folder + '/att_w.txt', attn_w_b)
            # # txt.write(str(attn_w_b))
            # txt.close()
            #
            # plt.matshow(att, fignum=48, cmap='Blues', vmin=-1, vmax=1)
            # plt.colorbar(cax=None, ax=None, shrink=0.8)
            # plt.show()


            # plt.matshow(attn_w_b, fignum=48, cmap='Blues', vmin=0, vmax=1)
            # plt.colorbar(cax=None, ax=None, shrink=0.8)
            # plt.show()
            # plt.matshow(attn_w_c, fignum=48, cmap='pink', vmin=0, vmax=1)
            # plt.colorbar(cax=None, ax=None, shrink=0.8)
            # plt.show()
            # plt.matshow(attn_w_id, fignum=48, cmap='Greens', vmin=0, vmax=1)
            # plt.colorbar(cax=None, ax=None, shrink=0.8)
            # # plt.title("Foursquare")
            # # plt.savefig('D:\桌面\\Foursquare_time_sim_matrix.png')
            # plt.show()

            att = torch.cat([attn_w_id, attn_w_b, attn_w_c], dim=2)
            att_w = att_w.cpu()
            att_w = att_w.detach().numpy()
            att = att.cpu()
            att = att.detach().numpy()
            # attn_w_id = attn_w_id.numpy()
            # attn_w_b = attn_w_b.numpy()
            # attn_w_c = attn_w_c.numpy()

            folder = args.dataset + '_' + args.train_dir
            np.save(folder + '/att_w.npy', att_w)
            np.save(folder + '/att.npy', att)

            fname = 'TiSASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done")