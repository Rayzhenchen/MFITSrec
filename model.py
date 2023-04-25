import numpy as np
import torch
import sys

FLOAT_MIN = -sys.float_info.max

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate): # wried, why fusion X 2?

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class TimeAwareMultiHeadAttention(torch.nn.Module):
    # required homebrewed mha layer for Ti/SASRec experiments
    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, abs_pos_K, abs_pos_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        # key masking, -2^32 lead to leaking, inf lead to nan
        # 0 * inf = nan, then reduce_sum([nan,...]) = nan

        # fixed a bug pointed out in https://github.com/pmixer/TiSASRec.pytorch/issues/2
        # time_mask = time_mask.unsqueeze(-1).expand(attn_weights.shape[0], -1, attn_weights.shape[-1])
        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) * (-2 ** 32 + 1)  # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights)  # True:pick padding
        attn_weights = torch.where(attn_mask, paddings, attn_weights)  # enforcing causality

        attn_weights = self.softmax(attn_weights)  # code as below invalids pytorch backward rules
        # attn_weights = torch.where(time_mask, paddings, attn_weights) # weird query mask in tf impl
        # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4
        # attn_weights[attn_weights != attn_weights] = 0 # rm nan for -inf into softmax case
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2)  # div batch_size

        return outputs


class MFITSRec(torch.nn.Module): # similar to torch.nn.MultiheadAttention
    def __init__(self, user_num, item_num, time_num,bnum,cnum, args):
        super(MFITSRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.b_num = bnum
        self.c_num = cnum
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.item_hidden_units, padding_idx=0)
        self.item_b = torch.nn.Embedding(self.b_num + 1, args.d_b, padding_idx=0)
        self.item_c = torch.nn.Embedding(self.c_num + 1, args.d_c, padding_idx=0)
        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)


        self.abs_pos_K_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.abs_pos_V_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.time_matrix_K_emb = torch.nn.Embedding(args.time_span+1, args.hidden_units)
        self.time_matrix_V_emb = torch.nn.Embedding(args.time_span+1, args.hidden_units)

        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = TimeAwareMultiHeadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate,
                                                            args.device)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def Item_Meta(self, seqs, item_meta):
        a = int(seqs.shape[0])
        b = int(seqs.shape[1])

        seq = seqs.reshape(a, b, 1)

        item = []
        item_seq = []
        for k in range(a):
            for i in range(b):
                x = list(seq[k][i])
                if x == [0]:
                    x.append(0)
                    x.append(0)
                else:
                    x.append(item_meta[x[0]][0])
                    x.append(item_meta[x[0]][1])
                item.append(x)
            item_seq.append(item)
            item = []

        item_seq = np.array(item_seq)
        item_ids = item_seq[:, :, 0:1].reshape(a, b)
        item_b = item_seq[:, :, 1:2].reshape(a, b)
        item_c = item_seq[:, :, 2:].reshape(a, b)
        return item_seq, item_ids, item_b, item_c

    def seq2feats(self, log_seqs, time_matrices, item_meta):
        item_seq, item_ids, item_b, item_c = self.Item_Meta(log_seqs, item_meta)
        seqs = self.item_emb(torch.LongTensor(item_ids).to(self.dev))
        seq_b = self.item_b(torch.LongTensor(item_b).to(self.dev))
        seq_c = self.item_c(torch.LongTensor(item_c).to(self.dev))
        # seqs = torch.cat([seqs, seq_c], dim=2)
        seqs = torch.cat([seqs, seq_b, seq_c], dim=2)
        seqs *= self.item_emb.embedding_dim ** 0.5
        seqs = self.item_emb_dropout(seqs)


        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        positions = torch.LongTensor(positions).to(self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)


        time_matrices = torch.LongTensor(time_matrices).to(self.dev)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        # mask 0th items(placeholder for dry-run) in log_seqs
        # would be easier if 0th item could be an exception for training
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            # Self-attention, Q=layernorm(seqs), K=V=seqs
            # seqs = torch.transpose(seqs, 0, 1) # (N, T, C) -> (T, N, C)
            Q = self.attention_layernorms[i](seqs) # PyTorch mha requires time first fmt
            mha_outputs = self.attention_layers[i](Q, seqs,
                                            timeline_mask, attention_mask,
                                            time_matrix_K, time_matrix_V, abs_pos_K, abs_pos_V)
            seqs = Q + mha_outputs
            # seqs = torch.transpose(seqs, 0, 1) # (T, N, C) -> (N, T, C)

            # Point-wise Feed-forward, actually 2 Conv1D for channel wise fusion
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, time_matrices, pos_seqs, neg_seqs,item_meta): # for training
        log_feats = self.seq2feats(log_seqs, time_matrices, item_meta)

        pos_seq, pos_ids, pos_b, pos_c = self.Item_Meta(pos_seqs, item_meta)
        neg_seq, neg_ids, neg_b, neg_c = self.Item_Meta(neg_seqs, item_meta)

        pos_embs = self.item_emb(torch.LongTensor(pos_ids).to(self.dev))
        pos_emb_b = self.item_b(torch.LongTensor(pos_b).to(self.dev))
        pos_emb_c = self.item_c(torch.LongTensor(pos_c).to(self.dev))
        # pos_embs = torch.cat([pos_embs, pos_emb_c], dim=2)
        pos_embs = torch.cat([pos_embs, pos_emb_b, pos_emb_c], dim=2)

        neg_embs = self.item_emb(torch.LongTensor(neg_ids).to(self.dev))
        neg_emb_b = self.item_b(torch.LongTensor(neg_b).to(self.dev))
        neg_emb_c = self.item_c(torch.LongTensor(neg_c).to(self.dev))
        # neg_embs = torch.cat([neg_embs, neg_emb_c], dim=2)
        neg_embs = torch.cat([neg_embs, neg_emb_b, neg_emb_c], dim=2)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, time_matrices, item_indices,item_meta): # for inference
        log_feats = self.seq2feats(log_seqs, time_matrices,item_meta)
        item_indices = item_indices.reshape(1, -1)
        seq, ids, b, c = self.Item_Meta(item_indices, item_meta)

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(ids).to(self.dev)) # (U, I, C)
        emb_b = self.item_b(torch.LongTensor(b).to(self.dev))
        emb_c = self.item_c(torch.LongTensor(c).to(self.dev))
        # item_embs = torch.cat([item_embs, emb_c], dim=2)
        item_embs = torch.cat([item_embs, emb_b, emb_c], dim=2)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
