import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weight(m):
    nn.init.xavier_normal_(m.weight)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)


class SelfAttention(nn.Module):

    def __init__(self, args):
        super(SelfAttention, self).__init__()
        self.hidden_dim = args.hidden_dim  # default: 512
        self.attention_dim = args.hidden_dim // args.n_head  # n_head =8(defalut)

        # Q, K, V 의 space 를 만들어줌(Linear: 모든 단어와 연관시키기 위해서 사용)
        self.q_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.k_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.v_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        # 초기화 Linear
        init_weight(self.q_w)
        init_weight(self.k_w)
        init_weight(self.v_w)

        self.dropout = nn.Dropout(args.dropout)
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).cuda(args.gpu)

    def forward(self, query, key, value, mask=None):

        # q, k ,v = [batch_size, sentence_length, attention_dim]
        q = self.q_w(query)
        # ("Linear_Q: ", q.size())
        # Encoder Linear_Q: [128, 8, 64] 64 == 512 // 8
        # Decoder Linear_Q: [128, 22, 64] 64 == 512 // 8
        # enc_dec [128, 22, 64]
        k = self.k_w(key)
        # ("Linear_K: ", k.size())
        # Encoder Linear_K: [128, 8, 64]
        # Decoder Linear_K: [128, 22, 64]
        # enc_dec [128, 8, 64]
        v = self.v_w(value)
        # ("Linear_V: ", v.size())
        # Encoder Linear_V: [128, 8, 64]
        # Decoder Linear_V: [128, 22, 64]
        # enc_dec [128, 8, 64]

        # paper) Attention = softmax(QK^T / sqrt(d_k)) * V
        self_attention = torch.bmm(q, k.permute(0, 2, 1))
        # ("Q*K: ", self_attention.size())
        # Encoder Q*K: [128, 8, 8]
        # Decoder Q*K: [128, 22, 22]
        # enc_dec [128, 22, 8]
        self_attention = self_attention / self.scale_factor
        # self_attention = [batch_size, sentence_length, sentence_length]

        if mask is not None:
            # todo #  mask 어떻게 생겼는지
            self_attention = self_attention.masked_fill(mask, -np.inf)
            # a.masked_fill(mask, value): a에 mask 값이 True 일 시 value값으로 대체
            # a = [0.5 0.1 -0.3] ==> a.maksed_fill(a > 0, -inf)
            # a = [-inf, -inf, -0.3]

        attention_score = F.softmax(self_attention, dim=-1)
        # ("Softmax(Q*K/scale): ", attention_score.size())
        # Encoder Softmax(Q*K/scale): [128, 8, 8]
        # Decoder Softmax(Q*K/scale): [128, 22, 22]
        # enc_dec [128, 22, 8]

        norm_attention_score = self.dropout(attention_score)
        # attention_score = [batch_size, sentence_length, sentence_length]

        weighted_v = torch.bmm(norm_attention_score, v)
        # ("Softmax(Q*K/scale)*V: ", weighted_v.size())
        # Encoder [128, 8, 64]
        # Decoder [128, 22, 64]
        # enc_dec [128, 22, 64]
        # [batch_size, sentence_length, attention_dim]
        # bmm => 내적곱이라고 생각 하면됨 (seq_length x seq_length)*(seq_length x attention_dim)
        # ==> (seq_length, attention_dim)

        return self.dropout(weighted_v), attention_score


class MultiHeadAttention(nn.Module):

    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        # default h_head = 8 / hidden_dim 512
        # assert args.hidden_dim % args.n_head == 0 # embed_dim 은 head 수만큼 나눌 예정이므로
        self.attentions = nn.ModuleList([SelfAttention(args)
                                         for _ in range(args.n_head)]) # h_head만큼 실행

        self.o_w = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        init_weight(self.o_w)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, query, key, value, mask=None):
        # ("----[Multi_Head_Attention]----")
        # q k ,v = [batch_size, sentence_length, hidden_dim]
        # 아직 linear 걸치기 전이라 같은 tensor임
        # sentence_length 는 6, 18 등 다양하게 변함

        # ("Q: ", query.size())
        # Encoder Q: [128, 8, 512]
        # Decoder Q [128, 22, 64]
        # enc_dec [128, 22, 512]
        # ("K: ", key.size())
        # Encoder K: [128, 8, 512]
        # Decoder Q [128, 22, 64]
        # enc_dec [128, 8, 512]
        # ("V: ", value.size())
        # Encoder V: [128, 8, 512]
        # Decoder Q [128, 22, 64]
        # enc_dec [128, 8, 512]
        # print("Q", query.size())

        self_attentions = [attention(query, key, value, mask) for attention in self.attentions]
        # self_attentions [batch_size, sentence_length, attention_dim] * n_head
        # (8, 2): 8 head 수 / softmax_dot_value 와 attention score 값들 저장
        # ("Scaled_dot_product_Attention_Output: ", np.array(self_attentions).shape)
        # Encoder (8,2)
        #

        weight_vs = [weighted_v[0] for weighted_v in self_attentions]
        # (8, ) = 8 head 수 / softmax_dot_value 값들
        # ("Softmax(Q*K/scale)*V list: ", np.array(weight_vs).shape)
        #  (8, )

        attentions = [weighted_v[1] for weighted_v in self_attentions]
        # (8, ) = 8 head 수 / attention score 값들
        # ("Softmax(Q*K/scale) list: ", np.array(attentions).shape)
        # Encoder (8, )

        # ("----[Concat - Linear]----")
        weighted_v = torch.cat(weight_vs, dim=-1)
        # ("Concat: ", weighted_v.size())
        # Encoder [128, 7, 512]
        # Decoder [128, 22, 512]
        # enc_dec [128, 22, 512]
        # [batch_size, sentence_length, hidden_dim]

        output = self.dropout(self.o_w(weighted_v))
        # ("Multi_Head_output(Linear): ", output.size())
        # Encoder [128, 7, 512]
        # Decoder [128, 22, 512]
        # enc_dec [128, 22, 512]
        # [batch_size, sentence_length, hidden_dim] =

        return output, attentions