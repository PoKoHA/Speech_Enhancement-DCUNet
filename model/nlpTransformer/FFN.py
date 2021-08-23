import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weight(m):
    nn.init.xavier_normal_(m.weight)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)


class PositionWiseFeedForward(nn.Module):

    def __init__(self, args):
        super(PositionWiseFeedForward, self).__init__()

        # default = hidden_dim=512 / feed_forward_dim=2048
        self.conv1 = nn.Conv1d(args.hidden_dim, args.feed_forward_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(args.feed_forward_dim, args.hidden_dim, kernel_size=1)
        # nn.Conv1d 의 input (N, C) => N: batch_size C: Channels

        init_weight(self.conv1)
        init_weight(self.conv2)

        # self.dropout = nn.Dropout(args.dropout)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        # ("----[PositionWiseFeedForward]----")
        # ("Input: ", input.size())
        # Encoder [128, 7, 512]
        # decoder [128, 22, 512]
        # input [batch_size, sentence_length, hidden_dim]

        # input 을 conv input 형태로 바꿔줌
        input = input.permute(0, 2, 1)  # [batch_size, hidden_dim, sentence_length]
        conv1 = self.conv1(input)  # [batch_size, feed_forward_dim, sentence_length]
        # ("Conv1: ", conv1.size())
        # Encoder [128, 2048, 8]
        # decoder [128, 2048, 22]
        conv1_relu = F.relu(conv1)
        conv1_relu_drop = self.dropout(conv1_relu)

        conv2 = self.conv2(conv1_relu_drop)  # [batch_size, hidden_dim, sentence_length]
        # ("conv2: ", conv2.size())
        # Encoder [128, 512, 8]
        # Decoder [128, 512, 22]

        output = conv2.permute(0, 2, 1)  # [batch_size, sentence_length, hidden_dim]

        # ("FeedForward Output: ", output.size())
        # Encoder [128 , 8, 512]
        # Decoder [128 , 22, 512]
        return self.dropout(output)

