# -*- coding: UTF-8 -*-
'''
@Project ：transformer_for_data_prediction
@File    ：networks.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import math
import torch
import config as cfg
from torch import nn
from CustomLayers import (GConvMultiHeadAttention,
                          FormerDecoderAttention,
                          LatterDecoderAttention)


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 multihead_num: int,
                 kernel_size: int,
                 strides: int,
                 padding_size: int or str,
                 padding_mode: str,
                 drop_rate: float,
                 shuffle: bool,
                 return_attention: bool,
                 num_layers: int):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.multihead_num = multihead_num
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding_size = padding_size
        self.padding_mode = padding_mode
        self.drop_rate = drop_rate
        self.shuffle = shuffle
        self.return_attention = return_attention
        self.num_layers = num_layers

        self.enc_layers = nn.ModuleList([GConvMultiHeadAttention(in_channels=self.in_channels,
                                                                 out_channels=self.out_channels,
                                                                 multihead_num=self.multihead_num,
                                                                 kernel_size=self.kernel_size,
                                                                 strides=self.strides,
                                                                 padding_size=self.padding_size,
                                                                 padding_mode=self.padding_mode,
                                                                 drop_rate=self.drop_rate,
                                                                 shuffle=self.shuffle,
                                                                 return_attention=self.return_attention)
                                         for i in range(self.num_layers)])

    def forward(self, x):

        enc_outputs, attentions = [], []
        for i in range(self.num_layers):
            x, attention = self.enc_layers[i]([x, x, x])
            enc_outputs.append(x)
            attentions.append(attention)

        return enc_outputs, attentions


class DecoderLayer(nn.Module):
    """
    Eliminate FeedForward mechanism
    """
    def __init__(self,
                 in_channels: int,
                 multihead_num: int,
                 kernel_size: int,
                 strides: int,
                 padding_size: int or str,
                 padding_mode: str,
                 drop_rate: float,
                 shuffle: bool,
                 return_attention: bool):
        super(DecoderLayer, self).__init__()
        self.in_channels = in_channels
        self.multihead_num = multihead_num
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding_size = padding_size
        self.padding_mode = padding_mode
        self.drop_rate = drop_rate
        self.shuffle = shuffle
        self.return_attention = return_attention

        self.former_attn = FormerDecoderAttention(in_channels=self.in_channels,
                                                  multihead_num=self.multihead_num,
                                                  kernel_size=self.kernel_size,
                                                  strides=self.strides,
                                                  drop_rate=self.drop_rate,
                                                  shuffle=self.shuffle,
                                                  return_attention=self.return_attention)

        self.latter_attn = LatterDecoderAttention(in_channels=self.in_channels,
                                                  multihead_num=self.multihead_num,
                                                  kernel_size=self.kernel_size,
                                                  strides=self.strides,
                                                  padding_size=self.padding_size,
                                                  padding_mode=self.padding_mode,
                                                  drop_rate=self.drop_rate,
                                                  shuffle=self.shuffle,
                                                  return_attention=self.return_attention)

    def forward(self, input, init_weight, enc_output, mask=None):

        x, _ = self.former_attn(input, init_weight, mask)
        x, attention = self.latter_attn(x, enc_output)

        return x, attention


class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 multihead_num: int,
                 kernel_size: int,
                 strides: int,
                 padding_size: int or str,
                 padding_mode: str,
                 drop_rate: float,
                 shuffle: bool,
                 return_attention: bool,
                 num_layers: int):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.multihead_num = multihead_num
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding_size = padding_size
        self.padding_mode = padding_mode
        self.drop_rate = drop_rate
        self.shuffle = shuffle
        self.return_attention = return_attention
        self.num_layers = num_layers

        self.dec_layers = nn.ModuleList([DecoderLayer(in_channels=self.in_channels,
                                                      multihead_num=self.multihead_num,
                                                      kernel_size=self.kernel_size,
                                                      strides=self.strides,
                                                      padding_size=self.padding_size,
                                                      padding_mode=self.padding_mode,
                                                      drop_rate=self.drop_rate,
                                                      shuffle=self.shuffle,
                                                      return_attention=self.return_attention)
                                         for i in range(self.num_layers)])

    def forward(self, x, init_weight, enc_outputs, mask):

        raise Exception("关注并联系作者获取decoder完整方法, e-mail:m13541280433@163.com")

class CreateModel(nn.Module):
    """
    Unlike the standard Transformer
    Padding_mask, Positional_encoding and other mechanisms are eliminated
    Use linear transformation instead of embedding operation
    """
    def __init__(self,
                 target_size: int,
                 embedding_size: int,
                 multihead_num: int,
                 kernel_size: int,
                 strides: int,
                 padding_size: int or str,
                 padding_mode: str,
                 drop_rate: float,
                 shuffle: bool,
                 return_attention: bool,
                 num_layers: int):
        super(CreateModel, self).__init__()
        self.target_size = target_size
        self.embedding_size = embedding_size
        self.multihead_num = multihead_num
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding_size = padding_size
        self.padding_mode = padding_mode
        self.drop_rate = drop_rate
        self.shuffle = shuffle
        self.return_attention = return_attention
        self.num_layers = num_layers

        self.linear_enc = nn.Linear(in_features=self.target_size,
                                    out_features=self.embedding_size)
        self.linear_dec = nn.Linear(in_features=self.target_size,
                                    out_features=self.embedding_size)
        self.linear = nn.Linear(in_features=self.embedding_size,
                                out_features=self.target_size)

        self.encoder = Encoder(in_channels=self.embedding_size,
                               out_channels=self.embedding_size,
                               multihead_num=self.multihead_num,
                               kernel_size=self.kernel_size,
                               strides=self.strides,
                               padding_size=self.padding_size,
                               padding_mode=self.padding_mode,
                               drop_rate=self.drop_rate,
                               shuffle=self.shuffle,
                               return_attention=self.return_attention,
                               num_layers=self.num_layers)

        self.decoder = Decoder(in_channels=self.embedding_size,
                               multihead_num=self.multihead_num,
                               kernel_size=self.kernel_size,
                               strides=self.strides,
                               padding_size=self.padding_size,
                               padding_mode=self.padding_mode,
                               drop_rate=self.drop_rate,
                               shuffle=self.shuffle,
                               return_attention=self.return_attention,
                               num_layers=self.num_layers)

        self.weight = "初始序列特征"

        self.init_params()

    def sequence_mask(self, seq_len):

        seq_mask = torch.triu(torch.ones(size=(seq_len, )*2), diagonal=1)
        seq_mask = seq_mask.unsqueeze(dim=0)

        return seq_mask

    def forward(self, src_seq, tgt_seq):
        """
        For training, use parallel inference
        For prediction, use sequential inference
        训练、预测的时序列输入不同,
        联系作者获取
        """

        enc_input = self.linear_enc(src_seq).transpose(2, 1)

        enc_outputs, _ = self.encoder(enc_input)

        dec_output, _ = self.decoder(dec_input, self.weight, enc_outputs, mask)

        output = self.linear(dec_output.transpose(2, 1))

        raise Exception("关注并联系作者获取完整方法, e-mail:m13541280433@163.com")

    def init_params(self):

        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name == 'weight':
                    torch.nn.init.normal_(param)
                elif name in ['linear_enc.weight', 'linear_dec.weight', 'linear.weight']:
                    stddev = math.sqrt(2/(sum(param.size()[:2])))
                    torch.nn.init.normal_(param, mean=0, std=stddev)
                elif name in ['linear_enc.bias', 'linear_dec.bias', 'linear.bias']:
                    torch.nn.init.zeros_(param)
                else:
                    continue

    def get_weights(self):

        weights = []
        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] == 'weight':
                    weights.append(param)
                else:
                    continue
        return weights
