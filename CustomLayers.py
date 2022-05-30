# -*- coding: UTF-8 -*-
'''
@Project ：transformer_for_data_prediction
@File    ：CustomLayers.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class SeqShuffleUnit(nn.Module):
    """
    Feature shuffling by matrix transposition
    """
    def __init__(self,
                 g: int=None):
        super(SeqShuffleUnit, self).__init__()
        self.g = g

    def forward(self, x):

        raise Exception("关注并联系作者获取该类, e-mail:m13541280433@163.com")


class FormerDecoderAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 multihead_num: int,
                 kernel_size: int or tuple,
                 strides: int or tuple,
                 drop_rate: float=None,
                 shuffle: bool=False,
                 return_attention: bool=False):
        super(FormerDecoderAttention, self).__init__()
        assert in_channels
        assert not in_channels % multihead_num

        self.q_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=kernel_size, stride=strides, padding='valid',
                                bias=False, groups=multihead_num)

        self.k_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=kernel_size, stride=strides, padding='valid',
                                bias=False, groups=multihead_num)

        self.v_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=kernel_size, stride=strides, padding='valid',
                                bias=False, groups=multihead_num)

        self.layer_norm = nn.LayerNorm(normalized_shape=in_channels)

        self.multihead_num = multihead_num
        self.multihead_size = int(in_channels / multihead_num)
        if shuffle:
            self.q_shuffle = SeqShuffleUnit(g=self.multihead_size)
            self.k_shuffle = SeqShuffleUnit(g=self.multihead_size)
            self.v_shuffle = SeqShuffleUnit(g=self.multihead_size)

        self.return_attention = return_attention
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.drop_rate = drop_rate
        self.shuffle = shuffle

        self.init_params()

    def forward(self, input, init_weight, mask=None):

        """
        gconv1d于训练、预测时计算输出不同, 需使用特殊padding机制
        联系作者获取
        """

        q = self.q_conv(input)
        k = self.k_conv(input)
        v = self.v_conv(input)

        if self.shuffle:
            q = self.q_shuffle(q)
            k = self.k_shuffle(k)
            v = self.v_shuffle(v)

        q = torch.cat(torch.split(q, split_size_or_sections=self.multihead_size, dim=1), dim=0)
        k = torch.cat(torch.split(k, split_size_or_sections=self.multihead_size, dim=1), dim=0)
        v = torch.cat(torch.split(v, split_size_or_sections=self.multihead_size, dim=1), dim=0)

        attention = torch.matmul(q.transpose(2, 1), k)
        attention /= torch.sqrt(torch.tensor(self.multihead_num, dtype=attention.dtype))

        if mask is not None:
            attention -= 1e+9 * mask
        attention = torch.softmax(attention, dim=-1)

        attention = torch.dropout(attention, p=self.drop_rate, train=True)

        o = torch.matmul(attention, v.transpose(2, 1))

        o = torch.cat(torch.split(o, split_size_or_sections=batch_size, dim=0), dim=-1)

        # if self.source_size != self.embed_size:
        #     o = self.main_conv(o.transpose(2, 1))

        o = torch.dropout(o, p=self.drop_rate, train=True)

        feat = o.transpose(2, 1) + init
        feat = self.layer_norm(feat.transpose(2, 1))

        if self.return_attention:
            raise Exception("关注并联系作者获取完整方法, e-mail:m13541280433@163.com")
        raise Exception("关注并联系作者获取完整方法, e-mail:m13541280433@163.com")

    def init_params(self):

        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[0] == 'layer_norm':
                    continue
                else:
                    # glorot normal
                    stddev = math.sqrt(2/(sum(param.size()[:2])))
                    torch.nn.init.normal_(param, mean=0, std=stddev)


class LatterDecoderAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 multihead_num: int,
                 kernel_size: int or tuple,
                 strides: int or tuple,
                 padding_size: int or str=None,
                 padding_mode: str = 'reflect',
                 drop_rate: float=None,
                 shuffle: bool=False,
                 return_attention: bool=False):
        super(LatterDecoderAttention, self).__init__()
        assert in_channels
        assert not in_channels % multihead_num
        assert padding_mode in ['zeros', 'reflect', 'replicate', 'circular']
        if isinstance(padding_size, str):
            assert padding_size == 'same'

        self.q_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=kernel_size, stride=strides, padding='valid',
                                padding_mode=padding_mode, bias=False, groups=multihead_num)

        self.k_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=kernel_size, stride=strides, padding=padding_size,
                                padding_mode=padding_mode, bias=False, groups=multihead_num)

        self.v_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=kernel_size, stride=strides, padding=padding_size,
                                padding_mode=padding_mode, bias=False, groups=multihead_num)

        self.layer_norm = nn.LayerNorm(normalized_shape=in_channels)

        self.multihead_num = multihead_num
        self.multihead_size = int(in_channels / multihead_num)
        if shuffle:
            self.q_shuffle = SeqShuffleUnit(g=self.multihead_size)
            self.k_shuffle = SeqShuffleUnit(g=self.multihead_size)
            self.v_shuffle = SeqShuffleUnit(g=self.multihead_size)

        self.return_attention = return_attention
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.drop_rate = drop_rate
        self.shuffle = shuffle

        self.init_params()

    def forward(self, init, enc_output, mask=None):
        """
        gconv1d于训练、预测时计算输出不同, 需使用特殊padding机制
        联系作者获取
        """

        q = self.q_conv(input)
        k = self.k_conv(enc_output)
        v = self.v_conv(enc_output)

        if self.shuffle:
            q = self.q_shuffle(q)
            k = self.k_shuffle(k)
            v = self.v_shuffle(v)

        q = torch.cat(torch.split(q, split_size_or_sections=self.multihead_size, dim=1), dim=0)
        k = torch.cat(torch.split(k, split_size_or_sections=self.multihead_size, dim=1), dim=0)
        v = torch.cat(torch.split(v, split_size_or_sections=self.multihead_size, dim=1), dim=0)

        attention = torch.matmul(q.transpose(2, 1), k)
        attention /= torch.sqrt(torch.tensor(self.multihead_num, dtype=attention.dtype))

        if mask is not None:
            attention -= 1e+9 * mask
        attention = torch.softmax(attention, dim=-1)

        attention = torch.dropout(attention, p=self.drop_rate, train=True)

        o = torch.matmul(attention, v.transpose(2, 1))

        o = torch.cat(torch.split(o, split_size_or_sections=batch_size, dim=0), dim=-1)

        # if self.source_size != self.embed_size:
        #     o = self.main_conv(o.transpose(2, 1))

        o = torch.dropout(o, p=self.drop_rate, train=True)

        feat = o.transpose(2, 1) + init
        feat = self.layer_norm(feat.transpose(2, 1))

        if self.return_attention:
            raise Exception("关注并联系作者获取完整方法, e-mail:m13541280433@163.com")
        raise Exception("关注并联系作者获取完整方法, e-mail:m13541280433@163.com")

    def init_params(self):

        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[0] == 'layer_norm':
                    continue
                else:
                    # glorot normal
                    stddev = math.sqrt(2/(sum(param.size()[:2])))
                    torch.nn.init.normal_(param, mean=0, std=stddev)


class GConvMultiHeadAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 multihead_num: int,
                 kernel_size: int or tuple,
                 strides: int or tuple,
                 padding_size: int or str=None,
                 padding_mode: str = 'reflect',
                 drop_rate: float=None,
                 shuffle: bool=False,
                 return_attention: bool=False):
        super(GConvMultiHeadAttention, self).__init__()
        assert in_channels and out_channels
        assert not (in_channels % multihead_num or out_channels % multihead_num)
        assert padding_mode in ['zeros', 'reflect', 'replicate', 'circular']
        if isinstance(padding_size, str):
            assert padding_size == 'same'

        self.q_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=strides, padding=padding_size,
                                padding_mode=padding_mode, bias=False, groups=multihead_num)

        self.k_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=strides, padding=padding_size,
                                padding_mode=padding_mode, bias=False, groups=multihead_num)

        self.v_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=strides, padding=padding_size,
                                padding_mode=padding_mode, bias=False, groups=multihead_num)

        if out_channels != in_channels:
            self.main_conv = nn.Conv1d(in_channels=out_channels, out_channels=in_channels,
                                       kernel_size=kernel_size, stride=strides, padding=padding_size,
                                       padding_mode=padding_mode, bias=False)

        self.layer_norm = nn.LayerNorm(normalized_shape=in_channels)

        self.multihead_num = multihead_num
        self.multihead_size = int(out_channels / multihead_num)
        if shuffle:
            self.q_shuffle = SeqShuffleUnit(g=self.multihead_size)
            self.k_shuffle = SeqShuffleUnit(g=self.multihead_size)
            self.v_shuffle = SeqShuffleUnit(g=self.multihead_size)

        self.return_attention = return_attention
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_rate = drop_rate
        self.shuffle = shuffle

        self.init_params()

    def init_params(self):

        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[0] == 'layer_norm':
                    continue
                else:
                    # glorot normal
                    stddev = math.sqrt(2/(sum(param.size()[:2])))
                    torch.nn.init.normal_(param, mean=0, std=stddev)

    def forward(self, inputs, mask=None):
        assert isinstance(inputs, list)
        assert inputs.__len__() >= 2
        q, init = inputs[0], inputs[0]
        k = inputs[1]
        v = inputs[2] if inputs.__len__() > 2 else k
        batch_size = init.size(0)

        q = self.q_conv(q)
        k = self.k_conv(k)
        v = self.v_conv(v)

        if self.shuffle:
            q = self.q_shuffle(q)
            k = self.k_shuffle(k)
            v = self.v_shuffle(v)

        q = torch.cat(torch.split(q, split_size_or_sections=self.multihead_size, dim=1), dim=0)
        k = torch.cat(torch.split(k, split_size_or_sections=self.multihead_size, dim=1), dim=0)
        v = torch.cat(torch.split(v, split_size_or_sections=self.multihead_size, dim=1), dim=0)

        attention = torch.matmul(q.transpose(2, 1), k)
        attention /= torch.sqrt(torch.tensor(self.multihead_num, dtype=attention.dtype))

        if mask is not None:
            attention -= 1e+9 * mask
        attention = torch.softmax(attention, dim=-1)

        attention = torch.dropout(attention, p=self.drop_rate, train=True)

        o = torch.matmul(attention, v.transpose(2, 1))

        o = torch.cat(torch.split(o, split_size_or_sections=batch_size, dim=0), dim=-1)

        if self.in_channels != self.out_channels:
            o = self.main_conv(o.transpose(2, 1))

        o = torch.dropout(o, p=self.drop_rate, train=True)

        feat = o.transpose(2, 1) + init
        feat = self.layer_norm(feat.transpose(2, 1))

        if self.return_attention:
            return feat.transpose(2, 1), attention
        return feat.transpose(2, 1)
