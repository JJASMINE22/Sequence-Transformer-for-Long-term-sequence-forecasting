# -*- coding: UTF-8 -*-
'''
@Project ：transformer_for_data_prediction
@File    ：Transformer.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import config as cfg
import matplotlib.pyplot as plt
from torch import nn
from networks import CreateModel


class TransFormer:
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
                 num_layers: int,
                 weight_decay: float,
                 learning_rate: float,
                 load_ckpt: bool,
                 ckpt_path: str,
                 device):

        self.device = device
        self.weight_decay = weight_decay

        self.model = CreateModel(target_size=target_size,
                                 embedding_size=embedding_size,
                                 multihead_num=multihead_num,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding_size=padding_size,
                                 padding_mode=padding_mode,
                                 drop_rate=drop_rate,
                                 shuffle=shuffle,
                                 return_attention=return_attention,
                                 num_layers=num_layers)
        if self.device:
            self.model = self.model.to(self.device)

        if load_ckpt:
            ckpt = torch.load(ckpt_path)
            ckpt_dict = ckpt['state_dict']
            self.model.load_state_dict(ckpt_dict)

        self.loss = nn.MSELoss()

        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=learning_rate)

        self.train_loss, self.val_loss = 0, 0

    def train(self, sources, logits, targets):
        if self.device:
            sources = torch.tensor(sources, dtype=torch.float).to(self.device)
            logits = torch.tensor(logits, dtype=torch.float).to(self.device)
            targets = torch.tensor(targets, dtype=torch.float).to(self.device)

        self.optimizer.zero_grad()

        predictions = self.model(sources, logits)
        loss = self.loss(predictions, targets)
        for weight in self.model.get_weights():
            loss += self.weight_decay * torch.sum(torch.square(weight))

        loss.backward()
        self.optimizer.step()

        self.train_loss += loss.data.item()

    def validate(self, sources, logits, targets):

        if self.device:
            sources = torch.tensor(sources, dtype=torch.float).to(self.device)
            logits = torch.tensor(logits, dtype=torch.float).to(self.device)
            targets = torch.tensor(targets, dtype=torch.float).to(self.device)

        predictions = self.model(sources, logits)
        loss = self.loss(predictions, targets)
        for weight in self.model.get_weights():
            loss += self.weight_decay * torch.sum(torch.square(weight))

        self.val_loss += loss.data.item()

    def generate_sample(self, sources, logits, targets, batches):

        if self.device:
            sources = torch.tensor(sources, dtype=torch.float).to(self.device)
            logits = torch.tensor(logits, dtype=torch.float).to(self.device)
        samples = self.model(sources, logits).cpu().detach().numpy()
        for i in range(cfg.target_seq):
            for k in range(cfg.target_size):
                plt.subplot(cfg.target_size, 1, k + 1)
                plt.plot(samples[:, i, k], color='r', marker='*',
                         linewidth=0.5, label='feature_{:0>1d}_prediction'.format(k + 1),
                         linestyle="--")
                plt.plot(targets[:, i, k], color='y', marker='o',
                         linewidth=0.5, label='feature_{:0>1d}_real'.format(k + 1))
                plt.grid(True)
                plt.legend(loc='upper right', fontsize='xx-small')
                if k == 4:
                    plt.ylabel('Factor Dimensions')
            plt.xlabel('Batch Size')
            plt.savefig(fname=cfg.sample_path + '\\batch{:0>3d}_seq{:d}.jpg'.format(batches, i), dpi=300)
            plt.close()
