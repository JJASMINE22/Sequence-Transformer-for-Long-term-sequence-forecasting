# -*- coding: UTF-8 -*-
'''
@Project ：transformer_for_data_prediction
@File    ：config.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch

# ===data loader===
text_path='C:\\DATASET\\time_series_data\\weather_data.xlsx'
epoches=200
batch_size=8
ratio=0.7
source_seq=7
target_seq=6

# ===model===
target_size=9
embedding_size=256
multihead_num=4
kernel_size=3
strides=1
padding_size='same'
padding_mode='reflect'
shuffle=True
return_attention=True
num_layers=3
drop_rate=0.
learning_rate = 3e-4
weight_decay = 5e-4
per_sample_interval = 50
device = torch.device('cuda') if torch.cuda.is_available() else None
checkpoint_path = '.\\saved\\checkpoint'
sample_path = '.\\sample'
load_ckpt = True

# ===prediction===
roll_time = 10
