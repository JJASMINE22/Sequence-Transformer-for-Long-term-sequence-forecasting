# -*- coding: UTF-8 -*-
'''
@Project ：transformer_for_data_prediction
@File    ：data_generator.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os
import math
import datetime
import numpy as np
import pandas as pd
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.feature_bagging import FeatureBagging


class Generator(object):
    def __init__(self,
                 txt_path: str,
                 batch_size: int,
                 ratio: float,
                 source_seq: int,
                 target_seq: int
                 ):
        self.txt_path = txt_path
        self.batch_size = batch_size
        self.ratio = ratio
        self.source_seq = source_seq
        self.target_seq = target_seq
        self.radian = np.pi / 180
        self.train_source, self.train_target, self.val_source, self.val_target = self.split_train_val()

    def split_train_val(self):

        total_data, seq_source, target = self.preprocess()
        index = np.arange(len(seq_source))
        np.random.shuffle(index)
        train_source = seq_source[index[:int(len(index)*self.ratio)]]
        train_target = target[index[:int(len(index)*self.ratio)]]
        val_source = seq_source[index[int(len(index)*self.ratio):]]
        val_target = target[index[int(len(index)*self.ratio):]]

        return train_source, train_target, val_source, val_target

    def erase_default_value(self, x, t):

        row_index, col_index = np.where(np.equal(x, ''))
        total_index = list(np.arange(x.shape[0]))

        for default_index in list(set(row_index)):
            total_index.remove(default_index)
        return x[np.array(total_index)],\
               t[np.array(total_index)]

    def erase_anomal_value(self, x, t):

        clf = FeatureBagging(base_estimator=KNN(), max_features=x.shape[-1])
        clf.fit(x)
        position_index = 1 - clf.predict(x)

        return x[np.array(position_index).astype('bool')], \
               t[np.array(position_index).astype('bool')]

    def preprocess(self):
        df = pd.read_excel(self.txt_path, keep_default_na=False)

        time_stamp = df['时间']
        df = df.drop(columns=['城市', '时间', '天气', '风向', '风级(级)', '日降雨量(mm)', '平均总云量(%)'])

        values = df.values
        # erase default values
        values, time_stamp = self.erase_default_value(values, time_stamp)
        # erase anomal values
        values, time_stamp = self.erase_anomal_value(values, time_stamp)
        values = np.concatenate([np.array(time_stamp)[:, np.newaxis],
                                 np.array(values)], axis=-1)
        df = pd.DataFrame(data=values, columns=['时间'] + [*df.keys()])

        # recover the time stamp
        df['时间'] = df['时间'].apply(lambda x: datetime.datetime.strptime(x.split(' ')[0], '%Y-%m-%d'))
        df = df.set_index('时间')
        index = pd.date_range(df.index[0], df.index[-1], freq='D')
        df = df.reindex(index, fill_value=np.nan)
        df = df.astype(float)
        df = df.interpolate(method='polynomial', order=5)

        # divide the wind speed
        x_speed = np.array(list(map(lambda i:
                                    np.array(df['风速(m/s)'])[i] * np.cos(np.array(df['风向角度(度)'])[i] * self.radian),
                                    np.arange(len(df)))))

        y_speed = np.array(list(map(lambda i:
                                    np.array(df['风速(m/s)'])[i] * np.sin(np.array(df['风向角度(度)'])[i] * self.radian),
                                    np.arange(len(df)))))

        df.insert(loc=4, column='横向风速', value=x_speed)
        df.insert(loc=5, column='纵向风速', value=y_speed)
        df = df.drop(columns=['风速(m/s)'])

        # -1~1 normalize
        df = 2 * (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0)) - 1
        assign_source = np.array([np.array(df)[i:i + self.source_seq]
                                  for i in range(len(df) - self.source_seq - self.target_seq + 1)])
        assign_target = np.array([np.array(df)[i + self.source_seq: i + self.source_seq + self.target_seq]
                                  for i in range(len(df) - self.source_seq - self.target_seq + 1)])

        return np.array(df), assign_source, assign_target

    def get_train_len(self):
        train_len = len(self.train_source)
        if not train_len % self.batch_size:
            return train_len//self.batch_size
        else:
            return train_len//self.batch_size + 1

    def get_val_len(self):
        val_len = len(self.val_source)
        if not val_len % self.batch_size:
            return val_len//self.batch_size
        else:
            return val_len//self.batch_size + 1

    def generate(self, training=True):
        """
        data generation
        """
        while True:
            if training:
                idx = np.arange(len(self.train_source))
                np.random.shuffle(idx)
                train_source = self.train_source[idx]
                train_target = self.train_target[idx]
                targets, sources = [], []
                for i, (src, tgt) in enumerate(zip(train_source, train_target)):
                    sources.append(src)
                    targets.append(tgt)
                    if np.equal(len(targets), self.batch_size) or np.equal(i+1, train_source.shape[0]):
                        annotation_targets, annotation_sources = targets.copy(), sources.copy()
                        targets.clear()
                        sources.clear()
                        yield np.array(annotation_sources), np.array(annotation_targets)

            else:
                idx = np.arange(len(self.val_source))
                np.random.shuffle(idx)
                val_source = self.val_source[idx]
                val_target = self.val_target[idx]
                targets, sources = [], []
                for i, (src, tgt) in enumerate(zip(val_source, val_target)):
                    sources.append(src)
                    targets.append(tgt)
                    if np.equal(len(targets), self.batch_size) or np.equal(i+1, val_source.shape[0]):
                        annotation_targets, annotation_sources = targets.copy(), sources.copy()
                        targets.clear()
                        sources.clear()
                        yield np.array(annotation_sources), np.array(annotation_targets)
