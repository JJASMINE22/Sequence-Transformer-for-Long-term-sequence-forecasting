## Sequence Transformer based on Gconv-MultiHeadAttention --Pytorch
---

## 目录
1. [所需环境 Environment](#所需环境)
2. [注意力结构 Attention Structure](#注意力结构)
2. [模型结构 Model Structure](#模型结构)
3. [注意事项 Cautions](#注意事项)
4. [文件下载 Download](#文件下载)
5. [训练步骤 How2train](#训练步骤)
6. [预测效果 predict](#位置编码)
7. [参考资料 Reference](#参考资料)

## 所需环境
1. Python3.7
2. PyTorch>=1.10.1+cu113
3. numpy==1.19.5
4. pandas==1.2.4
5. pyod==0.9.8
6. matplotlib==3.2.2
6. CUDA 11.0+  

## 注意力结构
![image](https://github.com/JJASMINE22/Sequence-Transformer-for-Long-term-sequence-forecasting/blob/main/attention%20structure/attention.jpg)  

## 模型结构
Encoder  
由全连接层、一维分组卷积多头注意力机制组成  
![image](https://github.com/JJASMINE22/Sequence-Transformer-for-Long-term-sequence-forecasting/blob/main/model%20structure/encoder.jpg)  

Decoder  
由全连接层、一维分组卷积多头注意力机制组成  
![image](https://github.com/JJASMINE22/Sequence-Transformer-for-Long-term-sequence-forecasting/blob/main/model%20structure/decoder.jpg) 

Sequence Transformer  
合并Encoder-Decoder，拼接全连接层  
![image](https://github.com/JJASMINE22/Sequence-Transformer-for-Long-term-sequence-forecasting/blob/main/model%20structure/transformer.jpg) 

## 注意事项
1. 时序数据推理，删除了标准Transformer的位置掩码、位置编码、前馈层等机制
2. 使用一个正态分布变量替代起始序列特征
3. 将Linear MultiHeadAttention替换为GConv MultiHeadAttention
4. 训练时，并行推理解码序列；预测时，贯续推理解码序列
5. 提出特殊的边界序列填充方法，克服卷积操作引发的差异性，保证训练、预测阶段的运算机制相同
6. 保留三角掩码，防止特征泄露
7. 加入权重正则化操作，防止过拟合

## 文件下载    
链接：https://pan.baidu.com/s/13T1Qs4NZL8NS4yoxCi-Qyw 
提取码：sets 
下载解压后放置于config.py中设置的路径即可。

## 训练步骤
运行train.py即可开始训练。  

## 预测效果
sequence_1  
![image](https://github.com/JJASMINE22/Sequence-Transformer-for-Long-term-sequence-forecasting/blob/main/sample/sequence1.jpg)  

sequence_2  
![image](https://github.com/JJASMINE22/Sequence-Transformer-for-Long-term-sequence-forecasting/blob/main/sample/sequence2.jpg) 

sequence_3  
![image](https://github.com/JJASMINE22/Sequence-Transformer-for-Long-term-sequence-forecasting/blob/main/sample/sequence3.jpg)  

## 参考资料
https://arxiv.org/pdf/1706.03762.pdf

