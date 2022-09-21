# RoBERTa-Shopname-Classification

利用RoBERTa做的商品名分类，RoBERTa以huggingface hub上的预训练模型做锚点


<br />

<p align="center">
  

  <h3 align="center">"RoBERTa分类</h3>
  <p align="center">
    <br />
    <a href="https://github.com/clock-uni/RoBERTa-Shopname-Classification/"><strong>探索本项目的文档 »</strong></a>
    <br />
    <br />
    <a href="https://github.com/clock-uni/RoBERTa-Shopname-Classification/edit/main/README.md">查看Demo</a>
    ·
    <a href="https://github.com/clock-uni/RoBERTa-Shopname-Classification/edit/main/README.md/issues">报告Bug</a>
    ·
    <a href="https://https://github.com/clock-uni/RoBERTa-Shopname-Classification/edit/main/README.md/issues">提出新特性</a>
  </p>

</p>


 
## 目录

- [环境配置](#环境配置)
- [模型说明](#模型说明)
- [文件目录说明](#文件目录说明)

### 环境配置

1. trasformers == 4.18.0
2. pytorch 1.11.0-gpu

### 模型说明
RoBERTa引用了来自huggingface hub的预训练锚点 https://huggingface.co/uer/roberta-base-finetuned-jd-full-chinese  
最后直接一层fc输出维度，本来在中间插了层BiLSTM但效果反而下降了。。。  

### 文件目录说明
eg:

```
filetree 
├── main.py
├── README.md
├── readfile.py
├── model.py
├── train.py
├── nlpdataset
  ├── shops
    ├── data_202106c.csv


```





