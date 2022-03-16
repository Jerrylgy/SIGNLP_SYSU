## 文件

```
- utils.py  工具函数
- eval.py  评测主文件
```

## 评测

命令格式：
```
python eval.py [model_type] [pooling] [task_name] [n_components]
```

使用例子：
```
python eval.py BERT cls OPPO 256
```

其中四个参数必须传入，含义分别如下：
```
- model_type: 模型，必须是['BERT', 'RoBERTa']之一；
- pooling: 池化方式，必须是['first-last-avg', 'last-avg', 'cls', 'pooler']之一；
- task_name: 评测数据集，仅在OPPO数据集上做实验，因此必须为"OPPO"
- n_components: 保留的维度，如果是0，则不进行whitening-transformation，如果是负数，则保留全部维度，如果是正数，则按照所给的维度保留（本工作中，若进行降维，则取）；
```


