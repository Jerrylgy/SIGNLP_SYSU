#! -*- coding: utf-8 -*-


from utils import *
import sys
import jieba

jieba.initialize()

# 基本参数
model_type, pooling, task_name, n_components = sys.argv[1:]
assert model_type in ['BERT', 'RoBERTa']
assert pooling in ['first-last-avg', 'last-avg', 'cls', 'pooler']
assert task_name in ["OPPO"]

n_components = int(n_components)
if n_components < 0:
    n_components = 768

maxlen = 64

# 加载数据集
data_path = './datasets'
#无监督任务，本质上不需要训练验证集之类的
datasets = {
    '%s-%s' % (task_name, f):
    load_data('%s/%s/%s' % (data_path,task_name,f))
    for f in ['dev']
}

# bert配置，RoBERTa效果好于BERT，更大更牛逼
model_name = {
    'BERT': 'chinese_L-12_H-768_A-12',
    'RoBERTa': 'chinese_roberta_wwm_ext_L-12_H-768_A-12'
}[model_type]

config_path = './pretrainModels/%s/bert_config.json' % model_name
checkpoint_path = './pretrainModels/%s/bert_model.ckpt' % model_name
dict_path = './pretrainModels/%s/vocab.txt' % model_name

# 建立分词器
tokenizer = get_tokenizer(dict_path)

# 建立模型

encoder = get_encoder(config_path, checkpoint_path, pooling=pooling)

# 语料向量化
all_names, all_weights, all_vecs, all_labels = [], [], [], []
for name, data in datasets.items():
    a_vecs, b_vecs, labels = convert_to_vecs(data, tokenizer, encoder, maxlen)
    all_names.append(name)
    all_weights.append(len(data))
    all_vecs.append((a_vecs, b_vecs))
    all_labels.append(labels)

# 计算变换矩阵和偏置项
if n_components == 0:
    kernel, bias = None, None
else:
    kernel, bias = compute_kernel_bias([v for vecs in all_vecs for v in vecs])
    kernel = kernel[:, :n_components]

# 变换，标准化，相似度，相关系数
all_corrcoefs = []
for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
    a_vecs = transform_and_normalize(a_vecs, kernel, bias)
    b_vecs = transform_and_normalize(b_vecs, kernel, bias)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(labels, sims)
    all_corrcoefs.append(corrcoef)

all_corrcoefs.extend([
    np.average(all_corrcoefs),
    np.average(all_corrcoefs, weights=all_weights)
])

for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
    print('%s: %s' % (name, corrcoef))
