from transformers import BertTokenizer, BertModel
import torch
import pickle
import csv

import pickle


def read_pickle_and_print(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        print(data)

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 读取实体描述文件
data = []
with open('entity_description.txt', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        data.append(row[1])

# 逐批次进行分词和编码
batch_size = 16  # 根据内存情况调整批次大小
text_features_list = []

for i in range(0, len(data), batch_size):
    print(i);

    batch_data = data[i:i + batch_size]

    # 对文本进行分词和编码
    encoded_data = tokenizer(batch_data, padding=True, truncation=True, return_tensors='pt')

    # 使用BERT模型获取文本特征
    with torch.no_grad():
        outputs = model(**encoded_data)

    # 提取最后一层隐藏状态作为文本特征
    text_features = outputs.last_hidden_state
    text_features_list.append(text_features)

    # 清理GPU缓存
    torch.cuda.empty_cache()

# 合并所有批次的文本特征
text_features = torch.cat(text_features_list, dim=0)

# 将文本特征保存为pickle文件
with open('text_features.pkl', 'wb') as f:
    pickle.dump(text_features, f)
