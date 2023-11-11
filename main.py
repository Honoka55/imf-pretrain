from transformers import BertTokenizer, BertModel
import torch
import pickle
import csv

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 读取实体描述文件
data = []
with open('entity_description.txt', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        data.append(row[1])

# 对文本进行分词和编码
encoded_data = tokenizer(data, padding=True, truncation=True, return_tensors='pt')

# 使用BERT模型获取文本特征
with torch.no_grad():
    outputs = model(**encoded_data)

# 提取最后一层隐藏状态作为文本特征
text_features = outputs.last_hidden_state

# 将文本特征保存为pickle文件
with open('text_features.pkl', 'wb') as f:
    pickle.dump(text_features, f)