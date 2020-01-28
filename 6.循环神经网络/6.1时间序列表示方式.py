import torch
from torch import nn
from torchnlp.word_to_vector import Glove
"""
语音文字
[word_num, b, word_vec]  分别是单词数量，序列数量， 表达方式

"""

word_to_ix = {"hellow":0, "world":1}

lookup_tensor = torch.tensor([word_to_ix["hellow"]], dtype=torch.long)

embeds = nn.Embedding(2, 5)  # 创建一个表，里面包含两个单词，每个单词用用一个五维度的数据表示
hello_embed = embeds(lookup_tensor)
print(hello_embed)






vectors = Glove()

print(vectors["hellow"])








