---
layout: post
title: 【学习笔记】Bert Word Embedding
categories: ["学习笔记", "代码demo"]
keywords: bert
mathjax: true
---

总是需要快速搞一个bert的demo试试。虽然写过好几遍了，但从没有整理下来一份干净的代码。这次跟着[BERT Word Embeddings Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)练习一遍。

# Use Pre-trained BERT

使用huggingface的接口，安装：

```shell
pip install transformers
```

然后加载BERT的分词器。

```python
import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```

BERT预训练的时候数据有一些特殊的格式，我们输入文本的时候需要处理一下原数据：
1. [SEP]：用于标记一个句子的结束或者两个句子的间隔。
2. [CLS]：文本的开始，可以用于文本分类。但无论是什么任务，最好都加上这个。
3. 需要用BERT的分词器分词。

用BERT的分词器分个句子试试：

```python
text = "Here is the sentence I want embeddings for."
marked_text = "[CLS] " + text + " [SEP]"
tokenized_text = tokenizer.tokenize(marked_text)
print (tokenized_text)
```

```
['[CLS]', 'here', 'is', 'the', 'sentence', 'i', 'want', 'em', '##bed', '##ding', '##s', 'for', '.', '[SEP]']
```

接着，我们可以将分词后的结果映射到词汇表的id上

```python
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
for tup in zip(tokenized_text, indexed_tokens):
    print('{:<12} {:>6,}'.format(tup[0], tup[1]))
```

```
[CLS]           101
here          2,182
is            2,003
the           1,996
sentence      6,251
i             1,045
want          2,215
em            7,861
##bed         8,270
##ding        4,667
##s           2,015
for           2,005
.             1,012
[SEP]           102
```

BERT还需要接收一个叫 Segment ID 的东西，来标记每个词属于哪个句子。如果只有一个句子，那么全输入1就行了。如果有多个句子，可以00001110000111这样穿插。

```python
segments_ids = [1] * len(tokenized_text)
print (segments_ids)
```

```
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

然后，我们需要将输入都转成tensor才能输入BERT模型中。

```python
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
```

现在，加载BERT预训练模型，开启eval模式防止dropout。

```python
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True)
model.eval()
```

使用BERT！

```python
with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]
```

# Output Analysis

在outputs中，有三个东西，是：odict_keys(['last_hidden_state', 'pooler_output', 'hidden_states'])。我们这里只取了hidden states。
其实还可以输出每一层的注意力权重，但是要把BertModel.from_pretrained的参数output_attentions设为True。它在output里叫attentions。

1. last_hidden_state: 模型最后一层的hidden states。(batch_size, sequence_length, hidden_size)
2. pooler_output: 序列的第一个token的最后一层的hidden state，[CLS]。(batch_size, hidden_size)
3. hidden states: 一个元组，第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(batch_size, sequence_length, hidden_size)。

```python
print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
layer_i = 0

print ("Number of batches:", len(hidden_states[layer_i]))
batch_i = 0

print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))
```

```
Number of layers: 13   (initial embeddings + 12 BERT layers)
Number of batches: 1
Number of tokens: 14
Number of hidden units: 768
```

现在，hidden states的维度是[# layers, # batches, # tokens, # features]这样的，但我们需要将输出处理成我们需要的格式：[# tokens, # layers, # features]。

hidden states是一个元组，现在我们将元组里的两个元素拼起来形成一个大的tensor。原来是（1*，12*），变成13*。

```python
token_embeddings = torch.stack(hidden_states, dim=0)
token_embeddings.size()
```

```
torch.Size([13, 1, 14, 768])
```

因为这里我们只有一条数据，扔掉batch维度：

```python
token_embeddings = torch.squeeze(token_embeddings,dim=1)
token_embeddings.size()
```

```
torch.Size([13, 14, 768])
```

然后交换token和layer两个维度，最后的结果就是[tokens * layers * features]

```python
token_embeddings = token_embeddings.permute(1,0,2)
token_embeddings.size()
```

```
torch.Size([14, 13, 768])
```



# Get Emebdding

得到了所有的输出后，我们最终需要的其实是 word embedding 和sentence embedding 。但可以有很多种得到embedding的方法。

## Word Embedding

### 最后四层进行拼接

为每个token提供一个4 * 768 = 3072维度的词向量。

```python
token_vecs_cat = []
for token in token_embeddings:
    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    token_vecs_cat.append(cat_vec)
print ('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))
``` 

```
Shape is: 14 x 3072
```

### 最后四层相加

```python
token_vecs_sum = []
for token in token_embeddings:
    sum_vec = torch.sum(token[-4:], dim=0)
    token_vecs_sum.append(sum_vec)

print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))
```

```
Shape is: 14 x 768
```

### 最后一层

```python
token_vecs_last = []
for token in token_embeddings:
    last_vec = token[-1]
    token_vec_last.append(last_vec)
print ('Shape is: %d x %d' % (len(token_vecs_last), len(token_vecs_last[0])))
```

```
Shape is: 14 x 768
```

## Sentence Embedding

将每个token的倒数第二层hidden state进行平均，就得到这句话的sentence embedding

```python
token_vecs = hidden_states[-2][0]
sentence_embedding = torch.mean(token_vecs, dim=0)
```

那么sentence embedding就是一个维度为torch.Size([768])的东西。


# Clean Code

将一些必要的操作封装成函数，计算几个例子最后一层vector的余弦相似度。

```python
import torch
from transformers import BertTokenizer, BertModel

def load_bert():
    # loading bert and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased',
                                    output_hidden_states = True)
    model.eval()
    return tokenizer, model


def bert_text_preparation(text, tokenizer):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors, model, mode="last"):
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

    # 整理维度，去掉batch维度
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)

    token_vecs = []
    if mode == "cat_four":
        for token in token_embeddings:
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            token_vecs.append(cat_vec)

    elif mode == "sum_four":
        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs.append(sum_vec)


    elif mode == "last":
        for token in token_embeddings:
            last_vec = token[-1]
            token_vecs.append(last_vec)

    return token_vecs
```

```python
if __name__ == "__main__":
    # for single sentence
    text = "Here is the sentence I want embeddings for."
    tokenizer, model = load_bert()
    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
    token_vecs = get_bert_embeddings(tokens_tensor, segments_tensors, model, mode="last")
    print(len(token_vecs), token_vecs[0].shape)
    # 14 torch.Size([768])
```


计算两个东西的余弦相似度，需要
```python
from scipy.spatial.distance import cosine
```

cosine直接计算的是距离，余弦相似度需要1减去。这里展示的是计算不同上下文上的bank的余弦相似度。

```python
if __name__ == "__main__":
    # for calculating similarity
    texts = ["bank",
      "The river bank was flooded.",
      "The bank vault was robust.",
      "He had to bank on her for support.",
      "The bank was out of money.",
      "The bank teller was a man."]
    target_word_embeddings = []
    for text in texts:
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
        list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
        
        # Find the position 'bank' in list of tokens
        word_index = tokenized_text.index('bank')
        # Get the embedding for bank
        word_embedding = list_token_embeddings[word_index]

        target_word_embeddings.append(word_embedding)

    list_of_distances = []
    for text1, embed1 in zip(texts, target_word_embeddings):
        for text2, embed2 in zip(texts, target_word_embeddings):
            cos_dist = 1 - cosine(embed1, embed2)
            list_of_distances.append([text1, text2, cos_dist])
    distances_df = pd.DataFrame(list_of_distances, columns=['text1', 'text2', 'distance'])
    distances_df #[distances_df.text1 == 'bank']
```

```
	text1	text2	distance
0	bank	bank	1.000000
1	bank	The river bank was flooded.	0.338063
2	bank	The bank vault was robust.	0.494099
3	bank	He had to bank on her for support.	0.256140
4	bank	The bank was out of money.	0.469942
5	bank	The bank teller was a man.	0.466021
6	The river bank was flooded.	bank	0.338063
7	The river bank was flooded.	The river bank was flooded.	1.000000
8	The river bank was flooded.	The bank vault was robust.	0.523326
9	The river bank was flooded.	He had to bank on her for support.	0.331584
10	The river bank was flooded.	The bank was out of money.	0.512161
11	The river bank was flooded.	The bank teller was a man.	0.519274
12	The bank vault was robust.	bank	0.494099
13	The bank vault was robust.	The river bank was flooded.	0.523326
14	The bank vault was robust.	The bank vault was robust.	1.000000
``` 