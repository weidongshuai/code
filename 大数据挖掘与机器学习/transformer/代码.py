import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


# 更复杂的句子数据集
sentences = [
    # enc_input           dec_input         dec_output
    ['ich mochte wirklich ein sehr gutes bier am Abend P P P', 'S i really want a very good beer in the evening .', 'i really want a very good beer in the evening . E'],
    ['ich mochte gerne eine leckere cola nach dem Essen P P', 'S i would like a delicious coke after the meal .', 'i would like a delicious coke after the meal . E'],
    ['ich brauche dringend ein Glas klares Wasser jetzt P', 'S i urgently need a glass of clear water right now .', 'i urgently need a glass of clear water right now . E'],
    ['ich möchte am liebsten einen frischen Apfel zum Frühstück', 'S i would most like a fresh apple for breakfast .', 'i would most like a fresh apple for breakfast . E'],
    ['ich hätte gern eine warme Banane als Snack P P P', 'S i would like a warm banana as a snack .', 'i would like a warm banana as a snack . E'],
    ['ich wünsche mir eine saftige Orange zu Mittag P P', 'S i wish for a juicy orange for lunch .', 'i wish for a juicy orange for lunch . E'],
    ['ich möchte ein dickes Brot mit Butter P P P P', 'S i want a thick bread with butter .', 'i want a thick bread with butter . E'],
    ['ich trinke gern eine kalte Milch vor dem Schlafen', 'S i like to drink a cold milk before sleeping .', 'i like to drink a cold milk before sleeping . E']
]

# Padding Should be Zero
src_vocab = {
    'P': 0, 'Abend': 1, 'Apfel': 2, 'Banane': 3, 'Brot': 4, 'Butter': 5, 
    'Essen': 6, 'Frühstück': 7, 'Glas': 8, 'Milch': 9, 'Mittag': 10, 'Orange': 11, 
    'Schlafen': 12, 'Snack': 13, 'Wasser': 14, 'als': 15, 'am': 16, 'bier': 17, 
    'brauche': 18, 'cola': 19, 'dem': 20, 'dickes': 21, 'dringend': 22, 'ein': 23, 
    'eine': 24, 'einen': 25, 'frischen': 26, 'gern': 27, 'gerne': 28, 'gutes': 29, 
    'hätte': 30, 'ich': 31, 'jetzt': 32, 'kalte': 33, 'klares': 34, 'leckere': 35, 
    'liebsten': 36, 'mir': 37, 'mit': 38, 'mochte': 39, 'möchte': 40, 'nach': 41, 
    'saftige': 42, 'sehr': 43, 'trinke': 44, 'vor': 45, 'warme': 46, 'wirklich': 47,
    'wünsche': 48, 'zu': 49, 'zum': 50
}
# 检查 src_vocab 中的索引是否正确
max_index = max(src_vocab.values())
if max_index >= len(src_vocab):
    print(f"Error: src_vocab 中存在超出范围的索引，最大索引为 {max_index}，词表大小为 {len(src_vocab)}")
    # 修正索引，重新构建词表
    unique_words = sorted(set(word for sentence in sentences for word in sentence[0].split()))
    new_src_vocab = {word: i for i, word in enumerate(unique_words)}
    # 确保 'P' 索引为 0
    new_src_vocab['P'] = 0
    src_vocab = new_src_vocab
    print("重新构建后的 src_vocab:")
    print(src_vocab)
src_vocab_size = len(src_vocab)

tgt_vocab = {'P': 0, 'i': 1, 'really': 2, 'want': 3, 'a': 4, 'very': 5, 'good': 6, 
             'beer': 7, 'in': 8, 'the': 9, 'evening': 10, 'would': 11, 'like': 12, 
             'delicious': 13, 'coke': 14, 'after': 15, 'meal': 16, 'urgently': 17, 
             'need': 18, 'glass': 19, 'of': 20, 'clear': 21, 'water': 22, 'right': 23,
             'now': 24, 'most': 25, 'fresh': 26, 'apple': 27, 'for': 28, 'breakfast': 29,
             'warm': 30, 'banana': 31, 'as': 32, 'snack': 33, 'wish': 34, 'juicy': 35, 
             'orange': 36, 'lunch': 37, 'thick': 38, 'bread': 39, 'with': 40, 'butter': 41, 
             'drink': 42, 'cold': 43, 'before': 44, 'sleeping': 45, 'S': 46, 'E': 47, '.': 48}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

# 计算最大长度
src_len = max([len(sentence[0].split()) for sentence in sentences])
tgt_len = max([len(sentence[1].split()) for sentence in sentences])

# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
num_experts = 4  # 专家数量


def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [src_vocab.get(n, 0) for n in sentences[i][0].split()]
        # 检查索引是否超出范围
        for idx in enc_input:
            if idx >= src_vocab_size:
                print(f"Error: Encoder input index {idx} out of range in sentence {sentences[i][0]}")
        enc_input = enc_input + [0] * (src_len - len(enc_input))  # 填充到最大长度
        dec_input = [tgt_vocab.get(n, 0) for n in sentences[i][1].split()]
        # 检查索引是否超出范围
        for idx in dec_input:
            if idx >= tgt_vocab_size:
                print(f"Error: Decoder input index {idx} out of range in sentence {sentences[i][1]}")
        dec_input = dec_input + [0] * (tgt_len - len(dec_input))  # 填充到最大长度
        dec_output = [tgt_vocab.get(n, 0) for n in sentences[i][2].split()]
        # 检查索引是否超出范围
        for idx in dec_output:
            if idx >= tgt_vocab_size:
                print(f"Error: Decoder output index {idx} out of range in sentence {sentences[i][2]}")
        dec_output = dec_output + [0] * (tgt_len - len(dec_output))  # 填充到最大长度

        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


enc_inputs, dec_inputs, dec_outputs = make_data(sentences)


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model)(output + residual), attn


class Expert(nn.Module):
    def __init__(self):
        super(Expert, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, x):
        return self.fc(x)


class MOE(nn.Module):
    def __init__(self):
        super(MOE, self).__init__()
        self.experts = nn.ModuleList([Expert() for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        gates = nn.Softmax(dim=-1)(self.gate(x))  # [batch_size, seq_len, num_experts]
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [batch_size, seq_len, d_model, num_experts]
        moe_output = torch.einsum('bsn,bsdn->bsd', gates, expert_outputs)
        return moe_output


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.moe = MOE()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = enc_outputs + self.moe(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.moe = MOE()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = dec_outputs + self.moe(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)  # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


model = Transformer()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

for epoch in range(1000):
    total_loss = 0
    for enc_inputs, dec_inputs, dec_outputs in loader:
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]
        '''
        enc_inputs, dec_inputs, dec_outputs = enc_inputs, dec_inputs, dec_outputs
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(loader)
    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(avg_loss))


def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype)], -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab["."]:
            terminal = True
        print(next_word)
    return dec_input


# Test
for enc_input in enc_inputs:
    greedy_dec_input = greedy_decoder(model, enc_input.view(1, -1), start_symbol=tgt_vocab["S"])
    predict, _, _, _ = model(enc_input.view(1, -1), greedy_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    input_sentence = [list(src_vocab.keys())[list(src_vocab.values()).index(idx)] for idx in enc_input.tolist()]
    output_sentence = [idx2word[n.item()] for n in predict.squeeze()]
    print(' '.join(input_sentence), '->', ' '.join(output_sentence))
    