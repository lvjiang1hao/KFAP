import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.config import *
from utils.utils_general import _cuda
import math


class ContextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, n_layers=1):
        super(ContextRNN, self).__init__()
        self.input_size = input_size        # input_size是词表的维度
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.W = nn.Linear(2 * hidden_size, hidden_size)

    def get_state(self, bsz):  # 获得隐状态
        """Get cell states and hidden states."""
        return _cuda(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # contiguous（）.view相当与reshape
        # input_seqs为[conv_arr]，size是[max_len, batch_size, 4（每个词为四元组，对于KVR是6）]
        # (max_len, batch_size, 4) -> (max_len, batch_size * 4, embedding_size)，size(0)为该批中句子最大长度
        # 举例，conv_arr最长10个词，batch为9，每个词有四个元组，则conv_arr的size为（10，8，4），经过view(input_seqs.size(0),-1)后为(10, 8*4),经过Embedding后为(10, 8*4, hidden_size)
        embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long())
        # (max_len, batch_size * 4, embedding_size) -> (max_len, batch_size, 4, embedding_size)
        embedded = embedded.view(input_seqs.size() + (embedded.size(-1),))
        # 对数据处理后得到的四元组的Embedding相加
        embedded = torch.sum(embedded, 2).squeeze(2)  # 该输出结构(max_len, bsz, embsz)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))  # 初始化hidden，全0
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)    # pack_padded_sequence统计每一个时间步对应有多少个batch
        outputs, hidden = self.gru(embedded, hidden)  # outputs包含每一时间步的输出h_t，(seq_len, batch_size, 2 * hidden); hidden是最后时间步的输出h_n, (2 * layer, bsz, hsz)
        if input_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        hidden = self.W(torch.cat((hidden[0], hidden[1]), dim=1)).unsqueeze(0)  # unsqueeze(0)为在第一维上增加一维，结构（1,bsz,hsz）,拼接hidden[0]和hidden[1]是将正向和反向的隐状态拼起来
        outputs = self.W(outputs)  # (seq_len, bsz, 2 * hsz)->(seq_len,bsz,hsz)
        return outputs.transpose(0, 1), hidden      # (bsz, seq_len, hsz), (1, bsz, hsz)


class ExternalKnowledge(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout):
        super(ExternalKnowledge, self).__init__()
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)  # 初始化权重均值为0，方差0.1的正态分布
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5,
                                    padding=2)  # 一维卷积函数Conv1d：参数分别为(进出通道 卷积核的大小 输入的每一条边补充0的层数)

    def add_lm_embedding(self, full_memory, kb_len, conv_len, hiddens):
        # 添加hidden到memory
        # full_memory (batch_size,max_len+1,embedding_size)
        for bi in range(full_memory.size(0)):  # batch
            start, end = kb_len[bi], kb_len[bi] + conv_len[bi]
            full_memory[bi, start:end, :] = full_memory[bi, start:end, :] + hiddens[bi, :conv_len[bi], :]
        return full_memory

    def load_memory(self, story, kb_len, conv_len, hidden, dh_outputs):
        # Forward multiple hop mechanism
        u = [hidden.squeeze(0)]  # hidden(num_layers * num_directions, batch_size, hidden_size)去掉hidden的第一维即（bsz,hsz），得到文中的q，即输入的隐状态作为查询向量，注意此处u为一个列表
        story_size = story.size()  # story为对话历史序列，（batch_size, 该批context_arr_lengths最大值max_len+1, memory_size）,加的1是空标记$$$$
        self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story_size[0], -1))  # .long()) # b * (len * m) * emb
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * len * m * emb
            embed_A = torch.sum(embed_A, 2).squeeze(2)  # b * len * emb, m=seq_len?记忆用词袋表示, 注意此处的squeeze没用
            if not args["ablationH"]:  # 是否做向记忆中添加hidden的消融实验
                embed_A = self.add_lm_embedding(embed_A, kb_len, conv_len,
                                                dh_outputs)  # dh为dialog history,(batch_size,max_len,embedding_size)
            embed_A = self.dropout_layer(embed_A)

            if (len(list(u[-1].size())) == 1):
                u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(embed_A)  # 调整查询向量到可以与memory相乘的维度，每次都是取列表中的最后一个u,本实验中hidden维度和embedding维度相同
            prob_logit = torch.sum(embed_A * u_temp, 2)  # 将Embedding加起来了，(bsz, len)
            prob_ = self.softmax(prob_logit)        # (bsz, len),对第一维即 len 计算了softmax

            embed_C = self.C[hop + 1](story.contiguous().view(story_size[0], -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            embed_C = torch.sum(embed_C, 2).squeeze(2)          # (bsz, len, emb)
            if not args["ablationH"]:
                embed_C = self.add_lm_embedding(embed_C, kb_len, conv_len, dh_outputs)

            prob = prob_.unsqueeze(2).expand_as(embed_C)        # 将prob转换为与embed_c相同的维度
            o_k = torch.sum(embed_C * prob, 1)          # (bsz, len, emb)->(bsz, emb)
            u_k = u[-1] + o_k       # 更新u_k
            u.append(u_k)       # 将新的u_k放入u的列表中
            self.m_story.append(embed_A)
        self.m_story.append(embed_C)
        return self.sigmoid(prob_logit), u[-1], self.m_story[-1]          # 前者为全局指针，后者为外部知识的读取输出

    def forward(self, query_vector, global_pointer, reason, rea, memory_mask, attention, local_attention):
        u = [query_vector]          # (bsz, hsz)
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]         # # (bsz, maxlen, emb)
            if not args["ablationG"]:       # 记忆信息 * 全局记忆指针 根据权重对记忆信息进行处理（即指针指向的位置权重几乎不变，未指向的位置信息大幅度衰减）
                m_A = m_A * global_pointer.unsqueeze(2).expand_as(m_A)          # global_pointer为(bsz, len),故要转换到 m_A的维度

            if attention:
                m_A = m_A * local_attention.unsqueeze(2).expand_as(m_A)

            if (len(list(u[-1].size())) == 1):
                u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_logits = torch.sum(m_A * u_temp, 2)        # prob_logits即查询向量与记忆的点乘, (bsz, maxlen)

            if rea:
                for bi in range(prob_logits.size()[0]):
                    for i in range(prob_logits.size()[1]):
                        if i not in reason[bi]:
                            prob_logits[bi][i] = -np.inf


            prob_logits = prob_logits + (1 - memory_mask) * -1e10

            prob_soft = self.softmax(prob_logits)       # prob_soft即得到的注意力分数, (bsz, len)
            m_C = self.m_story[hop + 1]
            if not args["ablationG"]:
                m_C = m_C * global_pointer.unsqueeze(2).expand_as(m_C)

            if attention:
                m_C = m_C * local_attention.unsqueeze(2).expand_as(m_C)

            prob = prob_soft.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        return prob_soft, prob_logits, u[-1]       # 最后返回的是决定查询向量的记忆相关度的软记忆注意和未经Softmax的logits, (bsz, len), (bsz, len)


class LocalMemoryDecoder(nn.Module):
    def __init__(self, shared_emb, lang, embedding_dim, hop, dropout):
        super(LocalMemoryDecoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.C = shared_emb  # shared的emb保存为C，后面可知shared的embedding为encoder的embedding
        self.softmax = nn.Softmax(dim=1)
        self.sketch_rnn = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        self.relu = nn.ReLU()
        self.projector = nn.Linear(2 * embedding_dim, embedding_dim)
        self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(2 * embedding_dim, embedding_dim)
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.V = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, extKnow, story_size, story_lengths, copy_list, encode_hidden, target_batches, max_target_length,
                batch_size, use_teacher_forcing, get_decoded_words, global_pointer, dh_outputs, m_soft, m_logits, m_out, conv_len, memory_mask):
        # Initialize variables for vocab and pointer
        all_decoder_outputs_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))       # (max_target_length, batch_size, self.num_vocab)
        all_decoder_outputs_ptr = _cuda(torch.zeros(max_target_length, batch_size, story_size[1]))      # (max_target_length, batch_size, max_input_len), story_size(bsz, len, mem)
        decoder_input = _cuda(torch.LongTensor([SOS_token] * batch_size))
        memory_mask_for_step = _cuda(torch.ones(story_size[0], story_size[1]))  # mask用来防止生成相同的槽，即文中的R，(batch_size, max_input_len)
        decoded_fine, decoded_coarse = [], []       # coarse为粗糙的，含有如 @address这种词，fine将 @address替换为了具体地址

        hidden = self.relu(self.projector(encode_hidden)).unsqueeze(0)  # (1, batch_size, hidden_size)


        m_soft = m_soft.transpose(0, 1)
        m_logits = m_logits.transpose(0, 1)
        m_out = m_out.transpose(0, 1)
        context_mask = _cuda(torch.zeros(dh_outputs.size()[0], dh_outputs.size()[1]))
        for bi in range(len(conv_len)):
            context_mask[bi, :conv_len[bi]] = 1

        # Start to generate word-by-word
        for t in range(max_target_length):
            # hidden的生成在前四行，不同循环的不同变量只有decoder_input
            embed_q = self.dropout_layer(self.C(decoder_input))  # b * e
            if len(embed_q.size()) == 1: embed_q = embed_q.unsqueeze(0)
            _, hidden = self.sketch_rnn(embed_q.unsqueeze(0), hidden)
            query_vector = hidden[0]        # hidden:(1, bsz, hsz)，取hidden[0]目的是降维，转为(bsz, hsz),因为是单步，故hidden和output效果一样

            context_soft, context_attention = self.attention(hidden.squeeze(0).unsqueeze(1),
                                                             dh_outputs, context_mask)


            p_vocab = self.attend_vocab(self.C.weight, hidden.squeeze(0))       # C为encoder和decoder共享的嵌入矩阵，(vocab_size, embedding_size)，hidden维度到词表维度
            all_decoder_outputs_vocab[t] = p_vocab          # 每一步得到的词表中所有单词的概率
            _, topvi = p_vocab.data.topk(1)         # topk函数：得到前k个元素，返回两个tensor，第一个为数值，第二个为下标,此处是得到数值最大的元素的下标


            # 修改
            local_attention = context_soft.unsqueeze(1).bmm(m_logits)       # (bsz, 1, kb_len)
            local_attention = local_attention.squeeze(1)  # (bsz, kb_len)
            local_attention = self.sigmoid(local_attention)


            # query the external konwledge using the hidden state of sketch RNN
            prob_soft, prob_logits, out = extKnow(query_vector, global_pointer, None, False, memory_mask, True, local_attention)          # extKnow为ExternalKnowledge的forward函数
            all_decoder_outputs_ptr[t] = prob_logits        # prob_logits为(bsz, maxlen), all_decoder_outputs_ptr则为(max_target_length, bsz, maxlen)

            if use_teacher_forcing:
                decoder_input = target_batches[:, t]
            else:
                decoder_input = topvi.squeeze()

            if get_decoded_words:       # 训练时为False，测试时为True

                search_len = min(5, min(story_lengths))         # 最大搜索长度为min（5，对话历史最小长度）
                prob_soft = prob_soft * memory_mask_for_step
                _, toppi = prob_soft.data.topk(search_len)          # 取注意的前search_len元素，作为填入槽的object预备
                temp_f, temp_c = [], []         # temp_c包含所有生成的单词，temp_f将temp_c中标记词替换为object

                for bi in range(batch_size):
                    token = topvi[bi].item()        # topvi[:,0][bi].item()，取下标  item()取出单元素张量的元素值并返回该值，保持原元素类型不变,即取出第bi批的生成的单词的index
                    temp_c.append(self.lang.index2word[token])      # 转为单词并保存

                    if '@' in self.lang.index2word[token]:          # 如果是草稿标记
                        cw = 'UNK'
                        for i in range(search_len):
                            if toppi[:, i][bi] < story_lengths[bi] - 1:         # 如果存在于对话历史中存在，则copy word
                                cw = copy_list[bi][toppi[:, i][bi].item()]          # 复制object
                                break
                        temp_f.append(cw)

                        if args['record']:          # 若设置该选项，则会将已copy的单词屏蔽，防止多次复制
                            memory_mask_for_step[bi, toppi[:, i][bi].item()] = 0
                    else:
                        temp_f.append(self.lang.index2word[token])      # 若没有草图标记，也将其保存入temp_f

                decoded_fine.append(temp_f)         # 使用copy机制后的输出
                decoded_coarse.append(temp_c)         # 未进行copy的输出

        return all_decoder_outputs_vocab, all_decoder_outputs_ptr, decoded_fine, decoded_coarse

    def attend_vocab(self, seq, cond):          # 每个解码步中生成的hidden转换到词表维度，得到每个词的分数，注释掉softmax的原因是无影响，因为只取最大的词
        scores_ = cond.matmul(seq.transpose(1, 0))      # 首先对输入的矩阵转置，然后进行矩阵乘法
        # scores = F.softmax(scores_, dim=1)
        return scores_

    def attention(self, query, value, mask):
        trans_query = self.tanh(self.W_q(query))
        e = self.tanh(self.V(trans_query * value))
        e = e.squeeze(2)
        masked_score = e + (1 - mask) * -1e10  # pad masked position
        score = torch.softmax(masked_score, dim=-1)
        attention = score.unsqueeze(1).bmm(value)


        return score, attention.squeeze(1)         # (bsz, len), (bsz, hsz)

class Reason(nn.Module):
    def __init__(self, vocab, embedding_dim):
        super(Reason, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
        self.linear1 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.num_attention_heads = 8
        self.attention_head_size = 16
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.drop = nn.Dropout(p=0.2)
        self.linearsoft = nn.Linear(embedding_dim, 1)


    def forward(self, dh_outputs, dh_hidden, extKnow, global_pointer, batch_size, story, domain, context_len, kb_len, conv_len, memory_mask, memory_story):
        reason = []
        for bi in range(batch_size):
            reason.append([])

        conv_mask = _cuda(torch.zeros(dh_outputs.size()[0], dh_outputs.size()[1]))
        for bi in range(batch_size):
            conv_mask[bi][:conv_len[bi]] = 1

        # i = dh_hidden.squeeze(0)
        i = torch.zeros_like(dh_hidden.squeeze(0))

        prob_soft_, prob_logits_, out = extKnow(i, global_pointer, None, False, memory_mask, False, None)
        re = torch.zeros_like(prob_logits_)
        memory_mask_for_step = _cuda(
            torch.ones(story.size()[0], story.size()[1]))  # mask用来防止生成相同的槽，即文中的R，(batch_size, max_input_len)

        dh_hidden_new = dh_hidden.squeeze(0)
        # q = self.linear2(self.drop(self.tanh(self.linear1(self.drop(torch.cat((dh_hidden_new.unsqueeze(1).expand_as(dh_outputs), dh_outputs), dim=2))))))
        q = self.linear2((self.tanh(self.linear1((torch.cat((dh_hidden_new.unsqueeze(1).expand_as(dh_outputs), dh_outputs), dim=2))))))
        q = self.softmax(q)  # (bsz, len, hsz)
        i = torch.sum(q * dh_outputs, dim=1)

        prob_soft, prob_logits, out = extKnow(i, global_pointer, None, False, memory_mask, False, None)

        for bi in range(batch_size):
            prob_logits[bi, kb_len[bi]:(context_len[bi] - 1)] = -np.inf
            prob_logits[bi, context_len[bi]:] = -np.inf
        # logits = self.softmax(prob_logits) * memory_mask_for_step
        logits = self.sigmoid(prob_logits) * memory_mask_for_step
        reason_len = min(12, min(context_len))
        top, toppi = logits.topk(reason_len)

        for bi in range(batch_size):
            for pi in range(reason_len):
                # if toppi[bi][pi] >= 0.4:
                # if top[bi][pi] >= 0.3:
                reason[bi].append(toppi[bi][pi].item())
                # re[bi][toppi[bi][pi].item()] = 1

        return reason, i  # (bsz, len)

    def transform(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)




class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
