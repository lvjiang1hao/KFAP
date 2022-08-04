import json
import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *
import ast
import os

from utils.utils_general import *
from utils.utils_temp import entityList, get_type_dict


def read_langs(file_name, global_entity, type_dict, max_line = None):
    # print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr = [], [], [], []
    max_resp_len, sample_counter = 0, 0
    with open(file_name) as fin:
        cnt_lin = 1
        for line in fin:
            line = line.strip()     # 移除头尾字符
            if line:
                nid, line = line.split(' ', 1)      # 分开轮次和句子
                # print("line", line)
                if '\t' in line:        # 当\t在一行中，根据\t将一行划分成user和response
                    u, r = line.split('\t')
                    gen_u = generate_memory(u, "$u", str(nid))      # 为user话语中的每个词添加说话者和时态信息
                    context_arr += gen_u        # 包含对话和知识
                    conv_arr += gen_u       # 仅包含对话
                    ptr_index, ent_words = [], []
                    
                    # Get local pointer position for each word in system response
                    # 局部指针，当回复中的词为实体并在上下文中出现，该词的指针指向出现位置
                    for key in r.split():
                        if key in global_entity and key not in ent_words: 
                            ent_words.append(key)
                        # 若系统回复中的单词在过往对话中出现且为全局实体，获取其index，否则为空
                        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in global_entity)]
                        # 如果存在这样的index则取序号最大值，否则取长度（超出索引，即没有）
                        index = max(index) if index else len(context_arr)
                        ptr_index.append(index)
                    
                    # Get global pointer labels for words in system response, the 1 in the end is for the NULL token
                    # 当上下文中的词出现在实体词或回复中，全局指针标签为1，否则为0，+[1]处理没有内容的情况
                    selector_index = [1 if (word_arr[0] in ent_words or word_arr[0] in r.split()) else 0 for word_arr in context_arr] + [1]

                    sketch_response = generate_template(global_entity, r, type_dict)
                    
                    data_detail = {
                        'context_arr': list(context_arr+[['$$$$']*MEM_TOKEN_SIZE]),  # $$$$ is NULL token    包含对话和知识
                        'response': r,
                        'sketch_response': sketch_response,
                        'ptr_index': ptr_index+[len(context_arr)],
                        'selector_index': selector_index,
                        'ent_index': ent_words,
                        'ent_idx_cal': [],
                        'ent_idx_nav': [],
                        'ent_idx_wet': [],
                        'conv_arr': list(conv_arr),       # 仅包含对话
                        'kb_arr': list(kb_arr),
                        'id': int(sample_counter),      # 对话轮数
                        'ID': int(cnt_lin),          # 对话次数
                        'domain': ""}
                    data.append(data_detail)

                    gen_r = generate_memory(r, "$s", str(nid))      # 为系统回复中的每个词添加说话者和时态信息
                    context_arr += gen_r
                    conv_arr += gen_r
                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:       # 当句子不是user和response时，为KB信息
                    r = line
                    kb_info = generate_memory(r, "", str(nid))
                    context_arr = kb_info + context_arr         # 当一行为KB信息时，context_arr前面部分加入KB信息，直到遇到对话时，进入上面部分建立data_detail
                    kb_arr += kb_info                           # 所以当有KB信息时，context_arr以KB信息开头
            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr = [], [], []
                if(max_line and cnt_lin>=max_line):
                    break

    return data, max_resp_len


def generate_memory(sent, speaker, time):       # 为上下文的每个词添加说话者信息和对话轮次信息
    sent_new = []
    sent_token = sent.split(' ')
    if speaker=="$u" or speaker == "$s":      # 为对话添加信息
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn'+str(time), 'word'+str(idx)] + ["PAD"]*(MEM_TOKEN_SIZE-4)
            sent_new.append(temp)
    else:       # 为知识添加信息，当有“R_rating”时，顺序，没有时，逆序
        if sent_token[1] == "R_rating":
            sent_token = sent_token + ["PAD"]*(MEM_TOKEN_SIZE-len(sent_token))
        else:
            sent_token = sent_token[::-1] + ["PAD"]*(MEM_TOKEN_SIZE-len(sent_token))
        sent_new.append(sent_token)
    return sent_new


def generate_template(global_entity, sentence, type_dict):      # 将一句话转化成sketch形式
    sketch_response = []
    for word in sentence.split():
        if word in global_entity:       # 当某个词属于知识中的某一类实体，将这个词替换为实体类别
            ent_type = None
            for kb_item in type_dict.keys():
                if word in type_dict[kb_item]:
                    ent_type = kb_item
                    break
            sketch_response.append('@'+ent_type)
        else:
            sketch_response.append(word)        # 当这个词是普通词时，直接使用这个词
    sketch_response = " ".join(sketch_response)
    return sketch_response


def prepare_data_seq(task, batch_size=100):
    data_path = 'data/dialog-bAbI-tasks/dialog-babi'
    file_train = '{}-task{}trn.txt'.format(data_path, task)
    file_dev = '{}-task{}dev.txt'.format(data_path, task)
    file_test = '{}-task{}tst.txt'.format(data_path, task)
    kb_path = data_path+'-kb-all.txt'
    file_test_OOV = '{}-task{}tst-OOV.txt'.format(data_path, task)
    type_dict = get_type_dict(kb_path, dstc2=False)
    global_ent = entityList('data/dialog-bAbI-tasks/dialog-babi-kb-all.txt', int(task))     # 所有实体

    pair_train, train_max_len = read_langs(file_train, global_ent, type_dict)           # 这里的pair_train为单词形式的data
    pair_dev, dev_max_len = read_langs(file_dev, global_ent, type_dict)
    pair_test, test_max_len = read_langs(file_test, global_ent, type_dict)
    pair_testoov, testoov_max_len = read_langs(file_test_OOV, global_ent, type_dict)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len, testoov_max_len) + 1
    
    lang = Lang()

    train = get_seq(pair_train, lang, batch_size, True)         # 将上面的单词形式的pair_train转换为id形式，并得到data_loader
    dev   = get_seq(pair_dev, lang, 100, False)         # 验证集与测试集则不需要将单词转换为id, 故为False
    test  = get_seq(pair_test, lang, batch_size, False)
    testoov = get_seq(pair_testoov, lang, batch_size, False)

    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))  
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, testoov, lang, max_resp_len


def get_data_seq(file_name, lang, max_len, task=5, batch_size=1):
    data_path = 'data/dialog-bAbI-tasks/dialog-babi'
    kb_path = data_path+'-kb-all.txt'
    type_dict = get_type_dict(kb_path, dstc2=False)
    global_ent = entityList(kb_path, int(task))
    pair, _ = read_langs(file_name, global_ent, type_dict)
    # print("pair", pair)
    d = get_seq(pair, lang, batch_size, False)
    return d







