import json
import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *
import ast

from utils.utils_general import *


def read_langs(file_name, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr = [], [], [], []
    max_resp_len = 0

    with open('data/MULTIWOZ2.1/global_entities.json') as f:
        global_entity = json.load(
            f)  # global_entity包含数据集中的所有实体，是一个dict，包含event,time,date,party,room,agenda,location,weekly_time,temperature,weather_attribute,traffic_info,poi_type,poi,distance

    with open(file_name) as fin:
        cnt_lin, sample_counter = 1, 1  # 对话的个数，对话轮数
        for line in fin:
            line = line.strip()  # 移除头尾字符
            if line:
                if '#' in line:  # 如果有#在句子中，则#后面的内容为任务的类型，即navigate，weather，calender
                    line = line.replace("#", "")
                    task_type = line
                    continue

                nid, line = line.split(' ', 1)  # nid为对话的轮数
                if '\t' in line:  # 若有\t在句子中，则这句话为输入，回复，真实实体
                    u, r, gold_ent = line.split('\t')
                    gen_u = generate_memory(u, "$u", str(
                        nid))  # 将输入转化为[['thank', '$u', 'turn1', 'word0', 'PAD', 'PAD'], ['you', '$u', 'turn1', 'word1', 'PAD', 'PAD']]的形式
                    context_arr += gen_u  # context有知识和对话，conv只有对话
                    conv_arr += gen_u

                    # Get gold entity for each domain
                    gold_ent = ast.literal_eval(gold_ent)  # ast.literal_eval函数将字符串转化成list或dict等格式的数据
                    ent_idx_hot, ent_idx_res, ent_idx_att = [], [], []
                    if task_type == "hotel":
                        ent_idx_hot = gold_ent
                    elif task_type == "restaurant":
                        ent_idx_res = gold_ent
                    elif task_type == "attraction":
                        ent_idx_att = gold_ent
                    ent_index = list(set(ent_idx_hot + ent_idx_res + ent_idx_att))  # set中无重复的元素，此操作目的为去掉重复的元素

                    # Get local pointer position for each word in system response
                    ptr_index = []
                    for key in r.split():
                        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in ent_index)]
                        if (index):
                            index = max(index)
                        else:
                            index = len(context_arr)
                        ptr_index.append(index)

                    # Get global pointer labels for words in system response, the 1 in the end is for the NULL token
                    selector_index = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for word_arr in
                                      context_arr] + [1]
                    # selector_index_kb = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for word_arr
                    #                      in kb_arr]

                    sketch_response = generate_template(global_entity, r, gold_ent, kb_arr, task_type)

                    data_detail = {
                        'context_arr': list(context_arr + [['$$$$'] * MEM_TOKEN_SIZE]),  # $$$$ is NULL token
                        'response': r,
                        'sketch_response': sketch_response,
                        'ptr_index': ptr_index + [len(context_arr)],
                        'selector_index': selector_index,
                        # 'selector_index_kb': selector_index_kb,
                        'ent_index': ent_index,
                        'ent_idx_hot': list(set(ent_idx_hot)),
                        'ent_idx_res': list(set(ent_idx_res)),
                        'ent_idx_att': list(set(ent_idx_att)),
                        'conv_arr': list(conv_arr),
                        'kb_arr': list(kb_arr),
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': task_type}
                    data.append(data_detail)

                    gen_r = generate_memory(r, "$s", str(nid))
                    context_arr += gen_r
                    conv_arr += gen_r
                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:  # 当一句话中没有\t时，这句话为知识
                    r = line
                    kb_info = generate_memory(r, "", str(nid))
                    context_arr = kb_info + context_arr
                    kb_arr += kb_info
            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr = [], [], []
                if (max_line and cnt_lin >= max_line):
                    break

    return data, max_resp_len


def generate_template(global_entity, sentence, sent_ent, kb_arr, domain):
    """
    Based on the system response and the provided entity table, the output is the sketch response. 
    """
    sketch_response = []
    if sent_ent == []:
        sketch_response = sentence.split()
    else:
        for word in sentence.split():
            if word not in sent_ent:
                sketch_response.append(word)
            else:
                ent_type = None
                # if domain != 'weather':
                for kb_item in kb_arr:
                    if word == kb_item[0]:
                        ent_type = kb_item[1]
                        break
                # if ent_type == None:  # weather有区别，实体的类别不能通过每段话前面的kb_arr来得到，而是要通过全局实体得到
                #     for key in global_entity.keys():
                #         if key != 'poi':
                #             global_entity[key] = [x.lower() for x in global_entity[key]]
                #             if word in global_entity[key] or word.replace('_', ' ') in global_entity[key]:
                #                 ent_type = key
                #                 break
                #         else:  # poi和其他实体有些区别，[{'address': '593 Arrowhead Way', 'poi': 'Chef_Chu_s', 'type': 'chinese restaurant'},{...}]
                #             poi_list = [d['poi'].lower() for d in global_entity['poi']]
                #             if word in poi_list or word.replace('_', ' ') in poi_list:
                #                 ent_type = key
                #                 break
                sketch_response.append('@' + ent_type)
    sketch_response = " ".join(sketch_response)
    return sketch_response


def generate_memory(sent, speaker, time):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker == "$u" or speaker == "$s":  # 添加说话者信息和时态信息
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn' + str(time), 'word' + str(idx)] + ["PAD"] * (MEM_TOKEN_SIZE - 4)
            sent_new.append(temp)
    else:
        if len(sent_token) > 4:
            sent_token = sent_token[::-1][:2]
            sent_token = sent_token + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
            sent_new.append(sent_token)
        else:
            sent_token = sent_token[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
            sent_new.append(sent_token)
    return sent_new


def prepare_data_seq(task, batch_size=100):
    file_train = 'data/MULTIWOZ2.1/train.txt'
    file_dev = 'data/MULTIWOZ2.1/dev.txt'
    file_test = 'data/MULTIWOZ2.1/test.txt'

    pair_train, train_max_len = read_langs(file_train, max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None)
    pair_test, test_max_len = read_langs(file_test, max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1

    lang = Lang()

    train = get_seq(pair_train, lang, batch_size, True)
    dev = get_seq(pair_dev, lang, batch_size, False)
    test = get_seq(pair_test, lang, batch_size, False)

    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, [], lang, max_resp_len


def get_data_seq(file_name, lang, max_len, batch_size=1):
    pair, _ = read_langs(file_name, max_line=None)
    # print(pair)
    d = get_seq(pair, lang, batch_size, False)
    return d
