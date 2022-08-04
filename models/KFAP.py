import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import os
import json
import copy

from utils.measures import wer, moses_multi_bleu, compute_bleu, compute_distinct
from utils.masked_cross_entropy import *
from utils.config import *
from models.modules import *


class KFAP(nn.Module):
    def __init__(self, hidden_size, lang, max_resp_len, path, task, dataset, lr, n_layers, dropout):
        super(KFAP, self).__init__()
        self.name = "KFAP"
        self.task = task
        self.dataset = dataset
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_resp_len = max_resp_len
        self.decoder_hop = n_layers
        self.softmax = nn.Softmax(dim=0)

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th')
                self.extKnow = torch.load(str(path) + '/enc_kb.th')
                self.decoder = torch.load(str(path) + '/dec.th')
                self.reason = torch.load(str(path) + '/rea.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th', lambda storage, loc: storage)
                self.extKnow = torch.load(str(path) + '/enc_kb.th', lambda storage, loc: storage)
                self.decoder = torch.load(str(path) + '/dec.th', lambda storage, loc: storage)
                self.reason = torch.load(str(path) + '/rea.th', lambda storage, loc: storage)
        else:
            self.encoder = ContextRNN(lang.n_words, hidden_size, dropout)
            self.extKnow = ExternalKnowledge(lang.n_words, hidden_size, n_layers, dropout)
            self.decoder = LocalMemoryDecoder(self.encoder.embedding, lang, hidden_size, self.decoder_hop,
                                              dropout)  # Generator(lang, hidden_size, dropout)
            self.reason = Reason(lang.n_words, hidden_size)

        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.extKnow_optimizer = optim.Adam(self.extKnow.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.reason_optimizer = optim.Adam(self.reason.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)
        self.criterion_bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.reset()

        if USE_CUDA:
            self.encoder.cuda()
            self.extKnow.cuda()
            self.decoder.cuda()
            self.reason.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_g = self.loss_g / self.print_every
        print_loss_v = self.loss_v / self.print_every
        print_loss_l = self.loss_l / self.print_every
        print_loss_r = self.loss_r / self.print_every
        self.print_every += 1
        return 'L:{:.2f},LE:{:.2f},LG:{:.2f},LP:{:.2f}, RE:{:.2f}'.format(print_loss_avg, print_loss_g, print_loss_v,
                                                                          print_loss_l, print_loss_r)

    def save_model(self, dec_type, epoch):
        if self.dataset == 'kvr':
            name_data = "KVR/"
        elif self.dataset == 'mul':
            name_data = "MUL/"
        elif self.dataset == 'cam':
            name_data = "CAM/"
        layer_info = str(self.n_layers)
        directory = 'save/KFAP-' + args["addName"] + name_data + str(self.task) + 'EP' + str(epoch) + 'HDD' + str(
            self.hidden_size) + 'BSZ' + str(args['batch']) + 'DR' + str(self.dropout) + 'L' + layer_info + 'lr' + str(
            self.lr) + str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.extKnow, directory + '/enc_kb.th')
        torch.save(self.decoder, directory + '/dec.th')
        torch.save(self.reason, directory + '/rea.th')

    def reset(self):
        self.loss, self.print_every, self.loss_g, self.loss_v, self.loss_l, self.loss_r = 0, 1, 0, 0, 0, 0

    def _cuda(self, x):
        if USE_CUDA:
            return torch.Tensor(x).cuda()
        else:
            return torch.Tensor(x)

    def train_batch(self, data, clip, reset=0):
        if reset: self.reset()
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.extKnow_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.reason_optimizer.zero_grad()

        # Encode and Decode
        # 以一定概率使用真正的目标输出作为下一个输入，而不是使用解码器的猜测作为下一个输入。可以加快收敛，但训练好的网络可能会表现出不稳定性。
        use_teacher_forcing = random.random() < args['teacher_forcing_ratio']
        max_target_length = max(data['response_lengths'])
        all_decoder_outputs_vocab, all_decoder_outputs_ptr, _, _, global_pointer, re, reason = self.encode_and_decode(
            data,
            max_target_length,
            use_teacher_forcing,
            False)

        # Loss calculation and backpropagation
        loss_g = self.criterion_bce(global_pointer, data['selector_index'])  # BCELoss为二元交叉熵损失,训练全局指针
        loss_v = masked_cross_entropy(  # 交叉熵损失训练解码器生成的词
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),
            data['sketch_response'].contiguous(),
            data['response_lengths'])
        loss_l = masked_cross_entropy(  # 交叉熵训练局部指针
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(),
            # (max_target_length, bsz, maxlen)->(bsz, max_target_length, maxlen)
            data['ptr_index'].contiguous(),  # (bsz, max_target_length)
            data['response_lengths'])

        new_index = copy.deepcopy(data['selector_index'])
        for bi in range(data['selector_index'].size()[0]):
            new_index[bi, data['kb_arr_lengths'][bi]:data['context_arr_lengths'][bi] - 1] = 0

        loss = loss_g + loss_v + loss_l
        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        ec = torch.nn.utils.clip_grad_norm_(self.extKnow.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        rc = torch.nn.utils.clip_grad_norm_(self.reason.parameters(), clip)


        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.extKnow_optimizer.step()
        self.decoder_optimizer.step()
        self.reason_optimizer.step()
        self.loss += loss.item()
        self.loss_g += loss_g.item()
        self.loss_v += loss_v.item()
        self.loss_l += loss_l.item()


    def encode_and_decode(self, data, max_target_length, use_teacher_forcing, get_decoded_words):
        # Build unknown mask for memory
        if args['unk_mask'] and self.decoder.training:  # 初始化mask矩阵，对应论文里OOV设置的mask部分输入为UNK
            story_size = data['context_arr'].size()
            rand_mask = np.ones(story_size)  # (batch_size, seq_length, memory_size)
            bi_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
            rand_mask[:, :, 0] = rand_mask[:, :, 0] * bi_mask
            kb_rand_mask = np.ones(data['kb_arr'].size())
            conv_rand_mask = np.ones(data['conv_arr'].size())  # (seq_length, batch_size, memory_size)
            for bi in range(story_size[0]):
                start, end = data['kb_arr_lengths'][bi], data['kb_arr_lengths'][bi] + data['conv_arr_lengths'][bi]
                kb_rand_mask[:start, bi, :] = rand_mask[bi, :start, :]
                conv_rand_mask[:end - start, bi, :] = rand_mask[bi, start:end, :]
            rand_mask = self._cuda(rand_mask)
            kb_rand_mask = self._cuda(kb_rand_mask)
            conv_rand_mask = self._cuda(conv_rand_mask)
            kb_story = data['kb_arr'] * kb_rand_mask.long()
            conv_story = data['conv_arr'] * conv_rand_mask.long()
            story = data['context_arr'] * rand_mask.long()
        else:
            story, kb_story, conv_story = data['context_arr'], data['kb_arr'], data[
                'conv_arr']  # conv_story（max_len,batch_size,memory_size）

        # Encode dialog history and KB to vectors
        dh_outputs, dh_hidden = self.encoder(conv_story, data['conv_arr_lengths'])

        global_pointer, kb_readout, memory_story = self.extKnow.load_memory(story, data['kb_arr_lengths'], data['conv_arr_lengths'],
                                                              dh_hidden, dh_outputs)

        memory_mask = (torch.zeros_like(global_pointer)).cuda()
        for bi in range(global_pointer.size()[0]):
            memory_mask[bi, :data['context_arr_lengths'][bi]] = 1

        m_soft = []
        m_logits = []
        m_out = []
        global_pointer_ = torch.ones_like(global_pointer)
        for li in range(dh_outputs.size()[1]):
            soft, logits, out = self.extKnow(dh_outputs[:, li, :], global_pointer, None, False, memory_mask, False, None)
            m_soft.append(soft)
            m_logits.append(logits)
            m_out.append(out)
        m_soft = torch.stack(m_soft)  # (context_len, bsz, kb_len)
        m_logits = torch.stack(m_logits)
        m_out = torch.stack(m_out)  # (context_len, bsz, hsz)

        # 修改
        reason, r = self.reason(dh_outputs, dh_hidden, self.extKnow, global_pointer, data['context_arr'].size()[0],
                                 story, data['domain'], data['context_arr_lengths'],
                                 data['kb_arr_lengths'], data['conv_arr_lengths'], memory_mask, memory_story)  # (bsz, len)

        # prob_soft_reason, prob_logits_reason, out_reason = self.extKnow(dh_hidden.squeeze(0), global_pointer, reason, True, memory_mask, False, None)
        prob_soft_reason, prob_logits_reason, out_reason = self.extKnow(dh_hidden.squeeze(0), global_pointer, reason, True, memory_mask, False, None)


        # encoded_hidden = torch.cat((dh_hidden.squeeze(0), kb_readout), dim=1)  # 拼接编码器隐状态与知识读出作为解码器的输入
        encoded_hidden = torch.cat((dh_hidden.squeeze(0), out_reason), dim=1)

        # Get the words that can be copy from the memory
        batch_size = len(data['context_arr_lengths'])
        self.copy_list = []
        for elm in data['context_arr_plain']:  # 将知识的对象部分和历史上下文中的所有单词加入self.copy_list
            elm_temp = [word_arr[0] for word_arr in elm]
            self.copy_list.append(elm_temp)

        outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse = self.decoder.forward(
            self.extKnow,
            story.size(),
            data['context_arr_lengths'],
            self.copy_list,
            encoded_hidden,
            data['sketch_response'],
            max_target_length,
            batch_size,
            use_teacher_forcing,
            get_decoded_words,
            global_pointer,
            dh_outputs,
            m_soft,
            m_logits,
            m_out,
            data['conv_arr_lengths'],
            memory_mask)

        return outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse, global_pointer, r, reason

    def evaluate(self, epoch, dev, matric_best, early_stop=None):
        global result
        print("STARTING EVALUATION")
        # Set to not-training mode to disable dropout
        self.encoder.train(False)  # 因为train函数的默认为True，即开启dropout
        self.extKnow.train(False)
        self.decoder.train(False)
        self.reason.train(False)


        ref, hyp = [], []
        acc, total = 0, 0
        dialog_acc_dict = {}
        F1_pred, F1_cal_pred, F1_nav_pred, F1_wet_pred = 0, 0, 0, 0
        F1_count, F1_cal_count, F1_nav_count, F1_wet_count = 0, 0, 0, 0
        F1_hot_pred, F1_res_pred, F1_att_pred = 0, 0, 0
        F1_hot_count, F1_res_count, F1_att_count = 0, 0, 0
        pbar = tqdm(enumerate(dev), total=len(dev))
        new_precision, new_recall, new_f1_score = 0, 0, 0

        if self.dataset == 'kvr':
            fw = open('./result/result_kvr.txt', 'w', encoding='utf-8')
            fj = open('./result/result_kvr.json', 'w')
        elif self.dataset == 'cam':
            fw = open('./result/result_cam.txt', 'w', encoding='utf-8')
            fj = open('./result/result_cam.json', 'w')
        elif self.dataset == 'mul':
            fw = open('./result/result_mul.txt', 'w', encoding='utf-8')
            fj = open('./result/result_mul.json', 'w')

        json_dicts = []
        sample_id = 0

        if args['dataset'] == 'kvr':
            with open('data/KVR/kvret_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ', '_') for item in
                                               global_entity[key]]  # lower为小写
                    else:
                        for item in global_entity[
                            'poi']:  # 'poi'的形式{'address': '593 Arrowhead Way', 'poi': 'Chef_Chu_s', 'type': 'chinese restaurant'}
                            global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
                global_entity_list = list(set(global_entity_list))

        if args['dataset'] == 'mul':
            with open('data/MULTIWOZ2.1/global_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]  # lower为小写
                global_entity_list = list(set(global_entity_list))

        if args['dataset'] == 'cam':
            with open('data/CamRest/camrest676-entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]  # lower为小写
                global_entity_list = list(set(global_entity_list))

        for j, data_dev in pbar:
            # Encode and Decode
            _, _, decoded_fine, decoded_coarse, global_pointer, re, reason = self.encode_and_decode(data_dev,
                                                                                                    self.max_resp_len,
                                                                                                    False, True)
            decoded_coarse = np.transpose(decoded_coarse)  # 转为array格式
            decoded_fine = np.transpose(decoded_fine)
            for bi, row in enumerate(decoded_fine):
                st = ''
                for e in row:
                    if e == 'EOS':
                        break
                    else:
                        st += e + ' '
                st_c = ''
                for e in decoded_coarse[bi]:
                    if e == 'EOS':
                        break
                    else:
                        st_c += e + ' '
                pred_sent = st.lstrip().rstrip()  # 去除两边的空格
                pred_sent_coarse = st_c.lstrip().rstrip()
                gold_sent = data_dev['response_plain'][bi].lstrip().rstrip()  # 正确答案
                ref.append(gold_sent)
                hyp.append(pred_sent)

                if args['dataset'] == 'kvr':
                    # compute F1 SCORE
                    single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_cal'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_cal_pred += single_f1
                    F1_cal_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_nav'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_nav_pred += single_f1
                    F1_nav_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_wet'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_wet_pred += single_f1
                    F1_wet_count += count
                elif args['dataset'] == 'mul':
                    # compute F1 SCORE
                    single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_hot'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_hot_pred += single_f1
                    F1_hot_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_res'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_res_pred += single_f1
                    F1_res_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_att'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_att_pred += single_f1
                    F1_att_count += count
                elif args['dataset'] == 'cam':
                    # compute F1 SCORE
                    single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count
                else:
                    # compute Dialogue Accuracy Score
                    current_id = data_dev['ID'][bi]
                    if current_id not in dialog_acc_dict.keys():
                        dialog_acc_dict[current_id] = []
                    if gold_sent == pred_sent:
                        dialog_acc_dict[current_id].append(1)
                    else:
                        dialog_acc_dict[current_id].append(0)

                # compute Per-response Accuracy Score
                total += 1
                if (gold_sent == pred_sent):
                    acc += 1

                if args['genSample']:
                    json_dict = self.print_examples(bi, data_dev, pred_sent, pred_sent_coarse, gold_sent, fw, fj, reason[bi], data_dev['ent_index'], sample_id)
                    json_dicts.append(json_dict)
                sample_id += 1
        json.dump(json_dicts, fj, indent=1)
        # Set back to training mode
        self.encoder.train(True)
        self.extKnow.train(True)
        self.decoder.train(True)
        self.reason.train(True)

        bleu1_score, bleu2_score = compute_bleu(np.array(hyp), np.array(ref))
        distinct_1, distinct_2 = compute_distinct(np.array(hyp))
        acc_score = acc / float(total)  # 每次回复准确率
        print("ACC SCORE:\t" + str(acc_score))

        if args['dataset'] == 'kvr':
            F1_score = F1_pred / float(F1_count)
            print("F1 SCORE:\t{}".format(F1_pred / float(F1_count)))
            print("\tCAL F1:\t{}".format(F1_cal_pred / float(F1_cal_count)))
            print("\tWET F1:\t{}".format(F1_wet_pred / float(F1_wet_count)))
            print("\tNAV F1:\t{}".format(F1_nav_pred / float(F1_nav_count)))
            print("BLEU_1 SCORE:\t" + str(bleu1_score))
            print("BLEU_2 SCORE:\t" + str(bleu2_score))
            print("DISTINCT_1 SCORE:\t" + str(distinct_1))
            print("DISTINCT_2 SCORE:\t" + str(distinct_2))
        elif args['dataset'] == 'mul':
            F1_score = F1_pred / float(F1_count)
            print("F1 SCORE:\t{}".format(F1_pred / float(F1_count)))
            print("\tHOT F1:\t{}".format(F1_hot_pred / float(F1_hot_count)))
            print("\tRES F1:\t{}".format(F1_res_pred / float(F1_res_count)))
            print("\tATT F1:\t{}".format(F1_att_pred / float(F1_att_count)))
            print("BLEU_1 SCORE:\t" + str(bleu1_score))
            print("BLEU_2 SCORE:\t" + str(bleu2_score))
            print("DISTINCT_1 SCORE:\t" + str(distinct_1))
            print("DISTINCT_2 SCORE:\t" + str(distinct_2))
        elif args['dataset'] == 'cam':
            F1_score = F1_pred / float(F1_count)
            print("F1 SCORE:\t{}".format(F1_pred / float(F1_count)))
            print("BLEU_1 SCORE:\t" + str(bleu1_score))
            print("BLEU_2 SCORE:\t" + str(bleu2_score))
            print("DISTINCT_1 SCORE:\t" + str(distinct_1))
            print("DISTINCT_2 SCORE:\t" + str(distinct_2))
        else:  # 每次对话准确率
            dia_acc = 0
            for k in dialog_acc_dict.keys():
                if len(dialog_acc_dict[k]) == sum(dialog_acc_dict[k]):
                    dia_acc += 1
            print("Dialog Accuracy:\t" + str(dia_acc * 1.0 / len(dialog_acc_dict.keys())))

        if (early_stop == 'BLEU'):
            # if (bleu1_score >= matric_best):
            self.save_model('ENTF1-{:.4f}'.format(F1_score) + 'BLEU-' + str(bleu1_score), epoch)
            print("MODEL SAVED")
            return bleu1_score
        elif (early_stop == 'ENTF1'):
            # if (F1_score >= matric_best):
            self.save_model('ENTF1-{:.4f}'.format(F1_score) + 'BLEU-' + str(bleu1_score), epoch)
            print("MODEL SAVED")
            return F1_score
        else:
            if (acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(acc_score))
                print("MODEL SAVED")
            return acc_score

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0  # 分别是预测的多少是正确的，预测的多少是不正确的，多少正确的未被预测
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0  # 计算准确率
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0  # 计算召回率
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0  # 计算F1分数
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count

    def print_examples(self, batch_idx, data, pred_sent, pred_sent_coarse, gold_sent, fw, fj, reason, entity, sample_id):
        kb_len = len(data['context_arr_plain'][batch_idx]) - data['conv_arr_lengths'][batch_idx] - 1
        # print("{}: ID{} id{} ".format(data['domain'][batch_idx], data['ID'][batch_idx], data['id'][batch_idx]))

        temp_dict = {}
        temp_dict['ID'] = data['id'][batch_idx]
        temp_dict['id'] = sample_id
        temp_dict['knowledge'] = []
        # temp_dict['filter'] = []
        temp_dict['context'] = []


        for i in range(kb_len):
            kb_temp = [w for w in data['context_arr_plain'][batch_idx][i] if w != 'PAD']  # context_arr_plain前面部分为KB信息
            kb_temp = kb_temp[::-1]
            temp_dict['knowledge'].append(' '.join(kb_temp))
            if 'poi' not in kb_temp:
                fw.write(str(kb_temp) + '\n')
                temp_dict['knowledge'].append(str(kb_temp))
                # print(kb_temp)
        flag_uttr, uttr = '$u', []
        for word_idx, word_arr in enumerate(data['context_arr_plain'][batch_idx][kb_len:]):
            if word_arr[1] == flag_uttr:
                uttr.append(word_arr[0])
            else:
                fw.write(str(flag_uttr) + ': ' + " ".join(uttr) + '\n')
                # print(flag_uttr, ': ', " ".join(uttr))
                temp_dict['context'].append(' '.join(uttr))
                flag_uttr = word_arr[1]
                uttr = [word_arr[0]]
        fw.write('\n')
        fw.write('reason: ' + str(reason) + '\n')
        # fw.write('re: ' + str(re[:data['kb_arr_lengths'][batch_idx]]) + '\n')
        fw.write('re_plain: ' + '\n')
        for i in range(len(reason)):
            fw.write('          ' + str(data['context_arr_plain'][batch_idx][reason[i]]) + '\n')
            # temp_dict['filter'].append(' '.join(data['context_arr_plain'][batch_idx][reason[i]]))
        fw.write('\n')
        # fw.write(
        #     'selector_index: ' + str(data['selector_index'][batch_idx][:data['kb_arr_lengths'][batch_idx]]) + '\n')
        # fw.write('selector_index_kb: ' + str(data['selector_index_kb'][batch_idx]) + '\n')
        fw.write('Sketch System Response : ' + str(pred_sent_coarse) + '\n')
        fw.write('Final System Response : ' + str(pred_sent) + '\n')
        fw.write('Gold System Response : ' + str(gold_sent) + '\n')
        fw.write('Gold Entity:           ' + str(entity[batch_idx]) + '\n')
        fw.write('\n')
        fw.write('\n')
        # print('Sketch System Response : ', pred_sent_coarse)
        # print('Final System Response : ', pred_sent)
        # print('Gold System Response : ', gold_sent)
        # print('\n')
        temp_dict['entity       '] = entity[batch_idx]
        temp_dict['response     '] = gold_sent
        temp_dict['sketch_result'] = pred_sent_coarse
        temp_dict['result       '] = pred_sent

        return temp_dict