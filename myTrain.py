from tqdm import tqdm

from utils.config import *
from models.KFAP import *

'''
Command:

python myTrain.py -ds= -dec= -bsz= -t= -hdd= -dr= -l= -lr=

'''

early_stop = args['earlyStop']
if args['dataset'] == 'kvr':
    from utils.utils_Ent_kvr import *

    # early_stop = 'ENTF1'
    early_stop = 'BLEU'
elif args['dataset'] == 'mul':
    from utils.utils_Ent_multi import *

    # early_stop = 'ENTF1'
    early_stop = 'BLEU'
elif args['dataset'] == 'cam':
    from utils.utils_Ent_camrest import *

    early_stop = 'BLEU'
    # early_stop = 'ENTF1'


else:
    print("[ERROR] You need to provide the --dataset information")

# Configure models and load data
avg_best, cnt, acc = 0.0, 0, 0.0
train, dev, test, testOOV, lang, max_resp_len = prepare_data_seq(args['task'], batch_size=int(args['batch']))

model = globals()[args['decoder']](
    int(args['hidden']),
    lang,
    max_resp_len,
    args['path'],
    args['task'],
    args['dataset'],
    lr=float(args['learn']),
    n_layers=int(args['layer']),
    dropout=float(args['drop']))

CUDA_LAUNCH_BLOCKING = '1'

# for epoch in range(200):
for epoch in range(200):
    print("Epoch:{}".format(epoch))
    # Run the train function
    pbar = tqdm(enumerate(train), total=len(train))
    for i, data in pbar:
        model.train_batch(data, int(args['clip']), reset=(i == 0))
        pbar.set_description(model.print_loss())  # 设置tqdm描述为自定义打印loss
        # break
    if ((epoch + 1) % int(args['evalp']) == 0):  # 评估周期，默认为每一轮评估一次
        acc = model.evaluate(epoch, dev, avg_best, early_stop)
        model.scheduler.step(acc)

        if (acc >= avg_best):
            avg_best = acc
            cnt = 0
        else:
            cnt += 1

        if (cnt == 8 or (acc == 1.0 and early_stop == None)):  # 连续八轮没有超过最好的acc，或acc=1.0，则提前停止
            print("Ran out of patient, early stop...")
            break

# data['context_arr'].size() = (bsz, len, mem)
# data['kb_arr'].size() = (len, bsz, mem)
# data['conv_arr'].size() = (len. bsz, mem)
# data['response'].size() = (bsz, len)
