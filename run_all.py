import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import random
import numpy as np
import torch
import json


seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(seed)

data_list = ['Structured/Amazon-Google',
             'Structured/BeerAdvo-RateBeer',
             'Structured/DBLP-ACM',
             'Structured/Fodors-Zagats',
             'Structured/iTunes-Amazon',
             'Dirty/DBLP-ACM',
             'Dirty/iTunes-Amazon',
             'Textual/Abt-Buy']

for data in data_list:
    configs = json.load(open('configs.json'))
    configs = {conf['name']: conf for conf in configs}
    config = configs[data]

    cmd = """python train.py \
    --data_name %s --n_epoch %d --seed %d""" % (data, config['epoch'], seed)
    if config['literal_channel']:
        cmd += ' --literal'
    if config['digital_channel']:
        cmd += ' --digital'
    if config['structure_channel']:
        cmd += ' --structure'
    if config['name_channel']:
        cmd += ' --name'
    print(cmd)
    os.system(cmd)





