import os
import sys
import json
import random
import time
from tqdm import tqdm
from logger import setup_logging
import copy
from copy import deepcopy

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
 
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import Dataset
from datasets import load_metric

from transformers import pipeline
from transformers import AutoConfig
from transformers import AdapterConfig
from transformers import get_scheduler
from transformers import TrainerCallback
from transformers import BertTokenizer, BertModel
from transformers.adapters.composition import Stack
from transformers import AutoTokenizer, AutoAdapterModel
from transformers import TrainingArguments, AdapterTrainer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


TIME = int(time.time())
folder = f'models/{TIME}'

if not os.path.exists(folder):
   os.makedirs(folder)

logger = setup_logging (console_log_level='debug', logfile_log_level='debug', log_file_path=os.path.join(folder,'log.log'))

params = {}
params['train_path'] = 'data/train.tsv'
params['test_path'] = 'data/test_kaggle_trans.tsv'
params['model_path'] = 'model'
params['my_model_file'] = os.path.join(params['model_path'], 'cs1190722_cs5190768.pkl')
params['my_model_file_nli'] = os.path.join(params['model_path'], 'cs1190722_cs5190768_nli.pkl')
params['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# params['load_model'] = 'models/1669729888/my_model.pkl'    # with nli and adapter fine tuning
# params['load_model'] = 'models/1669728712/my_model.pkl'  # with adapter fine tuning only
# params['load_model'] = 'models/1669726610/my_model.pkl' # without adapter fine tuning
params['load_model'] = params['my_model_file']

# Training:
params['lr'] = 5e-4
params['lr_ft'] = 5e-5
params['train_bs'] = 64
params['val_bs'] = 64
params['test_bs'] = 64
params['num_workers'] = 1

params['train_epochs'] = 10
params['grad_clip'] = 5.0
params['seed'] = 69

params['label_map'] = {
    'neutral': 0,
    'contradiction' : 1,
    'entailment': 2
}

params['inv_label_map'] = {
    0: 'neutral',
    1: 'contradiction',
    2: 'entailment'
}