import os
import re
import sys
import json
import random
import time
import statistics
from math import sqrt
from tqdm import tqdm
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# import nltk
# # nltk.download ('stopwords')
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))

import numpy as np
import pandas as pd

from sklearn import metrics
# from sklearn.metrics import precision_recall_fscore_support

# from torchcrf import CRF

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils.rnn import pad_sequence
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# import torchtext
# from torchtext.legacy.data import BucketIterator
# import torchcrf 
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from torch.optim import AdamW
from transformers import get_scheduler
from datasets import load_metric
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import pipeline
from transformers import TrainerCallback
from logger import setup_logging
import time
import copy
from copy import deepcopy
from transformers import AutoConfig
from transformers import AdapterConfig
from transformers.adapters.composition import Stack
from transformers import AutoTokenizer, AutoAdapterModel
from transformers import TrainingArguments, AdapterTrainer

import sklearn



TIME = int(time.time())
folder = f'models/{TIME}'

if not os.path.exists(folder):
   os.makedirs(folder)

logger = setup_logging (console_log_level='debug', logfile_log_level='debug', log_file_path=os.path.join(folder,'log.log'))

params = {}
params['train_path'] = 'data/train.tsv'
params['test_path'] = 'data/test_kaggle_trans.tsv'
# params['val_path'] = 'data/dev.txt'
# params['labels_path'] = 'data/labels.json'
params['model_path'] = folder
params['model_file'] = os.path.join(params['model_path'], 'model.pkl')
params['my_model_file'] = os.path.join(params['model_path'], 'my_model.pkl')
params['my_adapter_file'] = os.path.join(params['model_path'], 'my_nli.pkl')
params['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# params['load_model'] = 'models/1669729888/my_model.pkl'    # with nli and adapter fine tuning
params['load_model'] = 'models/1669728712/my_model.pkl'  # with adapter fine tuning only
# params['load_model'] = 'models/1669726610/my_model.pkl' # without adapter fine tuning

# params['dropout'] = 0.2

# Training:
params['lr'] = 5e-4
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
