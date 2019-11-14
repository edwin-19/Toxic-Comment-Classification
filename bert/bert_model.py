import numpy as np
import pandas as pd 

import torch
import torch.optim as optim

import fastai
from fastai.vision import *
from fastai.text import *
from torchvision.models import *
import pretrainedmodels

import sys

from fastai.callbacks.tracker import EarlyStoppingCallback
from fastai.callbacks.tracker import SaveModelCallback
from fastai.utils import *
from torch.utils import *
from sklearn.model_selection import train_test_split
from fastai.callback import *

from transformers import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertForNextSentencePrediction, BertForMaskedLM 

class FastAiBertTokenizer(BaseTokenizer):
    """Wrapper around BertTokenizer to be compatible with fast.ai"""
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __call__(self, *args, **kwargs):
        return self
    
    def tokenizer(self, t:str) -> List[str]:
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]

def load_bert_model(model_path):
    data_root = '../data/toxic_comment/cleaned/'
    bert_tok = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )
    
    train_df = pd.read_csv(data_root + 'train.csv')
    test_df = pd.read_csv(data_root + 'test.csv')
    
    # split 8:2 ratio
    train_df.fillna('no comment', inplace=True)
    test_df.fillna('no comment', inplace=True)
    train, val = train_test_split(
        train_df, shuffle=True, test_size=0.2, random_state=42
    )
    
    fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))
    
    label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    fastai_tokenizer = Tokenizer(
        tok_func=FastAiBertTokenizer(
            bert_tok, max_seq_len=256
        ), pre_rules=[], post_rules=[]
    )

    data_bunch_train = TextClasDataBunch.from_df(
        model_path, train, val,
        tokenizer=fastai_tokenizer,
        vocab=fastai_bert_vocab,
        include_bos=False,
        include_eos=False,
        text_cols="comment_text",
        label_cols=label_columns,
        bs=12,
        collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
    )
    
    bert_model_class = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=6
    )
    
    loss_func = nn.BCEWithLogitsLoss()
    acc_02 = partial(accuracy_thresh, thresh=0.25)
    model = bert_model_class
    
    learner = Learner(
        data_bunch_train, model,
        loss_func=loss_func, model_dir='model/', metrics=acc_02
    )
    
    return learner