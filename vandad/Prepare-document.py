import math
import os
import numpy
import torch

import numpy
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from torch.utils.data import Dataset
from tqdm import tqdm

import pandas as pd
import pickle

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
tokenizer = BertTokenizer('_cache/vocabs/bert-base-multilingual-uncased-vocab.txt')


# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


with open('dataset/trainingandtestdata/training_data.pkl', 'rb') as f:
    data = pickle.load(f)



#model = BertModel.from_pretrained('_cache/models/bert-base-multilingual-uncased.tar.gz', cache_dir='_cache')


import nltk.data

tokenizer = nltk.data.load('_cache/nltk/tokenizers/punkt/english.pickle')


f= open("_cache/document.txt","w+", encoding="utf-8")

for i in range(len(data)):
    seperator = ' '
    text = seperator.join(data[i]);
    text = text.replace('. <repeated>', 'REP')
    text = text.replace('. ! <repeated>', 'REP1')
    text = text.replace('? ! <repeated>', 'REP2')
    text = text.replace('! ? <repeated>', 'REP3')
    text = text.replace('? !', 'REP4')
    text = text.replace('! ?', 'REP5')
    text = text.replace('! <repeated>', 'REP6')
    text = text.replace('? <repeated>', 'REP7')
    text = text.replace('. . .', 'REP8')
    
    text = '\n'.join(tokenizer.tokenize(text))
    
    text = text.replace('REP', '. <repeated>\n')
    text = text.replace('REP1', '. ! <repeated>\n')
    text = text.replace('REP2', '? ! <repeated>\n')
    text = text.replace('REP3', '! ? <repeated>\n')
    text = text.replace('REP4', '? !\n')
    text = text.replace('REP5', '! ?\n')
    text = text.replace('REP6', '! <repeated>\n')
    text = text.replace('REP7', '? <repeated>\n')
    text = text.replace('REP8', '. <repeated>\n')
    
    text = text.replace('\n ','\n')
    
    sentence_list = text.split('\n')
    j = 0
    while j < len(sentence_list)-1:
        if len(sentence_list[j+1].split(' ')) < 4:
            sentence_list[j] = sentence_list[j] + ' ' + sentence_list[j+1]
            for k in range(j+1,len(sentence_list)-1):
                sentence_list[k] = sentence_list[k+1]
            sentence_list = sentence_list[:-1]
            j -= 1
        j += 1
    text = '\n'.join(sentence_list)
    
    
    f.write(text + '\n\n')

f.close() 




