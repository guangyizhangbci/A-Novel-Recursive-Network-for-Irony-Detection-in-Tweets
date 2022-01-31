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


import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
tokenizer = BertTokenizer('_cache/vocabs/bert-base-multilingual-uncased-vocab.txt')


df = pd.read_csv("dataset/trainingandtestdata/training-data.csv", encoding='latin-1', sep=',', names =['id', 'id2', 'g1', 'g2', 'data', 'g3'])
texts = df.iloc[:,5].values.tolist()



preprocessor = TextPreProcessor(
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                       'time', 'date', 'number'],
            annotate={"hashtag", "elongated", "allcaps", "repeated",
                      'emphasis', 'censored'},
            all_caps_tag="wrap",
            fix_text=True,
            segmenter="twitter_2018",
            corrector="twitter_2018",
            unpack_hashtags=True,
            unpack_contractions=True,
            spell_correct_elong=True,
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts=[emoticons]
        ).pre_process_doc

name = 'all_data'
desc = "PreProcessing dataset {}...".format(name)
data = [preprocessor(x) for x in tqdm(texts, desc=desc)]



import pickle
with open('dataset/trainingandtestdata/training_data.pkl', 'wb') as f:
    pickle.dump(data, f)


