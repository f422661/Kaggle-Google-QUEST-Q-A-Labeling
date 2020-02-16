# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:37:58 2019

@author: khyeh
"""

import random; random.seed(0)
import numpy as np; np.random.seed(0)
#import tensorflow as tf; tf.set_random_seed(0)
import pandas as pd
from contextlib import contextmanager
import os
import random
from tqdm import tqdm
import gc; gc.enable(); #gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import time
import re
import warnings; warnings.filterwarnings('ignore')
import sys
import psutil
import shutil
from scipy.stats import spearmanr
from urllib.parse import urlparse
from math import floor, ceil

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

SUBMIT_MODE = True # change to true when submitting to kernels
DEV_MODE = False 
TRAIN_FINAL_MODEL = False

#############################
### don't change this
#############################
GENERATE_OOF = not DEV_MODE and not SUBMIT_MODE

if not SUBMIT_MODE:
    sys.path.insert(0, "apex/")
    from apex import amp
else:
    os.system("pip install ../input/sacremoses/sacremoses-master/ > /dev/null")
    # limit tensorflow gpu memory
    import tensorflow as tf
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpu_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    
import keras
import keras.backend as K
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(K.tensorflow_backend._get_available_gpus())

sys.path.insert(0, "../input/transformers/transformers-master/")
import transformers
  
############################# 
### Define Vaiables
#############################

INPUT_PATH = '../input/'
MODEL_DIR = INPUT_PATH+'kh-gqa-models/'
FAST_EMBEDDING_BIN_PATH = INPUT_PATH+'fasttext-common-crawl-bin-model-pkl/fasttext-common-crawl-bin-model/cc.en.300.bin'
CRAWL_EMBEDDING_PATH = INPUT_PATH+'fasttextcrawl300d2m/crawl-300d-2M.vec'
GLOVE_EMBEDDING_PATH = INPUT_PATH+'glove840b300dtxt/glove.840B.300d.txt'
PARA_EMBEDDING_PATH  = INPUT_PATH+'paragram-300-sl999/paragram_300_sl999/paragram_300_sl999.txt'
SUBMIT_MODE_RNN_EMBED_CACHE_PATH = 'submit_mode_rnn_embeddings.pkl'
SUBMIT_MODE_USE_FEATS_CACHE_PATH = 'pretrained_embedding_features.pkl'


TRAIN_PATH = INPUT_PATH+'google-quest-challenge/train.csv'
TEST_PATH = INPUT_PATH+'google-quest-challenge/test.csv'

SEED = 423
embed_size = 300
max_features = 100000
maxlens = {
    'question_title': 100, 
    'question_body': 300, 
    'answer': 300
}

maxlens_bert = {
    'question_title': 80, 
    'question_body': 256, 
    'answer': 256
}

text_cols = ['question_title','question_body','answer']
target_cols = [
        'question_asker_intent_understanding',
        'question_body_critical',
        'question_conversational',
        'question_expect_short_answer',
        'question_fact_seeking',
        'question_has_commonly_accepted_answer',
        'question_interestingness_others',
        'question_interestingness_self',
        'question_multi_intent',
        'question_not_really_a_question',
        'question_opinion_seeking',
        'question_type_choice',
        'question_type_compare',
        'question_type_consequence',
        'question_type_definition',
        'question_type_entity',
        'question_type_instructions',
        'question_type_procedure',
        'question_type_reason_explanation',
        'question_type_spelling',
        'question_well_written',
        'answer_helpful',
        'answer_level_of_information',
        'answer_plausible',
        'answer_relevance',
        'answer_satisfaction',
        'answer_type_instructions',
        'answer_type_procedure',
        'answer_type_reason_explanation',
        'answer_well_written'    
    ]

exclude_regex = re.compile(u'[^A-Za-z!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~]+')
                              
misspell_dict = {"aren't": "are not", "can't": "cannot", "couldn't": "could not",
                 "didn't": "did not", "doesn't": "does not", "don't": "do not",
                 "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                 "he'd": "he would", "he'll": "he will", "he's": "he is",
                 "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not",
                 "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",
                 "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
                 "she'd": "she would", "she'll": "she will", "she's": "she is",
                 "shouldn't": "should not", "that's": "that is", "there's": "there is",
                 "they'd": "they would", "they'll": "they will", "they're": "they are",
                 "they've": "they have", "we'd": "we would", "we're": "we are",
                 "weren't": "were not", "we've": "we have", "what'll": "what will",
                 "what're": "what are", "what's": "what is", "what've": "what have",
                 "where's": "where is", "who'd": "who would", "who'll": "who will",
                 "who're": "who are", "who's": "who is", "who've": "who have",
                 "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                 "you'll": "you will", "you're": "you are", "you've": "you have",
                 "'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying",
                 "'s": ""}


############################# 
### APIs
#############################
@contextmanager
def timer(msg):
    t0 = time.time()
    process = psutil.Process(os.getpid())
    print('[{}] start.'.format(msg))
    yield
    elapsed_time = (time.time() - t0)/60
    process_GB = process.memory_info().rss/(1024**3) # in GB
    print('[{}] done in {:.2f} min, current ram: {:.2f} gb'.format(msg, elapsed_time, process_GB))
    
def seed_everything(seed=SEED):
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    #tf.set_random_seed(seed)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    model_dir = '/'.join(path.split('/')[:-1])+"/"
    model_path = path.split('/')[-1]
    pickle_path = model_dir+".".join(model_path.split(".")[:-1])+".pkl"
    print('Checking :', pickle_path)
    
    if os.path.isfile(pickle_path):
        import joblib
        res = joblib.load(pickle_path)
        return res
    else:
        f = open(path, 'r', encoding="utf-8", errors='ignore')
        if not SUBMIT_MODE:
            res = dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f) ) #if len(line)>100)
        else:
            res = dict(get_coefs(*line.strip().split(' ')) for line in f)
        del f; gc.collect()
        
        if not SUBMIT_MODE:
            import joblib
            joblib.dump((res), pickle_path)
        return res

def build_matrix_from_bin(word_index_list, path):
    
    model_dir = '/'.join(path.split('/')[:-1])+"/"
    model_path = path.split('/')[-1]
    pickle_path = model_dir+".".join(model_path.split(".")[:-1])+".pkl"
    print('Checking :', pickle_path)
    
    if os.path.isfile(pickle_path):
        import joblib
        model = joblib.load(pickle_path)
    else:
        from gensim.models.wrappers import FastText
        model = FastText.load_fasttext_format(path)
        if not SUBMIT_MODE:
            import joblib
            joblib.dump((model), pickle_path)
        
    embedding_matrix_list = []
    for word_index in word_index_list:
        embedding_matrix = np.zeros((len(word_index) + 1, embed_size), dtype=np.float32)
       
        unknown_words = []
        for key, i in tqdm(word_index.items()): 
            try:
                embedding_matrix[i] =  model.wv[key]
            except:
                unknown_words += [key]
        print('# of unknown_words=', len(unknown_words))
              
        del unknown_words; gc.collect()
        embedding_matrix_list.append(embedding_matrix)
        
    del model; gc.collect()
    return embedding_matrix_list
   
def build_matrix(word_index_list, path):
    embeddings_index = load_embeddings(path)
    
    '''
    ss = SymSpell(max_edit_distance=2)
    ss.create_dictionary_from_arr_with_chk(list(embeddings_index.keys()), 
                                           word_length_lim=8) # limit number of max words for speed control
    '''
    
    embedding_matrix_list = []
    for word_index in word_index_list:
        embedding_matrix = np.zeros((len(word_index) + 1, embed_size), dtype=np.float32)
        unknown_words = []
        for key, i in tqdm(word_index.items()):
            
            word = key
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                continue       
            word = key.lower()
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                continue        
            word = key.capitalize()
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                continue
            '''         
            if len(key) > 1:
                bw = ss.best_word(key, silent=True)
                if bw is not None:
                    embedding_matrix[i] = embeddings_index.get(bw)
                    continue           
            word = key.upper()
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                continue      
            word = ps.stem(key)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                continue
            word = lc.stem(key)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                continue
            word = sb.stem(key)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                continue    
            word = wordnet_lemmatizer.lemmatize(key)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                continue
            '''
            unknown_words.append((key, i))
            
        print('# of unknown_words=', len(unknown_words))
        del unknown_words; gc.collect()
        embedding_matrix_list.append(embedding_matrix)
        
    del embeddings_index; gc.collect()
    return embedding_matrix_list

def calc_spearmanr(y_pred, y_true):
    return np.nan_to_num(spearmanr(y_true, y_pred).correlation)
    
def calc_spearmanr_metric(y_pred, y_true):
    score = 0
    for i in range(30):
        score +=  calc_spearmanr(y_true[:, i], y_pred[:, i])/ 30
        # valid_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
    return score

def calc_spearmanr_metric_list(y_pred, y_true):
    score = []
    for i in range(30):
        score += [np.nan_to_num(spearmanr(y_true[:, i], y_pred[:, i]).correlation)]
        # valid_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
    return score

def rank_gauss(x):
    # map the values to the same distribution for ranking
    from scipy.special import erfinv
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x

def rank_gauss_norm(x):
    for i in np.arange(x.shape[1]):
        x[:, i] = rank_gauss(x[:, i])
        print(x[:, i].min(), x[:, i].max())
    return x

class TextDataset(Dataset):

    def __init__(self, title_data, question_data, answer_data, meta_data, idxs, targets=None):
        self.title_data = title_data[idxs]
        self.question_data = question_data[idxs]
        self.answer_data = answer_data[idxs]
        self.meta_data = meta_data[idxs]
        self.targets = targets[idxs] if targets is not None else np.zeros((self.question_data.shape[0], 30))

    def __getitem__(self, idx):
        title = self.title_data[idx]
        question = self.question_data[idx]
        answer = self.answer_data[idx]
        meta = self.meta_data[idx]
        target = self.targets[idx]
        
        return title, question, answer, meta, target

    def __len__(self):
        return len(self.question_data)

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class RNN_No_Embed(nn.Module):
    def __init__(self,
                 hidden_size: int = 128,
                 hidden_size_title: int=32,
                 max_lens: list = maxlens,
                 embed_size: int = 300,
                 meta_size: int = 64,
                 dropout: float = 0.0,
                 dropout_title: float = 0.0,):
        
        super(RNN_No_Embed, self).__init__()
        
        self.sp_dropout_q = SpatialDropout(dropout)
        self.sp_dropout_a = SpatialDropout(dropout)
        
        self.gru_q = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru_q2 = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)
        self.gru_q_att = Attention(hidden_size * 2, max_lens['question_title']+max_lens['question_body'])
        
        self.gru_a = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru_a2 = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)
        self.gru_a_att = Attention(hidden_size * 2, max_lens['answer'])
        
        self.linear1_q = nn.Linear(hidden_size * 2, 512)
        self.droupout_q = nn.Dropout(0.1)
        self.linear_out_q = nn.Linear(512, len(target_cols)-9)
        
        self.linear1_a = nn.Linear(hidden_size * 2 * 2 + 512, 512)
        self.droupout_a = nn.Dropout(0.1)
        self.linear_out_a = nn.Linear(512, 9)
        
    def forward(self, embed_t, embed_q, embed_a):
        
        embed_q = torch.cat([embed_t, embed_q], 1)
        
        embed_q = self.sp_dropout_q(embed_q)
        h_gru_q, _ = self.gru_q(embed_q)
        h_gru_q, _ = self.gru_q2(h_gru_q)
        h_gru_q_att = self.gru_q_att(h_gru_q)
        
        
        embed_a = self.sp_dropout_a(embed_a)
        h_gru_a, _ = self.gru_a(embed_a)
        h_gru_a, _ = self.gru_a2(h_gru_a)
        h_gru_a_att = self.gru_a_att(h_gru_a)
        
        q_features = h_gru_q_att
        x = self.droupout_q(nn.ELU()(self.linear1_q(q_features)))
        out_q = self.linear_out_q(x)
        
        a_features = torch.cat((h_gru_q_att, h_gru_a_att, x), 1)
        x = self.droupout_a(nn.ELU()(self.linear1_a(a_features)))
        out_a = self.linear_out_a(x)
        
        return torch.cat([out_q, out_a], 1)
    
class RNN_No_Embed_Pretrain_Features(nn.Module):
    def __init__(self,
                 hidden_size: int = 128,
                 hidden_size_title: int=32,
                 max_lens: list = maxlens,
                 embed_size: int = 300,
                 pretrain_feature_size: int = 64,
                 dropout: float = 0.0,
                 dropout_title: float = 0.0,):
        
        super(RNN_No_Embed_Pretrain_Features, self).__init__()
        
        self.sp_dropout_q = SpatialDropout(dropout)
        self.sp_dropout_a = SpatialDropout(dropout)
        
        self.gru_q = nn.GRU(embed_size+pretrain_feature_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru_q2 = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)
        self.gru_q_att = Attention(hidden_size * 2, max_lens['question_title']+max_lens['question_body'])
        
        self.gru_a = nn.GRU(embed_size+pretrain_feature_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru_a2 = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)
        self.gru_a_att = Attention(hidden_size * 2, max_lens['answer'])
        
        self.linear1_q = nn.Linear(hidden_size * 2, 512)
        self.droupout_q = nn.Dropout(0.1)
        self.linear_out_q = nn.Linear(512, len(target_cols)-9)
        
        self.linear1_a = nn.Linear(hidden_size * 2 * 2 + 512, 512)
        self.droupout_a = nn.Dropout(0.1)
        self.linear_out_a = nn.Linear(512, 9)
        
    def forward(self, embed_t, embed_q, embed_a, pretrain_feats):
        
        pretrain_feats = pretrain_feats.view((-1, 1, pretrain_feats.shape[1]))
        
        embed_q = torch.cat([embed_t, embed_q], 1)
        embed_q = torch.cat([embed_q, pretrain_feats.repeat(1, embed_q.shape[1], 1)], 2)
        embed_q = self.sp_dropout_q(embed_q)
        
        h_gru_q, _ = self.gru_q(embed_q)
        h_gru_q, _ = self.gru_q2(h_gru_q)
        h_gru_q_att = self.gru_q_att(h_gru_q)
        
        
        embed_a = torch.cat([embed_a, pretrain_feats.repeat(1, embed_a.shape[1], 1)], 2)
        embed_a = self.sp_dropout_a(embed_a)
        h_gru_a, _ = self.gru_a(embed_a)
        h_gru_a, _ = self.gru_a2(h_gru_a)
        h_gru_a_att = self.gru_a_att(h_gru_a)
        
        q_features = h_gru_q_att
        x = self.droupout_q(nn.ELU()(self.linear1_q(q_features)))
        out_q = self.linear_out_q(x)
        
        a_features = torch.cat((h_gru_q_att, h_gru_a_att, x), 1)
        x = self.droupout_a(nn.ELU()(self.linear1_a(a_features)))
        out_a = self.linear_out_a(x)
        
        return torch.cat([out_q, out_a], 1)
    
class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
    
class RNN(nn.Module):
    def __init__(self,
                 embedding_matrixs, #embedding matrixs from [title,question,anaswer]
                 dropout: float = 0.0,
                 dropout_title: float = 0.0,
                 hidden_size: int = 128,
                 hidden_size_title: int=32,
                 max_lens: list = maxlens,
                 embed_size: int = 300,
                 meta_size: int = 64):
        super(RNN, self).__init__()
        
        
        self.embedding_t = nn.Embedding(*embedding_matrixs[0].shape)
        self.embedding_t.weight = nn.Parameter(torch.tensor(embedding_matrixs[0], dtype=torch.float32))
        self.embedding_t.weight.requires_grad = False
                
        self.embedding_q = nn.Embedding(*embedding_matrixs[1].shape)
        self.embedding_q.weight = nn.Parameter(torch.tensor(embedding_matrixs[1], dtype=torch.float32))
        self.embedding_q.weight.requires_grad = False

        self.embedding_a = nn.Embedding(*embedding_matrixs[2].shape)
        self.embedding_a.weight = nn.Parameter(torch.tensor(embedding_matrixs[2], dtype=torch.float32))
        self.embedding_a.weight.requires_grad = False
        
        self.rnn_no_embed = RNN_No_Embed(hidden_size = hidden_size,
                                         hidden_size_title = hidden_size_title,
                                         max_lens = max_lens,
                                         embed_size = embed_size,
                                         meta_size = meta_size,
                                         dropout = dropout,
                                         dropout_title = dropout_title,)

    def get_saving_model(self):
        return self.rnn_no_embed
    
    def forward(self, title, question, answer):
        h_embedding_t = self.embedding_t(title)
        h_embedding_q = self.embedding_q(question)
        h_embedding_a = self.embedding_a(answer)
        
        return self.rnn_no_embed(h_embedding_t, h_embedding_q, h_embedding_a)

class RNN_Pretrain_Features(nn.Module):
    def __init__(self,
                 embedding_matrixs, #embedding matrixs from [title,question,anaswer]
                 dropout: float = 0.0,
                 dropout_title: float = 0.0,
                 hidden_size: int = 128,
                 hidden_size_title: int=32,
                 max_lens: list = maxlens,
                 embed_size: int = 300,
                 pretrain_feature_size: int = 64):
        super(RNN_Pretrain_Features, self).__init__()
        
        self.embedding_t = nn.Embedding(*embedding_matrixs[0].shape)
        self.embedding_t.weight = nn.Parameter(torch.tensor(embedding_matrixs[0], dtype=torch.float32))
        self.embedding_t.weight.requires_grad = False
                
        self.embedding_q = nn.Embedding(*embedding_matrixs[1].shape)
        self.embedding_q.weight = nn.Parameter(torch.tensor(embedding_matrixs[1], dtype=torch.float32))
        self.embedding_q.weight.requires_grad = False

        self.embedding_a = nn.Embedding(*embedding_matrixs[2].shape)
        self.embedding_a.weight = nn.Parameter(torch.tensor(embedding_matrixs[2], dtype=torch.float32))
        self.embedding_a.weight.requires_grad = False
        
        self.rnn_no_embed = RNN_No_Embed_Pretrain_Features(hidden_size = hidden_size,
                                                           hidden_size_title = hidden_size_title,
                                                           max_lens = max_lens,
                                                           embed_size = embed_size,
                                                           pretrain_feature_size = pretrain_feature_size,
                                                           dropout = dropout,
                                                           dropout_title = dropout_title,)
        

    def get_saving_model(self):
        return self.rnn_no_embed
    
    def forward(self, title, question, answer, pretrain_feats):
        h_embedding_t = self.embedding_t(title)
        h_embedding_q = self.embedding_q(question)
        h_embedding_a = self.embedding_a(answer)
        
        return self.rnn_no_embed(h_embedding_t, h_embedding_q, h_embedding_a, pretrain_feats)
    
    
class Modeling_Pipeline():
    def __init__(self, train_path, test_path, config):
        self.train_path = train_path
        self.test_path = test_path
        self.config = config

    def get_netloc(self, df):
        find = re.compile(r"^[^.]*")
        return df['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])
    
    def get_pickle_ohe_path(self):
        return MODEL_DIR+'ohe_v1.pkl'
    
class CyclicLR(lr_scheduler._LRScheduler):

    def __init__(self, optimizer, base_lr, max_lr, step_size, gamma=0.99, mode='triangular', last_epoch=-1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.gamma = gamma
        self.mode = mode
        assert mode in ['triangular', 'triangular2', 'exp_range']
        super(CyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        new_lr = []
        # make sure that the length of base_lrs doesn't change. Dont care about the actual value
        for base_lr in self.base_lrs:
            cycle = np.floor(1 + self.last_epoch / (2 * self.step_size))
            x = np.abs(float(self.last_epoch) / self.step_size - 2 * cycle + 1)
            if self.mode == 'triangular':
                lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
            elif self.mode == 'triangular2':
                lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
            elif self.mode == 'exp_range':
                lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * (self.gamma ** (
                    self.last_epoch))
            new_lr.append(lr)
        return new_lr
        
class RNN_Pipeline(Modeling_Pipeline):
    def __init__(self, *args):
        super(RNN_Pipeline, self).__init__(*args)
        
    @staticmethod
    def replace_(match):
        return misspell_dict[match.group(0)]
    
    @staticmethod
    def clean_special_chars(txt):
        for p in ".,!?:;~“”’\'\"`":
            txt = txt.replace(p, ' '+p+' ') 
        return txt
    
    @staticmethod
    def clean_(txt):
        txt = re.compile('(%s)' % '|'.join(misspell_dict.keys())).sub(RNN_Pipeline.replace_, txt)
        txt = RNN_Pipeline.clean_special_chars(txt)
        return txt
    
    @staticmethod
    def clean_sentences(data):
        return data.fillna('Amazing ~').astype(str).apply(RNN_Pipeline.clean_)

    def select_data_and_clean(self):
        with timer('Clean Texts'):
            from sklearn.preprocessing import OneHotEncoder
            
            self.x_train = {tc: None for tc in text_cols}
            self.x_test = {tc: None for tc in text_cols}
            self.target = None
            
            meta_features = ['netloc', 'category']
            if not SUBMIT_MODE:
                train_df = pd.read_csv(self.train_path)
                self.target = train_df[target_cols].values
                for tc in text_cols:
                    self.x_train[tc] = RNN_Pipeline.clean_sentences(train_df[tc])
                
                ohe = OneHotEncoder(dtype=np.float32, handle_unknown='ignore')
                train_df['netloc'] = self.get_netloc(train_df)
                self.meta_train = ohe.fit_transform(train_df[meta_features]).toarray()
                pd.to_pickle(ohe, self.get_pickle_ohe_path())
                
                del train_df; gc.collect()
            
            test_df = pd.read_csv(self.test_path)
            for tc in text_cols:
                self.x_test[tc] = RNN_Pipeline.clean_sentences(test_df[tc])
            
            ohe = pd.read_pickle(self.get_pickle_ohe_path())
            test_df['netloc'] = self.get_netloc(test_df)
            self.meta_test = ohe.transform(test_df[meta_features]).toarray()
            
            del test_df; gc.collect()
    
    def pad_sequences(self, sequences, maxlen, pre_prop=.3, post_prop=.7, pad_val=0, dtype='int32'):
        res = np.full((len(sequences), maxlen), pad_val, dtype=dtype)
        
        assert (pre_prop+post_prop) == 1.
        pre_len = int(maxlen*pre_prop)
        post_len = maxlen-pre_len
        #print(sequences)
        for i, s in enumerate(sequences):
            if not len(s):
                continue  # empty list/array was found
            
            if len(s) <= maxlen:
                res[i, -len(s):] = s
            else:
                res[i, :pre_len] = s[:pre_len]
                res[i, -post_len:] = s[-post_len:]
                
        return res
    
    def tokenize_and_pad_seq(self):
        with timer('Tokenizing and Padding seq'):
            from keras.preprocessing.text import Tokenizer
            from keras.preprocessing.sequence import pad_sequences
            #CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
            CHARS_TO_REMOVE = '#$%&()*+-/<=>@[\\]^_{|}\t\n∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
            self.tokenizer = {tc: Tokenizer(filters=CHARS_TO_REMOVE, lower=False) for tc in text_cols}
            
            for tc in text_cols:
                if not SUBMIT_MODE:
                    self.tokenizer[tc].fit_on_texts(list(self.x_train[tc])+list(self.x_test[tc]))
                
                    self.x_train[tc] = self.tokenizer[tc].texts_to_sequences(self.x_train[tc]); gc.collect()
                    self.x_test[tc] = self.tokenizer[tc].texts_to_sequences(self.x_test[tc]); gc.collect()
                
                    #self.x_train[tc] = pad_sequences(self.x_train[tc], maxlen=maxlens[tc]); gc.collect()
                    #self.x_test[tc] = pad_sequences(self.x_test[tc], maxlen=maxlens[tc]); gc.collect()
                    self.x_train[tc] = self.pad_sequences(self.x_train[tc], maxlen=maxlens[tc]); gc.collect()
                    self.x_test[tc] = self.pad_sequences(self.x_test[tc], maxlen=maxlens[tc]); gc.collect()
                    
                else:
                    self.tokenizer[tc].fit_on_texts(list(self.x_test[tc]))       
                    self.x_test[tc] = self.tokenizer[tc].texts_to_sequences(self.x_test[tc]); gc.collect()
                    self.x_test[tc] = self.pad_sequences(self.x_test[tc], maxlen=maxlens[tc]); gc.collect()
                
                print(tc, '# of word index = ', len(self.tokenizer[tc].word_index))
            
    def prepare_embeddings(self):
        with timer('Prepare embeddings'):
            if SUBMIT_MODE and os.path.isfile(SUBMIT_MODE_RNN_EMBED_CACHE_PATH):
                self.embedding_matrix = pd.read_pickle(SUBMIT_MODE_RNN_EMBED_CACHE_PATH)
                del self.tokenizer; gc.collect()
                return
            
            if not SUBMIT_MODE and os.path.isfile(self.config['embed_path_tmp']):
                self.embedding_matrix = pd.read_pickle(self.config['embed_path_tmp'])
                del self.tokenizer; gc.collect()
                return
            
            self.embedding_matrix = {tc: None for tc in text_cols}
            word_index_list = [self.tokenizer[tc].word_index for tc in text_cols]
            
            for p in self.config['embed_paths']:
                
                with timer('Building Embedding from {}'.format(p)):
                
                    if '.bin' in p:
                        embedding_matrix_list = build_matrix_from_bin(word_index_list, p)
                    else:
                        embedding_matrix_list = build_matrix(word_index_list, p)
                    
                    
                    for tc, embedding_matrix in zip(text_cols, embedding_matrix_list):
                        if self.embedding_matrix[tc] is None:
                            self.embedding_matrix[tc] = embedding_matrix.copy()
                        else:
                            self.embedding_matrix[tc] = np.concatenate([self.embedding_matrix[tc], embedding_matrix.copy()], axis=1)
                    
                    del embedding_matrix_list; gc.collect()
            
            if SUBMIT_MODE:
                pd.to_pickle(self.embedding_matrix, SUBMIT_MODE_RNN_EMBED_CACHE_PATH)
                
            del self.tokenizer, word_index_list; gc.collect()
            
    def preprocess(self): # for both training and inference mode
        self.select_data_and_clean()
        self.tokenize_and_pad_seq()
        self.prepare_embeddings()
    
    def _train(self, tr_ind, val_ind, fold=0, bag_id=0):
        
        oof_ckpt_preds = []
        weights = []
        
        model = RNN([self.embedding_matrix[tc] for tc in text_cols], #embedding matrixs from [title,question,anaswer]
                    dropout = self.config['dropout'],
                    dropout_title = self.config['dropout_title'],
                    hidden_size = self.config['hidden_size'],
                    hidden_size_title = self.config['hidden_size_title'],
                    max_lens = maxlens,
                    embed_size = self.embedding_matrix[text_cols[0]].shape[1],
                    meta_size = self.meta_train.shape[1]).cuda()
        model_to_save = model.get_saving_model()
        
        train_loader = DataLoader(TextDataset(self.x_train['question_title'], 
                                              self.x_train['question_body'], 
                                              self.x_train['answer'], 
                                              self.meta_train, 
                                              tr_ind, self.target),
                                  batch_size=self.config['batch_size'], shuffle=True, 
                                  pin_memory=True)
        valid_loader = DataLoader(TextDataset(self.x_train['question_title'], 
                                              self.x_train['question_body'], 
                                              self.x_train['answer'], 
                                              self.meta_train,
                                              val_ind, self.target),
                                  batch_size=self.config['pred_batch_size'], shuffle=False, 
                                  pin_memory=True)
        
        optimizer = torch.optim.Adam(model.parameters(), self.config['max_lr'])
        #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.config['epochs']-2, 
                                        gamma=self.config['min_lr']/self.config['max_lr'])
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
        
        best_score = 0
        patience = 10
        
        for epoch in range(self.config['epochs']):
            torch.manual_seed(self.config['seed']+epoch)
            start_time = time.time()
            avg_loss = 0.
    
            model_path = self.config['model_path_format'].format(bag_id, fold, epoch)
            
            if DEV_MODE or not os.path.isfile(model_path): # not trained before
                # training
                for param in model_to_save.parameters():
                    param.requires_grad=True
                model=model.train()
                optimizer.zero_grad()
                
                '''
                steps = len(train_loader)
                #steps_up = int(steps*self.config['warmup_ratio'])
                #steps_down = steps-steps_up
                scheduler = CyclicLR(optimizer, 
                                     base_lr=self.config['min_lr'],
                                     max_lr=self.config['max_lr'],
                                     step_size=steps,
                                     mode='triangular2')
                '''
                for title, question, answer, meta, y_batch in tqdm(train_loader, disable=True):
                    title = title.long().cuda()
                    question = question.long().cuda()
                    answer = answer.long().cuda()
                    #meta = meta.cuda()
                    y_batch = y_batch.cuda()
                    y_pred = model(title, question, answer)
        
                    loss = loss_fn(y_pred.double(), y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    optimizer.zero_grad()
    
                    avg_loss += loss.item() / len(train_loader)
                    #scheduler.step()
            else:
                model_to_save.load_state_dict(torch.load(model_path))
                
            # evaluation
            for param in model_to_save.parameters():
                param.requires_grad=False
            model=model.eval()
            
            avg_val_loss = 0.
            preds = []
            original = []
            for i, (title, question, answer, meta, y_batch) in enumerate(valid_loader):
                title = title.long().cuda()
                question = question.long().cuda()
                answer = answer.long().cuda()
                #meta = meta.cuda()
                y_batch = y_batch.cuda()
                y_pred = model(title, question, answer).detach()
    
                avg_val_loss += loss_fn(y_pred.double(), y_batch).item() / len(valid_loader)
                preds.append(torch.sigmoid(y_pred).cpu().numpy())
                original.append(y_batch.cpu().numpy())
            
            preds = np.concatenate(preds)
            original = np.concatenate(original)
            
            score = calc_spearmanr_metric(preds, original)
            
            oof_ckpt_preds.append(preds)
            weights.append(2.**epoch)
            
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t spearman={:.4f} \t time={:.2f}s'.format(
                epoch + 1, self.config['epochs'], avg_loss, avg_val_loss, score, elapsed_time))
    
            #scheduler.step(avg_val_loss)
            scheduler.step()
    
            valid_score = score
            if valid_score > best_score:
                best_score = valid_score
                p = 0
    
            # check if validation loss didn't improve
            if valid_score <= best_score:
                p += 1
                print(f'{p} epochs of non improving score')
                #if p > patience:
                #    print('Stopping training')
                #    break
            
            if DEV_MODE or not os.path.isfile(model_path): # not saved before
                torch.save(model_to_save.state_dict(), model_path)
        
        oof_pred = np.average(oof_ckpt_preds, weights=weights, axis=0)
        
        del train_loader, valid_loader, optimizer, scheduler, loss_fn
        del model, model_to_save; gc.collect(); torch.cuda.empty_cache(); 

        return oof_pred
    
    def train_model(self):
        def bce(y_true, y_pred, eps=1e-15):
            y_pred = np.clip(y_pred, eps, 1-eps)
            return np.mean(-(y_true*np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
        
        if not SUBMIT_MODE:
            scores = []
            fold_num = self.config['fold_num']
            splits = list(KFold(n_splits=self.config['fold_num'], random_state=self.config['seed'], shuffle=True) \
                          .split(self.target, self.target))
            
            oof_preds = np.zeros(self.target.shape)
            for bag in range(self.config['bag_size']):
                for fold in np.arange(fold_num):
                    tr_ind, val_ind = splits[fold]
                    oof_pred = self._train(tr_ind, val_ind, fold=fold, bag_id=bag)
                    oof_preds[val_ind] += oof_pred/self.config['bag_size']
                    score = calc_spearmanr_metric(oof_pred, self.target[val_ind])
                    scores.append(score)
                    print('Bag {} Fold {} score: {:.5f}'.format(bag, fold, score))
            
            overall_score = bce(self.target, oof_preds)
            overall_metric = calc_spearmanr_metric(oof_preds, self.target)
            print('overall bce = {:.5f} spearmanr = {:.5f}'.format(overall_score, overall_metric))
            print('score details:', scores)
            
            return oof_preds
        else:
            print('SUBMIT MODE = TRUE WILL NOT TRAIN MODEL')

    def inference_model(self):        
        model = RNN([self.embedding_matrix[tc] for tc in text_cols], #embedding matrixs from [title,question,anaswer]
                    dropout = self.config['dropout'],
                    dropout_title = self.config['dropout_title'],
                    hidden_size = self.config['hidden_size'],
                    hidden_size_title = self.config['hidden_size_title'],
                    max_lens = maxlens,
                    embed_size = self.embedding_matrix[text_cols[0]].shape[1],
                    meta_size = self.meta_test.shape[1]).cuda()
        model_to_save = model.get_saving_model()
        for param in model_to_save.parameters():
            param.requires_grad=False
        model=model.eval()
        
        test_loader = DataLoader(TextDataset(self.x_test['question_title'], 
                                             self.x_test['question_body'], 
                                             self.x_test['answer'], 
                                             self.meta_test, 
                                             np.arange(len(self.x_test['answer'])), None),
                                batch_size=self.config['pred_batch_size'], shuffle=False, 
                                pin_memory=True)
        
        fold_num = self.config['fold_num']
        
        oof_preds = np.zeros((len(self.x_test['answer']), len(target_cols)))
        
        for bag in range(self.config['bag_size']):
            for fold in np.arange(fold_num):
                weights = []
                oof_ckpt_preds = []
                
                for epoch in range(self.config['epochs']):
                    torch.manual_seed(self.config['seed']+epoch)
                    
                    model_path = self.config['model_path_format'].format(bag, fold, epoch)
                    assert os.path.isfile(model_path)
                
                    print('Loading model:', model_path)
                    model_to_save.load_state_dict(torch.load(model_path))
                    
                    preds = []
                    for i, (title, question, answer, meta, _) in enumerate(test_loader):
                        title = title.long().cuda()
                        question = question.long().cuda()
                        answer = answer.long().cuda()
                        y_pred = model(title, question, answer).detach()
                        preds.append(torch.sigmoid(y_pred).cpu().numpy())
                    preds = np.concatenate(preds)
                    
                    oof_ckpt_preds.append(preds)
                    weights.append(2.**epoch)
            
                oof_pred = np.average(oof_ckpt_preds, weights=weights, axis=0)
                oof_preds += oof_pred/(self.config['bag_size']*fold_num)
                
        del test_loader, model, model_to_save; gc.collect(); torch.cuda.empty_cache()
        return oof_preds
        
    def close(self):
        if not SUBMIT_MODE:
            del self.x_train
        del self.x_test, self.embedding_matrix; gc.collect()
        
        torch.cuda.empty_cache() 

class RNN_Pretrain_Features_Pipeline(RNN_Pipeline):
    def __init__(self, *args):
        super(RNN_Pretrain_Features_Pipeline, self).__init__(*args)
    
    def prepare_embeddings(self):
        with timer('Prepare embeddings'):
            if SUBMIT_MODE and os.path.isfile(SUBMIT_MODE_RNN_EMBED_CACHE_PATH):
                self.embedding_matrix = pd.read_pickle(SUBMIT_MODE_RNN_EMBED_CACHE_PATH)
                del self.tokenizer; gc.collect()
                return
            
            if not SUBMIT_MODE and os.path.isfile(self.config['embed_path_tmp']):
                self.embedding_matrix = pd.read_pickle(self.config['embed_path_tmp'])
                del self.tokenizer; gc.collect()
                return
            
            self.embedding_matrix = {tc: None for tc in text_cols}
            word_index_list = [self.tokenizer[tc].word_index for tc in text_cols]
            
            for p in self.config['embed_paths']:
                
                with timer('Building Embedding from {}'.format(p)):
                
                    if '.bin' in p:
                        embedding_matrix_list = build_matrix_from_bin(word_index_list, p)
                    else:
                        embedding_matrix_list = build_matrix(word_index_list, p)
                    
                    
                    for tc, embedding_matrix in zip(text_cols, embedding_matrix_list):
                        if self.embedding_matrix[tc] is None:
                            self.embedding_matrix[tc] = embedding_matrix.copy()
                        else:
                            self.embedding_matrix[tc] = np.concatenate([self.embedding_matrix[tc], embedding_matrix.copy()], axis=1)
                    
                    del embedding_matrix_list; gc.collect()
            
            if SUBMIT_MODE:
                pd.to_pickle(self.embedding_matrix, SUBMIT_MODE_RNN_EMBED_CACHE_PATH)
                
            del self.tokenizer, word_index_list; gc.collect()
    
    def do_fe(self, x, meta):
        embeddings = np.concatenate([
                            #x['dst_bert_title'],
                            x['use_large_title'],
                            #x['dst_bert_question'],
                            x['use_large_question'],
                            #x['dst_bert_answer'],
                            x['use_large_answer'],
                            #x['use_qa_title'],
                            #x['use_qa_question'],
                            #x['use_qa_answer'],
                        ], axis=1)
    
        return embeddings
    
    def fetch_pretrain_dstbert_features(self, string_list, batch_size=64, is_question=True):
        with timer('Extracting distilled bert features'):
            def chunks(l, n):
                """Yield successive n-sized chunks from l."""
                for i in range(0, len(l), n):
                    yield l[i:i + n]
                    
            # inspired by https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
            DEVICE = torch.device("cuda")
            tokenizer = transformers.DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")
            model = transformers.DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")
            model.to(DEVICE)
        
            fin_features = []
            for data in chunks(string_list, batch_size):
                tokenized = []
                for x in data:
                    x = " ".join(x.strip().split()[:300])
                    tok = tokenizer.encode(x, add_special_tokens=True)
                    tokenized.append(tok[:512])
        
                max_len = 512
                padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
                attention_mask = np.where(padded != 0, 1, 0)
                input_ids = torch.tensor(padded).to(DEVICE)
                attention_mask = torch.tensor(attention_mask).to(DEVICE)
        
                with torch.no_grad():
                    last_hidden_states = model(input_ids, attention_mask=attention_mask)
        
                features = last_hidden_states[0][:, 0, :].detach().cpu().numpy()
                fin_features.append(features)
        
            fin_features_ = np.vstack(fin_features) 
            del model, tokenizer, fin_features, tok, attention_mask, input_ids
            del features, last_hidden_states, tokenized, padded; 
            gc.collect(); torch.cuda.empty_cache(); 
            os.system('nvidia-smi >> log.txt')
            return fin_features_

    def fetch_use_large_features(self, string_list, batch_size = 4, is_question=True):
        with timer('Extracting USE-large features'):
            import tensorflow_hub as hub
            import tensorflow as tf
            
            module_url = "../input/universalsentenceencoderlarge4/"
            embed = hub.load(module_url)
            
            texts = string_list.str.replace('?', '.').str.replace('!', '.').tolist()
            
            curr_emb = []
            ind = 0
            while ind*batch_size < len(texts):
                curr_emb.append(embed(texts[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
                ind += 1
            
            res = np.vstack(curr_emb)
            del curr_emb, embed, texts; gc.collect(); K.clear_session();
            os.system('nvidia-smi >> log.txt')
            return res
    
    def fetch_use_features(self, string_list, batch_size = 4, is_question=True):
        with timer('Extracting USE features'):
            import tensorflow_hub as hub
            import tensorflow as tf
            
            module_url = "../input/universal-sentence-encoder/"
            embed = hub.load(module_url)
            
            texts = string_list.str.replace('?', '.').str.replace('!', '.').tolist()
            
            curr_emb = []
            ind = 0
            while ind*batch_size < len(texts):
                curr_emb.append(embed(texts[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
                ind += 1
            
            res = np.vstack(curr_emb)
            del curr_emb, embed, texts; gc.collect(); K.clear_session();
            os.system('nvidia-smi >> log.txt')
            return res
    
    def fetch_use_qa_features(self, string_list, batch_size = 4, is_question=True):
        with timer('Extracting USE-qa features'):
            import tensorflow_hub as hub
            import tensorflow as tf
            
            module_url = "../input/universal-sentence-encoder-qa/"
            embed = hub.load(module_url)
            embed.init_op()
            
            texts = string_list.tolist()

            curr_emb = []
            ind = 0
            if is_question:
                while ind*batch_size < len(texts):
                    #curr_emb.append(embed(texts[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
                    tmp = texts[ind*batch_size: (ind + 1)*batch_size]
                    curr_emb.append(embed.signatures['question_encoder'](tf.constant(tmp))["outputs"].numpy())
                    ind += 1
            else:
                while ind*batch_size < len(texts):
                    #curr_emb.append(embed(texts[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
                    tmp = texts[ind*batch_size: (ind + 1)*batch_size]
                    curr_emb.append(embed.signatures['response_encoder'](input=tf.constant(tmp), context=tf.constant([""]*len(tmp)))["outputs"].numpy())
                    ind += 1
            
            res = np.vstack(curr_emb)
            del curr_emb, embed, texts, tmp; gc.collect(); K.clear_session();
            os.system('nvidia-smi >> log.txt')
            return res
    
    def prepare_use_features(self):
        # load pretrained embedding features
        if not SUBMIT_MODE:
            self.x_pre_train = {} # dict of pretrained extracted embeddings
        self.x_pre_test = {}

        feat_sources = ['dst_bert', 'use_large', 'use_qa']
        postfixs = ['title', 'question', 'answer']
        
        if not SUBMIT_MODE:
            pkl_paths = [
                        '../input/dist_bert_features_fast_loading.pkl',
                        '../input/use_large_features_fast_loading.pkl',
                        #'../input/use_features_fast_loading.pkl',
                        '../input/use_qa_features_fast_loading.pkl',
                    ]
            
            for fs, pkl_path in zip(feat_sources, pkl_paths):
                assert os.path.isfile(pkl_path)
            
                embed_dict = pd.read_pickle(pkl_path)
            
                for prefix, dct in zip(['train', 'test'], [self.x_pre_train, self.x_pre_test]):
                    dct.update({fs+'_'+pf: embed_dict[prefix+'_'+pf].copy() for pf in postfixs})
                    
                del embed_dict; gc.collect()
                
            self.x_pre_train = self.do_fe(self.x_pre_train, self.meta_train)
            self.x_pre_test = self.do_fe(self.x_pre_test, self.meta_test)
        else:
            if os.path.isfile(SUBMIT_MODE_USE_FEATS_CACHE_PATH):
                self.x_pre_test = pd.read_pickle(SUBMIT_MODE_USE_FEATS_CACHE_PATH)
            else:
                test_df = pd.read_csv(self.test_path)
            
                extract_funcs = [
                            self.fetch_pretrain_dstbert_features,
                            self.fetch_use_large_features,
                            self.fetch_use_qa_features,
                        ]
                
                for fs, extract_func in zip(feat_sources, extract_funcs): 
                    self.x_pre_test.update({
                                fs+'_title': extract_func(test_df.question_title, is_question=True),
                                fs+'_question': extract_func(test_df.question_body, is_question=True),
                                fs+'_answer': extract_func(test_df.answer, is_question=False),
                            })
                pd.to_pickle(self.x_pre_test, SUBMIT_MODE_USE_FEATS_CACHE_PATH)
                del test_df; gc.collect()
                
            test_feats = self.do_fe(self.x_pre_test, self.meta_test)
            del self.x_pre_test; gc.collect()
            self.x_pre_test = test_feats
                
            
    def preprocess(self):
        super(RNN_Pretrain_Features_Pipeline, self).select_data_and_clean()
        super(RNN_Pretrain_Features_Pipeline, self).tokenize_and_pad_seq()
        self.prepare_embeddings()
        self.prepare_use_features()
        
    def _train(self, tr_ind, val_ind, fold=0, bag_id=0):
        
        oof_ckpt_preds = []
        weights = []
        
        model = RNN_Pretrain_Features([self.embedding_matrix[tc] for tc in text_cols], #embedding matrixs from [title,question,anaswer]
                    dropout = self.config['dropout'],
                    dropout_title = self.config['dropout_title'],
                    hidden_size = self.config['hidden_size'],
                    hidden_size_title = self.config['hidden_size_title'],
                    max_lens = maxlens,
                    embed_size = self.embedding_matrix[text_cols[0]].shape[1],
                    pretrain_feature_size = self.x_pre_train.shape[1]).cuda()
        model_to_save = model.get_saving_model()
        
        train_loader = DataLoader(TextDataset(self.x_train['question_title'], 
                                              self.x_train['question_body'], 
                                              self.x_train['answer'], 
                                              self.x_pre_train, 
                                              tr_ind, self.target),
                                  batch_size=self.config['batch_size'], shuffle=True, 
                                  pin_memory=True)
        valid_loader = DataLoader(TextDataset(self.x_train['question_title'], 
                                              self.x_train['question_body'], 
                                              self.x_train['answer'], 
                                              self.x_pre_train,
                                              val_ind, self.target),
                                  batch_size=self.config['pred_batch_size'], shuffle=False, 
                                  pin_memory=True)
        
        optimizer = torch.optim.Adam(model.parameters(), self.config['max_lr'])
        #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.config['epochs']-2, gamma=0.1)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
        
        best_score = 0
        patience = 10
        epoch_scores_list = []
        
        for epoch in range(self.config['epochs']):
            torch.manual_seed(self.config['seed']+epoch)
            start_time = time.time()
            avg_loss = 0.
    
            model_path = self.config['model_path_format'].format(bag_id, fold, epoch)
            
            if DEV_MODE or not os.path.isfile(model_path): # not trained before
                # training
                for param in model_to_save.parameters():
                    param.requires_grad=True
                model=model.train()
                optimizer.zero_grad()
                
                for title, question, answer, pretrain_feats, y_batch in tqdm(train_loader, disable=True):
                    title = title.long().cuda()
                    question = question.long().cuda()
                    answer = answer.long().cuda()
                    pretrain_feats = pretrain_feats.cuda()
                    y_batch = y_batch.cuda()
                    y_pred = model(title, question, answer, pretrain_feats)
        
                    loss = loss_fn(y_pred.double(), y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    optimizer.zero_grad()
    
                    avg_loss += loss.item() / len(train_loader)
                    #scheduler.step()
            else:
                model_to_save.load_state_dict(torch.load(model_path))
                
            # evaluation
            for param in model_to_save.parameters():
                param.requires_grad=False
            model=model.eval()
            
            avg_val_loss = 0.
            preds = []
            original = []
            for i, (title, question, answer, pretrain_feats, y_batch) in enumerate(valid_loader):
                title = title.long().cuda()
                question = question.long().cuda()
                answer = answer.long().cuda()
                pretrain_feats = pretrain_feats.cuda()
                y_batch = y_batch.cuda()
                y_pred = model(title, question, answer, pretrain_feats).detach()
    
                avg_val_loss += loss_fn(y_pred.double(), y_batch).item() / len(valid_loader)
                preds.append(torch.sigmoid(y_pred).cpu().numpy())
                original.append(y_batch.cpu().numpy())
            
            preds = np.concatenate(preds)
            original = np.concatenate(original)
            
            score = calc_spearmanr_metric(preds, original)
            scores = calc_spearmanr_metric_list(preds, original)
            epoch_scores_list+= [np.array(scores)]
            
            oof_ckpt_preds.append(preds)
            weights.append(2.**epoch)
            
            elapsed_time = time.time() - start_time
            print('Epoch {}/{}  loss={:.4f}  val_loss={:.4f}  spearman={:.4f}  time={:.2f}s'.format(
                epoch + 1, self.config['epochs'], avg_loss, avg_val_loss, score, elapsed_time))
    
            #for it, (t, s) in enumerate(zip(target_cols, scores)):
            #    print('{:40}: {:4f}  '.format(t, s), end='\n')
                
            #scheduler.step(avg_val_loss)
            scheduler.step()
    
            valid_score = score
            if valid_score > best_score:
                best_score = valid_score
                p = 0
    
            # check if validation loss didn't improve
            if valid_score <= best_score:
                p += 1
                print(f'{p} epochs of non improving score')
                #if p > patience:
                #    print('Stopping training')
                #    break
            
            if DEV_MODE or not os.path.isfile(model_path): # not saved before
                torch.save(model_to_save.state_dict(), model_path)
        
        oof_pred = np.average(oof_ckpt_preds, weights=weights, axis=0)
        
        del train_loader, valid_loader, optimizer, scheduler, loss_fn
        del model, model_to_save; gc.collect(); torch.cuda.empty_cache(); 

        #pd.to_pickle(np.vstack(epoch_scores_list), 'epoch_scores_list.pkl')
        #assert False
        
        return oof_pred
    
    def inference_model(self):        
        model = RNN_Pretrain_Features([self.embedding_matrix[tc] for tc in text_cols], #embedding matrixs from [title,question,anaswer]
                    dropout = self.config['dropout'],
                    dropout_title = self.config['dropout_title'],
                    hidden_size = self.config['hidden_size'],
                    hidden_size_title = self.config['hidden_size_title'],
                    max_lens = maxlens,
                    embed_size = self.embedding_matrix[text_cols[0]].shape[1],
                    pretrain_feature_size = self.x_pre_test.shape[1]).cuda()
        model_to_save = model.get_saving_model()
        for param in model_to_save.parameters():
            param.requires_grad=False
        model=model.eval()
        
        test_loader = DataLoader(TextDataset(self.x_test['question_title'], 
                                             self.x_test['question_body'], 
                                             self.x_test['answer'], 
                                             self.x_pre_test, 
                                             np.arange(len(self.x_test['answer'])), None),
                                batch_size=self.config['pred_batch_size'], shuffle=False, 
                                pin_memory=True)
        
        fold_num = self.config['fold_num']
        
        oof_preds = np.zeros((len(self.x_test['answer']), len(target_cols)))
        
        for bag in range(self.config['bag_size']):
            for fold in np.arange(fold_num):
                weights = []
                oof_ckpt_preds = []
                
                for epoch in range(self.config['epochs']):
                    torch.manual_seed(self.config['seed']+epoch)
                    
                    model_path = self.config['model_path_format'].format(bag, fold, epoch)
                    assert os.path.isfile(model_path)
                
                    print('Loading model:', model_path)
                    model_to_save.load_state_dict(torch.load(model_path))
                    
                    preds = []
                    for i, (title, question, answer, pretrain_feats, _) in enumerate(test_loader):
                        title = title.long().cuda()
                        question = question.long().cuda()
                        answer = answer.long().cuda()
                        pretrain_feats = pretrain_feats.cuda()
                        y_pred = model(title, question, answer, pretrain_feats).detach()
                        preds.append(torch.sigmoid(y_pred).cpu().numpy())
                    preds = np.concatenate(preds)
                    
                    oof_ckpt_preds.append(preds)
                    weights.append(2.**epoch)
            
                oof_pred = np.average(oof_ckpt_preds, weights=weights, axis=0)
                oof_preds += oof_pred/(self.config['bag_size']*fold_num)
                
        del test_loader, model, model_to_save; gc.collect(); torch.cuda.empty_cache()
        return oof_preds
    
    def close(self):
        if not SUBMIT_MODE:
            del self.x_pre_train
        del self.x_pre_test, self.embedding_matrix; gc.collect()
        
        torch.cuda.empty_cache() 
        
class Slanted_LR_Scheduler():
    def __init__(self, start_lr, max_lr, min_lr, start_ratio, tq_bar=None):
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.start_ratio = start_ratio
        self.tq_bar = tq_bar
    
    def update_tq_bar(self, tq_bar):
        self.tq_bar = tq_bar
        
    def __call__(self, current_ratio, warmup_ratio):
        current_ratio += self.start_ratio
        if current_ratio <= warmup_ratio:
            lr_diff = self.max_lr-self.start_lr
            lr = self.start_lr+current_ratio/warmup_ratio*lr_diff
        else:
            lr_diff = self.max_lr-self.min_lr
            lr = self.max_lr-(current_ratio-warmup_ratio)/(1.-warmup_ratio)*lr_diff    

        if self.tq_bar is not None:
            self.tq_bar.set_description("current_ratio={:.6f}, warmup_ratio={:.6f}, lr={:.8f}".format(
                    current_ratio, warmup_ratio, lr))
        
        #print('Updated', current_ratio, warmup_ratio, lr)
        return lr/self.max_lr    

class BertEmbedForGQA(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super(BertEmbedForGQA, self).__init__(config)
        config.output_hidden_states = True
        self.bert = transformers.BertModel(config)
        #self.dropout = nn.Dropout(0.1)#config.hidden_dropout_prob)
        #self.classifier = nn.Linear(config.hidden_size*3, len(target_cols))
        
        self.linear1_q = nn.Linear(config.hidden_size * 4, 512)
        self.droupout_q = nn.Dropout(0.2)
        self.linear_out_q = nn.Linear(512, len(target_cols)-9)
        
        self.linear1_a = nn.Linear(config.hidden_size * 4 * 2, 512)
        self.droupout_a = nn.Dropout(0.2)
        self.linear_out_a = nn.Linear(512, 9)
        
        self.apply(self._init_weights)

    def summarize_hidden_states(self, token_embeddings, attention_mask):
        
        #method == 'cls':
        return token_embeddings[:, 0, :]
        '''
        
        #method == 'mean pooling':
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, 1)
        sum_mask = attention_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return (sum_embeddings / sum_mask)
        '''
    def forward(self, title_ids, question_ids, answer_ids):
        qt_ids = torch.cat((title_ids, question_ids), 1)
        qt_token_ids = None #torch.zeros(qt_ids.size(), dtype=torch.long, device=qt_ids.device)
        qt_outputs = self.bert(qt_ids, token_type_ids=qt_token_ids, attention_mask=qt_ids>0)
        
        answer_token_ids = None #torch.ones(answer_ids.size(), dtype=torch.long, device=answer_ids.device)
        answer_outputs = self.bert(answer_ids, token_type_ids=answer_token_ids, attention_mask=answer_ids>0)
        
        qt_conc = torch.cat([self.summarize_hidden_states(qt_outputs[0], qt_ids>0), 
                             self.summarize_hidden_states(qt_outputs[2][-1], qt_ids>0), 
                             self.summarize_hidden_states(qt_outputs[2][-2], qt_ids>0), 
                             self.summarize_hidden_states(qt_outputs[2][-3], qt_ids>0)], 1) 
        
        a_conc = torch.cat([qt_conc,
                            self.summarize_hidden_states(answer_outputs[0], answer_ids>0),
                            self.summarize_hidden_states(answer_outputs[2][-1], answer_ids>0),
                            self.summarize_hidden_states(answer_outputs[2][-2], answer_ids>0),
                            self.summarize_hidden_states(answer_outputs[2][-3], answer_ids>0)], 1)

        x = self.droupout_q(nn.ELU()(self.linear1_q(qt_conc)))
        out_q = self.linear_out_q(x)
        
        x = self.droupout_a(nn.ELU()(self.linear1_a(a_conc)))
        out_a = self.linear_out_a(x)
        
        return torch.cat([out_q, out_a], 1)

    
class BERT_Pipeline_Pytorch(Modeling_Pipeline):
        
    def __init__(self, *args):
        super(BERT_Pipeline_Pytorch, self).__init__(*args)
        
    def get_tokens(self, texts, max_len, pre_prop=.3, add_cls=False, add_sep=False):
        tokenizer = transformers.BertTokenizer.from_pretrained(self.config['bert_pretrained_dir'], 
                                                  cache_dir=None, do_lower_case=self.config['do_lower'])
        
        res = []
        max_seq_length = max_len-int(add_cls)-int(add_sep)
        pre_len = int(max_seq_length*pre_prop)
        post_len = max_seq_length-pre_len
        for i in tqdm(range(len(texts))):
            x = " ".join(texts[i].strip().split())
            tokens_a = tokenizer.tokenize(x)
            if len(tokens_a)>max_seq_length:
                tokens_a = tokens_a[:pre_len]+tokens_a[-post_len:]
            
            if add_cls:
                tokens_a = ["[CLS]"] + tokens_a
            if add_sep:
                tokens_a = tokens_a + ["[SEP]"]
            tokens_a = tokenizer.convert_tokens_to_ids(tokens_a) + [0] * (max_len - len(tokens_a))
            res.append(np.array(tokens_a))
            
        return np.array(res)
        
    def preprocess(self):
        
        with timer('Generate Tokens'):
            from sklearn.preprocessing import OneHotEncoder
            
            self.x_train = {tc: None for tc in text_cols}
            self.x_test = {tc: None for tc in text_cols}
            self.target = None
            
            text_cls_sep = {
                'question_title': {'add_cls': True, 'add_sep': True},
                'question_body': {'add_cls': False, 'add_sep': True},
                'answer': {'add_cls': True, 'add_sep': True}
            }
            meta_features = ['netloc', 'category']
            
            if not SUBMIT_MODE:
                train_df = pd.read_csv(self.train_path)
                self.target = train_df[target_cols].values
                
                for tc in text_cols:
                    train_texts = train_df[tc].fillna("DUMMY_VALUE").values.tolist()
                    self.x_train[tc] = np.array(self.get_tokens(train_texts, maxlens_bert[tc], pre_prop=.3, 
                                                    add_cls=text_cls_sep[tc]['add_cls'],
                                                    add_sep=text_cls_sep[tc]['add_sep']))
                    del train_texts; gc.collect()
                    
                ohe = OneHotEncoder(dtype=np.float32, handle_unknown='ignore')
                train_df['netloc'] = self.get_netloc(train_df)
                self.meta_train = ohe.fit_transform(train_df[meta_features]).toarray()
                pd.to_pickle(ohe, self.get_pickle_ohe_path())
                
                del train_df; gc.collect()
            
            test_df = pd.read_csv(self.test_path)
            for tc in text_cols:
                test_texts = test_df[tc].fillna("DUMMY_VALUE").values.tolist()
                self.x_test[tc] = np.array(self.get_tokens(test_texts, maxlens_bert[tc], pre_prop=.3, 
                                               add_cls=text_cls_sep[tc]['add_cls'],
                                               add_sep=text_cls_sep[tc]['add_sep']))
                del test_texts; gc.collect()
                
            ohe = pd.read_pickle(self.get_pickle_ohe_path())
            test_df['netloc'] = self.get_netloc(test_df)
            self.meta_test = ohe.transform(test_df[meta_features]).toarray()
            
            del test_df; gc.collect()
            
    def prepare_model(self):
        if DEV_MODE:
            from transformers import convert_bert_original_tf_checkpoint_to_pytorch
            print(os.listdir(self.config['transformed_bert_dir']))
            if not os.path.isfile(self.config['transformed_bert_model']):
                convert_bert_original_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
                        self.config['bert_ckpt_file_path'],
                        self.config['bert_config_file_path'],
                        self.config['transformed_bert_model'])
                shutil.copyfile(self.config['bert_config_file_path'], self.config['transformed_bert_config'])
        
        device = torch.device('cuda')
        
        if self.config['feature'] == 'embed':
            model = BertEmbedForGQA.from_pretrained(self.config['transformed_bert_dir'])
        else:
            assert False
            
        model.zero_grad()
        model.to(device)
        return model, device
    
    def custom_loss(self, pred, targets, device):
        import torch
        ''' Define custom loss function for weighted BCE on 'target' column '''
        loss1 = torch.nn.functional.binary_cross_entropy_with_logits(pred, targets, reduction='mean')
        
        return loss1

    def _train(self, tr_ind, val_ind, fold=0, bag_id=0):
        oof_ckpt_preds = []
        weights = []
        
        train_loader = DataLoader(TextDataset(self.x_train['question_title'], 
                                              self.x_train['question_body'], 
                                              self.x_train['answer'], 
                                              self.meta_train, 
                                              tr_ind, self.target),
                                  batch_size=self.config['batch_size'], shuffle=True, 
                                  pin_memory=True)
        valid_loader = DataLoader(TextDataset(self.x_train['question_title'], 
                                              self.x_train['question_body'], 
                                              self.x_train['answer'], 
                                              self.meta_train,
                                              val_ind, self.target),
                                  batch_size=self.config['pred_batch_size'], shuffle=False, 
                                  pin_memory=True)
        
        #########
        # prepare model, optimizers, lr schedulers
        #########
        model, device = self.prepare_model()
        
        steps_per_epoch = int(np.ceil(len(train_loader)/self.config['accum_iters']))
        num_train_optimization_steps = self.config['epochs']*steps_per_epoch
        
        optimizer = transformers.optimization.AdamW(model.parameters(),
                                                    lr=self.config['max_lr'],
                                                    weight_decay=0.0,
                                                    correct_bias=False)
        '''
        scheduler = transformers.optimization.get_cosine_with_hard_restarts_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=int(steps_per_epoch*self.config['warmup_ratio']),
                    num_training_steps=num_train_optimization_steps,
                    num_cycles=self.config['epochs']
                )
        '''
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(
                        optimizer=optimizer,
                        num_warmup_steps=int(steps_per_epoch*self.config['warmup_ratio']),
                        num_training_steps=num_train_optimization_steps,
                    )
        
        #model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
        training_started = False
        
        for epoch in range(0, self.config['epochs']):
            torch.manual_seed(self.config['seed']+epoch)
            start_time = time.time()
            
            model_path = self.config['model_path_format'].format(self.config['version'], bag_id, fold, epoch)
            avg_loss = 0.
            
            if DEV_MODE or not os.path.isfile(model_path): # not trained before
                
                print('training_started:', training_started)
                
                # amp initialize after start training
                if not training_started:
                    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
                    training_started = True
                    
                #########
                # Training
                #########
                for param in model.parameters():
                    param.requires_grad=True
                model=model.train()
                tk_trn = tqdm(enumerate(train_loader),total=len(train_loader))
                lossf=None
                optimizer.zero_grad() 
                
                for i, (title, question, answer, meta, y_batch) in tk_trn:
                    title = title.long().cuda()
                    question = question.long().cuda()
                    answer = answer.long().cuda()
                    #meta = meta.cuda()
                    y_batch = y_batch.cuda()
                    
                    y_pred = model(title, question, answer)
                   
                    loss = self.custom_loss(y_pred.double(), y_batch, device)
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        
                    if (i+1) % self.config['accum_iters'] == 0:             # Wait for several backward steps
                        optimizer.step()                            # Now we can do an optimizer step
                        optimizer.zero_grad()
                        
                    if lossf:
                        lossf = 0.98*lossf+0.02*loss.item()
                    else:
                        lossf = loss.item()
                    
                    tk_trn.set_postfix(loss = lossf)
                    avg_loss += loss.item() / len(train_loader)
                    scheduler.step()
                    
                print('Loss: {:.6f}'.format(avg_loss))
            else:
                model.load_state_dict(torch.load(model_path))
                
                # skip through these steps
                for i in range(len(train_loader)):
                    scheduler.step()
                
            #########
            # Valdiation
            #########
            for param in model.parameters():
                param.requires_grad=False
            model.eval()
            avg_val_loss = 0.
            preds = []
            original = []
            tk_val = tqdm(enumerate(valid_loader),total=len(valid_loader))
            for i, (title, question, answer, meta, y_batch) in tk_val:
                title = title.long().cuda()
                question = question.long().cuda()
                answer = answer.long().cuda()
                #meta = meta.cuda()
                y_batch = y_batch.cuda()
                
                y_pred = model(title, question, answer)
                y_pred = y_pred.detach()
                
                loss = self.custom_loss(y_pred.double(), y_batch, device)
                avg_val_loss += loss.item() / len(valid_loader)
                preds.append(torch.sigmoid(y_pred).cpu().numpy())
                original.append(y_batch.cpu().numpy())
                
            preds = np.concatenate(preds)
            original = np.concatenate(original)
            
            score = calc_spearmanr_metric(preds, original)
            
            oof_ckpt_preds.append(preds)
            weights.append(2.**epoch)
            
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t spearman={:.4f} \t time={:.2f}s'.format(
                epoch + 1, self.config['epochs'], avg_loss, avg_val_loss, score, elapsed_time))
    
            if DEV_MODE or not os.path.isfile(model_path): # not saved before
                torch.save(model.state_dict(), model_path)
        
        oof_ckpt_preds = oof_ckpt_preds[self.config['model_range_start']:self.config['model_range_end']]
        weights = weights[self.config['model_range_start']:self.config['model_range_end']]
            
        oof_pred = np.average(oof_ckpt_preds, weights=weights, axis=0)
        
        del train_loader, valid_loader, optimizer, model, scheduler
        gc.collect()
        torch.cuda.empty_cache() 
        return oof_pred
    
    def train_model(self):
        def bce(y_true, y_pred, eps=1e-15):
            y_pred = np.clip(y_pred, eps, 1-eps)
            return np.mean(-(y_true*np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
        
        if not SUBMIT_MODE:
            scores = []
            fold_num = self.config['fold_num']
            splits = list(KFold(n_splits=self.config['fold_num'], random_state=self.config['seed'], shuffle=True) \
                          .split(self.target, self.target))
            
            oof_preds = np.zeros(self.target.shape)
            for bag in range(self.config['bag_size']):
                for fold in np.arange(fold_num):
                    tr_ind, val_ind = splits[fold]
                    oof_pred = self._train(tr_ind, val_ind, fold=fold, bag_id=bag)
                    oof_preds[val_ind] += oof_pred/self.config['bag_size']
                    score = calc_spearmanr_metric(oof_pred, self.target[val_ind])
                    scores.append(score)
                    print('Bag {} Fold {} score: {:.5f}'.format(bag, fold, score))
            
            overall_score = bce(self.target, oof_preds)
            overall_metric = calc_spearmanr_metric(oof_preds, self.target)
            print('overall bce = {:.5f} spearmanr = {:.5f}'.format(overall_score, overall_metric))
            print('score details:', scores)
            
            return oof_preds
        else:
            print('SUBMIT MODE = TRUE WILL NOT TRAIN MODEL')
   
    def inference_model(self):
        model, device = self.prepare_model()
        for param in model.parameters():
            param.requires_grad=False
        model=model.eval()
        
        test_loader = DataLoader(TextDataset(self.x_test['question_title'], 
                                             self.x_test['question_body'], 
                                             self.x_test['answer'], 
                                             self.meta_test, 
                                             np.arange(len(self.x_test['answer'])), None),
                                batch_size=self.config['pred_batch_size'], shuffle=False, 
                                pin_memory=True)
        
        fold_num = self.config['fold_num']
        
        oof_preds = np.zeros((len(self.x_test['answer']), len(target_cols)))
        
        for bag in range(self.config['bag_size']):
            for fold in np.arange(fold_num):
                weights = []
                oof_ckpt_preds = []
                
                for epoch in range(self.config['model_range_start'], self.config['model_range_end']):
                    torch.manual_seed(self.config['seed']+epoch)
                    
                    model_path = self.config['model_path_format'].format(self.config['version'], bag, fold, epoch)
                    assert os.path.isfile(model_path)
                
                    print('Loading model:', model_path)
                    model.load_state_dict(torch.load(model_path))
                    
                    preds = []
                    for i, (title, question, answer, meta, _) in enumerate(test_loader):
                        title = title.long().cuda()
                        question = question.long().cuda()
                        answer = answer.long().cuda()
                        y_pred = model(title, question, answer).detach()
                        preds.append(torch.sigmoid(y_pred).cpu().numpy())
                    preds = np.concatenate(preds)
                    
                    oof_ckpt_preds.append(preds)
                    weights.append(2.**epoch)
            
                oof_pred = np.average(oof_ckpt_preds, weights=weights, axis=0)
                oof_preds += oof_pred/(self.config['bag_size']*fold_num)
                
        del test_loader, model
        gc.collect()
        return oof_preds
        
    def close(self):
        if not SUBMIT_MODE:
            del self.x_train
        del self.x_test; gc.collect()
        torch.cuda.empty_cache() 
        
class Bert_RNN(nn.Module):
    def __init__(self,
                 hidden_size: int = 128,
                 max_lens: list = maxlens,
                 embed_size: int = 300,
                 meta_size: int = 64,
                 dropout: float = 0.0):
        
        super(Bert_RNN, self).__init__()
        
        self.sp_dropout_q = SpatialDropout(dropout)
        self.sp_dropout_a = SpatialDropout(dropout)
        
        self.gru_q = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru_q2 = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)
        self.gru_q_att = Attention(hidden_size * 2, max_lens['question_title']+max_lens['question_body'])
        
        self.gru_a = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru_a2 = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)
        self.gru_a_att = Attention(hidden_size * 2, max_lens['answer'])
        
        self.linear1_q = nn.Linear(hidden_size * 2, 512)
        self.droupout_q = nn.Dropout(0.1)
        self.linear_out_q = nn.Linear(512, len(target_cols)-9)
        
        self.linear1_a = nn.Linear(hidden_size * 2 * 2 + 512, 512)
        self.droupout_a = nn.Dropout(0.1)
        self.linear_out_a = nn.Linear(512, 9)
        
    def forward(self, embed_qt, embed_a):
        
        embed_q = self.sp_dropout_q(embed_qt)
        h_gru_q, _ = self.gru_q(embed_q)
        h_gru_q, _ = self.gru_q2(h_gru_q)
        h_gru_q_att = self.gru_q_att(h_gru_q)
        
        embed_a = self.sp_dropout_a(embed_a)
        h_gru_a, _ = self.gru_a(embed_a)
        h_gru_a, _ = self.gru_a2(h_gru_a)
        h_gru_a_att = self.gru_a_att(h_gru_a)
        
        q_features = h_gru_q_att
        x = self.droupout_q(nn.ELU()(self.linear1_q(q_features)))
        out_q = self.linear_out_q(x)
        
        a_features = torch.cat((h_gru_q_att, h_gru_a_att, x), 1)
        x = self.droupout_a(nn.ELU()(self.linear1_a(a_features)))
        out_a = self.linear_out_a(x)
        
        return torch.cat([out_q, out_a], 1)

class BertFeatureExtractForGQA(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super(BertFeatureExtractForGQA, self).__init__(config)
        config.output_hidden_states = True
        self.bert = transformers.BertModel(config)
        
        self.init_weights()

    def forward(self, title_ids, question_ids, answer_ids):
        qt_ids = torch.cat((title_ids, question_ids), 1)
        qt_token_ids = None #torch.zeros(qt_ids.size(), dtype=torch.long, device=qt_ids.device)
        qt_outputs = self.bert(qt_ids, token_type_ids=qt_token_ids, attention_mask=qt_ids>0)
        
        answer_token_ids = None #torch.ones(answer_ids.size(), dtype=torch.long, device=answer_ids.device)
        answer_outputs = self.bert(answer_ids, token_type_ids=answer_token_ids, attention_mask=answer_ids>0)
        
        qt_ret = torch.cat([qt_outputs[2][-1]], 2)
        a_ret = torch.cat([answer_outputs[2][-1]], 2)
        
        return qt_ret, a_ret
    
class BERT_RNN_Pipeline_Pytorch(BERT_Pipeline_Pytorch):
        
    def __init__(self, *args):
        super(BERT_RNN_Pipeline_Pytorch, self).__init__(*args)

    def preprocess(self):
        
        with timer('Generate Tokens'):
            from sklearn.preprocessing import OneHotEncoder
            
            self.x_train = {tc: None for tc in text_cols}
            self.x_test = {tc: None for tc in text_cols}
            self.target = None
            
            text_cls_sep = {
                'question_title': {'add_cls': True, 'add_sep': True},
                'question_body': {'add_cls': False, 'add_sep': True},
                'answer': {'add_cls': True, 'add_sep': True},
            }
            meta_features = ['netloc', 'category']
            
            if not SUBMIT_MODE:
                train_df = pd.read_csv(self.train_path)
                self.target = train_df[target_cols].values
                
                for tc in text_cols:
                    train_texts = train_df[tc].fillna("DUMMY_VALUE").values.tolist()
                    self.x_train[tc] = np.array(self.get_tokens(train_texts, maxlens[tc], pre_prop=.3, 
                                                    add_cls=text_cls_sep[tc]['add_cls'],
                                                    add_sep=text_cls_sep[tc]['add_sep']))
                    del train_texts; gc.collect()
                    
                ohe = OneHotEncoder(dtype=np.float32, handle_unknown='ignore')
                train_df['netloc'] = self.get_netloc(train_df)
                self.meta_train = ohe.fit_transform(train_df[meta_features]).toarray()
                pd.to_pickle(ohe, self.get_pickle_ohe_path())
                
                del train_df; gc.collect()
            
            test_df = pd.read_csv(self.test_path)
            for tc in text_cols:
                test_texts = test_df[tc].fillna("DUMMY_VALUE").values.tolist()
                self.x_test[tc] = np.array(self.get_tokens(test_texts, maxlens[tc], pre_prop=.3, 
                                               add_cls=text_cls_sep[tc]['add_cls'],
                                               add_sep=text_cls_sep[tc]['add_sep']))
                del test_texts; gc.collect()
                
            ohe = pd.read_pickle(self.get_pickle_ohe_path())
            test_df['netloc'] = self.get_netloc(test_df)
            self.meta_test = ohe.transform(test_df[meta_features]).toarray()
            
            del test_df; gc.collect()
            
    # prepare model for bert embedding extraction 
    def prepare_model(self):
        if DEV_MODE:
            from transformers import convert_bert_original_tf_checkpoint_to_pytorch
            print(os.listdir(self.config['transformed_bert_dir']))
            if not os.path.isfile(self.config['transformed_bert_model']):
                convert_bert_original_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
                        self.config['bert_ckpt_file_path'],
                        self.config['bert_config_file_path'],
                        self.config['transformed_bert_model'])
                shutil.copyfile(self.config['bert_config_file_path'], self.config['transformed_bert_config'])
        
        device = torch.device('cuda')
            
        model = BertFeatureExtractForGQA.from_pretrained(self.config['transformed_bert_dir'])    
        model.zero_grad()
        model.to(device)
        
        # make sure bert model is freezed
        for param in model.parameters():
            param.requires_grad=True
        model=model.eval()
        
        return model, device

    def _train(self, tr_ind, val_ind, fold=0, bag_id=0):
        oof_ckpt_preds = []
        weights = []
        
        train_loader = DataLoader(TextDataset(self.x_train['question_title'], 
                                              self.x_train['question_body'], 
                                              self.x_train['answer'], 
                                              self.meta_train, 
                                              tr_ind, self.target),
                                  batch_size=self.config['batch_size'], shuffle=True, 
                                  pin_memory=True)
        valid_loader = DataLoader(TextDataset(self.x_train['question_title'], 
                                              self.x_train['question_body'], 
                                              self.x_train['answer'], 
                                              self.meta_train,
                                              val_ind, self.target),
                                  batch_size=self.config['pred_batch_size'], shuffle=False, 
                                  pin_memory=True)
        
        #########
        # prepare model, optimizers, lr schedulers
        #########
        bert_model, device = self.prepare_model()
        
        model = Bert_RNN(dropout = self.config['dropout'],
                         hidden_size = self.config['hidden_size'],
                         max_lens = maxlens,
                         embed_size = self.config['embed_size'],
                         meta_size = self.meta_train.shape[1]).cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), self.config['max_lr'])
        #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.config['epochs']-2, 
                                        gamma=self.config['min_lr']/self.config['max_lr'])
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
        
        for epoch in range(0, self.config['epochs']):
            torch.manual_seed(self.config['seed']+epoch)
            start_time = time.time()
            
            model_path = self.config['model_path_format'].format(self.config['version'], bag_id, fold, epoch)
            avg_loss = 0.
            
            if DEV_MODE or not os.path.isfile(model_path): # not trained before
                
                #########
                # Training
                #########
                # training
                for param in model.parameters():
                    param.requires_grad=True
                model=model.train()
                optimizer.zero_grad()
                
                for title, question, answer, meta, y_batch in tqdm(train_loader, disable=False):
                    title = title.long().cuda()
                    question = question.long().cuda()
                    answer = answer.long().cuda()
                    #meta = meta.cuda()
                    y_batch = y_batch.cuda()
                    
                    with torch.no_grad():
                        qt_feats, a_feats = bert_model(title, question, answer)
                    
                    y_pred = model(qt_feats, a_feats)
                   
                    loss = loss_fn(y_pred.double(), y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    optimizer.zero_grad()
    
                    avg_loss += loss.item() / len(train_loader)
                    
                print('Loss: {:.6f}'.format(avg_loss))
            else:
                model.load_state_dict(torch.load(model_path))
                
            scheduler.step()
            
            #########
            # Valdiation
            #########
            for param in model.parameters():
                param.requires_grad=False
            model.eval()

            avg_val_loss = 0.
            preds = []
            original = []
            with torch.no_grad():
                for i, (title, question, answer, meta, y_batch) in enumerate(valid_loader):
                    title = title.long().cuda()
                    question = question.long().cuda()
                    answer = answer.long().cuda()
                    #meta = meta.cuda()
                    y_batch = y_batch.cuda()
                    
                    qt_feats, a_feats = bert_model(title, question, answer)
                    y_pred = model(qt_feats, a_feats)
                    y_pred = y_pred.detach()
                    
                    loss = self.custom_loss(y_pred.double(), y_batch, device)
                    avg_val_loss += loss.item() / len(valid_loader)
                    preds.append(torch.sigmoid(y_pred).cpu().numpy())
                    original.append(y_batch.cpu().numpy())
                
            preds = np.concatenate(preds)
            original = np.concatenate(original)
            
            score = calc_spearmanr_metric(preds, original)
            
            oof_ckpt_preds.append(preds)
            weights.append(1.**epoch)
            
            elapsed_time = time.time() - start_time
            print('Epoch {}/{}  loss={:.4f}  val_loss={:.4f}  spearman={:.4f}  time={:.2f}s'.format(
                epoch + 1, self.config['epochs'], avg_loss, avg_val_loss, score, elapsed_time))
    
            if DEV_MODE or not os.path.isfile(model_path): # not saved before
                torch.save(model.state_dict(), model_path)
        
        oof_ckpt_preds = oof_ckpt_preds[self.config['model_range_start']:self.config['model_range_end']]
        weights = weights[self.config['model_range_start']:self.config['model_range_end']]
            
        oof_pred = np.average(oof_ckpt_preds, weights=weights, axis=0)
        
        del train_loader, valid_loader, optimizer, model, bert_model, scheduler
        gc.collect()
        torch.cuda.empty_cache() 
        return oof_pred
        #return self._train_finetune(tr_ind, val_ind, fold, bag_id)
    
    
    def inference_model(self):
        bert_model, device = self.prepare_model()
        
        model = Bert_RNN(dropout = self.config['dropout'],
                         hidden_size = self.config['hidden_size'],
                         max_lens = maxlens,
                         embed_size = self.config['embed_size'],
                         meta_size = self.meta_test.shape[1]).cuda()
        
        for param in model.parameters():
            param.requires_grad=False
        model=model.eval()
        
        test_loader = DataLoader(TextDataset(self.x_test['question_title'], 
                                             self.x_test['question_body'], 
                                             self.x_test['answer'], 
                                             self.meta_test, 
                                             np.arange(len(self.x_test['answer'])), None),
                                batch_size=self.config['pred_batch_size'], shuffle=False, 
                                pin_memory=True)
        
        fold_num = self.config['fold_num']
        
        oof_preds = np.zeros((len(self.x_test['answer']), len(target_cols)))
        
        for bag in range(self.config['bag_size']):
            for fold in np.arange(fold_num):
                weights = []
                oof_ckpt_preds = []
                
                for epoch in range(self.config['model_range_start'], self.config['model_range_end']):
                    torch.manual_seed(self.config['seed']+epoch)
                    
                    model_path = self.config['model_path_format'].format(self.config['version'], bag, fold, epoch)
                    assert os.path.isfile(model_path)
                
                    print('Loading model:', model_path)
                    model.load_state_dict(torch.load(model_path))
                    
                    preds = []
                    with torch.no_grad():
                        for i, (title, question, answer, meta, _) in enumerate(test_loader):
                            title = title.long().cuda()
                            question = question.long().cuda()
                            answer = answer.long().cuda()
                            qt_feats, a_feats = bert_model(title, question, answer)
                            y_pred = model(qt_feats, a_feats).detach()
                            preds.append(torch.sigmoid(y_pred).cpu().numpy())
                    preds = np.concatenate(preds)
                    
                    oof_ckpt_preds.append(preds)
                    weights.append(1.**epoch)
            
                oof_pred = np.average(oof_ckpt_preds, weights=weights, axis=0)
                oof_preds += oof_pred/(self.config['bag_size']*fold_num)
                
        del test_loader, bert_model, model
        gc.collect()
        return oof_preds
        
        
class DistillBERT_Pipeline_Pytorch(BERT_Pipeline_Pytorch):
    def __init__(self, *args):
        super(DistillBERT_Pipeline_Pytorch, self).__init__(*args)
    
    def prepare_model(self):
        device = torch.device('cuda')

        class DistillBertForGQA(transformers.DistilBertModel):
            def __init__(self, config):
                config.output_hidden_states = True
                super(DistillBertForGQA, self).__init__(config)
                cfg = self.config
                self.dropout = nn.Dropout(0.2)
                self.linear1 = nn.Linear(cfg.dim*3*4, 512) # avg+max pool->attn
        
                self.linear_out = nn.Linear(512, len(target_cols))
                self.apply(self._init_weights)
                
            def summarize_hidden_states(self, token_embeddings, attention_mask):
                #method == 'cls':
                return token_embeddings[:, 0, :]
            
            def forward(self, title_ids, question_ids, answer_ids):
                title_outputs = super(DistillBertForGQA, self).forward(title_ids, attention_mask=title_ids>0)
                question_outputs = super(DistillBertForGQA, self).forward(question_ids, attention_mask=question_ids>0)
                answer_outputs = super(DistillBertForGQA, self).forward(answer_ids, attention_mask=answer_ids>0)
                conc = torch.cat([self.summarize_hidden_states(title_outputs[0], title_ids>0), 
                                  self.summarize_hidden_states(title_outputs[1][-1], title_ids>0), 
                                  self.summarize_hidden_states(title_outputs[1][-2], title_ids>0), 
                                  self.summarize_hidden_states(title_outputs[1][-3], title_ids>0), 
                                  self.summarize_hidden_states(question_outputs[0], question_ids>0),
                                  self.summarize_hidden_states(question_outputs[1][-1], question_ids>0),
                                  self.summarize_hidden_states(question_outputs[1][-2], question_ids>0),
                                  self.summarize_hidden_states(question_outputs[1][-3], question_ids>0),
                                  self.summarize_hidden_states(answer_outputs[0], answer_ids>0),
                                  self.summarize_hidden_states(answer_outputs[1][-1], answer_ids>0),
                                  self.summarize_hidden_states(answer_outputs[1][-2], answer_ids>0),
                                  self.summarize_hidden_states(answer_outputs[1][-3], answer_ids>0)], 1)
        
                conc = self.dropout(nn.ELU()(self.linear1(conc)))
                logits = self.linear_out(conc)
        
                #conc = self.dropout(conc)
                #logits = self.classifier(conc)
        
                return logits
            
        model = DistillBertForGQA.from_pretrained(self.config['transformed_bert_dir'])
        model.zero_grad()
        model.to(device)
        return model, device

class MLP_GQA(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout):
        
        super(MLP_GQA, self).__init__()
        
        self.linear1 = nn.Linear(feature_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.droupout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_size, len(target_cols))

    def forward(self, features):
        
        # add l2, cosine  distance
        x = self.droupout(nn.ELU()(self.linear1(features)))
        x = self.droupout(nn.ELU()(self.linear2(x)+x))
        out = self.linear_out(x)

        return out
    
class MLP_GQA_v2(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout):
        
        super(MLP_GQA_v2, self).__init__()
        
        self.tq_fsize = feature_size//2
        
        self.linear1_tq = nn.Linear(self.tq_fsize, hidden_size)
        self.linear2_tq = nn.Linear(hidden_size, hidden_size)
        self.droupout_tq = nn.Dropout(dropout)
        self.linear_out_tq = nn.Linear(hidden_size, len(target_cols)-9)

        self.linear1_a = nn.Linear(self.tq_fsize+hidden_size*2, hidden_size)
        self.linear2_a = nn.Linear(hidden_size, hidden_size)
        self.droupout_a = nn.Dropout(dropout)
        self.linear_out_a = nn.Linear(hidden_size, 9)
        
    def forward(self, features):
        
        # add l2, cosine  distance
        x_tq_s = self.droupout_tq(nn.ELU()(self.linear1_tq(features[:,:self.tq_fsize])))
        x_tq_d = self.droupout_tq(nn.ELU()(self.linear2_tq(x_tq_s)+x_tq_s))
        out_tq = self.linear_out_tq(x_tq_d)
        
        a_feats = torch.cat([features[:,self.tq_fsize:], x_tq_s, x_tq_d], 1)
        x_a = self.droupout_a(nn.ELU()(self.linear1_a(a_feats)))
        x_a = self.droupout_a(nn.ELU()(self.linear2_a(x_a)+x_a))
        out_a = self.linear_out_a(x_a)
        
        return torch.cat([out_tq, out_a], 1)
    
class Pretrain_Features_Pipelin_Pytorch(Modeling_Pipeline):
    def __init__(self, *args):
        super(Pretrain_Features_Pipelin_Pytorch, self).__init__(*args)

    def fetch_pretrain_dstbert_features(self, string_list, batch_size=64, is_question=True):
        with timer('Extracting distilled bert features'):
            def chunks(l, n):
                """Yield successive n-sized chunks from l."""
                for i in range(0, len(l), n):
                    yield l[i:i + n]
                    
            # inspired by https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
            DEVICE = torch.device("cuda")
            tokenizer = transformers.DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")
            model = transformers.DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")
            model.to(DEVICE)
        
            fin_features = []
            for data in chunks(string_list, batch_size):
                tokenized = []
                for x in data:
                    x = " ".join(x.strip().split()[:300])
                    tok = tokenizer.encode(x, add_special_tokens=True)
                    tokenized.append(tok[:512])
        
                max_len = 512
                padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
                attention_mask = np.where(padded != 0, 1, 0)
                input_ids = torch.tensor(padded).to(DEVICE)
                attention_mask = torch.tensor(attention_mask).to(DEVICE)
        
                with torch.no_grad():
                    last_hidden_states = model(input_ids, attention_mask=attention_mask)
        
                features = last_hidden_states[0][:, 0, :].detach().cpu().numpy()
                fin_features.append(features)
        
            fin_features_ = np.vstack(fin_features) 
            del model, tokenizer, fin_features, tok, attention_mask, input_ids
            del features, last_hidden_states, tokenized, padded; 
            gc.collect(); torch.cuda.empty_cache(); 
            os.system('nvidia-smi >> log.txt')
            return fin_features_

    def fetch_use_large_features(self, string_list, batch_size = 4, is_question=True):
        with timer('Extracting USE-large features'):
            import tensorflow_hub as hub
            import tensorflow as tf
            
            module_url = "../input/universalsentenceencoderlarge4/"
            embed = hub.load(module_url)
            
            texts = string_list.str.replace('?', '.').str.replace('!', '.').tolist()
            
            curr_emb = []
            ind = 0
            while ind*batch_size < len(texts):
                curr_emb.append(embed(texts[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
                ind += 1
            
            res = np.vstack(curr_emb)
            del curr_emb, embed, texts; gc.collect(); K.clear_session();
            os.system('nvidia-smi >> log.txt')
            return res
    
    def fetch_use_features(self, string_list, batch_size = 4, is_question=True):
        with timer('Extracting USE features'):
            import tensorflow_hub as hub
            import tensorflow as tf
            
            module_url = "../input/universal-sentence-encoder/"
            embed = hub.load(module_url)
            
            texts = string_list.str.replace('?', '.').str.replace('!', '.').tolist()
            
            curr_emb = []
            ind = 0
            while ind*batch_size < len(texts):
                curr_emb.append(embed(texts[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
                ind += 1
            
            res = np.vstack(curr_emb)
            del curr_emb, embed, texts; gc.collect(); K.clear_session();
            os.system('nvidia-smi >> log.txt')
            return res
    
    def fetch_use_qa_features(self, string_list, batch_size = 4, is_question=True):
        with timer('Extracting USE-qa features'):
            import tensorflow_hub as hub
            import tensorflow as tf
            
            module_url = "../input/universal-sentence-encoder-qa/"
            embed = hub.load(module_url)
            embed.init_op()
            
            texts = string_list.tolist()

            curr_emb = []
            ind = 0
            if is_question:
                while ind*batch_size < len(texts):
                    #curr_emb.append(embed(texts[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
                    tmp = texts[ind*batch_size: (ind + 1)*batch_size]
                    curr_emb.append(embed.signatures['question_encoder'](tf.constant(tmp))["outputs"].numpy())
                    ind += 1
            else:
                while ind*batch_size < len(texts):
                    #curr_emb.append(embed(texts[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
                    tmp = texts[ind*batch_size: (ind + 1)*batch_size]
                    curr_emb.append(embed.signatures['response_encoder'](input=tf.constant(tmp), context=tf.constant([""]*len(tmp)))["outputs"].numpy())
                    ind += 1
            
            res = np.vstack(curr_emb)
            del curr_emb, embed, texts, tmp; gc.collect(); K.clear_session();
            os.system('nvidia-smi >> log.txt')
            return res
    
    def prepare_meta_features_and_targets(self):
        with timer('Prepare Meta Features and Targets'):
            from sklearn.preprocessing import OneHotEncoder
            
            self.target = None
            
            meta_features = ['netloc', 'category']
            if not SUBMIT_MODE:
                train_df = pd.read_csv(self.train_path)
                self.target = train_df[target_cols].values
                
                ohe = OneHotEncoder(dtype=np.float32, handle_unknown='ignore')
                train_df['netloc'] = self.get_netloc(train_df)
                self.meta_train = ohe.fit_transform(train_df[meta_features]).toarray()
                pd.to_pickle(ohe, self.get_pickle_ohe_path())
                
                del train_df; gc.collect()
            
            test_df = pd.read_csv(self.test_path)
            ohe = pd.read_pickle(self.get_pickle_ohe_path())
            test_df['netloc'] = self.get_netloc(test_df)
            self.meta_test = ohe.transform(test_df[meta_features]).toarray()
            
            del test_df; gc.collect()
            
    def preprocess(self):
        self.prepare_meta_features_and_targets()
        if not SUBMIT_MODE:
            self.x_train = {} # dict of pretrained extracted embeddings
        self.x_test = {}

        feat_sources = ['dst_bert', 'use_large', 'use_qa']
        postfixs = ['title', 'question', 'answer']
        
        if not SUBMIT_MODE:
            pkl_paths = [
                        '../input/dist_bert_features_fast_loading.pkl',
                        '../input/use_large_features_fast_loading.pkl',
                        #'../input/use_features_fast_loading.pkl',
                        '../input/use_qa_features_fast_loading.pkl',
                    ]
            
            for fs, pkl_path in zip(feat_sources, pkl_paths):
                assert os.path.isfile(pkl_path)
            
                embed_dict = pd.read_pickle(pkl_path)
            
                for prefix, dct in zip(['train', 'test'], [self.x_train, self.x_test]):
                    dct.update({fs+'_'+pf: embed_dict[prefix+'_'+pf].copy() for pf in postfixs})
                    
                del embed_dict; gc.collect()
                
            self.x_train = self.do_fe(self.x_train, self.meta_train)
            self.x_test = self.do_fe(self.x_test, self.meta_test)
        else:
            if os.path.isfile(SUBMIT_MODE_USE_FEATS_CACHE_PATH):
                self.x_test = pd.read_pickle(SUBMIT_MODE_USE_FEATS_CACHE_PATH)
            else:
                
                test_df = pd.read_csv(self.test_path)
                
                extract_funcs = [
                            self.fetch_pretrain_dstbert_features,
                            self.fetch_use_large_features,
                            #self.fetch_use_features,
                            self.fetch_use_qa_features,
                        ]
                
                for fs, extract_func in zip(feat_sources, extract_funcs): 
                    self.x_test.update({
                                fs+'_title': extract_func(test_df.question_title, is_question=True),
                                fs+'_question': extract_func(test_df.question_body, is_question=True),
                                fs+'_answer': extract_func(test_df.answer, is_question=False),
                            })
    
                pd.to_pickle(self.x_test, SUBMIT_MODE_USE_FEATS_CACHE_PATH)
                del test_df; gc.collect()
                
            test_feats = self.do_fe(self.x_test, self.meta_test)
            del self.x_test; gc.collect()
            self.x_test = test_feats
                
    def do_fe(self, x, meta):
        embeddings = np.concatenate([
                            x['dst_bert_title'],
                            x['dst_bert_question'],
                            x['dst_bert_answer'],
                            x['use_large_title'],
                            x['use_large_question'],
                            x['use_large_answer'],
                            x['use_qa_title'],
                            x['use_qa_question'],
                            x['use_qa_answer'],
                        ], axis=1)
    
        cos_dist = lambda x, y: (x*y).sum(axis=1)

        dist_features = np.array([
            # intra-group inter-text_columns
            #cos_dist(x['dst_bert_title'], x['dst_bert_answer']),
            #cos_dist(x['dst_bert_question'], x['dst_bert_answer']),
            #cos_dist(x['dst_bert_title'], x['dst_bert_question']),
            #cos_dist(x['use_large_title'], x['use_large_answer']),
            #cos_dist(x['use_large_question'], x['use_large_answer']),
            #cos_dist(x['use_large_title'], x['use_large_question']),
            #cos_dist(x['use_qa_title'], x['use_qa_answer']),
            #cos_dist(x['use_qa_question'], x['use_qa_answer']),
            #cos_dist(x['use_qa_title'], x['use_qa_question'])
            #l2_dist(x['dst_bert_title'], x['dst_bert_answer']),
            #l2_dist(x['dst_bert_question'], x['dst_bert_answer']),
            #l2_dist(x['dst_bert_title'], x['dst_bert_question']),
            #l2_dist(x['use_large_title'], x['use_large_answer']),
            #l2_dist(x['use_large_question'], x['use_large_answer']),
            #l2_dist(x['use_large_title'], x['use_large_question']),
            #l2_dist(x['use_qa_title'], x['use_qa_answer']),
            #l2_dist(x['use_qa_question'], x['use_qa_answer']),
            #l2_dist(x['use_qa_title'], x['use_qa_question']),
            
            # inter-group intra-text_columns
            #cos_dist(x['dst_bert_title'], x['use_large_title']),
            #cos_dist(x['dst_bert_title'], x['use_qa_title']),
            cos_dist(x['use_large_title'], x['use_large_title']),
            #cos_dist(x['dst_bert_question'], x['use_large_question']),
            #cos_dist(x['dst_bert_question'], x['use_qa_question']),
            cos_dist(x['use_large_question'], x['use_qa_question']),
            #cos_dist(x['dst_bert_answer'], x['use_large_answer']),
            #cos_dist(x['dst_bert_answer'], x['use_qa_answer']),
            cos_dist(x['use_large_answer'], x['use_qa_answer']),
            #l2_dist(x['use_large_title'], x['use_large_title']),
            #l2_dist(x['use_large_question'], x['use_qa_question']),
            #l2_dist(x['use_large_answer'], x['use_qa_answer']),
        ]).T
    
        return np.concatenate([embeddings, dist_features], axis=1)
    
    def _train(self, tr_ind, val_ind, fold=0, bag_id=0):
        
        from torch.utils import data
        
        oof_ckpt_preds = []
        weights = []
        
        model = MLP_GQA(self.x_train.shape[1], 
                        self.config['hidden_size'], 
                        self.config['dropout']).cuda()
        
        train_dataset = data.TensorDataset(torch.tensor(self.x_train[tr_ind],dtype=torch.float), 
                                           torch.tensor(self.target[tr_ind]))
        valid_dataset = data.TensorDataset(torch.tensor(self.x_train[val_ind],dtype=torch.float), 
                                           torch.tensor(self.target[val_ind]))
        
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config['batch_size'], shuffle=True, 
                                  pin_memory=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=self.config['pred_batch_size'], shuffle=False, 
                                  pin_memory=True)
        
        optimizer = torch.optim.Adam(model.parameters(), self.config['max_lr'])
        #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.config['epochs']-2, gamma=0.1)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
        
        best_score = 0
        
        for epoch in range(self.config['epochs']):
            torch.manual_seed(self.config['seed']+epoch)
            start_time = time.time()
            avg_loss = 0.
    
            model_path = self.config['model_path_format'].format(bag_id, fold, epoch)
            
            if DEV_MODE or not os.path.isfile(model_path): # not trained before
                # training
                for param in model.parameters():
                    param.requires_grad=True
                model=model.train()
                optimizer.zero_grad()
                
                for x, y_batch in tqdm(train_loader, disable=True):
                    x = x.cuda()
                    y_batch = y_batch.cuda()
                    y_pred = model(x)
        
                    loss = loss_fn(y_pred.double(), y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    optimizer.zero_grad()
    
                    avg_loss += loss.item() / len(train_loader)
                    #scheduler.step()
            else:
                model.load_state_dict(torch.load(model_path))
                
            # evaluation
            for param in model.parameters():
                param.requires_grad=False
            model=model.eval()
            
            avg_val_loss = 0.
            preds = []
            original = []
            for i, (x, y_batch) in enumerate(valid_loader):
                x = x.cuda()
                #meta = meta.cuda()
                y_batch = y_batch.cuda()
                y_pred = model(x).detach()
    
                avg_val_loss += loss_fn(y_pred.double(), y_batch).item() / len(valid_loader)
                preds.append(torch.sigmoid(y_pred).cpu().numpy())
                original.append(y_batch.cpu().numpy())
            
            preds = np.concatenate(preds)
            original = np.concatenate(original)
            
            score = calc_spearmanr_metric(preds, original)
            
            oof_ckpt_preds.append(preds)
            weights.append(2.**epoch)
            
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} loss={:.4f} val_loss={:.4f} spearman={:.2f} time={:.2f}s'.format(
                epoch + 1, self.config['epochs'], avg_loss, avg_val_loss, score, elapsed_time))
    
            #scheduler.step(avg_val_loss)
            scheduler.step()
    
            valid_score = score
            if valid_score > best_score:
                best_score = valid_score
                p = 0
    
            # check if validation loss didn't improve
            if valid_score <= best_score:
                p += 1
                print(f'{p} epochs of non improving score')
                #if p > patience:
                #    print('Stopping training')
                #    break
            
            if DEV_MODE or not os.path.isfile(model_path): # not saved before
                torch.save(model.state_dict(), model_path)
        
        oof_pred = np.average(oof_ckpt_preds, weights=weights, axis=0)
        
        del train_loader, valid_loader, optimizer, scheduler, loss_fn
        del model; gc.collect(); torch.cuda.empty_cache()
        
        return oof_pred
    
    def train_model(self):
        def bce(y_true, y_pred, eps=1e-15):
            y_pred = np.clip(y_pred, eps, 1-eps)
            return np.mean(-(y_true*np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
        
        if not SUBMIT_MODE:
            scores = []
            fold_num = self.config['fold_num']
            splits = list(KFold(n_splits=self.config['fold_num'], random_state=self.config['seed'], shuffle=True) \
                          .split(self.target, self.target))
            
            oof_preds = np.zeros(self.target.shape)
            for bag in range(self.config['bag_size']):
                for fold in np.arange(fold_num):
                    tr_ind, val_ind = splits[fold]
                    oof_pred = self._train(tr_ind, val_ind, fold=fold, bag_id=bag)
                    oof_preds[val_ind] += oof_pred/self.config['bag_size']
                    score = calc_spearmanr_metric(oof_pred, self.target[val_ind])
                    scores.append(score)
                    print('Bag {} Fold {} score: {:.5f}'.format(bag, fold, score))
            
            overall_score = bce(self.target, oof_preds)
            overall_metric = calc_spearmanr_metric(oof_preds, self.target)
            print('overall bce = {:.5f} spearmanr = {:.5f}'.format(overall_score, overall_metric))
            print('score details:', scores)
            
            return oof_preds
        else:
            print('SUBMIT MODE = TRUE WILL NOT TRAIN MODEL')

    def inference_model(self):  
        from torch.utils import data
        
        model = MLP_GQA(self.x_test.shape[1], 
                        self.config['hidden_size'], 
                        self.config['dropout']).cuda()
        
        test_dataset = data.TensorDataset(torch.tensor(self.x_test,dtype=torch.float))
        
        for param in model.parameters():
            param.requires_grad=False
        model=model.eval()
        
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.config['pred_batch_size'], shuffle=False, 
                                 pin_memory=True)
        
        fold_num = self.config['fold_num']
        
        oof_preds = np.zeros((len(self.x_test), len(target_cols)))
        
        for bag in range(self.config['bag_size']):
            for fold in np.arange(fold_num):
                weights = []
                oof_ckpt_preds = []
                
                for epoch in range(self.config['epochs']):
                    torch.manual_seed(self.config['seed']+epoch)
                    
                    model_path = self.config['model_path_format'].format(bag, fold, epoch)
                    assert os.path.isfile(model_path)
                
                    print('Loading model:', model_path)
                    model.load_state_dict(torch.load(model_path))
                    
                    preds = []
                    for i, (x, ) in enumerate(test_loader):
                        x = x.cuda()
                        y_pred = model(x).detach()
                        preds.append(torch.sigmoid(y_pred).cpu().numpy())
                    preds = np.concatenate(preds)
                    
                    oof_ckpt_preds.append(preds)
                    weights.append(2.**epoch)
            
                oof_pred = np.average(oof_ckpt_preds, weights=weights, axis=0)
                oof_preds += oof_pred/(self.config['bag_size']*fold_num)
                
        del test_dataset, test_loader, model; gc.collect(); torch.cuda.empty_cache()
        return oof_preds
        
    def close(self):
        if not SUBMIT_MODE:
            del self.x_train
        del self.x_test; gc.collect(); torch.cuda.empty_cache()


class BertPublic(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super(BertPublic, self).__init__(config)
        config.output_hidden_states = True
        self.bert = transformers.BertModel(config)
        
        self.droupout_tq = nn.Dropout(0.2)
        self.linear_out_tq = nn.Linear(config.hidden_size * 3, len(target_cols)-9)
        
        self.droupout_qa = nn.Dropout(0.2)
        self.linear_out_qa = nn.Linear(config.hidden_size * 4, 9)
        #self.init_weights()

    def forward(self, ids, masks, segments, title_len, body_len, answer_len,
                tq_ids, tq_masks, tq_segments, tq_title_lens, tq_body_lens):
        # public: global avg pool 1d:                       0.3770
        # cls: :                                            0.3818
        # cls+sep+sep+sep:                                  0.3877, 0.3952, avg: 0.3935, 0.4045
        # cls+sep+sep+sep, token ids: 0-1-0:                0.3833, 0.3936, avg: 0.3913, 0.3998
        # cls+QBODY+ANS+sep:                                0.3822, 0.3911, avg: 0.3879, 0.3962
        # cls+sep+sep+sep, tq-a in different branches:      0.3967, 0.4077, avg: 0.4051, 0.4148
        #                           0.4070, 0.4067, 0.4091, 0.3950, 0.4077, avg: 0.4139, 0.4141, 0.4196, 0.4043, 0.4168
        # cls+sep+sep+sep, tq-a diff br, 0-1, 0-0-1:        0.3936, 0.4068, avg: 0.4026, 0.4146
        # cls+sep+sep+sep, tq-a diff br, depth=2:           0.3925, avg: 0.4030
        # cls+sep+sep+sep, tq-a diff br, depth=2, cls only: 0.3955, 0.4056, avg: 0.4037, 0.4141
        # interact: tqa add tq: 0.3914, avg: 0.3991
        # interact: tqa add tq, remove original tq:
        # interact: a only add tq: 0.3922, avg: 0.3992
        # cls+cls+cls+sep, tq-a diff br, 0-0-0:             0.3853, avg: 0.3925
        # cls+unk+sep+sep, tq-a diff br, 0-0-1:             0.3942, 0.4084 avg: 0.4024, 0.4160
        # cls+unk+unk+sep, tq-a diff br, 0-0-0:             0.3899,  avg: 0.3980, 
        # cls+unk+sep+sep, tqa combined, 0-0-1:             0.3906, 0.4000 avg: 0.3975, 0.4068
        # cls+unk+sep+sep, tqa combined, diff br, 0-0-1:    0.3884, 0. avg: 0.3974, 0.
        # cls+unk+sep+sep, tqa combined, 0-0-1, depth=2:    0.3880, avg: 0.3968
        # cls+unk+sep+sep, tqa combined, 0-0-1, +dense512:  0.3868, avg: 0.3924   
        # cls+unk+sep+sep, tq-a diff br, 0-0-1, v2:         0.3869, avg: 0.3961
        # change seq len (tqa: longer title):               0.3956, 0.4042, avg: 0.4046, 0.4137
        # change seq len (tqa: shorter title):              0.3959, 0.4045, avg: 0.4040, 0.4147
        # use last -2 hidden layers instead of -1:          0.3916, 0., avg: 0.4005, 0.
        # use last -2 hidden layers average:                0.3967, 0., avg: 0.4041, 0.
        # more epochs (4->5)                                0.3956, 0.4019, avg: 0.4058, 0.4138
        # linear schd+adamw without weight update for bias: 0.4006, avg: 0.4001
        # adamw without weight update for bias:             0.3962, avg: 0.4031
        # sampling method (downsampling)                    0.3952, avg: 0.4043
        # focal loss:                                       0.3946, avg: 0.4022
        # pseudo labeling:                                  0.3956, 0.4054, avg: 0.4045, 0.4165
        # accum grad to bs 64
        # more bagging
        # tqa combined, one target one new token:           0.3934, 0.4062, avg: 0.4002, 0.4097
        # tqa combined, t-q-a separate tokens:              0.3889, avg: 0.3961
        # tqa seperated, one target one new token:          0.3969, 0.4113 avg: 0.4045, 0.4179
        # tqa seperated, one target one new token, last 2 hidden: 0.4000, 0.4118 avg: 0.4068, 0.4161  
        # tqa combined, one target one new token, last 2 hidden: 0.3946, avg: 0.4008
        # tqa combined, one target one new token, add cls features: 0.3890, avg: 0.3957
        # tqa combined, predict sentence type
        # tqa combined, add meta
        # tqa separate, one target one new token, t later sep: 0.3995, 0.4079, avg: 0.4055, 0.4141
        
        ###########
        # title + body branch
        ###########
        t_sep_ofst = tq_title_lens+1 # cls
        b_sep_ofst = tq_title_lens+tq_body_lens+2 # cls+sep
        
        outputs = self.bert(tq_ids, token_type_ids=tq_segments, attention_mask=tq_masks)
        
        #x = self.droupout_qa(torch.mean(outputs[2][-1], 1))
        # cls
        
        last_hidden = outputs[2][-1]
        cls_feat = last_hidden[:,0,:] # B * S * H,  # cls
        
        t_sep_feats = []
        q_sep_feats = []
        for i in torch.arange(last_hidden.shape[0]):
            t_sep_feats += [last_hidden[[i], t_sep_ofst[i], :]] # 1 * H
            q_sep_feats += [last_hidden[[i], b_sep_ofst[i], :]]
            
        t_sep_feats = torch.cat(t_sep_feats, 0) # B * H
        q_sep_feats = torch.cat(q_sep_feats, 0) # B * H
    
        tq_feats = torch.cat([cls_feat, t_sep_feats, q_sep_feats], 1) # B * H * 4
        x = self.droupout_tq(tq_feats)
        out_tq = self.linear_out_tq(x)
        
        ###########
        # answer branch
        ###########
        t_sep_ofst = title_len+1 # cls
        b_sep_ofst = title_len+body_len+2 # cls+sep
        a_sep_ofst = title_len+body_len+answer_len+3 # cls+sep+sep
        
        outputs = self.bert(ids, token_type_ids=segments, attention_mask=masks)
        
        #x = self.droupout_qa(torch.mean(outputs[2][-1], 1))
        # cls
        last_hidden = outputs[2][-1]        
        cls_feat = last_hidden[:,0,:] # B * S * H,  # cls
        
        t_sep_feats = []
        q_sep_feats = []
        a_sep_feats = []
        for i in torch.arange(last_hidden.shape[0]):
            t_sep_feats += [last_hidden[[i], t_sep_ofst[i], :]] # 1 * H
            q_sep_feats += [last_hidden[[i], b_sep_ofst[i], :]]
            a_sep_feats += [last_hidden[[i], a_sep_ofst[i], :]]
            
        t_sep_feats = torch.cat(t_sep_feats, 0) # B * H
        q_sep_feats = torch.cat(q_sep_feats, 0) # B * H
        a_sep_feats = torch.cat(a_sep_feats, 0) # B * H
        
        tqa_feats = torch.cat([cls_feat, t_sep_feats, q_sep_feats, a_sep_feats], 1) # B * H * 4
        
        x = self.droupout_qa(tqa_feats)
        out_qa = self.linear_out_qa(x)
        return torch.cat((out_tq, out_qa), 1)
 
class BERT_Public_Pipeline_Pytorch(BERT_Pipeline_Pytorch):
    def __init__(self, *args):
        super(BERT_Public_Pipeline_Pytorch, self).__init__(*args)
    
    def _get_masks(self, tokens, max_seq_length):
        """Mask for padding"""
        if len(tokens)>max_seq_length:
            raise IndexError("Token length more than max seq length!")
        return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))
    
    def _get_segments(self, tokens, max_seq_length, skip_first_sep=True):
        """Segments: 0 for the first sequence, 1 for the second"""
        if len(tokens)>max_seq_length:
            raise IndexError("Token length more than max seq length!")
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                #pass
                #current_segment_id = 1 - current_segment_id
                
                if skip_first_sep:
                    skip_first_sep = False 
                else:
                    current_segment_id = 1
                
        return segments + [0] * (max_seq_length - len(tokens))
    
    def _get_ids(self, tokens, tokenizer, max_seq_length):
        """Token ids from Tokenizer vocab"""
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
        return input_ids
    
    def _trim_input(self, title, question, answer, max_sequence_length, tokenizer, 
                    t_max_len=30, q_max_len=239, a_max_len=239):
    
        t = tokenizer.tokenize(title)
        q = tokenizer.tokenize(question)
        a = tokenizer.tokenize(answer)
        
        t_len = len(t)
        q_len = len(q)
        a_len = len(a)
    
        if (t_len+q_len+a_len+4) > max_sequence_length:
            
            if t_max_len > t_len:
                t_new_len = t_len
                a_max_len = a_max_len + floor((t_max_len - t_len)/2)
                q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
            else:
                t_new_len = t_max_len
          
            if a_max_len > a_len:
                a_new_len = a_len 
                q_new_len = q_max_len + (a_max_len - a_len)
            elif q_max_len > q_len:
                a_new_len = a_max_len + (q_max_len - q_len)
                q_new_len = q_len
            else:
                a_new_len = a_max_len
                q_new_len = q_max_len
                
                
            if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:
                raise ValueError("New sequence length should be %d, but is %d" 
                                 % (max_sequence_length, (t_new_len+a_new_len+q_new_len+4)))
            
            t = t[:t_new_len]
            q = q[:q_new_len]
            a = a[:a_new_len]
        
        return t, q, a
    
    def _trim_input_tq_only(self, title, question, max_sequence_length, tokenizer, 
                            t_max_len=100, q_max_len=409):
    
        t = tokenizer.tokenize(title)
        q = tokenizer.tokenize(question)
        
        t_len = len(t)
        q_len = len(q)
    
        if (t_len+q_len+3) > max_sequence_length:
            
            if t_max_len > t_len:
                t_new_len = t_len
                q_max_len = q_max_len + (t_max_len - t_len)
            else:
                t_new_len = t_max_len
          
            if q_max_len > q_len:
                q_new_len = q_len
            else:
                q_new_len = q_max_len
                
                
            if t_new_len+q_new_len+3 != max_sequence_length:
                raise ValueError("New sequence length should be %d, but is %d" 
                                 % (max_sequence_length, (t_new_len+q_new_len+3)))
            
            t = t[:t_new_len]
            q = q[:q_new_len]
        
        return t, q
    
    def _convert_to_bert_inputs(self, title, question, answer, tokenizer, max_sequence_length):
        """Converts tokenized input to ids, masks and segments for BERT"""
        
        stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]
    
        input_ids = self._get_ids(stoken, tokenizer, max_sequence_length)
        input_masks = self._get_masks(stoken, max_sequence_length)
        input_segments = self._get_segments(stoken, max_sequence_length, skip_first_sep=True)
    
        return [input_ids, input_masks, input_segments]
    
    def _convert_to_bert_inputs_tq_only(self, title, question, tokenizer, max_sequence_length):
        """Converts tokenized input to ids, masks and segments for BERT"""
        
        stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"]
        
        input_ids = self._get_ids(stoken, tokenizer, max_sequence_length)
        input_masks = self._get_masks(stoken, max_sequence_length)
        input_segments = self._get_segments(stoken, max_sequence_length, skip_first_sep=True)
    
        return [input_ids, input_masks, input_segments]
    
    def compute_input_arays(self, df, columns, tokenizer, max_sequence_length):
        input_ids, input_masks, input_segments, t_len, q_len, a_len = [], [], [], [], [], []
        input_ids_tq, input_masks_tq, input_segments_tq, t_len_tq, q_len_tq = [], [], [], [], []
        
        for _, instance in tqdm(df[columns].iterrows()):
            t, q, a = instance.question_title, instance.question_body, instance.answer
            t, q, a = self._trim_input(t, q, a, max_sequence_length, tokenizer)
            ids, masks, segments = self._convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)
            t_len.append(len(t))
            q_len.append(len(q))
            a_len.append(len(a))
            
            t, q = instance.question_title, instance.question_body
            t, q = self._trim_input_tq_only(t, q, max_sequence_length, tokenizer)
            ids, masks, segments = self._convert_to_bert_inputs_tq_only(t, q, tokenizer, max_sequence_length)
            input_ids_tq.append(ids)
            input_masks_tq.append(masks)
            input_segments_tq.append(segments)
            t_len_tq.append(len(t))
            q_len_tq.append(len(q))
            
            
        return {'ids': np.asarray(input_ids), 
                'masks': np.asarray(input_masks), 
                'segments': np.asarray(input_segments),
                'title_lens': np.asarray(t_len),
                'body_lens': np.asarray(q_len),
                'answer_lens': np.asarray(a_len),
                
                'tq_ids': np.asarray(input_ids_tq), 
                'tq_masks': np.asarray(input_masks_tq), 
                'tq_segments': np.asarray(input_segments_tq),
                'tq_title_lens': np.asarray(t_len_tq),
                'tq_body_lens': np.asarray(q_len_tq)}
    
    def preprocess(self):
        
        with timer('Generate Tokens'):
            from sklearn.preprocessing import OneHotEncoder
            
            self.x_train = None
            self.x_test = None
            self.target = None
            
            meta_features = ['netloc', 'category']
            
            tokenizer = transformers.BertTokenizer.from_pretrained(self.config['bert_pretrained_dir'], 
                                                                   cache_dir=None, do_lower_case=self.config['do_lower'])
            
            if not SUBMIT_MODE:
                train_df = pd.read_csv(self.train_path)
                self.target = train_df[target_cols].values
                
                self.x_train = self.compute_input_arays(train_df, text_cols, tokenizer, 512)
                '''
                inputs = pd.read_pickle('public_inputs.pkl')
                self.x_train = {
                        'ids': inputs[0], 
                        'masks': inputs[1], 
                        'segments': inputs[2]
                        }
                '''
                ohe = OneHotEncoder(dtype=np.float32, handle_unknown='ignore')
                train_df['netloc'] = self.get_netloc(train_df)
                self.meta_train = ohe.fit_transform(train_df[meta_features]).toarray()
                pd.to_pickle(ohe, self.get_pickle_ohe_path())
                
                del train_df; gc.collect()
            
            test_df = pd.read_csv(self.test_path)
            self.x_test = self.compute_input_arays(test_df, text_cols, tokenizer, 512)
                
            #ohe = pd.read_pickle(self.get_pickle_ohe_path())
            #test_df['netloc'] = self.get_netloc(test_df)
            #self.meta_test = ohe.transform(test_df[meta_features]).toarray()
            
            del test_df, tokenizer; gc.collect()
    
    def prepare_model(self):
        if DEV_MODE:
            from transformers import convert_bert_original_tf_checkpoint_to_pytorch
            print(os.listdir(self.config['transformed_bert_dir']))
            if not os.path.isfile(self.config['transformed_bert_model']):
                convert_bert_original_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
                        self.config['bert_ckpt_file_path'],
                        self.config['bert_config_file_path'],
                        self.config['transformed_bert_model'])
                shutil.copyfile(self.config['bert_config_file_path'], self.config['transformed_bert_config'])
        
        
        device = torch.device('cuda')
      
        
        model = BertPublic.from_pretrained(self.config['transformed_bert_dir'])
        
        '''
        from transformers import BertConfig
        bert_config = BertConfig(self.config['transformed_bert_config'])
        model = BertPublic(bert_config)
        model.bert.load_state_dict(torch.load(self.config['transformed_bert_model']))
        '''   
        model.zero_grad()
        model.to(device)
        
        #from torchsummary import summary
        #summary(model, input_size=[(512,), (512,), (512,)])
        
        return model, device
    
    def _train(self, tr_ind, val_ind, fold=0, bag_id=0):
        from torch.utils import data
        oof_ckpt_preds = []
        weights = []
               
        train_dataset = data.TensorDataset(torch.tensor(self.x_train['ids'][tr_ind]),
                                           torch.tensor(self.x_train['masks'][tr_ind]),
                                           torch.tensor(self.x_train['segments'][tr_ind]),
                                           torch.tensor(self.x_train['title_lens'][tr_ind]),
                                           torch.tensor(self.x_train['body_lens'][tr_ind]),
                                           torch.tensor(self.x_train['answer_lens'][tr_ind]),
                                           torch.tensor(self.x_train['tq_ids'][tr_ind]),
                                           torch.tensor(self.x_train['tq_masks'][tr_ind]),
                                           torch.tensor(self.x_train['tq_segments'][tr_ind]),
                                           torch.tensor(self.x_train['tq_title_lens'][tr_ind]),
                                           torch.tensor(self.x_train['tq_body_lens'][tr_ind]),
                                           torch.tensor(self.target[tr_ind]))
        
        valid_dataset = data.TensorDataset(torch.tensor(self.x_train['ids'][val_ind]),
                                           torch.tensor(self.x_train['masks'][val_ind]),
                                           torch.tensor(self.x_train['segments'][val_ind]),
                                           torch.tensor(self.x_train['title_lens'][val_ind]),
                                           torch.tensor(self.x_train['body_lens'][val_ind]),
                                           torch.tensor(self.x_train['answer_lens'][val_ind]),
                                           torch.tensor(self.x_train['tq_ids'][val_ind]),
                                           torch.tensor(self.x_train['tq_masks'][val_ind]),
                                           torch.tensor(self.x_train['tq_segments'][val_ind]),
                                           torch.tensor(self.x_train['tq_title_lens'][val_ind]),
                                           torch.tensor(self.x_train['tq_body_lens'][val_ind]),
                                           torch.tensor(self.target[val_ind]))
        
        
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config['batch_size'], shuffle=True, 
                                  pin_memory=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=self.config['pred_batch_size'], shuffle=False, 
                                  pin_memory=True)
        
        '''
        train_loader = DataLoader(TextDataset(self.x_train['ids'], 
                                              self.x_train['masks'], 
                                              self.x_train['segments'], 
                                              self.meta_train, 
                                              tr_ind, self.target),
                                  batch_size=self.config['batch_size'], shuffle=True, 
                                  pin_memory=True)
        valid_loader = DataLoader(TextDataset(self.x_train['ids'], 
                                              self.x_train['masks'], 
                                              self.x_train['segments'], 
                                              self.meta_train,
                                              val_ind, self.target),
                                  batch_size=self.config['pred_batch_size'], shuffle=False, 
                                  pin_memory=True)
        '''
        #########
        # prepare model, optimizers, lr schedulers
        #########
        model, device = self.prepare_model()
        
        steps_per_epoch = int(np.ceil(len(train_loader)/self.config['accum_iters']))
        num_train_optimization_steps = self.config['epochs']*steps_per_epoch
        
        optimizer = torch.optim.Adam(model.parameters(), self.config['max_lr'])
        '''
        optimizer = transformers.optimization.AdamW(model.parameters(),
                                                    lr=self.config['max_lr'],
                                                    weight_decay=0.0,
                                                    correct_bias=False)
        '''
        '''
        scheduler = transformers.optimization.get_cosine_with_hard_restarts_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=int(steps_per_epoch*self.config['warmup_ratio']),
                    num_training_steps=num_train_optimization_steps,
                    num_cycles=self.config['epochs']
                )
        
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(
                        optimizer=optimizer,
                        num_warmup_steps=int(steps_per_epoch*self.config['warmup_ratio']),
                        num_training_steps=num_train_optimization_steps,
                    )
        '''
        scheduler = transformers.optimization.get_constant_schedule(
                        optimizer=optimizer,
                    )
        #model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
        training_started = False
        
        for epoch in range(0, self.config['epochs']):
            torch.manual_seed(self.config['seed']+epoch)
            start_time = time.time()
            
            model_path = self.config['model_path_format'].format(self.config['version'], bag_id, fold, epoch)
            avg_loss = 0.
            
            if DEV_MODE or not os.path.isfile(model_path): # not trained before
                
                print('training_started:', training_started)
                
                # amp initialize after start training
                if not training_started:
                    model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
                    training_started = True
                    
                #########
                # Training
                #########
                
                for param in model.parameters():
                    param.requires_grad=True
                model=model.train()
                
                tk_trn = tqdm(enumerate(train_loader),total=len(train_loader))
                lossf=None
                optimizer.zero_grad() 
                            
                for i, (ids, masks, segments, title_len, body_len, answer_len, 
                        tq_ids, tq_masks, tq_segments, tq_title_lens, tq_body_lens, y_batch) in tk_trn:
                    ids = ids.long().cuda()
                    masks = masks.cuda()
                    segments = segments.long().cuda()
                    title_len = title_len.long().cuda()
                    body_len = body_len.long().cuda()
                    answer_len = answer_len.long().cuda()
                    
                    tq_ids = tq_ids.long().cuda()
                    tq_masks = tq_masks.cuda()
                    tq_segments = tq_segments.long().cuda()
                    tq_title_lens = tq_title_lens.long().cuda()
                    tq_body_lens = tq_body_lens.long().cuda()
                    
                    y_batch = y_batch.cuda()
                    
                    y_pred = model(ids, masks, segments, title_len, body_len, answer_len,
                                   tq_ids, tq_masks, tq_segments, tq_title_lens, tq_body_lens)
                   
                    loss = self.custom_loss(y_pred.double(), y_batch, device)
                    #loss.backward()
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    
                    if (i+1) % self.config['accum_iters'] == 0:             # Wait for several backward steps
                        optimizer.step()                            # Now we can do an optimizer step
                        optimizer.zero_grad()
                        
                    if lossf:
                        lossf = 0.98*lossf+0.02*loss.item()
                    else:
                        lossf = loss.item()
                    
                    tk_trn.set_postfix(loss = lossf)
                    avg_loss += loss.item() / len(train_loader)
                    scheduler.step()
                    
                print('Loss: {:.6f}'.format(avg_loss))
            else:
                model.load_state_dict(torch.load(model_path))
                
                # skip through these steps
                for i in range(len(train_loader)):
                    scheduler.step()
                
            #########
            # Valdiation
            #########
            
            for param in model.parameters():
                param.requires_grad=False
            model.eval()
            avg_val_loss = 0.
            preds = []
            original = []
            tk_val = tqdm(enumerate(valid_loader),total=len(valid_loader))
            for i, (ids, masks, segments, title_len, body_len, answer_len, 
                    tq_ids, tq_masks, tq_segments, tq_title_lens, tq_body_lens, y_batch) in tk_val:
                ids = ids.long().cuda()
                masks = masks.cuda()
                segments = segments.long().cuda()
                title_len = title_len.long().cuda()
                body_len = body_len.long().cuda()
                answer_len = answer_len.long().cuda()
                
                tq_ids = tq_ids.long().cuda()
                tq_masks = tq_masks.cuda()
                tq_segments = tq_segments.long().cuda()
                tq_title_lens = tq_title_lens.long().cuda()
                tq_body_lens = tq_body_lens.long().cuda()
                y_batch = y_batch.cuda()
                
                y_pred = model(ids, masks, segments, title_len, body_len, answer_len,
                               tq_ids, tq_masks, tq_segments, tq_title_lens, tq_body_lens)
                y_pred = y_pred.detach()
                
                loss = self.custom_loss(y_pred.double(), y_batch, device)
                avg_val_loss += loss.item() / len(valid_loader)
                preds.append(torch.sigmoid(y_pred).cpu().numpy())
                original.append(y_batch.cpu().numpy())
                
            preds = np.concatenate(preds)
            original = np.concatenate(original)
            
            score = calc_spearmanr_metric(preds, original)
            
            #preds = rank_gauss_norm(preds)
            #print(preds.min(), preds.max())
            
            oof_ckpt_preds.append(preds)
            weights.append(1.**epoch)
            
            oof_pred = np.average(oof_ckpt_preds, axis=0)
            score_avg = calc_spearmanr_metric(oof_pred, original)
            
            elapsed_time = time.time() - start_time
            print('Epoch {}/{}   loss={:.4f}   val_loss={:.4f}   spearman={:.4f}   spearman avg={:.4f}   time={:.2f}s'.format(
                epoch + 1, self.config['epochs'], avg_loss, avg_val_loss, score, score_avg, elapsed_time))
    
            if DEV_MODE or not os.path.isfile(model_path): # not saved before
                torch.save(model.state_dict(), model_path)
        
        oof_ckpt_preds = oof_ckpt_preds[self.config['model_range_start']:self.config['model_range_end']]
        weights = weights[self.config['model_range_start']:self.config['model_range_end']]
            
        oof_pred = np.average(oof_ckpt_preds, weights=weights, axis=0)
        
        del train_loader, valid_loader, optimizer, model, scheduler
        gc.collect()
        torch.cuda.empty_cache() 
        return oof_pred
    
    def train_model(self):
        def bce(y_true, y_pred, eps=1e-15):
            y_pred = np.clip(y_pred, eps, 1-eps)
            return np.mean(-(y_true*np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
        
        if not SUBMIT_MODE:
            
            if not TRAIN_FINAL_MODEL:
                scores = []
                fold_num = self.config['fold_num']
                
                splits = list(KFold(n_splits=self.config['fold_num'], random_state=self.config['seed'], shuffle=True) \
                              .split(self.target, self.target))
                '''
                train_question_body = pd.read_csv(TRAIN_PATH)['question_body'].values
                splits = list(GroupKFold(n_splits=self.config['fold_num']) \
                              .split(train_question_body, groups=train_question_body))
                '''
                oof_preds = np.zeros(self.target.shape)
                for bag in range(self.config['bag_size']):
                    for fold in np.arange(fold_num):
                        
                        #if fold <= 2: 
                        #    continue
                        
                        tr_ind, val_ind = splits[fold]
                        oof_pred = self._train(tr_ind, val_ind, fold=fold, bag_id=bag)
                        oof_preds[val_ind] += oof_pred/self.config['bag_size']
                        score = calc_spearmanr_metric(oof_pred, self.target[val_ind])
                        scores.append(score)
                        print('Bag {} Fold {} score: {:.5f}'.format(bag, fold, score))
                
                overall_score = bce(self.target, oof_preds)
                overall_metric = calc_spearmanr_metric(oof_preds, self.target)
                print('overall bce = {:.5f} spearmanr = {:.5f}'.format(overall_score, overall_metric))
                print('score details:', scores)
                
                return oof_preds
            else:
                for bag in range(self.config['bag_size']):
                    tr_ind, val_ind = np.arange(self.target.shape[0]), np.arange(self.target.shape[0])
                    self._train(tr_ind, val_ind, fold='all', bag_id=bag)
                return [0]
        else:
            print('SUBMIT MODE = TRUE WILL NOT TRAIN MODEL')
            
    def inference_model(self):
        from torch.utils import data
        model, device = self.prepare_model()
        for param in model.parameters():
            param.requires_grad=False
        model=model.eval()
        
        test_dataset = data.TensorDataset(torch.tensor(self.x_test['ids']),
                                           torch.tensor(self.x_test['masks']),
                                           torch.tensor(self.x_test['segments']),
                                           torch.tensor(self.x_test['title_lens']),
                                           torch.tensor(self.x_test['body_lens']),
                                           torch.tensor(self.x_test['answer_lens']),
                                           torch.tensor(self.x_test['tq_ids']),
                                           torch.tensor(self.x_test['tq_masks']),
                                           torch.tensor(self.x_test['tq_segments']),
                                           torch.tensor(self.x_test['tq_title_lens']),
                                           torch.tensor(self.x_test['tq_body_lens']))
        
        test_loader = DataLoader(test_dataset,
                                batch_size=self.config['pred_batch_size'], shuffle=False, 
                                pin_memory=True)
        
        oof_preds = np.zeros((len(self.x_test['ids']), len(target_cols)))
        
        for bag in range(self.config['bag_size']):
            weights = []
            oof_ckpt_preds = []
            
            for epoch in range(self.config['model_range_start'], self.config['model_range_end']):
                torch.manual_seed(self.config['seed']+epoch)
                
                model_path = self.config['model_path_format'].format(self.config['version'], bag, 'all', epoch)
                assert os.path.isfile(model_path)
            
                print('Loading model:', model_path)
                model.load_state_dict(torch.load(model_path))
                
                preds = []
                tk_tst = tqdm(enumerate(test_loader),total=len(test_loader))
                for i, (ids, masks, segments, title_len, body_len, answer_len, 
                        tq_ids, tq_masks, tq_segments, tq_title_lens, tq_body_lens) in tk_tst:
                    ids = ids.long().cuda()
                    masks = masks.cuda()
                    segments = segments.long().cuda()
                    title_len = title_len.long().cuda()
                    body_len = body_len.long().cuda()
                    answer_len = answer_len.long().cuda()
                    
                    tq_ids = tq_ids.long().cuda()
                    tq_masks = tq_masks.cuda()
                    tq_segments = tq_segments.long().cuda()
                    tq_title_lens = tq_title_lens.long().cuda()
                    tq_body_lens = tq_body_lens.long().cuda()
                    
                    y_pred = model(ids, masks, segments, title_len, body_len, answer_len,
                                   tq_ids, tq_masks, tq_segments, tq_title_lens, tq_body_lens)
                    y_pred = y_pred.detach()
                    
                    preds.append(torch.sigmoid(y_pred).cpu().numpy())
                preds = np.concatenate(preds)
                
                oof_ckpt_preds.append(preds)
                weights.append(1.**epoch)
        
            oof_pred = np.average(oof_ckpt_preds, weights=weights, axis=0)
            oof_preds += oof_pred/(self.config['bag_size'])
                
        del test_loader, model
        gc.collect()
        return oof_preds
        
def verify_submission(test_oof_path, submission):
    if (pd.read_pickle(test_oof_path) != submission[target_cols].values).sum() == 0:
        print('Submission Sanity Check Pass')
    else:
        print('Submission Sanity Check Fail')

def pipe_execute(pname, param, pipe, val_pred_queue, pred_queue):
    val_preds = []
    preds = []
    with timer('{}'.format(pname)):
        pipe_ = pipe(*param)
        with timer('{} Preprocessing'.format(pname)):
            pipe_.preprocess()
        
        if not SUBMIT_MODE:
            with timer('{} Training'.format(pname)):
                val_pred = pipe_.train_model()
                val_preds += [val_pred]
                
        with timer('{} Inferencing'.format(pname)):
            pred = pipe_.inference_model()
            preds += [pred]
                
        with timer('{} Closing'.format(pname)):
            pipe_.close()
            
        del pipe_; gc.collect()
    
    val_pred_queue.put(val_preds)
    pred_queue.put(preds)
    
if __name__ == '__main__':
    '''
    # enable multi-processing in Spyder
    if not KAGGLE_KERNEL:
        __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    '''
    start_time = time.time()
    
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(K.tensorflow_backend._get_available_gpus())
    
    # make sure repeatable
    seed_everything(SEED)
    
    ##### RNN Configuration
    # cv: 0.39591, lb: 0.
    rnn_pipeline_name = 'rnn_pipeline'
    rnn_model_folder = MODEL_DIR+'rnn/'
    rnn_config = {
            'fold_num': 5,
            'seed': SEED,
            'embed_paths': [FAST_EMBEDDING_BIN_PATH, CRAWL_EMBEDDING_PATH, 
                            GLOVE_EMBEDDING_PATH, PARA_EMBEDDING_PATH],
            'embed_path_tmp': 'fasttext_glove_subwords_para.pkl',
            'model_path_format': rnn_model_folder+'rnn_v19_dev_weights_bg_{}_fold_{}_ep_{}.hdf5',
            'version': 1,
            'hidden_size_title': 32,
            'hidden_size': 256,
            'dropout_title': 0.1,
            'dropout': 0.6,
            'batch_size': 64,
            'pred_batch_size': 128,
            'bag_size': 1,
            'epochs': 8,
            'max_lr': 0.0015,
            'min_lr': 0.00015,
            'warmup_ratio': 0.1,
            'concat': False,
            }
    rnn_param = [TRAIN_PATH, TEST_PATH, rnn_config]
    rnn_pipeline = RNN_Pipeline 
    
    ##### RNN + PRE-FEATS Configuration
    # cv: 0.40681, lb: 0.383
    rnn_pre_pipeline_name = 'rnn_pre_pipeline'
    rnn_model_folder = MODEL_DIR+'rnn_use/'
    rnn_pre_config = {
            'fold_num': 5,
            'seed': SEED,
            'embed_paths': [FAST_EMBEDDING_BIN_PATH, CRAWL_EMBEDDING_PATH, 
                            GLOVE_EMBEDDING_PATH, PARA_EMBEDDING_PATH],
            'embed_path_tmp': 'fasttext_glove_subwords_para.pkl',
            'model_path_format': rnn_model_folder+'rnn_use_v19_dev_weights_bg_{}_fold_{}_ep_{}.hdf5',
            'hidden_size_title': 32,
            'hidden_size': 256,
            'dropout_title': 0.6,
            'dropout': 0.6,
            'batch_size': 64,
            'pred_batch_size': 128,
            'bag_size': 1,
            'epochs': 8,
            'max_lr': 0.0015,
            'min_lr': 0.00015,
            'warmup_ratio': 0.1,
            }
    rnn_pre_param = [TRAIN_PATH, TEST_PATH, rnn_pre_config]
    rnn_pre_pipeline = RNN_Pretrain_Features_Pipeline
    
    ##### Pretrain Embedding + NN Configuration
    # cv: 0.39379, lb: 0.374
    ptembed_mlp_pipeline_name = 'pretrained_embedding_mlp_pipeline'
    ptembed_mlp_model_folder = MODEL_DIR+'ptembed_mlp/'
    ptembed_mlp_config = {
            'fold_num': 5,
            'seed': SEED,
            'model_path_format': ptembed_mlp_model_folder+'ptembed_v19_dev_weights_bg_{}_fold_{}_ep_{}.hdf5',
            'version': 1,
            'hidden_size': 512,
            'dropout': 0.2,
            'batch_size': 64,
            'pred_batch_size': 256,
            'bag_size': 1,
            'epochs': 8,
            'max_lr': 0.0015,
            'min_lr': 0.00015,
            'warmup_ratio': 0.1,
            }
    ptembed_mlp_param = [TRAIN_PATH, TEST_PATH, ptembed_mlp_config]
    ptembed_mlp_pipeline = Pretrain_Features_Pipelin_Pytorch 
    
    ##### BERT EMBED UNCASED Base Configuration
    # cv: 0.405, lb: 0.385
    uc_base_bert_embed_pipeline_name = 'bert_embed_pipeline'
    UC_BASE_BERT_PRETRAINED_DIR = INPUT_PATH+'pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
    UC_BASE_LARGE_BERT_TRANSFORMED_DIR = MODEL_DIR+'pp_bert_uncased/'
    uc_base_bert_embed_config = {
            'fold_num': 5,
            'seed': SEED,
            'version': 19, # v1: bs=16, v2: bs=512, v3: bs=64, v4: cased, v5: new schd, v6: 2 epoch, min=1e-5, v7: more dense layers
            'batch_size': 16,
            'pred_batch_size': 128,
            'bag_size': 1,
            'epochs': 4, #4
            'max_lr': 0.00003,
            'min_lr': 0.00001, #0, 0.000006
            'warmup_ratio': 0.1, #0.1, 0.1333
            'accum_iters': 1, #int(64/16), # try cumulate to 64
            'bert_pretrained_dir': UC_BASE_BERT_PRETRAINED_DIR,
            'bert_config_file_path': UC_BASE_BERT_PRETRAINED_DIR+'bert_config.json',
            'bert_ckpt_file_path': UC_BASE_BERT_PRETRAINED_DIR+'bert_model.ckpt',
            'bert_vocab_path': UC_BASE_BERT_PRETRAINED_DIR+'vocab.txt',
            'transformed_bert_dir':  UC_BASE_LARGE_BERT_TRANSFORMED_DIR,
            'transformed_bert_model': UC_BASE_LARGE_BERT_TRANSFORMED_DIR+'pytorch_model.bin',
            'transformed_bert_config': UC_BASE_LARGE_BERT_TRANSFORMED_DIR+'config.json',
            'model_path_format': UC_BASE_LARGE_BERT_TRANSFORMED_DIR+'bert_v{}_weights_bg_{}_fold_{}_ep_{}.bin',
            'do_lower': True,
            'per_layer_lr_decay': -1,
            'feature': 'embed',
            'model_range_start': 3,
            'model_range_end': 4
            }
    #(train, test, y_train_nn, y_aux_train_nn, y, y_identity, loss_weight, config)
    uc_base_bert_embed_param = [TRAIN_PATH, TEST_PATH, uc_base_bert_embed_config]
    uc_base_bert_embed_pipeline = BERT_Pipeline_Pytorch
    
    ##### BERT RNN UNCASED Base Configuration
    # cv: 0.41843, lb: .396
    uc_base_bert_rnn_pipeline_name = 'bert_rnn_pipeline'
    UC_BASE_BERT_PRETRAINED_DIR = INPUT_PATH+'pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
    UC_BASE_LARGE_BERT_TRANSFORMED_DIR = MODEL_DIR+'pp_bert_uncased/'
    bert_rnn_model_folder = MODEL_DIR+'bert_rnn/'
    uc_base_bert_rnn_config = {
            'fold_num': 5,
            'seed': SEED,
            'version': 19, # v1: bs=16, v2: bs=512, v3: bs=64, v4: cased, v5: new schd, v6: 2 epoch, min=1e-5, v7: more dense layers
            'hidden_size': 256,
            'dropout': 0.4,
            'embed_size': 768,
            'batch_size': 64,
            'pred_batch_size': 128,
            'bag_size': 1,
            'epochs': 8, #4
            'max_lr': 0.0015,
            'min_lr': 0.00015, #0, 0.000006
            'warmup_ratio': 0.1, #0.1, 0.1333
            'accum_iters': 1, #int(64/16), # try cumulate to 64
            'bert_pretrained_dir': UC_BASE_BERT_PRETRAINED_DIR,
            'bert_config_file_path': UC_BASE_BERT_PRETRAINED_DIR+'bert_config.json',
            'bert_ckpt_file_path': UC_BASE_BERT_PRETRAINED_DIR+'bert_model.ckpt',
            'bert_vocab_path': UC_BASE_BERT_PRETRAINED_DIR+'vocab.txt',
            'transformed_bert_dir':  UC_BASE_LARGE_BERT_TRANSFORMED_DIR,
            'transformed_bert_model': UC_BASE_LARGE_BERT_TRANSFORMED_DIR+'pytorch_model.bin',
            'transformed_bert_config': UC_BASE_LARGE_BERT_TRANSFORMED_DIR+'config.json',
            'model_path_format': bert_rnn_model_folder+'bert_rnn_v{}_dev_weights_bg_{}_fold_{}_ep_{}.hdf5',
            'do_lower': True,
            'per_layer_lr_decay': -1,
            'feature': 'embed',
            'model_range_start': 6,
            'model_range_end': 8
            }
    #(train, test, y_train_nn, y_aux_train_nn, y, y_identity, loss_weight, config)
    uc_base_bert_rnn_param = [TRAIN_PATH, TEST_PATH, uc_base_bert_rnn_config]
    uc_base_bert_rnn_pipeline = BERT_RNN_Pipeline_Pytorch
    
    ##### BERT RNN UNCASED Base Configuration
    # cv: 0.414, lb: 0.392 
    c_base_bert_rnn_pipeline_name = 'bert_cased_rnn_pipeline'
    C_BASE_BERT_PRETRAINED_DIR = INPUT_PATH+'pretrained-bert-including-scripts/cased_l-12_h-768_a-12/cased_L-12_H-768_A-12/'
    C_BASE_LARGE_BERT_TRANSFORMED_DIR = MODEL_DIR+'pp_bert_cased/'
    bert_rnn_model_folder = MODEL_DIR+'case_bert_rnn/'
    c_base_bert_rnn_config = {
            'fold_num': 5,
            'seed': SEED,
            'version': 19, # v1: bs=16, v2: bs=512, v3: bs=64, v4: cased, v5: new schd, v6: 2 epoch, min=1e-5, v7: more dense layers
            'hidden_size': 256,
            'dropout': 0.4,
            'embed_size': 768,
            'batch_size': 64,
            'pred_batch_size': 128,
            'bag_size': 1,
            'epochs': 8, #4
            'max_lr': 0.0015,
            'min_lr': 0.00015, #0, 0.000006
            'warmup_ratio': 0.1, #0.1, 0.1333
            'accum_iters': 1, #int(64/16), # try cumulate to 64
            'bert_pretrained_dir': C_BASE_BERT_PRETRAINED_DIR,
            'bert_config_file_path': C_BASE_BERT_PRETRAINED_DIR+'bert_config.json',
            'bert_ckpt_file_path': C_BASE_BERT_PRETRAINED_DIR+'bert_model.ckpt',
            'bert_vocab_path': C_BASE_BERT_PRETRAINED_DIR+'vocab.txt',
            'transformed_bert_dir':  C_BASE_LARGE_BERT_TRANSFORMED_DIR,
            'transformed_bert_model': C_BASE_LARGE_BERT_TRANSFORMED_DIR+'pytorch_model.bin',
            'transformed_bert_config': C_BASE_LARGE_BERT_TRANSFORMED_DIR+'config.json',
            'model_path_format': bert_rnn_model_folder+'bert_cased_rnn_v{}_dev_weights_bg_{}_fold_{}_ep_{}.hdf5',
            'do_lower': False,
            'per_layer_lr_decay': -1,
            'feature': 'embed',
            'model_range_start': 6,
            'model_range_end': 8
            }
    #(train, test, y_train_nn, y_aux_train_nn, y, y_identity, loss_weight, config)
    c_base_bert_rnn_param = [TRAIN_PATH, TEST_PATH, c_base_bert_rnn_config]
    c_base_bert_rnn_pipeline = BERT_RNN_Pipeline_Pytorch
    
    ##### BERT PUBLIC UNCASED Base Configuration
    uc_base_bert_pub_pipeline_name = 'bert_pub_pipeline'
    UC_BASE_BERT_PRETRAINED_DIR = INPUT_PATH+'pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
    UC_BASE_LARGE_BERT_TRANSFORMED_DIR = MODEL_DIR+'pp_bert_public/'
    uc_base_bert_pub_config = {
            'fold_num': 5,
            'seed': SEED,
            'version': '19_pub', # v1: bs=16, v2: bs=512, v3: bs=64, v4: cased, v5: new schd, v6: 2 epoch, min=1e-5, v7: more dense layers
            'batch_size': 8,
            'pred_batch_size': 32,
            'bag_size': 1,
            'epochs': 4, #4
            'max_lr': 0.00003,
            'min_lr': 0.00001, #0, 0.000006
            'warmup_ratio': 0.1, #0.1, 0.1333
            'accum_iters': 1, #int(64/16), # try cumulate to 64
            'bert_pretrained_dir': UC_BASE_BERT_PRETRAINED_DIR,
            'bert_config_file_path': UC_BASE_BERT_PRETRAINED_DIR+'bert_config.json',
            'bert_ckpt_file_path': UC_BASE_BERT_PRETRAINED_DIR+'bert_model.ckpt',
            'bert_vocab_path': UC_BASE_BERT_PRETRAINED_DIR+'vocab.txt',
            'transformed_bert_dir':  UC_BASE_LARGE_BERT_TRANSFORMED_DIR,
            'transformed_bert_model': UC_BASE_LARGE_BERT_TRANSFORMED_DIR+'pytorch_model.bin',
            'transformed_bert_config': UC_BASE_LARGE_BERT_TRANSFORMED_DIR+'config.json',
            'model_path_format': UC_BASE_LARGE_BERT_TRANSFORMED_DIR+'bert_v{}_weights_bg_{}_fold_{}_ep_{}.bin',
            'do_lower': True,
            'per_layer_lr_decay': -1,
            'feature': 'concat',
            'model_range_start': 0,
            'model_range_end': 4,
            'max_lens': maxlens
            }
    uc_base_bert_pub_param = [TRAIN_PATH, TEST_PATH, uc_base_bert_pub_config]
    uc_base_bert_pub_pipeline = BERT_Public_Pipeline_Pytorch
    
    ##### All Pipelines Configuration
    
    pipeline_names = [uc_base_bert_pub_pipeline_name]
    #[uc_base_bert_rnn_pipeline_name, rnn_pre_pipeline_name, rnn_pipeline_name, ptembed_mlp_pipeline_name, uc_base_bert_embed_pipeline_name]
    
    params = [uc_base_bert_pub_param]
    #[uc_base_bert_rnn_param, rnn_pre_param, rnn_param, ptembed_mlp_param, uc_base_bert_embed_param]
    
    pipelines = [uc_base_bert_pub_pipeline]
    #[uc_base_bert_rnn_pipeline, rnn_pre_pipeline, rnn_pipeline, ptembed_mlp_pipeline,uc_base_bert_embed_pipeline]
    
    train_oof_path = '../oof/use_v19_train_oof.pkl'
    test_oof_path = '../oof/use_v19_test_oof.pkl'
    val_preds = []
    preds = [] 
    from multiprocessing import Queue
    val_pred_queue = Queue()
    pred_queue = Queue()
    for pname, param, pipe in zip(pipeline_names, params, pipelines):
        pipe_execute(pname, param, pipe, val_pred_queue, pred_queue)
        val_preds += val_pred_queue.get()
        preds += pred_queue.get()
        
    if not SUBMIT_MODE:
        val_preds = np.concatenate(val_preds, axis=1)
        preds = np.concatenate(preds, axis=1)
        
        if GENERATE_OOF:
            pd.to_pickle(preds, test_oof_path)
            pd.to_pickle(val_preds, train_oof_path)        
    else:
        preds = np.concatenate(preds, axis=1)
        
    if SUBMIT_MODE:
        with timer('Generate Prediction'):
            submission = pd.read_csv(TEST_PATH, index_col='qa_id', usecols=['qa_id'])
            #submission.drop(text_cols, axis=1, inplace=True)
            
            ensemble_data = pd.read_pickle('../input/kh-gqa-models/ensemble_models.pkl')
                   
            # simple average :)
            for i, col in enumerate(target_cols):
                submission[col] = preds[:,i::len(target_cols)].mean(axis=1)
                #model = ensemble_data['models'][i]
                #features = ensemble_data['features'][i]
                #submission[col] = model.predict(preds[:,features])
                
            #submission[target_cols] = preds
            
            # clipping is necessary or we will get an error
            submission[target_cols] = np.clip(submission[target_cols].values, 0.00001, 0.999999)
            submission.to_csv('submission.csv')   
            #del submission; gc.collect()

    # show run-time
    run_time = time.time()-start_time
    print('{}h{}m{:.2f}s'.format(run_time//3600, (run_time%3600)//60, run_time%60))