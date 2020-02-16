import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings('ignore')

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

def calc_spearmanr(y_pred, y_true):
    return np.nan_to_num(spearmanr(y_true, y_pred).correlation)
  
def calc_spearmanr_metric(y_pred, y_true):
    score = 0
    for i in range(30):
        score +=  calc_spearmanr(y_true[:, i], y_pred[:, i])/ 30
        # valid_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
    return score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
    efi_x = (efi_x-efi_x.min())/(efi_x.max()-efi_x.min())
    return efi_x

def find_range(y_true, y_pred):
    min_, max_ = 0., 1.
    y_pred = np.clip(y_pred, min_, max_)
    
    unq_vals = sorted(pd.Series(y_true).unique())
    start = min_
    range_dict = {}
    
    original_spearmanr = calc_spearmanr(y_pred, y_true)
    pbar = tqdm(enumerate(unq_vals))
    pbar.set_description('original spr = {:.4f}'.format(original_spearmanr))
    
    for i, v in pbar:
        y_pred_ = y_pred.copy()
    
        # restore the grouped result
        for k, rang in range_dict.items():
            filt = (y_pred_>=rang[0]) & (y_pred_<rang[1])
            y_pred_[filt] = k
        
        # find best left boundary for center v
        if i == 0:
            start = min_ - 0.001
        else:
            start = range_dict[unq_vals[i-1]][1]
            
        bst_spr = None
        for l_ in np.arange(v, start, -0.001):
            filt = (y_pred_>=l_) & (y_pred_<=v)
            y_tmp = y_pred_.copy()
            y_tmp[filt] = v
            spr = calc_spearmanr(y_tmp, y_true)
            if bst_spr is None or spr > bst_spr:
                bst_spr = spr
                range_dict[v] = [l_, v]
                pbar.set_description('original spr = {:.4f}, tuned spr = {:.4f}'.format(original_spearmanr, bst_spr))
                
        # find best right boundary for center v
        if i == len(unq_vals) - 1:
            end = max_
        else:
            end = unq_vals[i+1]
                 
        bst_spr = None
        l_ = range_dict[v][0]
        for r_ in np.arange(v, end, 0.001):
            filt = (y_pred_>=l_) & (y_pred_<=r_)
            y_tmp = y_pred_.copy()
            y_tmp[filt] = v
            spr = calc_spearmanr(y_tmp, y_true)
            if bst_spr is None or spr > bst_spr:
                bst_spr = spr
                range_dict[v] = [l_, r_]
                pbar.set_description('original spr = {:.4f}, tuned spr = {:.4f}'.format(original_spearmanr, bst_spr))
            
            
    # store into y_pred the grouped result
    '''
    for k, rang in range_dict.items():
        filt = (y_pred>=rang[0]) & (y_pred<rang[1])
        y_pred[filt] = k     
    '''
    #print(range_dict)       
    return range_dict

if __name__ == '__main__':
    
    target_values = pd.read_csv('../input/google-quest-challenge/train.csv')[target_cols].values
    
    oof_mapping = {
        #'old_bert': '../dean_oof/y_oof_bert.pkl',
        #'old_roberta': '../dean_oof/y_oof_roberta.pkl',
        'new_bert': '../dean_oof/y_oof.pkl',
        'new_gpt2': '../dean_oof/GPT2-oof-f.pkl',
        'new_roberta': '../dean_oof/y_oof_roberta_0209.pkl',
        #'new_bert_large': '../dean_oof/bert_large_wwm_oof.pkl',
        'bert_rnn_oof': '../oof/bert_rnn_v19_train_oof.pkl',
        #'rnn_use_oof': '../oof/rnn_use_v19_train_oof.pkl',
        #'use_oof': '../oof/use_v19_train_oof.pkl'
    }
    
    df = pd.DataFrame()
    for k, v in oof_mapping.items():
        if k in ['old_bert', 'old_roberta']:
            oof = sigmoid(pd.read_pickle(v))
        else:
            oof = pd.read_pickle(v)
        for i, t in enumerate(target_cols):
            df[k+'_'+t] = oof[:, i]
        print('{:20} spearmanr cv = {:.4f}'.format(k, calc_spearmanr_metric(oof, target_values)))
        
    # simple blending
    simple_blend_res = np.zeros((df.shape[0], len(target_cols)))
    for i, t in enumerate(target_cols):
        simple_blend_res[:, i] = df.values[:,i::len(target_cols)].mean(axis=1)
    print('{:20} spearmanr cv = {:.4f}'.format('simple blending', calc_spearmanr_metric(simple_blend_res, target_values)))
            
    # rank gauss normalization and blending
    rank_gauss_blend_res = np.zeros((df.shape[0], len(target_cols)))
    for i, t in enumerate(target_cols):
        vals = df.values[:,i::len(target_cols)]
        # doing rank gauss normalization
        for j in np.arange(vals.shape[1]):
            vals[:, j] = rank_gauss(vals[:, j])
        
        rank_gauss_blend_res[:, i] = vals.mean(axis=1)
        
    print('{:20} spearmanr cv = {:.4f}'.format('rank gauss blending', calc_spearmanr_metric(rank_gauss_blend_res, target_values)))
     
    
    # brute-force prediction grouping
    tuned_range_dict = {}
    for i, t in enumerate(target_cols):
        print('find range for {:20}'.format(t))
        range_dict = find_range(target_values[:, i], simple_blend_res[:, i])
        tuned_range_dict[t] = range_dict
    pd.to_pickle(tuned_range_dict, 'tuned_range_dict.pkl')

    tuned_range_dict = pd.read_pickle('tuned_range_dict.pkl')
    for i, t in enumerate(target_cols):      
        for k, rang in tuned_range_dict[t].items():
            filt = (simple_blend_res[:, i]>=rang[0]) & (simple_blend_res[:, i]<rang[1])
            simple_blend_res[filt, i] = k 
    
    print('{:20} spearmanr cv = {:.4f}\n\n'.format('simple blending th-tuned', calc_spearmanr_metric(simple_blend_res, target_values)))    
    
    
    '''
    # brute-force prediction grouping
    tuned_range_dict = {}
    for i, t in enumerate(target_cols):
        print('find range for {:20}'.format(t))
        rank_gauss_blend_res[:, i], range_dict = find_range(target_values[:, i], rank_gauss_blend_res[:, i])
        tuned_range_dict[t] = range_dict
        
    print('{:20} spearmanr cv = {:.4f}\n\n'.format('rank gauss th-tuned', calc_spearmanr_metric(rank_gauss_blend_res, target_values)))    
    '''