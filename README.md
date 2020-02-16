First, I want to thanks my teammates @brightertiger @khyeh0719 @arvissu @qbenrosb00, we have great works. And thanks google and kaggle to host the good competition. I will briefly summarize our methods


## Preprocessing
We tried some preprocessing before trained BERT model and found the CV and LB are similar to no preprocessing. For the clean code, we didn't do any preprocessing to train SoTA models. Moreover, We used head and tail part of the texts as model input.

## Input 
As other teams, We train two BERT models for question labels(21) and answer labels(9).
* Question : `[cls]+ title+[sep]+question+[sep]`
* Answer : `[cls]+title+[sep]+question+[sep]+answer[sep]`

## Modeling

We tried BERT with different architectures.
* Vanilla BERT: 0.35x LB
* [Customized BERT+head and tail part of the texts:](https://www.kaggle.com/m10515009/customizedbert-pytorch-version-training) 0.392 LB
* Separate two BERT for question and answer: 0.396 LB
* Separate two BERT for the question and answer with special tokens: CV 415 LB 0.405

We find out using two BERT and special token get the best result on LB.
Therefore we apply the method to train GPT2 and RoBERTa.

We also used BERT outputs as embedding to train the LSTM and got great results
CV: 0.418, LB: 0.396

The summary of models we tried
* BERT-base CV 415 LB 0.405
* RoBERTa CV 0.411 LB 0.398
* GPT2  CV: 0.418 LB: 0.396
* BERT-large
* BERT-large-wwm
* BERT-large-wwm-squad
* BERT-RNN  CV: 0.418 LB: 0.396
* Pretrained Embedding + NN
* RNN+USE
## Ensemble

The models we used for ensemble are BERT, RoBERTa, GPT2, BERT-RNN. We only used simple average for ensemble. 

## Postprocessing
We only used the threshold method for postprocessing and got ~0.004 improvement in LB. I don't like the postpocessing method. However, in this competition, **postprocessing is all you need**.


```
def postProcessing(x):

    x = np.where(x&gt;=0.9241, 1.0, x)
    x = np.where(x&lt;=0.0759, 0.0, x)
    
    return x
    
targets = ['question_conversational',
           'question_type_compare', 
           'question_type_consequence', 
           'question_type_definition', 
           'question_type_entity', 
           'question_type_choice']
           
sub.loc[:, targets] = postProcessing(sub.loc[:, targets].values)
```

## Not work for us
* BERT-large
* Pretained BERT-base by SQuAD2 dataset.
* Gradient accumulation 
