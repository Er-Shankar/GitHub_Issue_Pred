from google.colab import drive
drive.mount('/content/gdrive/')

!pip install simpletransformers

import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import *

from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')
import gc
from scipy.special import softmax

from simpletransformers.classification.classification_model import ClassificationModel
from sklearn.metrics import accuracy_score

sample = pd.read_csv('/content/gdrive/My Drive/Machine Hack/Github Issue prediction/sample submission.csv')
train = pd.read_json('/content/gdrive/My Drive/Machine Hack/Github Issue prediction/embold_train.json')

train['text'] = train['title'] + ' ' + train['body']
train_data = train[['text', 'label']]
tr, val = train_test_split(train_data, test_size=0.2, random_state = 2020, stratify=train_data['label'])

def get_model(model_type, model_name, n_epochs = 2, train_batch_size = 32, eval_batch_size = 32, seq_len = 134, lr = 2e-5):
  model = ClassificationModel(model_type, model_name,num_labels=3, args={'train_batch_size':train_batch_size,
                                                                         "eval_batch_size": eval_batch_size,
                                                                         'reprocess_input_data': True,
                                                                         'overwrite_output_dir': True,
                                                                         'fp16': False,
                                                                         'do_lower_case': False,
                                                                         'num_train_epochs': n_epochs,
                                                                         'max_seq_length': seq_len,
                                                                         'regression': False,
                                                                         'manual_seed': 2,
                                                                         "learning_rate":lr,
                                                                         "save_eval_checkpoints": False,
                                                                         "save_model_every_epoch": False,})
  return model

model = get_model('roberta', 'roberta-base', n_epochs=2, train_batch_size = 32, eval_batch_size = 32, seq_len = 150, lr = 2e-5)
model.train_model(tr)

p = model.predict(val['text'].values)
accuracy_score(val['label'], p[0])

test = pd.read_json("/content/gdrive/My Drive/Machine Hack/Github Issue prediction/embold_test.json")
test['text'] = test['title'] + ' ' + test['body']
test_data = test[['text']]
p1 = model.predict(test_data['text'])
sub = pd.DataFrame(p1[0], columns = ['label'])
sub.to_csv('roberta_base_body_title_v1.csv', index=False)