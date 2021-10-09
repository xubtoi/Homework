import json
import numpy as np

from sklearn.metrics import confusion_matrix

import torch
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm_notebook, trange

# load the training data
train_data = json.load(open("./genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humor, 3 - Crime Fiction
docid = train_data['docid'] # these are the ids of the books which each training example came from
# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open("./genre_test.json", "r"))
Xt = test_data['X']
# load pretrained model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

def get_input_X(X,max_seq_len=100):
    tokenizer_out = tokenizer(X,padding=True,truncation=True,max_length=max_seq_len,return_tensors='pt')
    indexed_tokens = tokenizer_out['input_ids']
    seq_masks = tokenizer_out['attention_mask']
    seq_segments = tokenizer_out['token_type_ids']

    t_seqs = torch.tensor(indexed_tokens, dtype=torch.long)
    t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
    t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
    return t_seqs, t_seq_masks, t_seq_segments

train_len = int(len(X)*0.85)

# get train input
t_seqs, t_seq_masks, t_seq_segments=get_input_X(X[:train_len],100)

t_labels = torch.tensor(Y[:train_len], dtype = torch.long)
train_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)
train_dataloder = DataLoader(dataset= train_data, batch_size = 1)

# get val input
t_seqs, t_seq_masks, t_seq_segments=get_input_X(X[train_len:],100)

t_labels = torch.tensor(Y[train_len:], dtype = torch.long)
val_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)
val_dataloder = DataLoader(dataset= val_data, batch_size = 1)

# define loss function optimizer
loss_function = nn.CrossEntropyLoss()
loss_function.cuda()
model.cuda()
model.train()
optimizer=torch.optim.Adam(model.parameters(), lr=0.00001)

# training
loss_collect = []
for i in trange(4, desc='Epoch'):
    for step, batch_data in enumerate(tqdm_notebook(train_dataloder, desc='Iteration')):
        batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data
        batch_seqs = batch_seqs.cuda()
        batch_seq_masks = batch_seq_masks.cuda()
        batch_seq_segments = batch_seq_segments.cuda()
        batch_labels = batch_labels.cuda()
        
        optimizer.zero_grad()
        logits = model(batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
        logits=logits['logits']
        
        loss = loss_function(logits, batch_labels)
        loss.backward()
        loss_collect.append(loss.data.cpu().numpy().item())
        print("\r%f" % loss, end='')
        optimizer.step()
# val the model
pred_labels=[]
with torch.no_grad():
    for step, batch_data in enumerate(tqdm_notebook(val_dataloder, desc='Iteration')):
        batch_seqs, batch_seq_masks, batch_seq_segments = batch_data
        batch_seqs = batch_seqs.cuda()
        batch_seq_masks = batch_seq_masks.cuda()
        batch_seq_segments = batch_seq_segments.cuda()
        
        logits = model(batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
        logits = torch.nn.functional.softmax(logits['logits'],dim=1)
        pred_labels.append(logits.data.cpu().numpy().argmax())

matrix = confusion_matrix(pred_labels,Y[train_len:]).T
print('confustion matrix: ',matrix)

# get test input
t_seqs, t_seq_masks, t_seq_segments=get_input_X(Xt,100)

test_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments)
test_dataloder = DataLoader(dataset= test_data,batch_size = 1)

# get test output
pred_labels=[]
with torch.no_grad():
    for step, batch_data in enumerate(tqdm_notebook(test_dataloder, desc='Iteration')):
        batch_seqs, batch_seq_masks, batch_seq_segments = batch_data
        batch_seqs = batch_seqs.cuda()
        batch_seq_masks = batch_seq_masks.cuda()
        batch_seq_segments = batch_seq_segments.cuda()
        
        logits = model(batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
        logits = torch.nn.functional.softmax(logits['logits'],dim=1)
        pred_labels.append(logits.data.cpu().numpy().argmax())

# save test output
fout = open("out.csv", "w")
fout.write("Id,Y\n")
for i, line in enumerate(pred_labels):
    fout.write("%d,%d\n" % (i, line))
fout.close()
