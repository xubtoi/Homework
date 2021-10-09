
# NOTE: This file contains is a very poor model which looks for manually 
# chosen keywords and if none are found it predicts randomly according
# to the class distribution in the training set

import json
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import trange
from tqdm.notebook import tqdm
from sklearn.utils.extmath import softmax

def Encode(X, Max_Length = 500):
    token_seq = tokenizer(X, padding = 'max_length', truncation = True, max_length = Max_Length, return_tensors = 'pt')

    ids_seq = torch.LongTensor(token_seq['input_ids'])
    ids_attention_masks = torch.LongTensor(token_seq['attention_mask'])
    ids_token_type_ids = torch.LongTensor(token_seq['token_type_ids'])
    return ids_seq, ids_attention_masks, ids_token_type_ids

# load the training data
train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humor, 3 - Crime Fiction
docid = train_data['docid'] # these are the ids of the books which each training example came from

# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open("genre_test.json", "r"))
Xt = test_data['X']

#fit the model on the training data

epochs = 4
batch_size = 32

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)

ids, attention_masks, token_type_ids = Encode(X)
labels = torch.LongTensor(Y)
Train_Data = Data.TensorDataset(ids, attention_masks, token_type_ids, labels)
Train_Loader = Data.DataLoader(dataset = Train_Data, batch_size = batch_size, shuffle = True)

mark = False

if torch.cuda.is_available():
    device = torch.device('cuda')
    mark = True
else:
    device = torch.device('cpu')    

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 4)
optimizer = AdamW(model.parameters(), lr = 2e-5)
loss_func = nn.CrossEntropyLoss()

if mark:
    model.cuda()
    loss_func.cuda()
model.train()

for e in trange(epochs, desc = 'Epoch'):
    for i, batch in enumerate(tqdm(Train_Loader, desc = 'Iteration')):
        if mark:
            batch = tuple(t.cuda() for t in batch)
        else:
            batch = tuple(t.to(device) for t in batch)
        optimizer.zero_grad()
        output = model(input_ids = batch[0], attention_mask = batch[1], token_type_ids = batch[2], labels = None)['logits']
        loss = loss_func(output, batch[3])       
        loss.backward()
        print("\r%f" % loss, end = ' ')
        optimizer.step()

# predict on the test data
ids, attention_masks, token_type_ids = Encode(Xt)
Pred_Data = Data.TensorDataset(ids, attention_masks, token_type_ids)
Pred_Loader = Data.DataLoader(Pred_Data, batch_size = 1, shuffle = True)

Y_test_pred = []
model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(Pred_Loader, desc = 'Iteration')):
        if mark:
            batch = tuple(t.cuda() for t in batch)
        else:
            batch = tuple(t.to(device) for t in batch)
        output = model(input_ids = batch[0], attention_mask = batch[1], token_type_ids = batch[2], labels = None)
        output = softmax(output[0].detach().cpu().numpy())
        Y_test_pred += output.argmax(axis = 1).tolist()

# write out the csv file
# first column is the id, it is the index to the list of test examples
# second column is the predction as an integer
fout = open("out.csv", "w")
fout.write("Id,Y\n")
for i, line in enumerate(Y_test_pred): # Y_test_pred is in the same order as the test data
    fout.write("%d,%d\n" % (i, line))
fout.close()