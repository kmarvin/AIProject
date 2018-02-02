
# coding: utf-8

# In[280]:


import numpy as np
from easydict import EasyDict as edict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import heapq

# Import other python files


# #### Configuration / parameters to set

# In[281]:

def set_config(config_path = "config.txt", args = dict()):
    with open(config_path) as source:
        for line in source:
            line = line.strip()
            argLong, valueLong = line.split('=')
            arg = argLong.strip()
            value = valueLong.strip()
            if value == 'True':
                value = True
            elif value == 'False':
                value = False
            else:
                value = int(value)
            args[arg] = value
    return edict(args)


# In[282]:

config_path = 'config.txt'
args = {}
args = set_config(config_path, args)
print(args)


# ### Data Processing functions and classes

# In[389]:

def prepare_text(textsource):
    text = ''
    with open(textsource) as txtsource:
        for line in txtsource:
            line = line.strip().lower()
            line = line.replace(',', '').replace('.', '')
            line = line.replace('»', '').replace('«', '')
            line = line.replace('"', '')
            line = line.replace(u'\ufeff', '')
            text += ' ' + line
    text = text[:10000] #### nachher wieder rauslöschen!!!
    return text
# Chevrons müssen noch weg


# In[390]:

def prepare_data(text, seq_len, offset):
    # Get all the unique characters appearing in the text 
    chars = sorted(list(set(text)))
    char_idx = dict((c, i) for i, c in enumerate(chars))
    idx_char = dict((i, c) for i, c in enumerate(chars)) #### das brauchen wir später!!!
    no_classes = len(chars) # the nr. of unique characters corresponds to the nr. of classes
    
    # Define training samples by splitting the text
    sentences = []
    next_chars = []
    for i in range(0, len(text) - seq_len, offset):
        sentences.append(text[i: i + seq_len])
        next_chars.append(text[i + seq_len])
    print('nr training samples', len(sentences))
    
    # Generate features and labels using one-hot encoding
    X = np.zeros((len(sentences), seq_len, len(chars)), dtype='f')
    y = np.zeros((len(sentences)))
    
    for i, sentence in enumerate(sentences):
        for j, char in enumerate(sentence):
            X[i, j, char_idx[char]] = 1
        y[i] = char_idx[next_chars[i]]
        
    return X, y, char_idx, idx_char, no_classes


# In[391]:

class TextDataset(Dataset):
    ''' A text dataset class which implements the abstract class torch.utils.data.Dataset. '''
    def __init__(self, text, seq_len, offset):
        self.data, self.target, self.char_idx, self.idx_char, self.no_classes = prepare_data(text, seq_len, offset)
        
    def __getitem__(self, index):
        ''' Get the data for one training sample (by index) '''
        return self.data[index,:,:], self.target[index] 
    
    def __len__(self):
        ''' Get the number of training samples '''
        return self.data.shape[0]


# ### LSTM functions and classes

# In[392]:

class LSTM_RNN(nn.Module):
    
    def __init__(self, input_shape):
        super(LSTM_RNN, self).__init__()
        
        self.lstm = nn.LSTM(input_size = input_shape[0]*input_shape[1], hidden_size = args.hidden_size)
        self.linear = nn.Linear(in_features = args.hidden_size, out_features = input_shape[1])
        self.softmax = nn.Softmax()
        
        # LSTM needs hidden variable which is initialized in self.init_hidden(self)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        h0 = Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size))
        c0 = Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size))
        return (h0, c0)
    
    def forward(self, x):
        #x = x.type(torch.DoubleTensor)
        lstm_out, self.hidden = self.lstm(x, self.hidden) # (h0, c0 are set to default values)
        res = self.softmax(self.linear(lstm_out[-1])) # use only the output of the last layer of lstm
        return res


# In[393]:

# Training loop (one epoch)
def train(model, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss() # use the cross-entropy loss
    total_loss = 0.0 # compute total loss over one epoch
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(data.size(0), -1)
        if args.cuda:
            data, target = data.cuda(), target.cuda()   
        data, target = Variable(data), Variable(target)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.long()) # check how far away the output is from the original data
        loss.backward(retain_graph=True)
        optimizer.step()
        
        total_loss += loss.data[0]
    
    relative_loss = total_loss/len(train_loader.dataset)
    print('Relative loss over epoch %s: %s' %(epoch, relative_loss))
    return relative_loss # return the relative loss for later analysis
            


# In[394]:

# Prediction loop for ONE testdata tensor
def rnn_predict(model, testdata):
    ''' Note: testdata have to be submitted as a tensor'''
    testdata = torch.from_numpy(testdata)
    print("testdata:")
    print(testdata)
    model.eval()
    testdata = testdata.view(testdata.size(0), -1)
    if args.cuda:
        testdata = testdata.cuda()
    testdata = testdata.type(torch.FloatTensor)
    testdata = Variable(testdata)
    prediction = model(testdata)
    return prediction


# ### Other functions

# In[395]:

''' Function that returns the largest factor of number that isn't the number itself '''
def lfactor(num):
    for i in range(num - 1, 0, -1): # go backwards from num - 1 to 1
        if num % i == 0:            # if a number divides evenly
            return i                # it's the largest factor


# ### Marvins test functions

# In[396]:

# die funktion brauchen wir vllt gar nicht, je nachdem ob wir den test loader verwenden oder wie wir das auch immer machen
def prepare_input(text):
    X = np.zeros((1, args.seq_len, no_classes))  # array with one entry which have 20 lines, each 11 entrys
    for t, char in enumerate(text):
        X[0, t, char_idx[char]] = 1.
    return X

def sample(preds, top_n=1):
    print("test")
    preds = preds.data.numpy()[0]
    print(preds)
    print(preds.shape)
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)


def predict_completion(model, text):
    original_text = text
    generated = text
    completion = ''
    next_char = ''
    max_iterations = 100
    i = 0
    while next_char != ' ' and i < max_iterations:
        i += 1
        x = prepare_input(text)
        preds = rnn_predict(model, x)
        print(preds)
        next_index = sample(preds, top_n=1)[0]
        print(next_index)
        next_char = idx_char[next_index]
        print(next_char)
        text = text[1:] + next_char
        completion += next_char

    return completion


def predict_completions(model, text, n=3):
    x = prepare_input(text)
    preds = model.rnn_predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [idx_char[idx] + predict_completion(text[1:] + idx_char[idx]) for idx in next_indices]


# ### Main code

# In[397]:

# Generate train and test loader from our data
train_text = prepare_text('./Brown_Leseprobe.txt')
train_set = TextDataset(train_text, args.seq_len, args.offset)
train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle=True)

test_text = prepare_text('./Brown_Leseprobe_test.txt')
test_set = TextDataset(test_text, args.seq_len, args.offset)
test_loader = DataLoader(test_set, batch_size = args.batch_size, shuffle=True)

# set further parameters
char_idx = train_set.char_idx
idx_char = train_set.idx_char
no_classes = train_set.no_classes
input_shape = (args.seq_len, no_classes) # seq_len * nr. of unique characters 

# get len of data to determine the possible batch_size
args.batch_size = lfactor(len(train_set))
print(args.batch_size)


# In[386]:

# Generate model
rnn = LSTM_RNN(input_shape)
if args.cuda:
    rnn.cuda()
print(rnn)


# In[387]:

# Initialize the optimization algorithm
optimizer = optim.RMSprop(rnn.parameters(), lr=0.05)


# In[347]:

# Run training and store history
history = dict()
history['loss_train'] = []
history['loss_test'] = []

# wie wir die accuracy machen, weiß ich noch nicht...
#history['acc_train'] = []
#history['acc_test'] = []

for epoch in range(20):
    loss_train = train(rnn, epoch)        
    history['loss_train'].append(loss_train)      


# In[ ]:

# Try a prediction

#testdata = Variable(torch.from_numpy(test_set.data[0])) # get first element from the test set
#truth = test_set.target[0]
#print(testdata,truth)

#prediction = rnn(testdata)
## dann muss man hier noch auf die sizes achten, ach verdammt
#prepare_input("This is an example of input for our LSTM".lower(), train_set.data, char_idx)
#print(predict_completions(seq, 5))


# In[399]:


test = "hrend die historische Zahnrad "
predict_completion(rnn, test.lower())


# In[398]:

print(train_set.no_classes)
print(char_idx)
print(args.batch_size)


# In[ ]:



