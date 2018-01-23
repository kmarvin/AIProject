
# coding: utf-8

# In[281]:


import numpy as np
from easydict import EasyDict as edict
import load_config

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# #### Configuration / parameters to set

# In[282]:

def set_config(config_path = "config.txt", args = {}):
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


# In[283]:

config_path = 'config.txt'
args = set_config(config_path, args)
print(args)


# ### Data Processing functions and classes

# In[284]:

def prepare_text(textsource):
    text = ''
    with open(textsource) as txtsource:
        for line in txtsource:
            line = line.strip().lower()
            line = line.replace(',', '').replace('.', '')
            line = line.replace('»', '').replace('«', '')
            line = line.replace('"', '')
            text += ' ' + line
    text = text[:700] #### nachher wieder rauslöschen!!!
    return text
# Chevrons müssen noch weg


# In[285]:

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
        
    return X, y, no_classes


# In[286]:

class TextDataset(Dataset):
    ''' A text dataset class which implements the abstract class torch.utils.data.Dataset. '''
    def __init__(self, text, seq_len, offset):
        self.data, self.target, self.no_classes = prepare_data(text, seq_len, offset)
        
    def __getitem__(self, index):
        ''' Get the data for one training sample (by index) '''
        return self.data[index,:,:], self.target[index] 
    
    def __len__(self):
        ''' Get the number of training samples '''
        return self.data.shape[0]


# ### LSTM functions and classes

# In[287]:

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
        lstm_out, self.hidden = self.lstm(x, self.hidden) # (h0, c0 are set to default values)
        res = self.softmax(self.linear(lstm_out[-1])) # use only the output of the last layer of lstm
        return res


# In[288]:

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
    print('Total loss over epoch %s: %s' %(epoch, total_loss/len(train_loader.dataset)))


# ### Main code

# In[289]:

# test the functions defined above
textsource = './Brown_Leseprobe.txt'
text = prepare_text(textsource)
feat, true_pred, no_classes = prepare_data(text, args.seq_len, args.offset)
print(feat.shape)
print(no_classes)
print(text)


# In[290]:

# Generate train and test loader from our data
train_text = prepare_text('./Brown_Leseprobe.txt')
train_set = TextDataset(train_text, args.seq_len, args.offset)
train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle=True)
#dataiter = iter(train_loader)
#print(dataiter.next())

# set further parameters
no_classes = train_set.no_classes
input_shape = (args.seq_len, no_classes) # seq_len * nr. of unique characters 


# In[291]:

# Generate model
rnn = LSTM_RNN(input_shape)
if args.cuda:
    rnn.cuda()
print(rnn)


# In[294]:

# Initialize the optimization algorithm
optimizer = optim.RMSprop(rnn.parameters(), lr=0.01)


# In[ ]:

for epoch in range(5):
    train(rnn, epoch)


# In[ ]:




# In[49]:

rnn = nn.LSTM(10, 20, 2)
input = Variable(torch.randn(5, 3, 10))
h0 = Variable(torch.randn(2, 3, 20))
c0 = Variable(torch.randn(2, 3, 20))
output = rnn(input, (h0, c0))
print(type(output))


# In[50]:

output = Variable(torch.rand(1,10))
target = Variable(torch.LongTensor([1]))
print(output, target)
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
print(loss)


# In[ ]:



