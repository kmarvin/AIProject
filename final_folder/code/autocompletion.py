
# autocompletion project - schacherer, klaus, bek

# import libraries
import numpy as np
from easydict import EasyDict as edict
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
from matplotlib import pyplot as plt
import heapq
import random

## class definitions
class TextDataset(Dataset):
    ''' A text dataset class which implements the abstract class torch.utils.data.Dataset. '''

    def __init__(self, text, seq_len, offset, char_idx, idx_char):
        self.data, self.target = prepare_data(text, seq_len, offset, char_idx, idx_char)

    def __getitem__(self, index):
        ''' Get the data for one training sample (by index) '''
        return self.data[index, :, :], self.target[index]

    def __len__(self):
        ''' Get the number of training samples '''
        return self.data.shape[0]


# #### LSTM functions and classes

class LSTM_RNN(nn.Module):
    ''' Class defining a recurrent neural network for text autocompletion tasks. '''

    def __init__(self, no_classes):
        super(LSTM_RNN, self).__init__()

        self.lstm = nn.GRU(input_size=no_classes, hidden_size=args.hidden_size, num_layers=args.num_layers)
        self.linear = nn.Linear(in_features=args.hidden_size, out_features=no_classes)

        nn.init.normal(self.linear.weight, 0, 0.075)
        nn.init.normal(self.linear.bias, 0, 0.075)
        nn.init.xavier_normal(self.lstm.weight_hh_l0)
        nn.init.xavier_normal(self.lstm.weight_ih_l0)

        # LSTM needs hidden variables which is initialized in self.init_hidden(self)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        h0 = Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size))
        return h0

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)  # (h0, c0 are set to default values)
        linear_out = self.linear(lstm_out[-1])
        return linear_out, hidden

## routine definitions
def set_config(config_path = "config.txt", args = dict()): # simple config handler
    ''' Function that reads configuration parameters from a text file source. 
    Returns an edict containing all parameters and their respective values. '''
    
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
            elif '.' in value:
                value = float(value)
            else:
                value = int(value)
            args[arg] = value
    return edict(args)

# #### Data Processing functions and classes

def prepare_text(textsource):
    ''' Function that reads a text from a textfile with encoding utf8. 
    It removes all special characters, but keeps spaces. 
    in: 
        textsource: path of the textfile to read
    out:
        text: string containing the text in lower case and without any special characters. '''
    
    text = ''
    with open(textsource, encoding="utf8") as txtsource:
        for line in txtsource:
            line = line.strip().lower()
            line = ''.join(c for c in line if c.isalnum() or c == ' ')
            text += ' ' + line
    text = text[:64100]
    return text

def prepare_data(text, seq_len, offset, char_idx, idx_char):
    ''' Function that generates one-hot encoded training/test features and target vectors
    from the given text.
    in: 
        text: string containing the text from which the features should be generated from
        seq_len: sequence length of one training/test sample
        offset: offset by which two training/test samples should be spaced 
        char_idx: dictionary mapping unique chars to indices
        idx_char: dictionary mapping indices to unique chars 
        
    out:
        X: One-hot encoded vector of features
        y: vector of targets '''
    
    # Define training samples by splitting the text
    sentences = []
    next_chars = []
    for i in range(0, len(text) - seq_len, offset):
        sentences.append(text[i: i + seq_len])
        next_chars.append(text[i + seq_len])
    
    # Generate features and labels using one-hot encoding
    X = np.zeros((len(sentences), seq_len, len(chars)), dtype='f')
    y = np.zeros((len(sentences)))
    
    for i, sentence in enumerate(sentences):
        for j, char in enumerate(sentence):
            X[i, j, char_idx[char]] = 1
        y[i] = char_idx[next_chars[i]]
        
    return X, y

def train(model, epoch):
    ''' Training loop (one epoch) '''
    model.train()
    criterion = nn.CrossEntropyLoss() # use the cross-entropy loss
    total_loss = 0.0 # compute total loss over one epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.transpose(0, 1) #swap seq_len and batch_size
        
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        
        hidden = model.init_hidden()
        optimizer.zero_grad()
        loss = 0
    
        # Iterate over a sequence, predict every next character and accumulate the loss 
        for c in range(args.seq_len - 1):
            output, hidden = model(data[c, :, :].contiguous().view(1, -1, no_classes), hidden) 
            loss += criterion(output, decode(data[c+1, :, :])) # check how far away the output is from the original data
            
        output, hidden = model(data[args.seq_len-1, :, :].contiguous().view(1, -1, no_classes), hidden)
        loss += criterion(output, target.type(torch.LongTensor))
        loss.backward(retain_graph=True)
        optimizer.step()
        
        total_loss += loss.data[0]

    relative_loss = total_loss/float(len(train_loader)*args.seq_len)
    print('Mean loss over epoch %s: %s' %(epoch, relative_loss))
    return relative_loss # return the relative loss for later analysis

def evaluate(model, epoch):
    ''' Evaluation loop (one epoch)'''
    model.eval()
    criterion = nn.CrossEntropyLoss() # use the cross-entropy loss
    total_loss = 0.0 # compute total loss over one epoch
    correct = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        
        data = data.transpose(0, 1)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        
        hidden = model.init_hidden()
        optimizer.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output, target.type(torch.LongTensor)) 
        pred = output.data.max(1, keepdim=True)[1] # get index of max log prob
        correct += pred.eq(target.type(torch.LongTensor).data.view_as(pred)).cpu().sum() # check for true predictions

        total_loss += loss.data[0]
    
    model.train()
    relative_loss = total_loss/float(len(test_loader))
    accuracy = correct/(len(test_loader)*args.batch_size)
    print('Mean test loss over epoch %s: %s' %(epoch, relative_loss))#loss.data[0]))
    print('Accuracy: ' + str(accuracy) + '([{}%])'.format(accuracy*100))

    return relative_loss, accuracy # return the relative loss and accuracy for later analysis

# #### Prediction functions

def rnn_predict(model, testdata):
    ''' Prediction loop for ONE testdata array '''
    testdata = torch.from_numpy(testdata)
    model.eval()

    if args.cuda:
        testdata = testdata.cuda()

    testdata = testdata.type(torch.FloatTensor)
    testdata = Variable(testdata)
    hidden = model.init_hidden()
    prediction = model(testdata.unsqueeze(1), hidden)

    return prediction

def prepare_input(text):
    ''' Function to create an one-hot encoding for the given text '''
    X = np.zeros((len(text), no_classes)) 
    for t, char in enumerate(text):          
        X[t, char_idx[char]] = 1.
    return X

def sample(preds, top_n=1):
    ''' Function returning the element(s) of preds with the highest probability. '''
    preds = preds[-1].data.numpy()
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(len(preds), zip(preds, itertools.count()))

def predict_completion(model, text, topn=1, max_iterations=10, stop_on_space = True):
    ''' Function that iteratively predicts the following character until a space is predicted '''
    i = 0
    completion = ''
    next_char = '' 
    while next_char != ' ' and i < max_iterations:
        
        i += 1
        x = prepare_input(text)
        preds = rnn_predict(model, x) # make a prediction
        next_chars = sample(preds[0], top_n=topn) # find character with highest prob.
        text = text[1:] + idx_char[next_chars[0][1]]
        completion += idx_char[next_chars[0][1]]
        
        if stop_on_space:
            next_char = idx_char[next_chars[0][1]]
            if next_char == ' ':
                completion = completion[:-1]
                break
                
    return completion

def predict_completions(model, text, n=3):
    ''' Function to give multiple possible completions of an input text. '''
    x = prepare_input(text)
    preds = model.rnn_predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [idx_char[idx] + predict_completion(text[1:] + idx_char[idx]) for idx in next_indices]


def get_testcases(text, seq_lengths = (10,100), offset = 1, num_cases = 50):
    ''' Function to generate test instances of different length from a given text. '''
    validation_sentences = []
    endings = []
    for i in range(0, len(text) - seq_lengths[1], offset):
        seq_len = random.randint(seq_lengths[0], seq_lengths[1])
        validation_sentences.append(text[i: i + seq_len])
        
        ending = ''
        while text[i + seq_len] != ' ':
            ending += text[i + seq_len]
            seq_len += 1
        endings.append(ending)    
    
    idx = np.random.choice(np.arange(len(validation_sentences)), num_cases, replace=True)
    testcases = [validation_sentences[i] for i in idx]
    test_endings = [endings[i] for i in idx]
    
    return testcases, test_endings

# #### Further functions

def findOnes(sample):
    ''' Helper function to find the 1 in a one-hot encoded vector. '''
    arr = sample.data.numpy()
    i = 0
    flag = False
    for k in np.nditer(arr):
        if k == 1:
            flag = True
            break
        i += 1
    if flag == True:
        return i
    else:
        return 0

def decode(data):
    ''' Function to find the index of one-hot encoded vector where the 1 is located. '''
    class_tensor = torch.LongTensor(args.batch_size)
    for i in range(args.batch_size):
        class_tensor[i] = findOnes(data[i])
    
    return Variable(class_tensor)

def lfactor(num):
    ''' Function that returns the largest factor of number that isn't the number itself '''
    
    for i in range(num - 1, 0, -1): # go backwards from num - 1 to 1
        if num % i == 0:            # if a number divides evenly
            return i                # it's the largest factor

def plotLineData(header, yLabel, firstData, firstLabel, firstColor='b', xLabel='epoch'):
    ''' Plot function '''
    
    plt.plot(firstData, color=firstColor)
    plt.title(header)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.legend([firstLabel], loc='upper right')
    plt.savefig(str(header) + '.png')
    plt.show()

# ### Main code

# #### Data Preprocessing, training and testing

if __name__ == '__main__':
    # Load configurations
    config_path = 'config.txt'
    args = {}
    args = set_config(config_path, args)

    ''' Use the whole text to generate char indices map and indices char map '''
    text = prepare_text('./nietzsche_eng_edit.txt')
    chars = sorted(list(set(text))) # get all the unique characters appearing in the text
    char_idx = dict((c, i) for i, c in enumerate(chars))
    idx_char = dict((i, c) for i, c in enumerate(chars))
    no_classes = len(chars) # the nr. of unique characters corresponds to the nr. of classes

    ''' Set further parameter input shape defined as seq_ * nr. of unique characters '''
    input_shape = (args.seq_len, no_classes)

    ''' Generate train and test loader from our data '''
    train_text = prepare_text('./nietzsche_eng_train.txt')
    train_set = TextDataset(train_text, args.seq_len, args.offset, char_idx, idx_char)
    train_loader = DataLoader(train_set, batch_size = args.batch_size, shuffle=True)

    test_text = prepare_text('./nietzsche_eng_test.txt')
    test_set = TextDataset(test_text, args.seq_len, args.offset, char_idx, idx_char)
    test_loader = DataLoader(test_set, batch_size = args.batch_size, shuffle=True)

    # Generate model or reload model (then outcomment the last line in this block)
    if args.pretrained == True:
        rnn = torch.load("pred.pt")
    else:
        rnn = LSTM_RNN(no_classes)
        if args.cuda:
            rnn.cuda()
        print(rnn)

        # Initialize the optimization algorithm
        optimizer = optim.Adam(rnn.parameters(), lr=args.lr)

        # Run training and store history
        history = dict()
        history['loss_train'] = []
        history['loss_test'] = []
        history['acc'] = []

        # train for (args.stop_epoch) epochs
        for epoch in range(args.end_epoch+1):
            loss_train = train(rnn, epoch)
            loss_test, accuracy = evaluate(rnn, epoch)
            history['loss_train'].append(loss_train)
            history['loss_test'].append(loss_test)
            history['acc'].append(accuracy)
            torch.save(rnn, 'GRU-autocompletion-epoch#{}.pt'.format(epoch))

        # Create plots of train loss, test loss and accuracy
        plotLineData("Training loss", "loss", history['loss_train'], "loss")
        plotLineData("Test loss", "loss", history['loss_test'], "loss")
        plotLineData("Accuracy", "acc", history['acc'], "acc")

    # #### Test Predictions

    print('Predict Words: \n')
    # test 50 prefixes
    testcases, test_endings = get_testcases(test_text, seq_lengths=(10, 100), num_cases=3)
    correct = 0
    for i, case in enumerate(testcases):
        completion = predict_completion(rnn, case.lower())
        print('Prefix: ' + str(case) + '\n Completion: ' + str(completion))
        if completion == test_endings[i]:
            correct += 1

    print('Test: #words matching source text: {}'.format(correct))

    # text_generation from preselected word prefixes
    testcases_words = ["justice", "responsibility", "freedom", "psychologist"]
    # load specific text generation parameters
    rnn = torch.load("text_gen.pt")
    for case in testcases_words:
        print('Generated text from seed ' + str(case) + ': ' + str(predict_completion(rnn, case.lower(), max_iterations=300, stop_on_space=False)))