from flask import Flask, redirect, url_for, request, render_template, send_from_directory, jsonify
import torch
from easydict import EasyDict as edict
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import heapq
import itertools

def set_config(config_path = "../config.txt", args = dict()):
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

config_path = '../config.txt'
args = {}
args = set_config(config_path, args)

class LSTM_RNN(nn.Module):
    ''' Class defining a recurrent neural network for text autocompletion tasks. '''

    def __init__(self, no_classes):
        super(LSTM_RNN, self).__init__()

        self.lstm = nn.GRU(input_size = no_classes, hidden_size = args.hidden_size, num_layers = args.num_layers)
        self.linear = nn.Linear(in_features = args.hidden_size, out_features = no_classes)
        self.softmax = nn.Softplus()

        nn.init.normal( self.linear.weight, 0, 0.075)
        nn.init.normal(self.linear.bias, 0, 0.075)
        nn.init.xavier_normal(self.lstm.weight_hh_l0)
        nn.init.xavier_normal(self.lstm.weight_ih_l0)

        # LSTM needs hidden variables which is initialized in self.init_hidden(self)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        h0 = Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size))
        c0 = Variable(torch.zeros(args.num_layers, args.batch_size, args.hidden_size))
        return (h0)#,c0)#Variable(torch.zeros((args.num_layers, args.batch_size, args.hidden_size)))

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden) # (h0, c0 are set to default values)
        linear_out = self.linear(lstm_out[-1])
        return linear_out, hidden

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

text = prepare_text('../nietzsche_eng_edit.txt')
chars = sorted(list(set(text))) # get all the unique characters appearing in the text
char_idx = dict((c, i) for i, c in enumerate(chars))
idx_char = dict((i, c) for i, c in enumerate(chars))
no_classes = len(chars) # the nr. of unique characters corresponds to the nr. of classes

def load_model():
   return torch.load("../pred.pt")

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

def sample(preds):
    ''' Function returning the element(s) of preds with the highest probability. '''
    preds = preds[-1].data.numpy()
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(len(preds), zip(preds, itertools.count()))

def predict_completion(model, text, max_iterations=10, stop_on_space = True):
    ''' Function that iteratively predicts the following character until a space is predicted '''
    original_text = text
    processed = text


    i = 0
    completion = ''
    next_char = ''
    while next_char != ' ' and i < max_iterations:
        i += 1
        x = prepare_input(text)
        preds = rnn_predict(model, x) # make a prediction
        next_chars = sample(preds[0]) # find character with highest prob.
        text = text[1:] + idx_char[next_chars[0][1]]
        completion += idx_char[next_chars[0][1]]

        if stop_on_space:
            next_char = idx_char[next_chars[0][1]]
            if next_char == ' ':
                completion = completion[:-1]
                break
    return completion

def save_word_in_dict(dictionary, word, prob):
    '''
        saves a word in the word dictionary. if it exists just count the number up
    '''
    if word in dictionary:
        dictionary[word]["number"] += 1
        dictionary[word]["prob"] += prob
        if word[-1:] == ' ':
            dictionary[word]["finished"] = True
    else:
        finished = False
        if word == ' ':
            finished = True
        dictionary[word] = {"number": 1, "finished": finished, "prob": prob}
    return dictionary

def find_next_chars(model, different_words, word, text, number_suggestions, min_treshold, max_iterations=10):
    '''
        function to find the next chars given a text and a beginning of the current word
        if requested the function calculates more than one different words
    '''
    text += word
    text = text[:100]
    needed_number = different_words[word]["number"]
    x = prepare_input(text)
    preds = rnn_predict(model, x)
    next_chars = sample(preds[0])

    words = []
    probs = []
    probs_sum = 0
    different_words.pop(word, None) # delete key to add the new ones
    count_words = 1
    number_words = 0
    i = 0
    while number_words < needed_number and probs_sum < min_treshold and i < max_iterations:
        new_word = word + idx_char[next_chars[i][1]]
        words.append(new_word)

        probs_sum += next_chars[i][0]
        prob_format = "{0:.2f}".format(next_chars[i][0])
        probs.append(float(prob_format))
        i += 1

    result = []
    if(len(words) < needed_number):
        diff = 1 - np.sum(probs)
        probs[0] += diff
        result = np.random.choice(words, needed_number, p=probs)
    for word in result:
        different_words = save_word_in_dict(different_words, word, 0)

    return different_words

def predict_words(model, text, number_suggestions=1, min_treshold=0.90, max_iterations=20):
    ''' Function to give a number of fitting words '''
    text = text[:100]
    original_text = text
    different_words = {}

    # init words start letters
    x = prepare_input(text)
    preds = rnn_predict(model, x)
    next_chars = sample(preds[0])

    i = 0
    probs_sum = 0
    words = []
    probs = []
    while len(different_words) < number_suggestions and probs_sum < min_treshold:
        different_words = save_word_in_dict(different_words, idx_char[next_chars[i][1]], next_chars[i][0])
        words.append(idx_char[next_chars[i][1]])
        probs_sum += next_chars[i][0]
        prob_format = "{0:.2f}".format(next_chars[i][0])
        probs.append(float(prob_format))
        i += 1
    # status: words with one letter each, maybe to few

    result = []
    if(len(words) < number_suggestions):
        diff = 1 - np.sum(probs)
        probs[0] += diff
        result = np.random.choice(words, number_suggestions-len(words), p=probs)
    for word in result:
        different_words = save_word_in_dict(different_words, word, different_words[word]["prob"])

    # status: words with one letter each, maybe some letters multiple

    iter = 0
    while len(different_words) < number_suggestions and iter < max_iterations:
        for word in list(different_words):
            if different_words[word]["number"] > 1 and word != ' ':
                different_words = find_next_chars(model, different_words, word, original_text, number_suggestions, min_treshold)
        iter += 1
    # status: number_suggestions different word beginings in different_words

    # complete the not finished words
    words = []
    for word in different_words:
        if different_words[word]["finished"] == False:
            text = original_text + word
            text = text[:100]
            compl = predict_completion(model, text)
            full_word = word + compl
            words.append((different_words[word]["prob"], full_word))
        else:
            words.append((different_words[word]["prob"], word))

    # words with highest probability first
    words.sort(key=lambda tup: tup[0], reverse=True)
    format_words = list(map(lambda x: x[1], words))

    return format_words

# server implementation
app = Flask(__name__, template_folder='app', static_folder='app/static')

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/computeInput', methods=['POST'])
def generate():
    data = request.get_json()
    if "text" in data and "settings" in data and data["text"] != ' ':
        numberSuggestions = data["settings"]["numberSuggestions"]
        numberSuggestions = int(float(numberSuggestions))
        words = predict_words(model, data["text"].lower(), numberSuggestions)
        return jsonify(words)
    else:
        abort(500)

if __name__ == '__main__':
    app.run()
