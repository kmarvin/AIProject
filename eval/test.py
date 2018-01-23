import numpy as np
import heapq


SEQUENCE_LENGTH = 20    # length of text
chars = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]    # number different chars (same as numbers entry in char_indices)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))  # array with one entry which have 20 lines, each 11 entrys
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.
    print(x)
    return x


def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)


def predict_completion(model, text):
    original_text = text
    generated = text
    completion = ''
    next_char = ''
    while next_char != ' ':
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char

    return completion


def predict_completions(model, text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]


test_data = ["adgj"]
prepare_input("acj")
'''
for text in test_data:
    seq = text[:40].lower()
    print(seq)
    print(predict_completions(seq, 5))
    print()
'''
