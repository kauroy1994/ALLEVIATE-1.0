x = [["hello how are you"],["Hi I am fine"]]
y = [[["hello how are you"],["hello how are you"],["hello how are you"]],
     [["Hi I am fine"],["Hi I am fine"],["Hi I am fine"]]]

ALL_WORDS = ["hello","how","are","you","Hi","I","am","fine"]
ALL_SENTENCES = x

def word_to_index(word):
    return ALL_WORDS.index(word)

def sentence_to_index(sentence):
    return ALL_SENTENCES.index(sentence)

def word_to_tensor(word):
    one_hot_vector = [[float(0) for i in range(len(ALL_WORDS))]]
    one_hot_vector[0][word_to_index(word)] = float(1)
    return torch.tensor(one_hot_vector)

def sentence_to_tensor(sentence):
    one_hot_vector = [[float(0) for i in range(len(ALL_SENTENCES))]]
    one_hot_vector[0][sentence_to_index(sentence)] = float(1)
    return torch.tensor(one_hot_vector)



import torch
import torch.nn as nn

class RNN_1(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(RNN_1, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor),1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1,self.hidden_size)

class RNN_2(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(RNN_2, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor),1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1,self.hidden_size)



word_categories = ALL_WORDS
sentence_categories = ALL_SENTENCES

def word_category_from_output(output):
    category_idx = torch.argmax(output).item()
    return word_categories[category_idx]

n_hidden = 128
rnn1 = RNN_1(len(sentence_categories),n_hidden,len(word_categories))
rnn2 = RNN_2(len(word_categories),n_hidden,len(word_categories))

i1 = sentence_to_tensor(ALL_SENTENCES[0])
print (ALL_SENTENCES[0])
print (i1)
h1 = rnn1.init_hidden()
o1, nh1 = rnn1(i1,h1)
print (o1)
print (nh1)
print (word_category_from_output(o1))
i2 = word_to_tensor(word_category_from_output(o1))
h2 = rnn2.init_hidden()
o2,nh2 = rnn2(i2,h2)
print (o2)
