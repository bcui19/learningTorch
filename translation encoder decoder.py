'''
This torch module is currently not coded to be compileable with cuda because my local machine can't run CUDA


'''


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import time

use_cuda = torch.cuda.is_available()

class Config:
	START_TOKEN = 0
	END_TOKEN = 1
	mb_size = 1
	dropout = 0.1

	max_length = 10
	teacher_forcing_ratio = 0.5 #this basically chooses if the net uses it's own output or the golden output

	lr = 0.01
	n_iters = 1000
	hidden_size = 256


	plot_every = 100
	print_every = 1000

	


#helper class to keep track of language
class Lang:
	def __init__(self, name):
		self.name = name
		self.word_to_index = {}
		self.word_to_count = {}
		self.index_to_word = {0: Config.START_TOKEN, 1: Config.END_TOKEN}
		self.n_words = 2 #count of start and end tokens

	def addSentence(self, sentence = None):
		assert(sentence is not None)
		for word in sentence.split(' '):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word_to_index:
			self.word_to_index[word] = self.n_words
			self.word_to_count[word] = 1
			self.index_to_word[self.n_words] = word
			self.n_words += 1
		else:
			self.word_to_count[word] += 1



# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
	)

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
	s = unicodeToAscii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s

def readLangs(lang1, lang2, reverse = False):
	print ("Reading pairs...")

	lines = open('data/%s-%s.txt' %(lang1, lang2), encoding='utf-8').read().strip().split('\n')

	pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

	if reverse:
		pairs = [list(reversed(p)) for p in pairs]
		input_lang = Lang(lang2)
		output_lang = Lang(lang1)
	else:
		input_lang = Lang(lang1)
		output_lang = Lang(lang2)

	return input_lang, output_lang, pairs

eng_prefixes = (
	"i am ", "i m ",
	"he is", "he s ",
	"she is", "she s",
	"you are", "you re ",
	"we are", "we re ",
	"they are", "they re "
)


def filterPair(p):
	return len(p[0].split(' ')) < Config.max_length and \
		len(p[1].split(' ')) < Config.max_length and \
		p[0].startswith(eng_prefixes)


def filterPairs(pairs):
	return [pair for pair in pairs if filterPair(pair)]
	# for i, pair in enumerate(pairs):
		# pair = (str(pair[0]), str(pair[1]))
		# if 
		# print (pair, filterPair(pair))
		# return [pair]

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


import random

class runTranslation:
	def __init__(self, encoder = None, decoder = None):
		self.input_lang, self.output_lang, self.pairs = self.prepData()

		self.encoder = encoder(self.input_lang.n_words, Config.hidden_size)
		self.decoder = decoder(Config.hidden_size, self.output_lang.n_words, n_layers = 1, dropout = Config.dropout)

		self.train(self.encoder , self.decoder)

	#defaults to english and french bc datasets
	def prepData(self, lang1 = 'eng', lang2 = 'fra', reverse = False):
		input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
		print("Read %s sentence pairs" % len(pairs))
		pairs = filterPairs(pairs)
		print("Trimmed to %s sentence pairs" % len(pairs))
		print("Counting words...")
		for pair in pairs:
			input_lang.addSentence(pair[0])
			output_lang.addSentence(pair[1])
		print("Counted words:")
		print(input_lang.name, input_lang.n_words)
		print(output_lang.name, output_lang.n_words)
		return input_lang, output_lang, pairs

	def indexesFromSentences(self, lang, sentence):
		return [lang.word_to_index[word] for word in sentence.split(' ')]

	def variableFromSentence(self, lang, sentence):
		indexes = self.indexesFromSentences(lang, sentence)
		indexes.append(Config.END_TOKEN)

		result = Variable(torch.LongTensor(indexes).view(-1, 1)) #I think this needs to be changed if minibatching were to be implemented
		return result

	def variablesFromPair(self, pair):
		input_variable = self.variableFromSentence(self.input_lang, pair[0])
		target_variable = self.variableFromSentence(self.output_lang, pair[1])
		return (input_variable, target_variable)

	def doIter_train(self, input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=Config.max_length):
		encoder_hidden = encoder.initHidden()

		encoder_optimizer.zero_grad()
		decoder_optimizer.zero_grad()

		input_length = input_variable.size()[0]
		target_length = target_variable.size()[0]

		encoder_outputs = Variable(torch.zeros(Config.max_length, encoder.hidden_size))

		loss = 0

		for i in range(input_length):
			encoder_output, encoder_hidden = encoder(input_variable[i], encoder_hidden)
			encoder_outputs[i] = encoder_output[0][0]

		decoder_input = Variable(torch.LongTensor([[Config.START_TOKEN]]))

		decoder_hidden = encoder_hidden

		use_teacher_forcing = True if random.random() < Config.teacher_forcing_ratio else False

		if use_teacher_forcing:
			#Feed target as net input
			for i in range(target_length):
				decoder_output, decoder_hidden, decoder_attention = decoder(
					decoder_input, decoder_hidden, encoder_output, encoder_outputs)
				loss += criterion(decoder_output, target_variable[i])
				decoder_input = target_variable[i] #forcing next value
		else:
			for i in range(target_length):
				decoder_output, decoder_hidden, decoder_attention = decoder(
					decoder_input, decoder_hidden, encoder_output, encoder_outputs)
				topv, topi = decoder_output.data.topk(1)
				ni = top1[0][0]

				decoder_input = variable(torch.LongTensor([[ni]]))

				loss += criterion(decoder_output, target_variable[i])
				if ni == Config.END_TOKEN:
					break
		loss.backward()
		encoder_optimizer.step()
		decoder_optimizer.step()

		return loss.data[0] / target_length

	def train(self, encoder, decoder):
		start = time.time()
		plot_losses = []
		print_total_loss = 0
		plot_total_loss = 0

		encoder_optimizer = optim.SGD(encoder.parameters(), lr = Config.lr)
		decoder_optimizer = optim.SGD(decoder.parameters(), lr = Config.lr)

		training_pairs = [self.variablesFromPair(random.choice(self.pairs)) for i in range(Config.n_iters)]

		criterion = nn.NLLLoss()

		for i in range(1, Config.n_iters + 1):
			training_pair = training_pairs[i - 1]
			input_variable = training_pair[0]
			target_variable = training_pair[1]

			loss = self.doIter_train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

			print_total_loss += loss
			plot_total_loss += loss

			if i % Config.print_every == 0:
				print_loss_avg = print_loss_total / print_every
				print_loss_total = 0.
				print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iters),
										 i, i / Config.n_iters * 100, print_loss_avg))

			if i % Config.plot_every == 0:
				plot_loss_avg = plot_loss_total / plot_every
				plot_losses.append(plot_loss_avg)
				plot_loss_total = 0

		showPlot(plot_losses)






class rnnEncoder(nn.Module):
	def __init__(self, input_size, hidden_size, n_layers = 1):
		super(rnnEncoder, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size)

	def forward(self, input, hidden):
		embedded = self.embedding(input).view(1, Config.mb_size, -1)
		output = embedded
		for i in range(self.n_layers):
			output, hidden = self.gru(output, hidden) #forwards pass through a gru cell 
		return output, hidden

	def initHidden(self):
		result = Variable(torch.zeros(1, Config.mb_size, self.hidden_size))
		return result


class rnnDecoder(nn.Module):
	def __init__(self, hidden_size, output_size, n_layers = 1):
		super(rnnDecoder, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(output_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size) #input, output size
		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax()

	def forward(self, input, hidden):
		output = self.embedding(input).view(1, Config.mb_size, -1)
		for i in range(self.n_layers):
			output = F.relu(output)
			output, hidden = self.gru(output, hidden)

		output = self.softmax(self.out(output[0])) #take the first entry and softmax 
		return output, hidden

	def initHidden(self):
		result = Variable(torch.zeros(1, Config.mb_size, self.hidden_size))
		return result

class attentionRNNDecoder(nn.Module):
	def __init__(self, hidden_size, output_size, n_layers = 1, dropout = Config.dropout, max_length = Config.max_length):
		super(attentionRNNDecoder, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout_p = dropout
		self.max_length = max_length

		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
		self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.dropout = nn.Dropout(self.dropout_p)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, hidden, encoder_output, encoder_outputs):
		embedded = self.embedding(input).view(1, Config.mb_size, -1)
		embedded = nn.Dropout(embedded)

		attn_weights = F.softmax(
			self.attn(torch.cat((embedded[0], hidden[0]), 1))) #mb_size x max_length 
		attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

		output = torch.cat((embedded[0], attn_applied[0]), 1)
		output = self.attn_combine(output).unsqueeze(0)

		for i in range(self.n_layers):
			output = F.relu(output)
			output, hidden = self.gru(output, hidden)

		output = F.log_softmax(self.out(output[0]))
		return output, hidden, attn_weights

	def initHidden(self):
		result = Variable(torch.zeros(1, Config.mb_size, self.hidden_size))
		return result






def main():
	runTranslation(encoder = rnnEncoder, decoder = attentionRNNDecoder)

if __name__ == "__main__":
	main()






