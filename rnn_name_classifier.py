# -*- coding: utf-8 -*-
'''
This module is used for classifying names using an RNN esc framework

I believe that minibatching this isn't too hard, but there will need to be some early stopping done at training/test time
'''

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string

import torch
import torch.nn as nn
from torch.autograd import Variable

import random
import math
import time

#plotting modules
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters
	)

# Read a file and split into lines
def readLines(filename):
	lines = open(filename, encoding='utf-8').read().strip().split('\n')
	return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
	category = filename.split('/')[-1].split('.')[0]
	all_categories.append(category)
	lines = readLines(filename)
	category_lines[category] = lines

n_categories = len(all_categories)

# print('# categories:', n_categories, all_categories)
# print(unicodeToAscii("O'Néàl"))

#helper functions

class Config:
	mb_size = 1 #minibatch size 
	hidden_size = 128 #size of the hidden state
	lr = 0.005

	num_iter = 50000
	print_every = 1000
	plot_every = 250



def timeSince(since):
	now = time.time()
	s = now - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)



def letterToIndex(letter):
	return all_letters.find(letter)

def letterToTensor(letter):
	tensor = torch.zeros(Config.mb_size, n_letters)
	tensor[0][all_letters.find(letter)] = 1
	return tensor

# Form a 3D tensor with shape:
# length of line x mb_size x num_letters
#assumes the length of everything is uniform
def lineToTensor(lines):
	assert(len(lines) == Config.mb_size) #ensure that we're minibatching properly
	tensor = torch.zeros(len(lines[0]), Config.mb_size, n_letters)

	for i, line in enumerate(lines):
		for j, letter in enumerate(line):
			tensor[j, i, letterToIndex(letter)] = 1

	return tensor

# print (lineToTensor(['jones']))

def randomChoice(l):
	return l[random.randint(0, len(l)-1)]

def randomTrainingExample():
	category = randomChoice(all_categories)
	line = randomChoice(category_lines[category])
	category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
	line_tensor = Variable(lineToTensor([line]))
	return category, line, category_tensor, line_tensor


class runClassifier:
	def __init__(self, model):
		self.prepData()

		self.model = model(Config.n_letters, Config.hidden_size, Config.output_size)
		self.train()

	def prepData(self):
		Config.n_letters = n_letters
		Config.output_size = n_categories

	#takes an output from the rnn and returns the most likely category
	def output_to_category(self, output):
		v_top, i_top = output.data.topk(1, dim = 1)
		category_i = i_top[0][0]
		return all_categories[category_i], category_i

	def train(self):
		def prepTrain():
			self.optimizer = nn.NLLLoss()

		prepTrain()

		currLoss = 0.
		total_loss = []

		start = time.time()

		for i in range(1, Config.num_iter + 1):
			category, line, category_tensor, line_tensor = randomTrainingExample()
			output, loss = self.runSample(line_tensor, category_tensor)

			currLoss += loss[0]


			if i % Config.print_every == 0:
				guess, guess_i = self.output_to_category(output)
				correct = "YES" if guess == category else "NO %s" %(category)
				print('%d %d%% (%s) %.4f %s / %s %s' % (i, i / Config.num_iter * 100., timeSince(start), currLoss, line, guess, correct))

			if i %Config.plot_every == 0:
				total_loss.append(currLoss/Config.plot_every)
				currLoss = 0.
		
		plt.figure()
		plt.plot(total_loss)
		plt.show()




	def runSample(self, line_tensor, category_tensor):

		self.model.zero_grad()
		hidden = self.model.init_hidden()

		for i in range(line_tensor.size()[0]):
			output, hidden = self.model(line_tensor[i], hidden)

		loss = self.optimizer(output, category_tensor)
		loss.backward()

		for p in self.model.parameters():
			p.data.add_(-Config.lr, p.grad.data)

		return output, loss.data





class rnnClassifier(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(rnnClassifier, self).__init__()

		self.hidden_size = hidden_size

		self.input_output = nn.Linear(input_size + hidden_size, output_size)
		self.input_hidden = nn.Linear(input_size + hidden_size, hidden_size)

		self.softmax = nn.LogSoftmax()

	def forward(self, input, hidden):
		combined = torch.cat([input, hidden], dim = 1)
		hidden = self.input_hidden(combined)
		output = self.input_output(combined)
		output = self.softmax(output)
		return output, hidden

	def init_hidden(self):
		return Variable(torch.zeros(Config.mb_size, self.hidden_size))


def main():
	runClassifier(rnnClassifier)

if __name__ == "__main__":
	main()




