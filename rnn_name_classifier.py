# -*- coding: utf-8 -*-


'''
This module allows us to read in files and train a model to probabilistically 
determine where the country of origin is 
'''

from __future__ import unicode_literals, print_function, division
from io import open
import glob

def findFiles(path): return glob.glob(path)

import unicodedata
import string

import torch
import torch.nn as nn
from torch.autograd import Variable

import time #checking time it takes to run  
import math

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters)

#read a file and split it into lines
#external helper function
def readLines(filename):
	lines = open(filename, encoding = 'utf-8').read().strip().split('\n')
	return [unicodeToAscii(line) for line in lines]	

import random

#get a random choice from a list
def randomChoice(l):
	return l[random.randint(0, len(l)-1)]

def getTime(since):
	start = time.time()

	s = start - since
	m = s//60 #floor

	s -= m * 60
	return '%dm %ds' %(m, s) #returns number of minutes and seconds since starting


class Config:
	dropout = 0.1 #dropout probability
	minibatch_size = 1
	hidden_size = 29
	lr = 0.0005

	n_categories = None
	end_index = None

	#configurations for displaying results
	n_iters = 100000
	print_every = 5000
	plot_every = 500




class runClassifier:
	def __init__(self, model = None):
		self.initializeData()

		self.model = model(n_letters, Config.hidden_size, self.n_categories)

		self.losses = []
		self.total_loss = 0

		self.train()

	def loadData(self):
		for filename in findFiles('data/names/*.txt'):
			category = filename.split('/')[-1].split('.')[0]
			self.all_categories.append(category)
			lines = readLines(filename)
			self.category_boundaries[category] = lines

		self.n_categories = len(self.all_categories)
		Config.n_categories = self.n_categories #setting up the config file based on what's passed into the classifier 

	def initializeData(self):
		#building a list of names per language
		self.category_boundaries = {}
		self.all_categories = []

		self.loadData()
		print (self.category_boundaries)
		Config.end_index = len(n_letters) - 1

	#get a random category and random line from that category
	def randomTrainingPair(self):
		category = randomChoice(self.all_categories)
		line = randomChoice(category_lines[category])
		return category, line

	#for a given category create an appropriate one-hot tensor representing that category
	def create_CategoryTensor(self, category):
		li = self.all_categories.index(category)
		tensor = torch.zeros(Config.minibatch_size, self.n_categories)
		tensor[0][li] = 1
		return tensor

	#One-hot matrix of first to last letters 
	def create_inputTensor(self, line):
		tensor = torch.zeros(len(line), Config.minibatch_size, n_letters)

		for i in range(len(line)):
			letter = line[i]
			tensor[i][0][all_letters.find(letter)] = 1
		return tensor

	#creates a target tensor 
	def create_targetTensor(self, line):
		letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
		letter_indexes.append(Config.end_index) 
		return torch.LongTensor(letter_indexes)

	#get a random training sample from the training set
	def randomTrainingSample(self):
		category, line = self.randomTrainingPair()

		category_tensor = Variable(self.create_CategoryTensor(category))
		input_line_tensor = Variable(self.create_inputTensor(line))
		target_tensor = Variable(create_targetTensor(line))

		return category_tensor, input_line_tensor, target_tensor

	def train(self):
		#initialize training 
		def initialize_training():
			self.optimizer = nn.NLLLoss()

		initialize_training()
		start = time.time()
		
		for i in range(1, Config.n_iters + 1):
			output, loss = self.runSample(*randomTrainingSample())
			self.total_loss += loss

			if i % Config.print_every == 0:
				print ("%s (%d %d%%) %.4f" % (getTime(start), i, i / Config.n_iters * 100, loss))

			if i % Config.plot_every == 0:
				all_losses.append(total_loss / Config.plot_every)
				total_loss = 0

	def runSample(self, category_tensor, input_line_tensor, target_tensor):
		hidden = self.model.initHidden()

		self.model.zero_grad()

		loss = 0

		for i in range(input_line_tensor.size()[0]):
			output, hidden = self.model(category_tensor, input_line_tensor[i], hidden)
			loss += self.optimizer(output, target_tensor[i])

		loss.backward()

		for p in self.model.parameters():
			#I guess this is a new way of updating?
			#should be equivalent to other ways of loss optimization
			p.data.add_(-Config.lr, p.grad.data) 

		return output, loss.data[0] / input_line_tensor.size()[0] #output and mean loss







'''
This module essentially creates it's own RNN custom cell, or does a single pass of the rnn and all intermediate steps
'''
class rnnClassifier(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(rnnClassifier, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size

		self.hidden_linear = nn.Linear(Config.n_categories + input_size + hidden_size, hidden_size)
		self.output_linear = nn.Linear(Config.n_categories + input_size + hidden_size, Config.n_categories)

		self.outputOutput_linear = nn.Linear(Config.n_categories, Config.n_categories)

		self.dropout = nn.Dropout(Config.dropout)
		self.softmax = nn.LogSoftmax()

	def forward(self, category, input, hidden):
		concatenated_input = torch.cat([category, input, hidden], 1) #concatenate along the first dimension to allow for mb in the future
		hidden_input = self.hidden_linear(concatenated_input)

		output_input = self.output_linear(concatenated_input)

		output = torch.cat([output_input, hidden_input], 1) #again concatenate along the 1st axis to allow for mb

		output = self.outputOutput_linear(output)
		output = self.dropout(output)
		output = self.softmax(output)

		return output, hidden

	def initHidden(self):
		return Variable(torch.zeros(Config.minibatch_size, self.hidden_size)) #1 




		

def main():
	runClassifier(rnnClassifier)

if __name__ == "__main__":
	main()