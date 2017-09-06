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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

print ("NUMBER OF LETTERS IS: ", n_letters)

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

#config class for hyperparameter tuning and stuff 
class Config:
	dropout = 0.1 #dropout probability
	minibatch_size = 1
	hidden_size = 128
	lr = 0.005
	max_length = 20

	n_categories = None
	end_index = None


	#configurations for displaying results
	n_iters = 50000
	print_every = 1000
	plot_every = 500




class runClassifier:
	def __init__(self, model = None):
		self.initializeData()

		self.model = model(n_letters, Config.hidden_size, n_letters)

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
		print ("number of categories is: ", self.n_categories)
		Config.n_categories = self.n_categories #setting up the config file based on what's passed into the classifier 

	def initializeData(self):
		#building a list of names per language
		self.category_boundaries = {}
		self.all_categories = []

		self.loadData()
		Config.end_index = n_letters - 1

	#get a random category and random line from that category
	def randomTrainingPair(self):
		category = randomChoice(self.all_categories)
		line = randomChoice(self.category_boundaries[category])
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
		target_tensor = Variable(self.create_targetTensor(line))

		return category_tensor, input_line_tensor, target_tensor

	def train(self):
		#initialize training 
		def initialize_training():
			self.optimizer = nn.NLLLoss()

		initialize_training()
		start = time.time()
		
		for i in range(1, Config.n_iters + 1):
			output, loss = self.runSample(*self.randomTrainingSample())
			self.total_loss += loss

			if i == 10000:
				print ("old lr ", Config.lr)
				Config.lr /= 10
				print ("new lr ", Config.lr)

			if i % Config.print_every == 0:
				print ("%s (%d %d%%) %.4f" % (getTime(start), i, i / Config.n_iters * 100, self.total_loss/Config.plot_every))

			if i % Config.plot_every == 0:
				self.losses.append(self.total_loss / Config.plot_every)
				self.total_loss = 0
		
		self.sample("Russian", "R")

		plt.figure()
		plt.plot(range(len(self.losses)), self.losses)
		plt.show()

	def runSample(self, category_tensor, input_line_tensor, target_tensor):
		hidden = self.model.initHidden()

		self.model.zero_grad()

		loss = 0

		for i in range(input_line_tensor.size()[0]):
			output, hidden = self.model(category_tensor, input_line_tensor[i], hidden)
			# print ("target tensor is: ", target_tensor[i])
			# print ("output is: ", output)
			# print (" i is: ", i)
			loss += self.optimizer(output, target_tensor[i])

		loss.backward()

		for p in self.model.parameters():
			#I guess this is a new way of updating?
			#should be equivalent to other ways of loss optimization
			p.data.add_(-Config.lr, p.grad.data) 

		return output, loss.data[0] / input_line_tensor.size()[0] #output and mean loss

	def sample(self, category, start_letter = "A"):
		def convert_idx_to_char(output_words):
			print (output_words)
			return [''.join([all_letters[char_idx] for char_idx in currWord]) for currWord in output_words]

		category_tensor = Variable(self.create_CategoryTensor(category))
		input = Variable(self.create_inputTensor(start_letter))

		hidden = self.model.initHidden()

		output_words = [[all_letters.find(start_letter)]]

		for i in range(Config.max_length):
			output, hidden = self.model(category_tensor, input[0], hidden)

			top_v, idx = output.data.topk(1, dim = 1) #take the top k along the first axis

			index = idx[0]
			print (index)
			if index[0] == Config.end_index:
				break

			output_words.append(index) #append maximum index
			input = Variable(self.create_inputTensor(all_letters[index[0]]))

		print (convert_idx_to_char(output_words))






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
		self.output_linear = nn.Linear(Config.n_categories + input_size + hidden_size, output_size)

		self.outputOutput_linear = nn.Linear(output_size + hidden_size, output_size)

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