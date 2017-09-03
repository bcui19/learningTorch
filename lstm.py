'''
This module was used for testing the inner workings of pytorch lstm modules

Currently implemented a part of speech (POS) tagging system based on words


Important things to note:
	Hidden states in LSTMs are constructed as (hidden_dim, minibatch_size, input_size)
'''

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1) #seeding this

def test_lstm():
	lstm = nn.LSTM(3,3) #input dim is 3 output dim is 3
	inputs = [autograd.Variable(torch.randn((1, 3))) for _ in range(5)] #1x3 dimension

	hidden = (autograd.Variable(torch.randn(1, 1, 3)),
		autograd.Variable(torch.randn((1, 1, 3))))

	print hidden

	for i in inputs:
		out, hidden = lstm(i.view(1, 1, -1), hidden)

	#alternatively feed in the entire sequence at one
	inputs = torch.cat(inputs).view(len(inputs), 1, -1)

	hidden = (autograd.Variable(torch.randn(1, 1, 3)),
		autograd.Variable(torch.randn((1, 1, 3))))

	out, hidden = lstm(inputs, hidden)

	print out, hidden

#config class used for configurations for the nn
class Config:
	embedding_size = 7
	hidden_size = 13
	minibatch_size = 1
	epoch_num = 301
	num_lstm_layers = 1
	lr = 0.1

	#meant to be a sentence truncation length
	#I'll implement this later properly
	sentence_length = 4 


class Run_Part_Of_Speech:
	def __init__(self, model = None):
		assert(model is not None)
		self.word_to_idx = {} #initializing word to index dictionary
		self.tag_to_idx = {}
		self.prepareData()

		#note that word_to_idx and tag_to_idx must be stationary in length after this model is generated 
		self.model = model(Config.embedding_size, Config.hidden_size, len(self.word_to_idx), len(self.tag_to_idx))

		self.train()

	#for a given phrase/tag return the indices from a given index dict
	def dict_to_idx(self, sequence = None, ref_dict = None):
		assert(sequence is not None)
		assert(ref_dict is not None)
		idx = [ref_dict[word] for word in sequence]
		returnTensor = torch.LongTensor(idx)
		return autograd.Variable(returnTensor) #create a variable that can automatically be differentiated

	#idx is some list and then we reverse index 
	def idx_to_phrase(self, idx = None, ref_dict = None):
		assert(idx is not None)
		assert(ref_dict is not None)
		inverseDict = {ref_dict[i]: i for i in ref_dict}
		return [inverseDict[i.data.numpy()[0]] for i in idx]

	#adds a phrase to the dictionary, ensuring no overlaps
	def add_phrase_to_dict(self, phrase):
		for word in phrase:
			if word not in self.word_to_idx: self.word_to_idx[word] = len(self.word_to_idx)

	#adds a tag to the tag dictionary
	def add_tag_to_dict(self, tag):
		if tag not in self.tag_to_idx: self.tag_to_idx[tag] = len(self.tag_to_idx)


	def prepareData(self):
		self.trainData = [
	("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
	("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
	] #for every datapoint we have two elements one is the input sequence of words and the target is the output POS
		for phrase, tagList in self.trainData:
			self.add_phrase_to_dict(phrase)
			for tag in tagList:
				self.add_tag_to_dict(tag)


	def train(self):
		def training_initializer():
			self.loss_function = nn.NLLLoss()
			self.optimizer = optim.SGD(self.model.parameters(), lr = Config.lr)

		training_initializer() #initializes the loss function and optimizer 
		self.train_inputs = []
		self.train_goal = []
		for phrase, target in self.trainData:
			self.train_inputs.append(self.dict_to_idx(phrase, self.word_to_idx))
			self.train_goal.append(self.dict_to_idx(target, self.tag_to_idx))

		for i in range(Config.epoch_num):
			self.runEpoch()
			if i % 25 == 0:
				self.test()

	def runEpoch(self):
		#pairing up one to one goal and data
		for data, goal in zip(self.train_inputs, self.train_goal):
			self.model.zero_grad()

			#clearing out the models hidden memory 
			self.model.hidden = self.model.init_hidden()

			tag_probs = self.model(data)
			loss = self.loss_function(tag_probs, goal)
			loss.backward()
			self.optimizer.step()

	def test(self):
		print "TESTING\n"
		for data, goal in zip(self.train_inputs, self.train_goal):
			probs = self.model(data)
			prob_max, prob_idx = probs.max(1) #max returns tuple of the max value and the max index

			print self.idx_to_phrase(goal, self.tag_to_idx)
			print self.idx_to_phrase(prob_idx, self.tag_to_idx)
		print "DONE TESTING"



#right now as the model is implemented, we need to assume that 
#the sentences are of uniform length for minibatching to work properly
#this also means that end tokens will need to be appended to any input sentence
#in all honestly it doesn't even matter if end tokens exist
#it's just to make sure that we're not making pytorch mad becuase of dimension mismatches
class LSTM_tagger_model(nn.Module):
	def __init__(self, embedding_size, hidden_size, vocab_size, tag_size):

		super(LSTM_tagger_model, self).__init__() #initialization of LSTM model
		self.hidden_size = hidden_size

		self.word_embeddings = nn.Embedding(vocab_size, embedding_size) #again here because of left multiplication 

		#input size is embedding_size and outputs in hidden_size
		self.lstm = nn.LSTM(embedding_size, hidden_size)

		self.hidden_to_tag = nn.Linear(hidden_size, tag_size)
		self.hidden = self.init_hidden()

	'''
	initializing the hidden state of the lstm
	'''
	def init_hidden(self):
		#semantics are (num_layers, minibatch_size, hidden_size)
		return (autograd.Variable(torch.zeros(Config.num_lstm_layers, Config.minibatch_size, self.hidden_size)),
				autograd.Variable(torch.zeros(Config.num_lstm_layers, Config.minibatch_size, self.hidden_size)))

	#forward pass of the tagger
	def forward(self, sentence_idx):
		embeds = self.word_embeddings(sentence_idx)
		lstm_out, self.hidden_out = self.lstm(
			embeds.view(len(sentence_idx), Config.minibatch_size, -1), self.hidden) #inputting embeddings/hidden states into lstm

		tag_dist = self.hidden_to_tag(lstm_out.view(len(sentence_idx), -1)) #tag distribution
		tag_probs = F.log_softmax(tag_dist)
		return tag_probs






def main():
	# test_lstm()
	Run_Part_Of_Speech(LSTM_tagger_model)

if __name__ == "__main__":
	main()