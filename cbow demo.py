'''
Torch implementation of CBOW
able to minibatch 

was able to overfit wooo 
'''

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#configuration parameters for CBOW
class Config:
	window_size = 2 #window size
	lr = 0.1 #learning rate
	num_epochs = 10 #number of epochs
	embedding_size = 10 #size of embeddings
	minibatch_size = 5




class runCBOW:
	def __init__(self, model):

		self.model = model
		self.initData()
		self.initModel()
		self.train()

	def initModel(self):
		self.model = self.model(self.vocab_size, Config.embedding_size)


	def initData(self):
		self.trainData = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

		self.word_idx = {}
		self.vocab = set(self.trainData)
		self.vocab_size = len(self.vocab)
		self.train_set = self.createWindow(self.trainData)
		self.add_to_vocab(self.vocab)


	def add_to_vocab(self, wordList):
		for word in wordList:
			if word not in self.word_idx: self.word_idx[word] = len(self.word_idx)
	
	def get_phrase_idx(self, phrase):
		return [self.word_idx[word] for word in phrase]

	#creates
	def createWindow(self, phrase):
		windowSet = []
		for i in range(Config.window_size, len(phrase) - Config.window_size):
			currContext = [phrase[i-j] for j in reversed(range(-Config.window_size, Config.window_size+1)) if j != 0]
			target = phrase[i]
			windowSet.append((currContext, target))
		return windowSet

	def train(self):
		self.loss_function = nn.NLLLoss()
		self.optimizer = optim.SGD(self.model.parameters(), Config.lr)
		self.losses = []


		for i in range(Config.num_epochs):
			self.runEpoch_minibatch()
		print self.losses
		self.test_minibatch()

	def test(self):
		for inputs, target in self.train_set:
			inputs_idx = self.get_phrase_idx(inputs)
			inputs_var = autograd.Variable(torch.LongTensor(inputs_idx))

			log_probs = self.model(inputs_var)
			values, indices = log_probs.max(1)

			goal_index = self.get_phrase_idx([target])
			# print log_probs
			print "goal is: ", goal_index, " resu is: ", indices#, values

	def test_minibatch(self):
		counter = 0
		
		minibatch = []
		minibatch_target = []

		for inputs, target in self.train_set:
			if counter % Config.minibatch_size == 0 and len(minibatch) != 0:
				minibatch = autograd.Variable(torch.LongTensor(minibatch))
				log_probs = self.model(minibatch)

				values, indices = log_probs.max(1)

				print "goal is: ", minibatch_target, " resu is: ", indices

				minibatch = []
				minibatch_target = []

			minibatch.append(self.get_phrase_idx(inputs))
			minibatch_target.append(self.get_phrase_idx([target]))
			counter += 1




	def runEpoch(self):
		total_loss = torch.Tensor([0])
		counter = 0
		for inputs, target in self.train_set:
			inputs_idx = self.get_phrase_idx(inputs)
			inputs_var = autograd.Variable(torch.LongTensor(inputs_idx))

			self.model.zero_grad()

			log_probs = self.model(inputs_var)

			target_vec = autograd.Variable(torch.LongTensor(self.get_phrase_idx([target])))

			loss = self.loss_function(log_probs, target_vec)
			loss.backward()

			self.optimizer.step()

			total_loss += loss.data

		self.losses.append(total_loss)

	def runEpoch_minibatch(self):
		total_loss = torch.Tensor([0])
		counter = 0
		
		minibatch = []
		minibatch_target = []

		total_loss = torch.Tensor([0])

		for inputs, target in self.train_set:
			if counter % Config.minibatch_size == 0 and len(minibatch) != 0:
				self.model.zero_grad()
				minibatch = autograd.Variable(torch.LongTensor(minibatch))
				log_probs = self.model(minibatch)

				target_vec = autograd.Variable(torch.LongTensor(minibatch_target)).view(-1)


				loss = self.loss_function(log_probs, target_vec)
				loss.backward()

				self.optimizer.step()

				total_loss += loss.data
				minibatch = []
				minibatch_target = []


			inputs_idx = self.get_phrase_idx(inputs)
			minibatch.append(inputs_idx)
			minibatch_target.append(self.get_phrase_idx([target]))
			counter += 1

		self.losses.append(total_loss)

			




class CBowModel(nn.Module):
	def __init__(self, vocab_size, embedding_size):
		super(CBowModel, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embedding_size)


		self.linear = nn.Linear(embedding_size, vocab_size)


	def forward(self, inputs):
		# print inputs

		embeds = self.embeddings(inputs)
		embed_sum = embeds.sum(1).view(Config.minibatch_size, -1) #sum along 1st axis
		out = self.linear(embed_sum)
		log_probs = F.log_softmax(out)
		return log_probs






def main():
	runCBOW(CBowModel)

if __name__ == "__main__":
	main()

