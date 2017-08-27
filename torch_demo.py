import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


def makeTensors():
	V_data = [1., 2., 3.]
	V = torch.Tensor(V_data)
	# print V

	m_data = [[1., 2., 3.], [4., 5., 6.]]
	m_np = np.array(m_data)
	m = torch.Tensor(m_np)
	# print m

	T_data = [[[1., 2.], [3., 4.]],
		[[5., 6.], [7., 8.]]]
	T = torch.Tensor(T_data)
	# print T[:, 0, 0]

def tensor_reshape():
	x = torch.randn(2, 3, 4)
	print x
	print "x[0] is: ", x[0]
	print "x[1] is: ", x[1]
	
	#remember this makes things like an anchor 
	#e.g. minibatch size 
	print x.view(2, -1) #reshape like this


	print x.view(-1, 2) #this causes bad things to happen 

def backprop_demo():
	x = autograd.Variable(torch.Tensor([1., 2., 3.]), requires_grad = True)

	lr = 0.005

	y = autograd.Variable(torch.Tensor([4., 5., 6.]), requires_grad = True)

	z = x + y
	print z.data
	print z.grad_fn

	s = z.sum()
	print s
	print s.grad_fn

	s.backward()
	print x.grad #gradient of x after being backproped on the variable s, so updates are super easy

	# x -= lr * x.grad

	# print x

#Torch does left multiplication like np and tensorflow i think 
def linear_alg_demo():
	lin = nn.Linear(5,3) #linear map from R^5 to R^3
	data = autograd.Variable(torch.randn(3,5)) #matrix of shape 3 x 5
	print data
	print lin
	print lin(data)

def nonLinearity_demo():
	def crossEntropy(data):
		data_soft = F.softmax(data)
		data_sum = data_soft.sum(dim = 1)
		return data_soft/data_sum

	data = autograd.Variable(torch.randn(3, 3), requires_grad = True)
	print data
	print F.relu(data)
	data_ce =  crossEntropy(data)
	print data_ce
	print data_ce.sum(dim = 1)
	print -torch.log(data_ce)
	LL = F.log_softmax(data)
	print LL
	LL.sum().backward()

	print data.grad


class bagOfWordsClassifier(nn.Module):
	def __init__(self):
		self.word_idx = {}
		self.initData()
		#this is just some initializer always use it??
		super(bagOfWordsClassifier, self).__init__()



		self.linear = nn.Linear(self.vocab_size, self.num_labels)


	def initData(self):
		def word_to_idx(word):
			if word in self.word_idx: return
			#set it to the current length of the dictionary
			#this forces us to add the length
			self.word_idx[word] = len(self.word_idx) 

		self.data = [("me gusta comer en la cafeteria".split(), "SPANISH"), 
				("Give it to me".split(), "ENGLISH"),
				("No creo que sea una buena idea".split(), "SPANISH"),
				("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

		self.testData = [("Yo creo que si".split(), "SPANISH"),
				("it is lost on me".split(), "ENGLISH")]

		for phrase, classifier in self.data + self.testData:
			for word in phrase:
				word_to_idx(word)

		self.num_labels = 2 #just some static thing
		self.vocab_size = len(self.word_idx)

	#inputs a bag of words vector and perform a forwards pass through the neural net
	def forward(self, bow_vec):
		return F.log_softmax(self.linear(bow_vec))

	#leverages the word index
	def make_bow_vector(self, sentence):
		vec = torch.zeros(len(self.word_idx))
		#every vector is a count of the words
		for word in sentence:
			vec[self.word_idx[word]] += 1

		return vec.view(1, -1) #creating minibatches????

	def make_target(self, label, label_to_idx):
		return torch.LongTensor(label_to_idx[label])

	def test(self):
		for i, 

	def runModel(self):
		for i, datapoint in enumerate(self.data):
			input, label = datapoint
			bow_vec = self.make_bow_vector(input)
			log_probs = self.forward(autograd.Variable(bow_vec))
			print log_probs


def testBagOfWords():
	model = bagOfWordsClassifier() #initializing the graph
	#prints out the parameters
	# for i, param in enumerate(model.parameters()):
		# print i, param
	model.runModel()



def main():
	# makeTensors()
	# tensor_reshape()
	# backprop_demo()
	# linear_alg_demo()
	# nonLinearity_demo()
	testBagOfWords()

main()






