'''
Showing the usefulness of a dynamic architecture as presented in pytorch

Everything works and trains
can overfit

I'm going to need to come back to see how crf's actually work 


'''


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

#some helper functions pulled from tutorial

def to_scalar(var):
	# returns a python float
	return var.view(-1).data.tolist()[0]


def argmax(vec):
	# return the argmax as a python int
	_, idx = torch.max(vec, 1)
	return to_scalar(idx)


def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] for w in seq]
	tensor = torch.LongTensor(idxs)
	return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
	max_score = vec[0, argmax(vec)]
	max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
	return max_score + \
		torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class Config:
	embedding_size = 20
	hidden_size = 50
	num_epochs = 300
	num_layers = 1

	lr = 0.1
	weight_decay = 1e-4


	start_tag = None
	end_tag = None



class BiLSTM_CRF(nn.Module):
	def __init__(self, embedding_size, vocab_size, hidden_size, target_size, tag_to_idx = None):
		super (BiLSTM_CRF, self).__init__()
		self.embedding_size = embedding_size
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.target_size = target_size
		self.tag_to_idx = tag_to_idx

		self.initialize_net()

	def initialize_net(self):
		self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_size)
		self.lstm = nn.LSTM(self.embedding_size, self.hidden_size//2, #rounds down in python 3, treats as normal division in python 2
			num_layers = Config.num_layers, bidirectional = True)

		self.hidden_to_tag = nn.Linear(self.hidden_size, self.target_size)

		#matrix of transition parameters
		#score of transitioning from one state to another
		self.transitions = nn.Parameter(torch.randn(self.target_size, self.target_size))

		#these two ensure we don't transfer to a start token
		#and also make sure we don't transfer from an end token
		#these will also never be updated I think?
		#i, j => transitioning from token j to token i 
		self.transitions.data[Config.start_tag, :] = - 10000
		self.transitions.data[:, Config.end_tag] = -10000

		self.hidden = self.init_hidden()

	def init_hidden(self):
		return (autograd.Variable(torch.randn(2, 1, self.hidden_size // 2)),
			autograd.Variable(torch.randn(2, 1, self.hidden_size // 2)))

	def _forard_alg(self, feats):
		init_alphas = torch.Tensor(1, self.target_size).fill_(-10000.)

		init_alphas[0][Config.start_tag] = 0.

		forward_var = autograd.Variable(init_alphas)

		for feat in feats:
			alpha_t = [] #forard timestep variable
			for next_tag in range(self.target_size):
				emit_score = feat[next_tag].view(1, -1).expand(1, self.target_size)
				
				#the ith entry of the transition score is probability of transitioning to next_tag from i
				trans_score = self.transitions[next_tag].view(1, -1)

				next_tag_var = forward_var + trans_score + emit_score

				#forward variable is log-sum-exp of all scores
				alpha_t.append(log_sum_exp(next_tag_var))
			
			forward_var = torch.cat(alpha_t).view(1, -1) 

		terminal_var = forward_var + self.transitions[Config.end_tag]
		alpha = log_sum_exp(terminal_var)
		return alpha

	#from a given idx sequence produce this 
	def _get_lstm_features(self, sentence):
		self.hidden = self.init_hidden() #clearing out hidden cache
		embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
		lstm_out, self.hidden = self.lstm(embeds, self.hidden)

		lstm_out = lstm_out.view(len(sentence), self.hidden_size)

		dist = self.hidden_to_tag(lstm_out)
		return dist

	def _score_sentence(self, feats, tags):
		score = autograd.Variable(torch.Tensor([0]))
		tags = torch.cat([torch.LongTensor([Config.start_tag]), tags]) #concatenates a list of torch long tensors
		for i, feat in enumerate(feats):
			score = score + \
				self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]

		score = score + self.transitions[Config.end_tag, tags[-1]] #transitioning from the last tag to the end token tag
		return score

	def _viterbi_decode(self, feats):
		backpointers = []


		init_vvars = torch.Tensor(1, self.target_size).fill_(-10000.)
		init_vvars[0][Config.start_tag] = 0

		forward_var = autograd.Variable(init_vvars)

		for feat in feats:
			bptrs_t = [] # holds backpointers at this step
			viterbivars_t = [] # holds viterbi variables at this step

			for next_tag in range(self.target_size):
				next_tag_var = forward_var + self.transitions[next_tag]
				best_tag_id = argmax(next_tag_var)
				bptrs_t.append(best_tag_id)
				viterbivars_t.append(next_tag_var[0][best_tag_id])

			forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
			backpointers.append(bptrs_t)

		#transition to the stop tag
		terminal_var = forward_var + self.transitions[Config.start_tag]
		best_tag_id = argmax(terminal_var)
		path_score = terminal_var[0][best_tag_id]

		#following backpointers to decode path 
		best_path = [best_tag_id]
		for bptrs_t in reversed(backpointers):
			best_tag_id = bptrs_t[best_tag_id]
			best_path.append(best_tag_id)

		#pop off start tag
		start = best_path.pop()
		assert (start == Config.start_tag) #sanity check
		best_path.reverse()

		return path_score, best_path

	def neg_log_likelihood(self, sentence, tags):
		feats = self._get_lstm_features(sentence)
		forward_score = self._forard_alg(feats)
		gold_score = self._score_sentence(feats, tags)
		return forward_score - gold_score

	#different than _forward_alg above
	def forward(self, sentence):
		lstm_feats = self._get_lstm_features(sentence)

		#find best path given features
		score, tag_seq = self._viterbi_decode(lstm_feats)
		return score, tag_seq


class runCLF:
	def __init__(self, model):
		self.word_to_idx = {}
		self.tag_to_idx = {}

		self.initialize_run()

		self.model = model(Config.embedding_size, len(self.word_to_idx), Config.hidden_size, len(self.tag_to_idx))


		self.train()

	def initialize_run(self):
		self.trainingData = [(
	"the wall street journal reported today that apple corporation made money".split(),
	"B I I I O O O B I O O".split()
), (
	"georgia tech is a university in georgia".split(),
	"B I O O O O B".split()
)]
		for sentence, tags in self.trainingData:
			for word in sentence:
				if word not in self.word_to_idx: self.word_to_idx[word] = len(self.word_to_idx)
			for tag in tags:
				if tag not in self.tag_to_idx: self.tag_to_idx[tag] = len(self.tag_to_idx)

		self.tag_to_idx['<START>'] = len(self.tag_to_idx)
		self.tag_to_idx['<END>'] = len(self.tag_to_idx)

		Config.start_tag = self.tag_to_idx['<START>']
		Config.end_tag = self.tag_to_idx['<END>']


	def train(self):
		def initialize_train():
			self.optimizer = optim.SGD(self.model.parameters(), lr = Config.lr, weight_decay = Config.weight_decay)

		initialize_train()

		for i in range(Config.num_epochs):
			if i  % 25 == 0:
				self.test()

			self.run_epoch()

	def test(self):
		for sentence, tags in self.trainingData:
			sentence_input, targets = self.prep_inputs(sentence, tags)

			model_resu = self.model(sentence_input)

			reverseDict = {self.tag_to_idx[i]: i for i in self.tag_to_idx}

			resu = [reverseDict[i] for i in model_resu[1]]


			print "RESULT IS: ", resu

			print "TAGS ARE: ", tags

			# print neg_log_likelihood.max(1)[1], targets


	def run_epoch(self):
		for sentence, tags in self.trainingData:
			#clear out gradients 
			self.model.zero_grad()

			sentence_input, targets = self.prep_inputs(sentence, tags)

			#forwrad pass through model
			neg_log_likelihood = self.model.neg_log_likelihood(sentence_input, targets)

			#optimizer
			neg_log_likelihood.backward()
			self.optimizer.step()


	def prep_inputs(self, sentence, tags):
		sentence_input = prepare_sequence(sentence, self.word_to_idx)
		targets = torch.LongTensor([self.tag_to_idx[tag] for tag in tags])
		return sentence_input, targets
		



def main():
	runCLF(BiLSTM_CRF)


if __name__ == "__main__":
	main()