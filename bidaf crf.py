'''
Showing the usefulness of a dynamic architecture as presented in pytorch


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
	num_layers = 1
	start_tag = None
	end_tag = None



class BiLSTM_CRF(nn.module):
	def __init__(self, embedding_size, vocab_size, hidden_size, target_size, tag_to_idx):
		super (BiLSTM_CRF, self).__init__()
		self.embedding_size = embedding_size
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.target_size = target_size
		self.tag_to_idx = tag_to_idx

		self.initialize_net()

	def initialize_net(self):
		self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_size)
		self.lstm = nn.LSEM(self.embedding_size, self.hidden_dim//2, #rounds down in python 3, treats as normal division in python 2
			num_layers = Config.num_layers, bidirectional = True)

		self.hidden_to_tag = nn.Linear(self.hidden_dim, self.target_size)

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
		return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
			autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

	def _forard_alg(self, feats):
		init_alphas = torch.Tensor(1, self.target_size).fill_(-10000.)

		init_alphas[0][Config.start_tag] = 0.

		forward_var = autograd.Variable(init_alphas)

		for feat in feats:
			alpha_t = [] #forard timestep variable
			for next_tag in range(self.target_size):
				emit_score = feat[next_tag].view(1, -1).expand(1, self.target_size)
				
				#the ith entry of the transition score is probability of transitioning to next_tag from i
				trans_score = self.trainsitions[next_tag].view(1, -1)

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

		lstm_out = lstm_out.view(len(sentence), self.hidden_dim)

		dist = self.hidden_to_tag(lstm_out)
		return dist

	def _score_sentence(self, feats, tags):
		score = autograd.Variable(torch.tensor([0]))
		tags = torch.cat([torch.LongTensor([self.])])









def main():
	pass


if __name__ == "__main__":
	main()