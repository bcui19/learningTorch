import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def getWordEmbedding():
	word_idx = {"hello": 0, "world": 1}
	embeds = nn.Embedding(2,5) #2 words with dimension 5 embedding
	lookup_tensor = torch.LongTensor([word_idx['hello']])
	hello_embed = embeds(autograd.Variable(lookup_tensor))

	print embeds
	print hello_embed


class n_gram_model:
	class Config:
		context_size = 2
		embedding_size = 10
		epoch_num = 20
		lr = 0.00005


	def __init__(self, model):
		self.context_size = n_gram_model.Config.context_size
		self.embedding_size = n_gram_model.Config.embedding_size
		self.word_idx = {}

		self.model = model


		self.inputData()
		self.initializeModel() #initializes to be an instance
		self.train()

	def initializeModel(self):
		self.model = self.model(len(self.vocab), n_gram_model.Config.embedding_size, n_gram_model.Config.context_size)


	def makeTrigram(self, wordList):
		return [([wordList[i], wordList[i+1]], wordList[i+2]) for i in range(len(wordList)-2)]

	def add_to_vocab(self, wordList):
		for word in wordList:
			if word not in self.word_idx: self.word_idx[word] = len(self.word_idx)

	def inputData(self):
		self.test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
		self.test_trigram = self.makeTrigram(self.test_sentence)
		print self.test_trigram
		self.vocab = set(self.test_sentence)
		
		self.add_to_vocab(self.test_sentence)

	#assumes an input list of words that are split by spaces
	def get_phrase_idx(self, phrase):
		return [self.word_idx[word] for word in phrase]

	def train(self):
		self.loss_function = nn.NLLLoss()
		self.optimizer = optim.SGD(self.model.parameters(), n_gram_model.Config.lr)
		self.losses = []

		for i in range(n_gram_model.Config.epoch_num):
			self.run_epoch()
			return

		print self.losses


	def run_epoch(self):
		total_loss = torch.Tensor([0])
		for context, target in self.test_trigram:
			context_idx = self.get_phrase_idx(context)
			context_var = autograd.Variable(torch.LongTensor(context_idx))

			self.model.zero_grad()

			log_probs = self.model(context_var)
			# print log_probs
			# print autograd.Variable(torch.LongTensor(self.get_phrase_idx([target])))

			loss = self.loss_function(log_probs, autograd.Variable(torch.LongTensor(self.get_phrase_idx([target]))))
			loss.backward()
			# return
			self.optimizer.step()

			total_loss += loss.data

		self.losses.append(total_loss)




class n_gram_model_graph(nn.Module):
	def __init__(self, vocab_size, embedding_dim, context_size):
		super(n_gram_model_graph, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.linear1 = nn.Linear(context_size * embedding_dim, 128)

		self.linear2 = nn.Linear(128, vocab_size)

	#feed forwards pass log probability
	def forward(self, inputs):
		embeds = self.embeddings(inputs).view((1,-1)) #reshaping to 
		out = F.relu(self.linear1(embeds))
		out = self.linear2(out)
		log_probs = F.log_softmax(out)
		return log_probs






def main():
	n_gram_model(n_gram_model_graph) #can pass a class around like this

if __name__ == "__main__":
	main()
