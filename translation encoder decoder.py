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

use_cuda = torch.cuda.is_available()

class config:
	START_TOKEN = 0
	END_TOKEN = 1

	max_length = 10


#helper class to keep track of language
class Lang:
	def __init__(self, name):
		self.name = name
		self.word_to_index = {}
		self.word_to_count = {}
		self.index_to_word = {0: config.START_TOKEN, 1: config.END_TOKEN}
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
	return len(p[0].split(' ')) < config.max_length and \
		len(p[1].split(' ')) < config.max_length and \
		p[0].startswith(eng_prefixes)


def filterPairs(pairs):
	return [pair for pair in pairs if filterPair(pair)]
	# for i, pair in enumerate(pairs):
		# pair = (str(pair[0]), str(pair[1]))
		# if 
		# print (pair, filterPair(pair))
		# return [pair]

import random

class runTranslation:
	def __init__(self, model = None):
		self.input_lang, self.output_lang, self.pairs = self.prepData()

		# print (self.pairs)

		print(random.choice(self.pairs)) 

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




def main():
	runTranslation()

if __name__ == "__main__":
	main()






