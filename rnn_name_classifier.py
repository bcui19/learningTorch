# -*- coding: utf-8 -*-


'''
module is incomplete because some random bugs exist that I really don't want to deal with 
'''

from __future__ import unicode_literals, print_function, division
from io import open
import glob


def findFiles(path): return glob.glob(path)


# print (findFiles('data/names/*.txt'))

import unicodedata
import string

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


class runClassifier:
	def __init__(self, model = None):
		self.initializeData()

	def loadData(self):
		for filename in findFiles('data/names/*.txt'):
			category = filename.split('/')[-1].split('.')[0]
			self.all_categories.append(category)
			lines = readLines(filename)
			self.category_boundaries[category] = lines

		self.n_categories = len(self.all_categories)

	def initializeData(self):
		#building a list of names per language
		self.category_boundaries = {}
		self.all_categories = []

		self.loadData()
		print (self.category_boundaries)

def main():
	runClassifier()

if __name__ == "__main__":
	main()