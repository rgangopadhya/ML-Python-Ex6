def readFile(filename):
	f = open(filename)
	file_contents=f.read()
	f.close()
	return file_contents

def importVocab(filename):
	raw_text=readFile(filename)
	vocab_list = dict()
	for ind_word in raw_text.splitlines():
		ind, word = ind_word.strip().split('\t')
		vocab_list[word] = int(ind)
	return vocab_list		

def processEmail(file_contents):
	import re
	from stemming.porter2 import stem

	vocab_list= importVocab('/home/raja/Documents/MachineLearning/ex6/vocab.txt')

	processed_file = file_contents.lower()

	processed_file = re.sub('<[^<>]+>', '', processed_file)

	processed_file = re.sub('[0-9]+', 'number', processed_file)

	processed_file = re.sub('(http|https)://[^\s]*', 'httpaddr', processed_file)

	processed_file = re.sub('[^\s]+@[^\s]+', 'emailaddr', processed_file)

	processed_file = re.sub('[$]+', 'dollar', processed_file)

	splitter = re.compile('[\s@$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]+')

	word_indices = []

	for word in splitter.split(processed_file):
		word = re.sub('[^a-zA-Z0-9]', '', word)
		word = stem(word)

		if word in vocab_list:
			word_indices.append(vocab_list[word]-1)
		else: print word			
	return word_indices

def emailFeatures(word_indices):
	import numpy as np
	vocab_list= importVocab('/home/raja/Documents/MachineLearning/ex6/vocab.txt')

	x = np.zeros((len(vocab_list)))
	x[word_indices] = 1
	return x
