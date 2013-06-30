from spam_functions import *
from SVM_functions import *
def main():
	import sys
	import numpy as np
	from scipy.io import loadmat

	sys.path.append(r'/home/raja/Documents/MachineLearning/ex6')

	file_contents = readFile('/home/raja/Documents/MachineLearning/ex6/emailSample1.txt')
	print file_contents
	word_indices = processEmail(file_contents)
	features = emailFeatures(word_indices)	
	print len(features)
	print np.sum(features)

	data = loadmat('spamTrain.mat', matlab_compatible=True)
	pdata=dict()
	for key in data.keys():
		if key[0]!='_':
			pdata[key]=data[key].squeeze()
	C = 0.1
	model = svmTrain(pdata['X'], pdata['y'], C, 'linear', 1e-3, 20000)
	p = model.predict(pdata['X'])
	print np.mean(p==pdata['y'])

	data = loadmat('spamTest.mat', matlab_compatible=True)
	pdata=dict()
	for key in data.keys():
		if key[0]!='_':
			pdata[key]=data[key].squeeze()

	p = model.predict(pdata['Xtest'])
	print np.mean(p==pdata['ytest'])		

	w = model.coef_[0,:]
	print w
	w_s_ind=np.argsort(w)
	vocab_list= importVocab('/home/raja/Documents/MachineLearning/ex6/vocab.txt')
	#using python 2.6, dict comprehensions only in 2.7+
	inv_list = dict((v,k) for k, v in vocab_list.items())
	print inv_list
	for i in w_s_ind[:-16:-1]:
		print inv_list[i+1], w[i]	

if __name__=='__main__':
	main()	