from SVM_functions import *

def main():
	import matplotlib.pyplot as pyplot
	import numpy as np
	import sys

	sys.path.append(r'/home/raja/Documents/MachineLearning/ex6')

	pdata=import_plot('ex6data1.mat', 0)

	C=100	
	model=svmTrain(pdata['X'], pdata['y'], C, 'linear', 1e-3, 2000)

	visualizeBoundaryLinear(pdata['X'], pdata['y'], model, 1)

	pdata=import_plot("ex6data2.mat", 2)
	C=1
	sigma=0.1
	model=svmTrain(pdata['X'], pdata['y'], C, 'gaussian', 1e-3, 2000, sigma)
	visualizeBoundary(pdata['X'], pdata['y'], model, 3)
	
	pdata=import_plot("ex6data3.mat", 4)
	C, sigma, min_err = dataset3Params(pdata['X'], pdata['y'], pdata['Xval'], pdata['yval'])

	print (C, sigma, min_err)
	model = svmTrain(pdata['X'], pdata['y'], C, 'gaussian', 1e-3, 5000, sigma)
	visualizeBoundary(pdata['X'], pdata['y'], model, 5)

	pos=(pdata['yval']==1)
	neg=(pdata['yval']==0)
	pyplot.plot(pdata['Xval'][pos,0], pdata['Xval'][pos,1], 'k+', markerfacecolor='b', linewidth=1, markersize=7)
	pyplot.plot(pdata['Xval'][neg,0], pdata['Xval'][neg,1], 'ko', markerfacecolor='r', markersize=7)

	pyplot.show()

if __name__=='__main__':
	main()	