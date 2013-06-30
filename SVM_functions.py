def plotData(X, y, fig_num):
	"""
	plots the data points X and y into a new figure
	+ as positive examples, o for negative examples
	X assumed to be Mx2
	"""
	import matplotlib.pyplot as pyplot
	pos=(y==1)
	neg=(y==0)
	pyplot.figure(fig_num)
	pyplot.plot(X[pos,0], X[pos,1], 'k+', linewidth=1, markersize=7)
	pyplot.plot(X[neg,0], X[neg,1], 'ko', markerfacecolor='y', markersize=7)

def svmTrain(X, y, C, kernel, tol, max_passes, sigma=0):
	from sklearn import svm
	 		
	if kernel == 'linear':
		clf = svm.SVC(kernel=kernel, C=C, tol=tol, max_iter=max_passes)
	elif kernel == 'gaussian':
		clf = svm.SVC(kernel='rbf', C=C, gamma=1/(2*sigma**2), tol=tol, max_iter=max_passes)
	clf.fit(X, y)

	return clf

def visualizeBoundaryLinear(X, y, model, fig_num):
	import numpy as np
	import matplotlib.pyplot as pyplot
	w = model.coef_

	b = model.intercept_
	xp = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
	yp = - (w[0,0]*xp+b)/w[0,1]
	plotData(X, y, fig_num)
	pyplot.plot(xp, yp, '-b')

def import_plot(datafile, fig_num):
	from scipy.io import loadmat
	
	data = loadmat(datafile, matlab_compatible=True)

	pdata=dict()
	for key in data.keys():
		if key[0]!='_':
			pdata[key]=data[key].squeeze()

	plotData(pdata['X'], pdata['y'], fig_num)
	return pdata

def visualizeBoundary(X, y, model, fig_num):
	import numpy as np
	import matplotlib.pyplot as pyplot

	plotData(X, y, fig_num)
	x1plot = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
	x2plot = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)

	X1, X2 = np.meshgrid(x1plot, x2plot)
	vals = np.zeros(np.shape(X1))	
	#fix x1, vary x2, predict
	for i in xrange(np.shape(X1)[1]):
		this_X = np.reshape(np.hstack((X1[:, i], X2[:, i])), (np.shape(X1)[0],-1), 'F')
		vals[:, i] = model.predict(this_X)

	pyplot.contour(X1, X2, vals, levels=[0], color='b')

def dataset3Params(X, y, Xval, yval):
	import numpy as np
	vals=0.01*np.logspace(0, 20, base=3)
	sigma=0
	C=0
	min_err=1
	for C_v in vals:
		for sig_v in vals:
			model = svmTrain(X, y, C_v, 'gaussian', 1e-3, 5000, sig_v)
			pred_val = model.predict(Xval)	
			CV_err = np.mean(yval != pred_val)
			if CV_err < min_err:
				sigma = sig_v
				C = C_v
				min_err = CV_err
	return (C, sigma, min_err)	