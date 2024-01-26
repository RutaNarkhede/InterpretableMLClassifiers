from mcless import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time

def scale_factor(X):
    X = X.transpose()
    scale = np.zeros((len(X)))
    for i in range(len(X)):
        row = X[i]
        scale[i] = np.amax(np.abs(row))
    return scale

def scale_data(X,scale):
    X= X.transpose()
    scaled_X = np.zeros((X.shape))
    
    for i in range(len(X)):
        row = X[i]
        scaled_X[i] = X[i]/scale[i]
    
    return scaled_X.transpose()

if __name__ == "__main__":
	data1 = np.loadtxt('data/synthetic1.data', delimiter=',')
	X1= data1[:,0:2]
	y1=data1[:,2]

	data2= np.loadtxt('data/synthetic2.data', delimiter=',')
	X2= data2[:,0:2]
	y2=data2[:,2]

	data3 = load_iris()
	X3 = data3.data
	y3 = data3.target

	data4 = load_wine()
	X4 = data4.data
	y4 = data4.target
	
	data_X = [X1,X2,X3,X4]
	data_y = [y1,y2,y3,y4]
	data_names = ["Synthetic Data 1","Synthetic Data 2","Iris Data","Wine Data"]
	
	classifiers = [mCLESS(), LogisticRegression(max_iter = 1000),KNeighborsClassifier(5),SVC(kernel="rbf",gamma=2, C=1),
               RandomForestClassifier(max_depth=5, n_estimators=50, max_features=1)]
	
	names = [
		"mCLESS",
		"Logistic Regr",
		"KNeighbors-5 ",
		"RBF SVM ",
		"Random Forest"]
	
	N,d = X1.shape; labelset=set(y1)
	nclass=len(labelset);
	print('N,d,nclass=',N,d,nclass)

	rtrain = 0.7e0
	run = 100
	rtest = 1-rtrain

	for X,y,dataname in zip(data_X,data_y,data_names):
		acc_max = 0
		for name, clf in zip(names, classifiers):
			Acc = np.zeros([run,1])
			btime = time.time()

			for it in range(run):
				Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=rtest, random_state=it, stratify = y)
				scale = scale_factor(Xtrain)
				Xtrain_scaled = scale_data(Xtrain,scale)
				Xtest_scaled = scale_data(Xtest,scale)

				clf.fit(Xtrain_scaled, ytrain);
				Acc[it] = clf.score(Xtest_scaled, ytest)

			etime = time.time()-btime
			accmean = np.mean(Acc)*100
			print('%s: %s: Acc.(mean,std) = (%.2f,%.2f)%%; E-time= %.5f'%(dataname,name,accmean,np.std(Acc)*100,etime/run))
			if accmean>acc_max:
				acc_max= accmean; algname = name
		print('sklearn classifiers max: %s= %.2f\n' %(algname,acc_max))

	#model = mcless.mCLESS
	#model.fit(scaledTrain)
	#predictedClass = mcmodel.predict(scaledTest)

	#W = mcless.fit(scaledTrain)
	#predictedclass = mcless.predict(scaledTest, W)
	#mcless = confisionMatrix(predictedClass, actualClass)


	




