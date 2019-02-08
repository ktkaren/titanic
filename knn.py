import pandas as pd 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection as ms

if __name__ == '__main__':
	df = pd.read_csv('cleaned.csv')
	data = df.copy()
	#feature selection
	cols = ['Age','SibSp','Parch','male','female','class1',\
	'class2','class3','embarkedS','embarkedC','embarkedQ']
	X = data[cols]
	y = data['Survived']
	#train-test split
	X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3,random_state = 0)
	neigh = KNeighborsClassifier(3)
	neigh.fit(X_train,y_train)
	y_pred = neigh.predict(X_test)
	print(neigh.score(X_test,y_test))

	#cross validation
	kfold = ms.KFold(n_splits = 10,random_state = 0)
	model_cv = KNeighborsClassifier(3)
	results_cv = ms.cross_val_score(model_cv,X_train, y_train, cv = kfold,\
		scoring = 'accuracy')
	print(results_cv.mean()) 

	#is this underfitting??
	#0.7873134328358209
	#0.7977982590885817