import pandas as pd 
import numpy as np 
from sklearn import model_selection as ms
from sklearn.svm import SVC 

if __name__ == '__main__':
	df = pd.read_csv('cleaned.csv')
	# df_test = pd.read_csv('cleaned_test.csv')
	# data_test = df_test.copy()
	data = df.copy()
	#feature selection
	cols = ['Age','SibSp','Parch','male','female','class1',\
	'class2','class3','embarkedS','embarkedC','embarkedQ']
	X = data[cols]
	y = data['Survived']
	#train-test split
	X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3,random_state = 0)
	clf = SVC()
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	print(clf.score(X_test,y_test))

	#cross validation
	kfold = ms.KFold(n_splits = 10,random_state = 0)
	model_cv = SVC()
	results_cv = ms.cross_val_score(model_cv,X_train, y_train, cv = kfold,\
		scoring = 'accuracy')
	print(results_cv.mean()) 
