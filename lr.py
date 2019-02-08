import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection as ms





if __name__  == '__main__':
	df = pd.read_csv('cleaned.csv')
	data = df.copy()
	#feature selection
	# cols = ['Age','SibSp','Parch','male','female','class1',\
	# 'class2','class3','embarkedS','embarkedC','embarkedQ']
	cols = []
	for x in data.columns:
		if ~(np.isin(x,['PassengerId','Survived','Name','Cabin','Ticket'])):
			cols.append(x)	
	X = data[cols]
	y = data['Survived']
	#train-test split
	X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3,random_state = 0)
	logreg = LogisticRegression()
	logreg.fit(X_train,y_train)
	y_pred = logreg.predict(X_test)
	print(logreg.score(X_test,y_test))

	#cross validation
	kfold = ms.KFold(n_splits = 10,random_state = 0)
	model_cv = LogisticRegression()
	results_cv = ms.cross_val_score(model_cv,X_train, y_train, cv = kfold,\
		scoring = 'accuracy')
	print(results_cv.mean()) 
