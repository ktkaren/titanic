import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
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
	#train-valid split
	X_train, X_valid, y_train, y_valid = ms.train_test_split(X_train, y_train, test_size=0.3,random_state = 0)
	#pruning
	loe = []
	for i in range(len(X_valid)):
		clf = DecisionTreeClassifier()
		clf.fit(X,y)
		y_pred = clf.predict(X_test)
		loe.append(clf.score(X_test,y_test))
		X_test.append(X_valid[i])

	#cross validation
	kfold = ms.KFold(n_splits = 10,random_state = 0)
	model_cv = DecisionTreeClassifier()
	results_cv = ms.cross_val_score(model_cv,X_train, y_train, cv = kfold,\
		scoring = 'accuracy')
	print(results_cv.mean()) 

	#overfitting -> 1)add validation set 2)chi^2 test