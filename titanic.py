import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer


def survival_rate(x):
	rate = len(x[x.Survived == 1])/len(x)
	return rate


def dummy_encode(x):
	values = np.array(x)
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(values)
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	return onehot_encoded


def imp_value(x):
	m = x.median()
	imped = x.fillna(m)
	return imped

def ngram_one(x,n):
	#space and dash??
	result = []
	end = len(x) - n
	a, b = 0, n
	while a < end:
		result.append(x[a:b])
		a += 1
		b += 1
	result.append(x[end:])
	return result


def ngram_all(lst,n):
	result = []
	for x in lst:
		result.extend(ngram_one(x,n))
	return result


if __name__ == '__main__':
	df = pd.read_csv('train.csv')
	df_test = pd.read_csv('test.csv')
	test_data = df_test.copy()
	data = df.copy()
	#impute missing values
	data['Age'] = imp_value(data['Age'])
#bucket the age
	data['underr18'] = [1 if x < 18 else 0 for x in data['Age']]
	data['18to35'] = [1 if (x >= 18) & (x <= 35) else 0 for x in data['Age']]
	data['35to50'] = [1 if (x > 35) & (x <= 50) else 0 for x in data['Age']]
	data['50to70'] = [1 if (x > 50) & (x <= 70) else 0 for x in data['Age']]
	data['over70'] = [1 if x > 70 else 0 for x in data['Age']]			
	data['Embarked'] = data['Embarked'].fillna('S') #female, first class 
	data['alone'] = data['SibSp'] + data['Parch']
	data['alone'] = [1 if x == 0 else 0 for x in data['alone']]
	data['parch0'] = [1 if x == 0 else 0 for x in data['Parch']]
	data['sibsp0'] = [1 if x == 0 else 0 for x in data['SibSp']]

#bag-of-words (position of the ngrams?? e.g. abcd vs cdab)

	temp_last = [x.split(', ')[0] for x in data['Name']] \
	+ [x.split(', ')[0] for x in test_data['Name']]

	# bi_bag = np.unique(ngram_all(temp_last,2))
	# tri_bag = np.unique(ngram_all(temp_last,3))
	# four_bag = np.unique(ngram_all(temp_last,4))
	# #transform into binary features
	# temp_bi = dummy_encode(bi_bag)
	# temp_tri = dummy_encode(tri_bag)
	# temp_four = dummy_encode(four_bag)

	#WHYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY len(data) != len(bi_bag) == len(temp_bi)
	bi_vec = CountVectorizer(ngram_range = (2,2), analyzer = 'char',binary = True)
	bi_bag = bi_vec.fit_transform(temp_last).toarray()[:len(data)]
	tri_vec = CountVectorizer(ngram_range = (3,3), analyzer = 'char',binary = True)
	tri_bag = tri_vec.fit_transform(temp_last).toarray()[:len(data)]
	four_vec = CountVectorizer(ngram_range = (4,4), analyzer = 'char',binary = True)
	four_bag = four_vec.fit_transform(temp_last).toarray()[:len(data)]


	y = 0
	for b in bi_vec.get_feature_names():
		data[b] = [x[y] for x in bi_bag]
		y += 1

	s = 0
	for t in tri_vec.get_feature_names():
		data[t] = [x[s] for x in tri_bag]
		s += 1
	g = 0
	for f in four_vec.get_feature_names():
		data[f] = [x[g] for x in four_bag]
		y += 1		

	#honourific
	temp_hon = [x.split(', ')[1] for x in data['Name']]
	temp_hon_names = [x.split('.')[0] for x in temp_hon]
	temp_hon = dummy_encode(temp_hon_names)
	z = 0
	for p in np.unique(temp_hon_names):
		data[p] = [x[z] for x in temp_hon]
		z += 1
	#Sex

	temp_sex = dummy_encode(data['Sex'])
	data['male'] = [x[0] for x in temp_sex]
	data['female'] = [x[1] for x in temp_sex]
	#Pclass
	temp_pclass = dummy_encode(data['Pclass'])
	data['class1'] = [x[0] for x in temp_pclass]
	data['class2'] = [x[1] for x in temp_pclass]
	data['class3'] = [x[2] for x in temp_pclass]	
	#Embarked
	temp_embarked = dummy_encode(data['Embarked'].replace(np.nan,'n/a'))
	w = 0
	for k in data['Embarked'].unique():
		data['embarked{0}'.format(k)] = [x[w] for x in temp_embarked]
		w += 1
	del data['Embarked']
	del data['Pclass']
	del data['Sex']
	data.to_csv('cleaned.csv',index = False)

#add boosting

