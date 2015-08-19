import numpy as np 
import pandas as pd 
from pandas import Series, DataFrame
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.api as sm


def affair_check(x):
	if x != 0:
		return 1
	else:
		return 0



#create a data frame for the affair dataset
affair_df = sm.datasets.fair.load_pandas().data


#create an dummy categorical variable for occupation and husband's occupation
occ_dummies = pd.get_dummies(affair_df['occupation'])
hus_occ_dummies = pd.get_dummies(affair_df['occupation_husb'])

occ_dummies.columns = ['occ1','occ2','occ3','occ4','occ5','occ6']
hus_occ_dummies.columns = ['hocc1','hocc2','hocc3','hocc4','hocc5','hocc6']



affair_df['Had_affair'] = affair_df['affairs'].apply(affair_check)


X = affair_df.drop(['occupation','occupation_husb','Had_affair'],axis=1)
dummies = pd.concat([occ_dummies,hus_occ_dummies],axis=1)
X = pd.concat([X,dummies],axis=1)

Y = affair_df.Had_affair

X = X.drop('occ1',axis=1)
X = X.drop('hocc1',axis=1)
X = X.drop('affairs', axis=1)

Y = np.ravel(Y)

logistic_model = LogisticRegression()
logistic_model.fit(X,Y)
print logistic_model.score(X,Y)

print Y.mean()


coeff_df = DataFrame(zip(X.columns,np.transpose(logistic_model.coef_)))
print coeff_df


X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

logistic_model2 = LogisticRegression()
logistic_model2.fit(X_train,Y_train)
class_predict = logistic_model2.predict(X_test)

print metrics.accuracy_score(Y_test,class_predict)




