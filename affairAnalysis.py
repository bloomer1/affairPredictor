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
print affair_df.head() # see first five rows
print affair_df # see all of data in the datsets 6365 data points/rows


#create a new column had_affar and apply the affair_check function
affair_df['Had_affair'] = affair_df['affairs'].apply(affair_check)
print affair_df.head()

# lets just observe the mean values of all the field when grouped by had_affair
print affair_df.groupby('Had_affair').mean()

# lets see if age is a important feature 
sns.factorplot('age',data=affair_df,hue='Had_affair',palette='coolwarm',kind='count')
sns.plt.show()


# lets see if years married is an important feature
sns.factorplot('yrs_married',data=affair_df,hue='Had_affair',palette='coolwarm',kind='count')
sns.plt.show()

# lets see if children have an important impact on affair
sns.factorplot('children',data=affair_df,hue='Had_affair',palette='coolwarm',kind='count')
sns.plt.show()

# lets see if education level have an important impact on affair
sns.factorplot('educ',data=affair_df,hue='Had_affair',palette='coolwarm',kind='count')
sns.plt.show()

# lets see if religious level have an important impact on affair
sns.factorplot('religious',data=affair_df,hue='Had_affair',palette='coolwarm',kind='count')
sns.plt.show()


# lets see if marraige rating  have an important impact on affair
sns.factorplot('rate_marriage',data=affair_df,hue='Had_affair',palette='coolwarm',kind='count')
sns.plt.show()





