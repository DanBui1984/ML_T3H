import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pickle


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import tree

from sklearn.preprocessing import StandardScaler, PowerTransformer


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, classification_report



rawData = pd.read_csv('forestfires.csv')
print(rawData.isnull().sum(),"\n")
print(rawData.describe(),"\n")
print(rawData[rawData['rain'] > 0],"\n")
print(len(rawData[rawData['rain'] > 0]),"\n")

rawData.drop(labels=['rain'], axis = 1, inplace=True)
print(rawData.head(),"\n")

areaValues = rawData[['area']].values
areaFlag = []

for each in areaValues:
    
    if each[0] == 0:
        
        areaFlag.append(0)
        
    else:
        
        areaFlag.append(1)
rawData['burn'] = areaFlag

print(rawData.head())

X, Y = rawData[['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind']], rawData[['burn']]
XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.20, random_state=9892)
print ('Training Set: {} rows\n\nTest Set: {} rows'.format(XTrain.shape[0], XTest.shape[0]))


# reg = 0.01
# model = None
# model = SVC(C=1/reg, probability=True).fit(XTrain, YTrain['burn'])
#model = SVC(probability=True).fit(XTrain, YTrain['burn'])


filename = 'finalized_model.sav'
# ########save model trained
# pickle.dump(model, open(filename, 'wb'))

# #######load model trained
model = pickle.load(open(filename, 'rb'))

print (model)
print('\n')
# Xsample = XTest.sample(n=1, random_state=6)
# Ysample = YTest.sample(n=1, random_state=6)

Xsample = XTest.sample(n=1, random_state=5)
Ysample = YTest.sample(n=1, random_state=5)

print(Xsample)


prediction = None
predictions = model.predict(Xsample)
print('Burn Predict:', predictions)
print('Burn True:', Ysample)