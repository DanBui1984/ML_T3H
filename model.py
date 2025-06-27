import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import tree
from sklearn.metrics import f1_score

from sklearn.preprocessing import StandardScaler, PowerTransformer


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, classification_report



rawData = pd.read_csv('forestfires.csv')
print(rawData.isnull().sum(),"\n")
print(rawData.describe(),"\n")
print(len(rawData[rawData['rain'] > 0]),"\n")
print(rawData[rawData['rain'] > 0],"\n")
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

def evaluateModel(model, XTest, YTest):
    
    prediction = None
    predictions = model.predict(XTest)
    print(predictions)
    print(YTest)

    print('Accuracy: ', accuracy_score(YTest, predictions))
    print("Precision:",precision_score(YTest, predictions))
    print("Recall:",recall_score(YTest, predictions))
    
    return predictions
def confusionMatrix(model, XTest, YTest):

    prediction = None
    predictions = model.predict(XTest)


    # Print the confusion matrix
    cm = confusion_matrix(YTest, predictions)

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize = 20)
    plt.colorbar()

    classNames = ['no burn', 'burn']

    tick_marks = np.arange(len(classNames))

    plt.xticks(tick_marks, classNames, fontsize = 20)
    plt.yticks(tick_marks, classNames, fontsize = 20)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('int'))

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      color = "white" if cm[i, j] > threshold else "black"
      plt.text(j, i, labels[i, j], horizontalalignment="center", color=color, fontsize = 20)

    plt.tight_layout()
    plt.ylabel('True label', fontsize = 20)
    plt.xlabel('Predicted label', fontsize = 20)
    plt.show()

def ROCCurve(model, XTest, YTest):

    YScores = model.predict_proba(XTest)

    # calculate ROC curve
    fpr, tpr, thresholds = roc_curve(YTest, YScores[:,1])

    # plot ROC curve
    fig = plt.figure(figsize=(9, 9))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate', fontsize = 20)
    plt.ylabel('True Positive Rate', fontsize = 20)
    plt.title('ROC Curve', fontsize = 20)
    plt.show()

    auc = roc_auc_score(YTest,YScores[:,1])
    print('\nAUC: ' + str(auc))

# ########Model GradientBoostingClassifier########
# model = None
# model = GradientBoostingClassifier().fit(XTrain, YTrain['burn'])

# ##########Model SVC#######
reg = 0.01
model = None
model = SVC(C=1/reg, probability=True).fit(XTrain, YTrain['burn'])
# model = SVC(probability=True).fit(XTrain, YTrain['burn'])

print (model)
print('\n')

predsSVC = evaluateModel(model, XTest, YTest)
predic = model.predict(XTest)
print("F1 Score:", f1_score(YTest,predic))
confusionMatrix(model, XTest, YTest)
ROCCurve(model, XTest, YTest)

