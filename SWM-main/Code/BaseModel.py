# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:24:44 2022

@author: pkandala
"""

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
from matplotlib import pyplot
from sklearn.model_selection import * 
from sklearn.metrics import fbeta_score, accuracy_score,precision_recall_fscore_support, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
    

baseModelData = pd.read_csv('C:\dataset.csv')
baseModelData = baseModelData.sample(frac=1).reset_index(drop=True)

data_stats_unique = ""
for column in baseModelData:
    data_stats_unique +="{}: {} \n".format(column, baseModelData[column].unique().size)

fullData = baseModelData.shape[0]
nonFraudulentData = baseModelData[baseModelData.fraud == 0].step.count()
fraudulentData = fullData - nonFraudulentData

print("The non-fraud data comprises of {} datapoints, which is equivalent to {} % of the dataset".format(nonFraudulentData, round(100 *nonFraudulentData/fullData, 2)))
print("The fraudulent data comprises of {}datapoints, which is equivalent to {} % of the dataset".format(fraudulentData, round(100 *fraudulentData/fullData,2)))
classes = baseModelData.fraud
featuresBase = baseModelData.drop('fraud', axis =1)
featuresBase =  featuresBase.drop(['step','zipcodeOri', 'zipMerchant', 'customer'], axis = 1)
scaler = MinMaxScaler()
baseModelData[['amount', 'fraud']] = scaler.fit_transform(baseModelData[['amount', 'fraud']])
featuresBase.amount = baseModelData.amount
baseline_onehot  = pd.get_dummies(featuresBase)

def kMeansClassify(model,graphModifiedDF,classes):
    trainX, testX, trainY, testY = train_test_split(graphModifiedDF, classes, test_size=0.20)
    print(type(trainX))

    clustersN = len(np.unique(trainY))
    test = KMeans(clustersN = clustersN, random_state=42)
    test.fit(trainX)

    trainLabelsY = test.labels_
    testLabelsY = test.predict(testX)
    for i,j in enumerate(trainLabelsY):
        trainX[i] += trainLabelsY[i]
    for i,j in enumerate(testLabelsY):
        testX[i] += testLabelsY[i]
    print(testX.shape)

    clsy = model.fit(trainX, trainY)
    pred = clsy.predict(testX) 
    print(class_accuracy( list(testY), pred))
    print(classification_report(testY, pred))
    return clsy

def modelClassify(model,graphModifiedDF):

    trainX, testX, trainY, testY = train_test_split(graphModifiedDF, classes, test_size=0.20)

    clsy = model.fit(trainX, trainY)
    pred = model.predict(testX)
    print(class_accuracy( list(testY), pred))  
    print(classification_report(testY, pred))
    return clsy


def class_accuracy(original,predicted):
    correct_1=0
    correct_0=0
    total_1=0
    total_0=0
    for i in range(len(original)):
        if original[i]==1 :
            total_1+=1
            if predicted[i]==1:
                correct_1+=1
        elif original[i]==0 :
            total_0+=1
            if predicted[i]==0:
                correct_0+=1
    accuracy_1=(correct_1/total_1)
    accuracy_0=(correct_0/total_0)
    return [accuracy_1,accuracy_0]
def featureImpComp(clf_enhanced,graphDF):
    matplotlib.rcParams.update({'font.size': 22})
    imps = clf_enhanced.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf_enhanced.estimators_],
                axis=0)
    indices = np.argsort(imps)[::-1]
    test = list(graphDF.columns[indices[0:10]])
    plt.figure(figsize=(60,15))
    plt.xlabel("Feature Name", fontsize=30)
    plt.ylabel("Importance ", fontsize=30)
    plt.plot(test,imp_features[indices[0:10]],color='red',marker='o')
    plt.xlim([-1, 10])
    plt.show()
kf = KFold(n_splits=5)
betasFold = []
accuracyFold = []
accuracyFoldClasses=[]
kf.get_n_splits(baseline_onehot)

for train_index, test_index in kf.split(baseline_onehot):
    X_train, X_test = baseline_onehot.iloc[train_index], baseline_onehot.iloc[test_index]
    y_train, y_test = classes.iloc[train_index], classes.iloc[test_index]
    svmClsy = SVC()
    svmClsy.fit(X_train,y_train)
    rfClsy = RandomForestClassifier()
    rfClsy.fit(X_train,y_train)
    
    svmPred = svmClsy.predict(X_test)
    rfPred = rfClsy.predict(X_test)
    betasFold.append({"SVM": fbeta_score( y_test, svmPred, average='macro', beta=1),
                       "RF": fbeta_score( y_test, rfPred, average='macro', beta=1) })
    
    accuracyFold.append({"SVM": accuracy_score( y_test, svmPred),
                       "RF": accuracy_score(y_test, rfPred) })
    accuracyFoldClasses.append({"SVM": class_accuracy( list(y_test), svmPred),
                       "RF": class_accuracy(list(y_test), rfPred) })
    print('SVM Report')
    print(classification_report(y_test, svmPred))
    print('RF Report')
    print(classification_report(y_test, rfPred))
    
print(accuracyFoldClasses)

def avgFoldCalc(fold_data, model_names):
    model_1_sum = 0
    model_2_sum = 0

    for dat in fold_data:
        model_1_sum += dat[model_names[0]]/len(fold_data)
        model_2_sum += dat[model_names[1]]/len(fold_data)

    print("Model {} average: {}".format(model_names[0], model_1_sum))
    print("Model {} average: {}".format(model_names[1], model_2_sum))


def classesAvgCalc(fold_data_classes,model_names):
    model_1_sum_1 = 0
    model_1_sum_0 = 0
    model_2_sum_1 = 0
    model_2_sum_0 = 0

    for data in fold_data_classes:
        model_1_sum_1 += data[model_names[0]][0]/len(fold_data_classes)
        model_1_sum_0 += data[model_names[0]][1]/len(fold_data_classes)
        model_2_sum_1 += data[model_names[1]][0]/len(fold_data_classes)
        model_2_sum_0 += data[model_names[1]][1]/len(fold_data_classes)
    
    print('classes1')
    print("Model {} average: {}".format(model_names[0], model_1_sum_1))
    print('classes0')
    print("Model {} average: {}".format(model_names[0], model_1_sum_0))
    print('classes1')
    print("Model {} average: {}".format(model_names[1], model_2_sum_1))
    print('classes0')
    print("Model {} average: {}".format(model_names[1], model_2_sum_0))

print("---------- Betas Average ----------")
avgFoldCalc(betasFold, ["SVM", "RF"])
print("---------- Accuracy Averages -----------")
avgFoldCalc(accuracyFold, ["SVM", "RF"])
print("---------- Classes Accuracy Averages -----------")
classesAvgCalc(accuracyFoldClasses, ["SVM", "RF"])

XTrn, XTst, yTrn, yTst = train_test_split(baseline_onehot, 
                                                    classes, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

classifer = RandomForestClassifier()
classifer2 = RandomForestClassifier()
classifer2.fit(XTrn, yTrn)
pred = classifer2.predict(XTst)

arguments = {'n_estimators': [5, 10 , 100],
              'min_samples_split': [2, 10, 50],
              'max_features': ["sqrt", "log2"],
             }
f1_score = make_scorer(fbeta_score, beta=1)
gridObject = GridSearchCV(classifer, param_grid=arguments, scoring=f1_score)
fitGrid = gridObject.fit(XTrn, yTrn)
clsfrBest = fitGrid.best_estimator_
pred = (classifer.fit(XTrn, yTrn)).predict(XTst)
predRelevant = clsfrBest.predict(XTst)
print("\nFinal Report for base-line Model\n------")
print("Accuracy: {:.4f}".format(accuracy_score(yTst, predRelevant)))
print("F1-score: {:.4f}".format(fbeta_score(yTst, predRelevant, beta = 1)))
print("Accuracy - Class 1:"+str(class_accuracy(list(yTst), predRelevant)[0]))
print("Accuracy - Class 0:"+str(class_accuracy(list(yTst), predRelevant)[1]))
print('RF Report')
print(classification_report(yTst, predRelevant))
featureImpComp(clsfrBest,baseline_onehot)
rf_with_kmeans = kMeansClassify(clsfrBest,baseline_onehot.values,classes.values)
featureImpComp(rf_with_kmeans,baseline_onehot)
xgClassify = modelClassify(XGBClassifier(),baseline_onehot)


    

