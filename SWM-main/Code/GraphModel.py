# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:12:41 2022

@author: pkandala
"""

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import * 
from sklearn.metrics import fbeta_score, accuracy_score,precision_recall_fscore_support, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from xgboost import XGBClassifier

newGraphDf = pd.read_csv('C:/newDataset2.csv')
newGraphDf.isnull().values.any()
newGraphDf = newGraphDf.sample(frac=1).reset_index(drop=True)
classes=newGraphDf['fraud']
graphModelDF =  newGraphDf.drop(['step','age','gender','zipcodeOri', 'zipMerchant', 'customer','fraud'], axis = 1)
graphModelDF= pd.get_dummies(graphModelDF,columns=['category','merchant'])


dataNormalization = StandardScaler()

modifiedGraphDF = pd.DataFrame(dataNormalization.fit_transform(graphModelDF), columns = graphModelDF.columns)

def kmeansClsfy(model,modifiedGraphDF,classes):
    
    trainX, testX, trainY, testY = train_test_split(modifiedGraphDF, classes, test_size=0.20)
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
    print(accuracy_score(testY, pred) )
    print(classification_report(testY, pred))
    return clsy
def modelClsfy(model,modifiedGraphDF):

    trainX, testX, trainY, testY = train_test_split(modifiedGraphDF, classes, test_size=0.20)

    clsy = model.fit(trainX, trainY)
    pred = model.predict(testX)
    print(class_accuracy( list(testY), pred)) 
    print(accuracy_score(testY, pred) )
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
        
def featureImpComp(clf_enhanced,graphModelDF):

    matplotlib.rcParams.update({'font.size': 22})
    imp_features = clf_enhanced.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf_enhanced.estimators_],
                axis=0)
    indices = np.argsort(imp_features)[::-1]
    test = list(graphModelDF.columns[indices[0:10]])
    plt.figure(figsize=(60,15))
    plt.xlabel("Feature Name", fontsize=30)
    plt.ylabel("Importance ", fontsize=30)
    plt.plot(test,imp_features[indices[0:10]],color='blue',marker='o')
    plt.xlim([-1, 10])
    plt.show()

def foldAvgComp(fold_data, model_names):
    model_1_sum = 0
    model_2_sum = 0

    for dat in fold_data:
        model_1_sum += dat[model_names[0]]/len(fold_data)
        model_2_sum += dat[model_names[1]]/len(fold_data)

    print("Model {} average: {}".format(model_names[0], model_1_sum))
    print("Model {} average: {}".format(model_names[1], model_2_sum))
    
def classesAvgComp(fold_data_classes,model_names):
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
    
kf = KFold(n_splits=5)
betas_fold = []
accuracy_fold = []
accuracy_fold_classes=[]
kf.get_n_splits(modifiedGraphDF)

for train_index, test_index in kf.split(modifiedGraphDF):
    X_train, X_test = modifiedGraphDF.iloc[train_index], modifiedGraphDF.iloc[test_index]
    y_train, y_test = classes.iloc[train_index], classes.iloc[test_index]
    svmClsy = SVC()
    svmClsy.fit(X_train,y_train)
    rfClsy = RandomForestClassifier()
    rfClsy.fit(X_train,y_train)
    
    svmPred = svmClsy.predict(X_test)
    rfPred = rfClsy.predict(X_test)
    betas_fold.append({"SVM": fbeta_score( y_test, svmPred, average='macro', beta=1),
                       "RF": fbeta_score( y_test, rfPred, average='macro', beta=1) })
    
    accuracy_fold.append({"SVM": accuracy_score( y_test, svmPred),
                       "RF ": accuracy_score(y_test, rfPred) })
    accuracy_fold_classes.append({"SVM": class_accuracy( list(y_test), svmPred),
                       "RF": class_accuracy(list(y_test), rfPred) })
    print('SVM Report')
    print(classification_report(y_test, svmPred))
    print('RF Report')
    print(classification_report(y_test, rfPred))
    

print("---------- Classes Accuracy Averages -----------")
classesAvgComp(accuracy_fold_classes, ["SVM", "RF"])


xgClassify = modelClsfy(XGBClassifier(),modifiedGraphDF)

XTrn, XTst, yTrn, yTst = train_test_split(modifiedGraphDF, 
                                                    classes, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

clsfr = RandomForestClassifier()
clsfr2 = RandomForestClassifier()
clsfr2.fit(XTrn, yTrn)

pred = clsfr2.predict(XTst)

arguments = {'n_estimators': [5, 10 , 100],
              'min_samples_split': [2, 10, 50],
              'max_features': ["sqrt", "log2"],
             }
f1_score = make_scorer(fbeta_score, beta=1)

gridObject = GridSearchCV(clsfr, param_grid=arguments, scoring=f1_score)

fitGrid = gridObject.fit(XTrn, yTrn)

bestClsfr = fitGrid.best_estimator_
pred = (clsfr.fit(XTrn, yTrn)).predict(XTst)
predRelevant = bestClsfr.predict(XTst)
print("\nFinal Report for Graph-based Model\n------")
print("Accuracy: {:.4f}".format(accuracy_score(yTst, predRelevant)))
print("F-1 Score: {:.4f}".format(fbeta_score(yTst, predRelevant, beta = 1)))
print("Accuracy - Class 1:"+str(class_accuracy(list(yTst), predRelevant)[0]))
print("Accuracy - Class 0:"+str(class_accuracy(list(yTst), predRelevant)[1]))
print('RF Report')
print(classification_report(yTst, predRelevant))
featureImpComp(bestClsfr,modifiedGraphDF)
rf_with_kmeans = kmeansClsfy(bestClsfr,modifiedGraphDF.values,classes.values)
featureImpComp(rf_with_kmeans,modifiedGraphDF)