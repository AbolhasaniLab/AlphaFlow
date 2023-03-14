import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV, RidgeClassifierCV, MultiTaskLassoCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, GradientBoostingClassifier

import random as rnd

import statistics as stat

import numpy as np

#import tensorflow as tf
#from tensorflow.keras import models
#from tensorflow.keras import layers
#from tensorflow.keras import optimizers

def FitXScaler(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

def ApplyXScaler(scaler,X):
    X = scaler.transform(X)
    return X

def EnsembleMeanY(ensemble,X):
    Y = [];
    for model in ensemble:
        tY=model.predict(X)
        Y.append(tY)
    Y_avg=sum(Y)/len(Y)
    return Y_avg

def EnsembleMeanYRmvOutlier(ensemble,X):
    Y = []
    Y_avg=[]
    for model in ensemble:
        tY=model.predict(X)
        Y.append(tY)
    Y=trans(Y)
    for y in Y:
        ty=reject_outliers(np.array(y))
        Y_avg.append(sum(ty)/len(ty))
    return np.array(Y_avg)

def EnsembleSubsampleY(ensemble,X):
    tY=ensemble[rnd.randint(0,len(ensemble)-1)].predict(X)
    Y=tY
    return Y

def EnsembleStdevY(ensemble,X):
    Y = [];
    for model in ensemble:
        tY=model.predict(X)
        Y.append(tY)
    Y_var=(sum(abs(Y-EnsembleMeanY(ensemble,X)))/len(Y))**0.5
    return Y_var

def EnsembleMeanYinvYJ(ensemble,X,YJFit):
    Y = [];
    for model in ensemble:
        tY=model.predict(X)
        invY = YJFit.inverse_transform(tY.reshape(-1, 1))
        Y.append(invY.reshape(1, -1))
    Y_avg=sum(Y)/len(Y)
    return Y_avg

def EnsembleStdevYinvYJ(ensemble,X,YJFit):
    Y = [];
    for model in ensemble:
        tY=model.predict(X)
        invY = YJFit.inverse_transform(tY.reshape(-1, 1))
        Y.append(invY.reshape(1, -1))
    Y_var=(sum(abs(Y-EnsembleMeanYinvYJ(ensemble,X,YJFit)))/len(Y))**0.5
    return Y_var

def EnsembleClassProbability(ensemble,X):
    C = [];
    for model in ensemble:
        tC=model.predict(X)
        C.append(tC)
    P=sum(C)/len(C)
    return P

def ClassProbability(model,X):
    P=model.predict_proba(X)[:,1]
    # P=model.predict(X)[:]
    return P

def RegressionPredictY(model,X):
    Y=model.predict(X)
    return Y

def RegressionPredictYErr(model,X):
    Y=[]
    estimators=model.estimators_
    for estimator in estimators:
        tempY=estimator.predict(X)
        Y.append(tempY)
    Y_var=(sum(abs(Y-model.predict(X)))/len(Y))**0.5
    return Y_var

def TrainFeedForwardNet(X,Y):    
    mlp = models.Sequential()
    mlp.add(layers.Dense(20, activation='relu', input_shape=(X.shape[1],)))
    mlp.add(layers.Dense(20, activation='relu'))
    mlp.add(layers.Dense(20, activation='relu'))
    mlp.add(layers.Dense(Y.shape[1]))
    mlp.compile(optimizer=optimizers.RMSprop(lr=0.01),loss='mse',metrics=['mae'])
    mlp.fit(X,Y,epochs=50,batch_size=5,verbose=0)
    return mlp

def TrainEnsembleFeedForwardNet(X,Y,numModels,SubSamplingRate):
    Models = []
    for iModel in range(numModels):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
        mlp = models.Sequential()
        mlp.add(layers.Dense(20, activation='relu', input_shape=(X.shape[1],)))
        mlp.add(layers.Dense(20, activation='relu'))
        mlp.add(layers.Dense(20, activation='relu'))
        mlp.add(layers.Dense(Y.shape[1]))
        mlp.compile(optimizer=optimizers.RMSprop(lr=0.01),loss='mse',metrics=['mae'])
        mlp.fit(X_train,Y_train,epochs=50,batch_size=5,verbose=0)
        Models.append(mlp)
    return Models

def TrainDecisionTree(X,Y):
    Model=DecisionTreeRegressor()
    Model.fit(X,Y)
    return Model

def TrainEnsembleDecisionTree(X,Y,numModels,SubSamplingRate):
    Models = []
    for iModel in range(numModels):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
        Models.append(DecisionTreeRegressor())
        Models[iModel].fit(X_train,Y_train)
    return Models

def TrainEnsembleDecisionTreeClassifier(X,Y,numModels,SubSamplingRate):
    Models = []
    for iModel in range(numModels):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
        Models.append(DecisionTreeClassifier())
        Models[iModel].fit(X_train,Y_train)
    return Models

def TrainRandomForestClassifier(X,Y):
    Model=RandomForestClassifier()
    Y=np.ravel(Y)
    Model.fit(X,Y)
    return Model

def TrainRandomForestRegressor(X,Y):
    Model=RandomForestRegressor()
    Model.fit(X,Y)
    return Model

def TrainAdaboostRegressor(X,Y):
    Model=AdaBoostRegressor()
    Model.fit(X,Y)
    return Model

def TrainAdaboostClassifier(X,Y):
    Model=AdaBoostClassifier()
    Y=np.ravel(Y)
    Model.fit(X,Y)
    return Model

def TrainGradientBoostingClassifier(X,Y):
    Model=GradientBoostingClassifier()
    Y=np.ravel(Y)
    Model.fit(X,Y)
    return Model

def TrainEnsembleGaussianProcessRegressor(X,Y,numModels,SubSamplingRate):
    Models = []
    for iModel in range(numModels):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
        Models.append(GaussianProcessRegressor())
        Models[iModel].fit(X_train,Y_train)
    return Models

def TrainEnsembleGaussianProcessClassifier(X,Y,numModels,SubSamplingRate):
    Models = []
    for iModel in range(numModels):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
        Models.append(GaussianProcessClassifier())
        Models[iModel].fit(X_train,Y_train)
    return Models

def TrainStackedClassifier(X,Y,SubSamplingRate):
    Models = []
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
    Models.append(GaussianProcessClassifier())
    Models[0].fit(X_train,Y_train)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
    Models.append(RandomForestClassifier())
    Models[1].fit(X_train,Y_train)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
    Models.append(SVC())
    Models[2].fit(X_train,Y_train)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
    Models.append(RidgeClassifierCV())
    Models[3].fit(X_train,Y_train)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
    Models.append(KNeighborsClassifier())
    Models[4].fit(X_train,Y_train)  
    
    return Models

def TrainStackedRegressor(X,Y,SubSamplingRate):
    Models = []
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
    Models.append(GaussianProcessRegressor())
    Models[0].fit(X_train,Y_train)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
    Models.append(RandomForestRegressor())
    Models[1].fit(X_train,Y_train)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
    Models.append(RidgeCV())
    Models[2].fit(X_train,Y_train)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
    Models.append(LassoCV())
    Models[3].fit(X_train,Y_train)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
    Models.append(KNeighborsRegressor())
    Models[4].fit(X_train,Y_train)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
    Models.append(MLPRegressor(learning_rate='adaptive',max_iter=1000, solver='lbfgs',hidden_layer_sizes=(rnd.randint(50,100),rnd.randint(20,50))))
    Models[5].fit(X_train,Y_train)
    
    return Models

def TrainEnsembleSKLearnMLPRegressor(X,Y,numModels,SubSamplingRate):
    #models stored in array and for imodel in specified nummodels,models are trained on split data and returned
    Models = []
    for iModel in range(numModels):
        #splitting data for training:
        #if not enough data to train models with subsampling from data rate use all of the data
        try:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
        except:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1)
        Models.append(MLPRegressor(learning_rate='adaptive', solver='lbfgs',hidden_layer_sizes=(rnd.randint(100,200),rnd.randint(50,100),rnd.randint(20,50))))

        Models[iModel].fit(X_train,Y_train)
    return Models

def TrainEnsembleSKLearnMLPRegressorStratify(X,Y,numModels,SubSamplingRate):
    #models stored in array and for imodel in specified nummodels,models are trained on split data and returned
    Models = []
    StratClass=(Y>0.5)
    for iModel in range(numModels):
        #splitting data for training:
        #if not enough data to train models with subsampling from data rate use all of the data
        # try:
        #     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
        # except:
        #     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=SubSamplingRate, stratify=StratClass)

        Models.append(MLPRegressor(learning_rate='adaptive', solver='lbfgs',hidden_layer_sizes=(rnd.randint(100,200),rnd.randint(50,100),rnd.randint(20,50))))
        Models[iModel].fit(X_train,Y_train)
    return Models


def TrainEnsembleSKLearnMLPClassifier(X,Y,numModels,SubSamplingRate):
    Models = []
    for iModel in range(numModels):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = SubSamplingRate)
        Models.append(MLPClassifier(learning_rate='adaptive',max_iter=1000))
        Models[iModel].fit(X_train,Y_train)
    return Models

def reject_outliers(data, m=1):
    return data[abs(data - np.median(data)) < m * np.std(data)]

def trans(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]