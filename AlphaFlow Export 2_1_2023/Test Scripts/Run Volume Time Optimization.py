import os

os.chdir(os.path.dirname(__file__))

import time

import pandas as pd


import numpy as np


from Functions import Encoding
from Functions import BeliefModel
from Functions import ForwardMapping
from Functions import DecisionPolicy


np.set_printoptions(suppress=True)

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore", category=ConvergenceWarning)

import importlib

## to update files if any changes made
importlib.reload(Encoding)
importlib.reload(BeliefModel)
importlib.reload(ForwardMapping)
importlib.reload(DecisionPolicy)

# %%
xFilePath = 'Inputs Active.txt'
yFilePath = 'Reward Active.txt'

xPreTrainFilePath = 'Inputs Pretrain Vol Time.txt'
yPreTrainFilePath = 'Reward Pretrain Vol Time.txt'

MemorySteps = 4
BlockEncoding = 'One Hot'
SubSamplingRate = 0.25
RegressionModelStructure = 'Ensemble'
ClassModelStructure = 'SKLearn Ensemble'

xPreTrain = pd.read_csv(xPreTrainFilePath)
yPreTrain = pd.read_csv(yPreTrainFilePath)
YPreTrain = yPreTrain.values
XPreTrain = xPreTrain.values

#give matlab time to update RL suggestion and measured data files
yReading=True
while yReading==True:
    try:
        xnew = pd.read_csv(xFilePath)
        ynew = pd.read_csv(yFilePath)
        yReading=False
    except:
        time.sleep(0.5)


nX = 0


#updating RL suggestion and measured data file (w/ not empty initial file wait conditions)
# %%
while nX < 20000:

# %%
    xnew = pd.read_csv(xFilePath)
    ynew = pd.read_csv(yFilePath)

    Xnew = xnew.values
    Ynew = ynew.values

    #X and Y effectively master training data that are updated throughout campaigns
    #X= action/suggested action  (1,2,3,4, - not one hot encoded), Y=reward/response
    X = np.append(XPreTrain, Xnew, axis=0)
    Y = np.append(YPreTrain, Ynew, axis=0)

    #yeo johnson transformation
    Xdesc, descRangesX2, descRangesX3 = Encoding.DescretizeUniform(X, 7)

    nX = np.shape(X)[0]
    NX = nX + round(sum(X[:,2]) * 9)

    nY = np.shape(Y)[0]
    #if not waiting on updated input/measured data files
    if nY >= NX:
        tic = time.time()

        #encoding/formatting input and reward from measured data for use in models
        State, Response, ClassState, ClassResponse = Encoding.CIX23Y1toStateResponse(X, Y, MemorySteps)

        Model = BeliefModel.TrainEnsembleSKLearnMLPRegressor(State, Response, 20, SubSamplingRate)
        ClassModel = BeliefModel.TrainGradientBoostingClassifier(ClassState, ClassResponse)

        Objective, Nodes = ForwardMapping.SubsampleNodeMonteCarloCIX23Y1(
            Model, ClassModel, X, Y, 5000, 4, MemorySteps, RegressionModelStructure, ClassModelStructure, descRangesX2, descRangesX3)

        CurrentResponse = Response[-1:]

        RecommendedActionDesc = DecisionPolicy.UCBNodeRandomBranchSum(Objective, Nodes)


        RecommendedAction = Encoding.ValueinDescretizedRangeSample(RecommendedActionDesc, descRangesX2, descRangesX3)

        if X[-1,0] == 0:
            NewInjection = [1]
        else:
            NewInjection = Encoding.SelectNextPrecursor(State)

        RecommendedAction[0] = NewInjection[0]

        print(RecommendedAction)

        Xsave = np.vstack([Xnew, RecommendedAction])
        xsave = pd.DataFrame(Xsave, columns=['Block', 'Time', 'Volume'])
        xsave.to_csv(xFilePath, index=False)

        toc = time.time()
        print(toc - tic)
    else:
        time.sleep(5)

