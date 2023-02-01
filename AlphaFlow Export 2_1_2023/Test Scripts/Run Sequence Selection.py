import os

os.chdir(os.path.dirname(__file__))

import time
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import numpy as np
from Functions import Encoding
from Functions import BeliefModel
from Functions import ForwardMapping
from Functions import DecisionPolicy


np.set_printoptions(suppress=True)

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore", category=ConvergenceWarning)

# %%
import importlib
## to update files if any changes made
importlib.reload(Encoding)
importlib.reload(BeliefModel)
importlib.reload(ForwardMapping)
importlib.reload(DecisionPolicy)

# %%
xFilePath = 'Inputs Active.txt'
yFilePath = 'Reward Active.txt'

xPreTrainFilePath = 'Inputs Pretrain Sequence Select.txt'
yPreTrainFilePath = 'Reward Pretrain Sequence Select.txt'

#number of prior actions to include in the short term memory aspect of state
MemorySteps = 4
#define the encoding of actions steps
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

    #X and Y are master training data file (before encoding for model training) used to retrain models between agent decisions
    #X= action/suggested action  (i.e. actions 1,2,3,4, - not one hot encoded), Y=reward/response
    X = np.append(XPreTrain, Xnew, axis=0)
    Y = np.append(YPreTrain, Ynew, axis=0)

    #yeo johnson transformation of measured data
    YJFit = PowerTransformer(method='yeo-johnson')
    YJFit.fit(Y)
    Y = YJFit.transform(Y)

    nX = np.shape(X)[0]
    nY = np.shape(Y)[0]
    #if not waiting on updated input/measured data files
    if nY >= nX:
        tic = time.time()

        #encoding/formatting input and reward from measured data for use in models
        State, Response, ClassState, ClassResponse = Encoding.X1Y1toStateResponse(X, Y, MemorySteps, BlockEncoding)

        #belief model composed of regressor and classifier
        Model = BeliefModel.TrainEnsembleSKLearnMLPRegressor(State, Response, 20, SubSamplingRate)
        ClassModel = BeliefModel.TrainGradientBoostingClassifier(ClassState, ClassResponse)

        Objective, ObjectiveErr, Actions, PredictionState, PredictionClassState, PredictionResponse, PredictionProbability = ForwardMapping.SubsamplePermutationX1Y1RState(
            Model, ClassModel, X, Y, 20, 4, MemorySteps, BlockEncoding, RegressionModelStructure,
            ClassModelStructure)

        CurrentResponse = Response[-1:]

        RecommendedAction, iRecommendedAction = DecisionPolicy.UpperConfidenceBounds1EnsembleSubSampling(Objective,
                                                                                                    ObjectiveErr,
                                                                                                    Actions,
                                                                                                    CurrentResponse,
                                                                                                    Model,
                                                                                                    PredictionState,
                                                                                                    X)



        Xsave = np.vstack([Xnew, RecommendedAction])
        xsave = pd.DataFrame(Xsave, columns=['Block', 'Time', 'Volume'])
        xsave.to_csv(xFilePath, index=False)

        toc = time.time()
        print(toc - tic)
    else:
        time.sleep(2)

