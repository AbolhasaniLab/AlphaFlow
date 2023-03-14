import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy import optimize as sciopt
import random as rnd

from Functions import Encoding
from Functions import Plotting
from Functions import BeliefModel
from Functions import ForwardMapping
from Functions import DecisionPolicy
from Functions import OtherThing

# def FormatforBO(X,Y):
#     iZero
#     return

def LocalReward(Y):
    a_AP = 1
    a_PV = 0.5
    a_API = 0.25

    b1_AP = 470
    b2_AP = 600
    b_PV = 2
    b_API = 0.046

    AP = Y[:,0]
    PV = Y[:, 1]
    API = Y[:, 2]

    NDAP=a_AP*(AP-b1_AP)/(b2_AP-b1_AP)
    NDPV=a_PV*PV/b_PV
    NDAPI=a_API*API/b_API

    Reward = NDAP + NDPV + NDAPI
    return Reward

def GuassianProcessSelect(X,Y):
    return

def NelderMeadeSelect(X,Y,nInject):
    return

def UCBSelectNextCondition(x,R):
    nInject = int(len(x[0])/2)
    x0 = []
    bnds = []
    for ii in range(nInject * 2):
        x0.append(rnd.random())
        bnds.append((0, 1))

    R[np.isnan(R)] = -0.5

    Model = BeliefModel.TrainEnsembleSKLearnMLPRegressor(x,R,20,0.25)
    result = sciopt.basinhopping(PredictUCB,x0,T=1,minimizer_kwargs={"args":(Model),"bounds":bnds})

    xSuggest = result.x
    return xSuggest

def ExploitSelectNextCondition(x,R):
    # nInject = int(len(x)/2)
    nInject = 20
    x0 = []
    bnds = []
    for ii in range(nInject * 2):
        x0.append(rnd.random())
        bnds.append((0, 1))

    R[np.isnan(R)] = -0.5

    Model = BeliefModel.TrainEnsembleSKLearnMLPRegressor(x,R,20,0.25)
    result = sciopt.basinhopping(PredictExploit,x0,T=1,minimizer_kwargs={"args":(Model),"bounds":bnds})

    xSuggest = result.x
    return xSuggest

def UCBSelectNextConditionTtune(x,R):
    nInject = int(len(x[0])/2)
    x0 = []
    bnds = []
    for ii in range(nInject * 2):
        x0.append(rnd.random())
        bnds.append((0, 1))

    R[np.isnan(R)] = -0.5

    Model = BeliefModel.TrainEnsembleSKLearnMLPRegressor(x,R,20,0.25)

    Rmax = np.zeros((1, 4))[0]

    result = sciopt.basinhopping(PredictUCB,x0,T=100,minimizer_kwargs={"args":(Model),"bounds":bnds})
    Rmax[0] = -result.fun
    print('Basin Hopping T=100 Done')
    print(Rmax)

    result = sciopt.basinhopping(PredictUCB, x0, T=10, minimizer_kwargs={"args": (Model), "bounds": bnds})
    Rmax[1] = -result.fun
    print('Basin Hopping T=10 Done')
    print(Rmax)

    result = sciopt.basinhopping(PredictUCB, x0, T=1, minimizer_kwargs={"args": (Model), "bounds": bnds})
    Rmax[2] = -result.fun
    print('Basin Hopping T=1 Done')
    print(Rmax)

    result = sciopt.basinhopping(PredictUCB, x0, T=0.1, minimizer_kwargs={"args": (Model), "bounds": bnds})
    Rmax[3] = -result.fun
    print('Basin Hopping T=0.1 Done')
    print(Rmax)

    result = sciopt.basinhopping(PredictUCB, x0, T=0.01, minimizer_kwargs={"args": (Model), "bounds": bnds})
    Rmax[4] = -result.fun
    print('Basin Hopping T=0.01 Done')

    print(Rmax)

    return Rmax

def PredictUCB(x,Model):
    pred = BeliefModel.EnsembleMeanY(Model, [x])
    uncert = BeliefModel.EnsembleStdevY(Model, [x])
    UCB = pred + 1/(2**0.5)*uncert
    return -UCB

def PredictExploit(x,Model):
    pred = BeliefModel.EnsembleMeanY(Model, [x])
    return -pred