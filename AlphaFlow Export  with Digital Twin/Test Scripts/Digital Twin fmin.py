import os

os.chdir(os.path.dirname(__file__))
os.chdir('..')

import time
import pandas as pd
import math
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import statistics as stat
import random as rnd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression

from Functions import Encoding
from Functions import Plotting
from Functions import BeliefModel
from Functions import ForwardMapping
from Functions import DecisionPolicy
from Functions import OtherThing
from Functions import DigitalTwin
from Functions import DigitalTwinBOFunctions

from scipy import optimize as sciopt

np.set_printoptions(suppress=True)
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import power_transform
warnings.simplefilter("ignore", category=ConvergenceWarning)

import importlib
## to update files if any changes made
importlib.reload(Encoding)
importlib.reload(Plotting)
importlib.reload(BeliefModel)
importlib.reload(ForwardMapping)
importlib.reload(DecisionPolicy)
importlib.reload(OtherThing)
importlib.reload(DigitalTwin)
importlib.reload(DigitalTwinBOFunctions)

 # %%
maxDroplets = 10
maxInjectperDroplet = 20

ModelFolderPath = 'Saved Models 071822'
Models, YJFit = DigitalTwin.ImportModels(ModelFolderPath)

StartX = [0,0,0]
StartY = [478,1.68,0.041]

X = np.array([StartX])
Y = np.array([StartY])

DropletInjectionCount = 0
DropletCount = 1


nInject = 20
x0 = []
bnds = []
for ii in range(nInject*2):
    x0.append(rnd.random())
    bnds.append((0, 1))

tic = time.time()
# result = sciopt.minimize(DigitalTwin.SampleFromModelfminReward,x0,args=(Models,YJFit,nInject),method='L-BFGS-B',
#                          bounds=bnds,options={'maxiter':1, 'gtol':0.001, 'eps':0.0001, 'ftol':0.00001})
# result = sciopt.minimize(DigitalTwin.SampleFromModelfminReward,x0,args=(Models,YJFit,nInject),bounds=bnds)
# result = sciopt.minimize(DigitalTwin.SampleFromModelfminReward,x0,args=(Models,YJFit,nInject),bounds=bnds,method='Nelder-Mead',options={'xatol': 0.001})
result = sciopt.basinhopping(DigitalTwin.SampleFromModelfminReward,x0,minimizer_kwargs={"args":(Models,YJFit,nInject),"bounds":bnds})
toc = time.time()
print('Run time is ' + str(toc-tic) + ' seconds')
print([-result.fun,result.x])
print(DigitalTwin.SampleFromModelfminY(result.x,Models,YJFit,nInject))
