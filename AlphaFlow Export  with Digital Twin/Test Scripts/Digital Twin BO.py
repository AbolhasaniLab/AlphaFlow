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
maxDroplets = 100
maxInjectperDroplet = 20

ModelFolderPath = 'Saved Models 071822'
Models, YJFit = DigitalTwin.ImportModels(ModelFolderPath)

StartX = [0,0,0]
StartY = [0]

DropletInjectionCount = 0
DropletCount = 1

nInject = 20

tic = time.time()
x=[]
R=[]
Y=[]
X=[]

for iDroplet in range(maxDroplets):
    if iDroplet < 2:
        xnew=[]
        for ii in range(20):
            xnew=np.append(xnew,[rnd.random(),rnd.randint(0, 9)/9])
    else:
        xnew = DigitalTwinBOFunctions.UCBSelectNextCondition(x,R)

    Rnew = DigitalTwin.SampleFromModelfminReward(xnew,Models,YJFit,nInject)

    R = np.append(R, Rnew)

    if iDroplet==0:
        x=xnew
    else:
        x = np.row_stack((x, xnew))

        print(x)
        print(R)


toc = time.time()

print('Run time is ' + str(toc-tic) + ' seconds')
print(R)
