import os
os.chdir(os.path.dirname(__file__))
os.chdir('..')

import time
import numpy as np
import random as rnd

from Functions import Encoding
from Functions import Plotting
from Functions import BeliefModel
from Functions import ForwardMapping
from Functions import DecisionPolicy
from Functions import OtherThing
from Functions import DigitalTwin
from Functions import DigitalTwinBOFunctions

np.set_printoptions(suppress=True)
import warnings
from sklearn.exceptions import ConvergenceWarning
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
# maxTotalInjections = 525

ModelFolderPath = 'Saved Models 071822'
Models, YJFit = DigitalTwin.ImportModels(ModelFolderPath)

StartX = [0,0,0]
StartY = [478,1.68,0.041]

X = np.array([StartX,[1,rnd.random(),rnd.randint(0, 9)/9]])
Y = np.array([StartY])

DropletInjectionCount = 0
DropletCount = 1

tic = time.time()
# while len(X) <= maxTotalInjections:
while DropletCount <= maxDroplets:
    DropletInjectionCount = DropletInjectionCount + 1
    Y_New = DigitalTwin.SampleFromModel(X, Y, Models, YJFit)
    Y = np.append(Y,Y_New,axis=0)
    if np.isnan(Y[-1,0]) or DropletInjectionCount >= maxInjectperDroplet:
        DropletInjectionCount = 0
        DropletCount = DropletCount + 1
        Y = np.append(Y,[StartY],axis=0)
        X = np.append(X,[StartX],axis=0)

    R = DigitalTwin.SlopeReward(X,Y)
    if DropletCount < 3:
        X = np.append(X, [[1, rnd.random(), rnd.randint(0, 9) / 9]], axis=0)
    else:
        X_New = DigitalTwin.RLSelectNextInjectionUCB(X,R)
        X = np.append(X,[X_New],axis=0)
    print(X[-1],Y[-1])

toc = time.time()
print('Run time is ' + str(toc-tic) + ' seconds')
