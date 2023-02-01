import numpy as np

import scipy.optimize
from Functions import Encoding
from Functions import BeliefModel

import random as rnd
import math


def UpperConfidenceBoundsSum(Objective, ObjectiveErr, Actions):
    lamb = 1 / (2 ** 0.5)
    # lamb=10
    nBranches, nLevels, nDim = Objective.shape
    UCB = np.zeros((nBranches, nLevels))
    UCBSum = np.zeros((nBranches))
    for iBranch in range(nBranches):
        for iLevel in range(nLevels):
            UCB[iBranch, iLevel] = Objective[iBranch, iLevel] + lamb * (ObjectiveErr[iBranch, iLevel] ** 0.5)
            if any(UCB[iBranch, 0:iLevel] < 0.02):
                UCB[iBranch, iLevel] = 0
        UCBSum[iBranch] = sum(UCB[iBranch, :])
    iRecommendedAction = np.argmax(UCBSum)
    RecommendedAction = Actions[iRecommendedAction, 0, :]
    return RecommendedAction, iRecommendedAction


def ExploitationSum(Objective, ObjectiveErr, Actions):
    nBranches, nLevels, nDim = Objective.shape
    UCB = np.zeros((nBranches, nLevels))
    UCBSum = np.zeros((nBranches))
    for iBranch in range(nBranches):
        for iLevel in range(nLevels):
            UCB[iBranch, iLevel] = Objective[iBranch, iLevel]
            if any(UCB[iBranch, 0:iLevel] < 0.02):
                UCB[iBranch, iLevel] = 0
        UCBSum[iBranch] = sum(UCB[iBranch, :])
    iRecommendedAction = np.argmax(UCBSum)
    RecommendedAction = Actions[iRecommendedAction, 0, :]
    return RecommendedAction, iRecommendedAction


def MaxVarianceSum(Objective, ObjectiveErr, Actions):
    nBranches, nLevels, nDim = Objective.shape
    UCB = np.zeros((nBranches, nLevels))
    UCBSum = np.zeros((nBranches))
    for iBranch in range(nBranches):
        for iLevel in range(nLevels):
            UCB[iBranch, iLevel] = ObjectiveErr[iBranch, iLevel]
            if any(UCB[iBranch, 0:iLevel] < 0.02):
                UCB[iBranch, iLevel] = 0
        UCBSum[iBranch] = sum(UCB[iBranch, :])
    iRecommendedAction = np.argmax(UCBSum)
    RecommendedAction = Actions[iRecommendedAction, 0, :]
    return RecommendedAction, iRecommendedAction


def ExpectedImprovementSum(Objective, ObjectiveErr, Actions, CurrentResponse):
    lamb = 1 / (2 ** 0.5)
    # lamb=10
    nBranches, nLevels, nDim = Objective.shape
    UCB = np.zeros((nBranches, nLevels))
    UCBSum = np.zeros((nBranches))

    CurrentUCB = Encoding.ObjectiveFunction(CurrentResponse, [1])

    for iBranch in range(nBranches):
        # print(iBranch)
        for iLevel in range(nLevels):
            UCB[iBranch, iLevel] = Objective[iBranch, iLevel]
            UCB[iBranch, iLevel] = UCB[iBranch, iLevel] - CurrentUCB
            if UCB[iBranch, iLevel] < 0:
                UCB[iBranch, iLevel] = 0
            UCB[iBranch, iLevel] = UCB[iBranch, iLevel] + lamb * (ObjectiveErr[iBranch, iLevel] ** 0.5)
        #     print(UCB[iBranch,iLevel])
        # print('\n')
        UCBSum[iBranch] = sum(UCB[iBranch, :])
    iRecommendedAction = np.argmax(UCBSum)
    RecommendedAction = Actions[iRecommendedAction, 0, :]
    return RecommendedAction, iRecommendedAction


def ExpectedImprovementDiscreteAvg(Objective, ObjectiveErr, Actions, CurrentResponse):
    lamb = 1 / (2 ** 0.5)
    # lamb=10
    nBranches, nLevels, nDim = Objective.shape
    UCB = np.zeros((nBranches, nLevels))
    UCBSum = np.zeros((nBranches))

    CurrentUCB = Encoding.ObjectiveFunction(CurrentResponse, [1])

    for iBranch in range(nBranches):
        # print(iBranch)
        for iLevel in range(nLevels):
            UCB[iBranch, iLevel] = Objective[iBranch, iLevel]
            UCB[iBranch, iLevel] = UCB[iBranch, iLevel] - CurrentUCB
            # if UCB[iBranch,iLevel] <0:
            #     UCB[iBranch,iLevel]=0
            UCB[iBranch, iLevel] = UCB[iBranch, iLevel] + lamb * ObjectiveErr[iBranch, iLevel]
        #     print(UCB[iBranch,iLevel])
        UCBSum[iBranch] = max(UCB[iBranch, :])

    Q = [[], [], [], []]
    Blocks = Actions[:, 0, 0]
    for iBranch in range(nBranches):
        Q[int(Blocks[iBranch]) - 1].append(UCBSum[iBranch])
    Qavg = []
    for q in Q:
        Qavg.append(sum(q) / len(q))

    print(Qavg)
    iRecommendedAction = np.argmax(Qavg) + 1
    RecommendedAction = [iRecommendedAction, 0.5, 0.5]
    return RecommendedAction, iRecommendedAction


def ExpectedImprovementDiscreteAvgDynamicLam(Objective, ObjectiveErr, Actions, CurrentResponse):
    lamb = 1 / (2 ** 0.5)
    # lamb=10
    nBranches, nLevels, nDim = Objective.shape
    Improvement = False
    iteration = 0
    while not (Improvement):
        iteration = iteration + 1
        lamb = lamb * iteration

        UCB = np.zeros((nBranches, nLevels))
        UCBSum = np.zeros((nBranches))

        CurrentUCB = Encoding.ObjectiveFunction(CurrentResponse, [1])

        for iBranch in range(nBranches):
            # print(iBranch)
            for iLevel in range(nLevels):
                UCB[iBranch, iLevel] = Objective[iBranch, iLevel]
                UCB[iBranch, iLevel] = UCB[iBranch, iLevel] - CurrentUCB

                UCB[iBranch, iLevel] = UCB[iBranch, iLevel] + lamb * ObjectiveErr[iBranch, iLevel]

            # if UCB[iBranch,iLevel] <0:
            #         UCB[iBranch,iLevel]=0

            #     print(UCB[iBranch,iLevel])
            UCBSum[iBranch] = max(UCB[iBranch, :])

        Q = [[], [], [], []]
        Blocks = Actions[:, 0, 0]
        for iBranch in range(nBranches):
            Q[int(Blocks[iBranch]) - 1].append(UCBSum[iBranch])
        Qavg = []
        for q in Q:
            Qavg.append(sum(q) / len(q))

        Qavg = np.array(Qavg)
        Improvement = any(Qavg > 0.05)

    print([iteration, Qavg])
    iRecommendedAction = np.argmax(Qavg) + 1
    RecommendedAction = [iRecommendedAction, 0.5, 0.5]
    return RecommendedAction, iRecommendedAction


def ExpectedImprovementDiscreteAvgDynamicLamDelayedReward(Objective, ObjectiveErr, Actions, CurrentResponse):
    lamb = 1 / (2 ** 0.5)
    # lamb=10
    nBranches, nLevels, nDim = Objective.shape
    Improvement = False
    iteration = 0
    while not (Improvement):
        iteration = iteration + 1
        lamb = lamb * iteration

        UCB = np.zeros((nBranches, nLevels))
        UCBCSum = np.zeros((nBranches, nLevels))
        UCBSum = np.zeros((nBranches))

        # CurrentUCB=Encoding.ObjectiveFunction(CurrentResponse,[1])
        CurrentUCB = Encoding.ObjectiveFunctionY1(CurrentResponse, [1])

        for iBranch in range(nBranches):
            # print(iBranch)
            for iLevel in range(nLevels):
                UCB[iBranch, iLevel] = Objective[iBranch, iLevel]
                # UCB[iBranch,iLevel]=UCB[iBranch,iLevel]-CurrentUCB

                UCB[iBranch, iLevel] = UCB[iBranch, iLevel] + lamb * ObjectiveErr[iBranch, iLevel]
                UCBCSum[iBranch, iLevel] = sum(UCB[iBranch, :])
            # if UCB[iBranch,iLevel] <0:
            #         UCB[iBranch,iLevel]=0

            #     print(UCB[iBranch,iLevel])
            # UCBSum[iBranch]=UCB[iBranch,-1]
            UCBSum[iBranch] = max(UCBCSum[iBranch, :])
        print(UCBCSum)
        Q = [[], [], [], []]
        Blocks = Actions[:, 0, 0]
        for iBranch in range(nBranches):
            Q[int(Blocks[iBranch]) - 1].append(UCBSum[iBranch])
        Qavg = []
        for q in Q:
            Qavg.append(sum(q) / len(q))
            # Qavg.append(max(q))
        Qavg = np.array(Qavg)
        Improvement = any(Qavg > 2)

    #     Qavg=pd.Series(Qavg)
    #     C=Qavg.cumsum()
    #     Qavg=np.array(Qavg)
    #     C=np.array(C)
    #     C=C/C[-1]

    #     iRecommendedAction=1
    #     RandC=rnd.random()
    #     while RandC > C[iRecommendedAction-1]:
    #         iRecommendedAction=iRecommendedAction+1

    iRecommendedAction = np.argmax(Qavg) + 1
    RecommendedAction = [iRecommendedAction, rnd.random(), rnd.random()]
    print([iteration, iRecommendedAction, Qavg])

    return RecommendedAction, iRecommendedAction


def ExploitationDiscreteAvg(Objective, ObjectiveErr, Actions, CurrentResponse):
    lamb = 1 / (2 ** 0.5)
    # lamb=10
    nBranches, nLevels, nDim = Objective.shape
    UCB = np.zeros((nBranches, nLevels))
    UCBSum = np.zeros((nBranches))

    CurrentUCB = Encoding.ObjectiveFunction(CurrentResponse, [1])

    for iBranch in range(nBranches):
        # print(iBranch)
        for iLevel in range(nLevels):
            UCB[iBranch, iLevel] = Objective[iBranch, iLevel]
            UCB[iBranch, iLevel] = UCB[iBranch, iLevel] - CurrentUCB
            if UCB[iBranch, iLevel] < 0:
                UCB[iBranch, iLevel] = 0
        #     print(UCB[iBranch,iLevel])
        UCBSum[iBranch] = sum(UCB[iBranch, :])

    Q = [[], [], [], []]
    Blocks = Actions[:, 0, 0]
    for iBranch in range(nBranches):
        Q[int(Blocks[iBranch]) - 1].append(UCBSum[iBranch])
    Qavg = []
    for q in Q:
        Qavg.append(sum(q) / len(q))

    print(Qavg)
    iRecommendedAction = np.argmax(Qavg) + 1
    RecommendedAction = [iRecommendedAction, 0.5, 0.5]
    return RecommendedAction, iRecommendedAction


def ExpectedImprovementDiscreteAvgCSum(Objective, ObjectiveErr, Actions, CurrentResponse):
    lamb = 1 / (2 ** 0.5)
    # lamb=10
    nBranches, nLevels, nDim = Objective.shape

    UCB = np.zeros((nBranches, nLevels))
    UCBCSum = np.zeros((nBranches, nLevels))
    UCBSum = np.zeros((nBranches))

    # CurrentUCB=Encoding.ObjectiveFunction(CurrentResponse,[1])
    CurrentUCB = Encoding.ObjectiveFunctionY1(CurrentResponse, [1])

    for iBranch in range(nBranches):
        # print(iBranch)
        for iLevel in range(nLevels):
            UCB[iBranch, iLevel] = Objective[iBranch, iLevel]
            # UCB[iBranch,iLevel]=UCB[iBranch,iLevel]-CurrentUCB

            UCB[iBranch, iLevel] = UCB[iBranch, iLevel] + lamb * ObjectiveErr[iBranch, iLevel]
            UCBCSum[iBranch, iLevel] = sum(UCB[iBranch, :])
        # if UCB[iBranch,iLevel] <0:
        #         UCB[iBranch,iLevel]=0

        #     print(UCB[iBranch,iLevel])
        # UCBSum[iBranch]=UCB[iBranch,-1]
        UCBSum[iBranch] = max(UCBCSum[iBranch, :])
    Q = [[], [], [], []]
    Blocks = Actions[:, 0, 0]
    for iBranch in range(nBranches):
        Q[int(Blocks[iBranch]) - 1].append(UCBSum[iBranch])
    Qavg = []
    for q in Q:
        # Qavg.append(sum(q)/len(q))
        Qavg.append(max(q))
    Qavg = np.array(Qavg)

    #     Qavg=pd.Series(Qavg)
    #     C=Qavg.cumsum()
    #     Qavg=np.array(Qavg)
    #     C=np.array(C)
    #     C=C/C[-1]

    #     iRecommendedAction=1
    #     RandC=rnd.random()
    #     while RandC > C[iRecommendedAction-1]:
    #         iRecommendedAction=iRecommendedAction+1

    iRecommendedAction = np.argmax(Qavg) + 1
    RecommendedAction = [iRecommendedAction, rnd.random(), rnd.random()]
    print([iRecommendedAction, Qavg])

    return RecommendedAction, iRecommendedAction


def ExpectedImprovementDiscreteAvgCSumXTuning(Objective, ObjectiveErr, Actions, CurrentResponse, Model,
                                              PredictionState):
    lamb = 1 / (2 ** 0.5)
    # lamb=10
    nBranches, nLevels, nDim = Objective.shape

    UCB = np.zeros((nBranches, nLevels))
    UCBCSum = np.zeros((nBranches, nLevels))
    UCBSum = np.zeros((nBranches))

    # CurrentUCB=Encoding.ObjectiveFunction(CurrentResponse,[1])
    CurrentUCB = Encoding.ObjectiveFunctionY1(CurrentResponse, [1])

    for iBranch in range(nBranches):
        # print(iBranch)
        for iLevel in range(nLevels):
            UCB[iBranch, iLevel] = Objective[iBranch, iLevel]
            # UCB[iBranch,iLevel]=UCB[iBranch,iLevel]-CurrentUCB

            UCB[iBranch, iLevel] = UCB[iBranch, iLevel] + lamb * (ObjectiveErr[iBranch, iLevel] ** 2)
            UCBCSum[iBranch, iLevel] = sum(UCB[iBranch, :])
        # if UCB[iBranch,iLevel] <0:
        #         UCB[iBranch,iLevel]=0

        #     print(UCB[iBranch,iLevel])
        # UCBSum[iBranch]=UCB[iBranch,-1]
        UCBSum[iBranch] = max(UCBCSum[iBranch, :])
    Q = [[], [], [], []]
    Blocks = Actions[:, 0, 0]
    for iBranch in range(nBranches):
        Q[int(Blocks[iBranch]) - 1].append(UCBSum[iBranch])
    Qavg = []
    for q in Q:
        q = np.sort(q)
        n25 = int(len(q) * 0.25)
        q = q[-n25:]
        Qavg.append(sum(q) / len(q))
        # Qavg.append(max(q))
    Qavg = np.array(Qavg)

    RecommendedInjection = np.argmax(Qavg) + 1
    iRecommendedAction = np.argmax(UCBSum)

    fun = lambda X: -TuneXFunction(X, Model, iRecommendedAction, PredictionState)
    Xopt = scipy.optimize.minimize(fun, [0.5, 0.5], bounds=[[0, 1], [0, 1]])

    X1 = Xopt.x[0]
    X2 = Xopt.x[1]

    RecommendedAction = [RecommendedInjection, X1, X2]

    # RecommendedAction=[iRecommendedAction,rnd.random(),rnd.random()]
    print([RecommendedAction, Qavg])

    return RecommendedAction, iRecommendedAction


def ExpectedImprovementDiscreteAvgCSumX1Epsilon(Objective, ObjectiveErr, Actions, CurrentResponse, Model,
                                         PredictionState):
    lamb = 1 / (2 ** 0.5)
    # lamb=10
    nBranches, nLevels, nDim = Objective.shape

    UCB = np.zeros((nBranches, nLevels))
    UCBCSum = np.zeros((nBranches, nLevels))
    UCBSum = np.zeros((nBranches))

    # CurrentUCB=Encoding.ObjectiveFunction(CurrentResponse,[1])
    CurrentUCB = Encoding.ObjectiveFunctionY1(CurrentResponse, [1])

    for iBranch in range(nBranches):
        # print(iBranch)
        for iLevel in range(nLevels):
            UCB[iBranch, iLevel] = Objective[iBranch, iLevel]
            # UCB[iBranch,iLevel]=UCB[iBranch,iLevel]-CurrentUCB

            UCB[iBranch, iLevel] = UCB[iBranch, iLevel] + lamb * (ObjectiveErr[iBranch, iLevel] ** 2)
            UCBCSum[iBranch, iLevel] = sum(UCB[iBranch, :])
        # if UCB[iBranch,iLevel] <0:
        #         UCB[iBranch,iLevel]=0

        #     print(UCB[iBranch,iLevel])
        # UCBSum[iBranch]=UCB[iBranch,-1]
        UCBSum[iBranch] = max(UCBCSum[iBranch, :])
    Q = [[], [], [], []]
    Blocks = Actions[:, 0, 0]
    for iBranch in range(nBranches):
        Q[int(Blocks[iBranch]) - 1].append(UCBSum[iBranch])
    Qavg = []
    for q in Q:
        q = np.sort(q)
        n25 = int(len(q) * 0.25)
        q = q[-n25:]
        Qavg.append(sum(q) / len(q))
        # Qavg.append(max(q))
    Qavg = np.array(Qavg)

    RecommendedInjection = np.argmax(Qavg) + 1
    iRecommendedAction = np.argmax(UCBSum)

    # fun = lambda X: -TuneXFunction(X, Model, iRecommendedAction, PredictionState)
    # Xopt = scipy.optimize.minimize(fun, [0.5, 0.5], bounds=[[0, 1], [0, 1]])

    # X1 = Xopt.x[0]
    # X2 = Xopt.x[1]

    if rnd.random() < 0.3:
        RecommendedInjection = rnd.randint(1, 4)

    if RecommendedInjection == 1:
        X1 = 0.11111
    elif RecommendedInjection == 2:
        X1 = 0.22222
    else:
        X1 = 0.55556

    X2 = 0.44444
    RecommendedAction = [RecommendedInjection, X1, X2]

    # RecommendedAction=[iRecommendedAction,rnd.random(),rnd.random()]
    print([RecommendedAction, Qavg])

    return RecommendedAction, iRecommendedAction

def UpperConfidenceBoundAvgCSumX1(Objective, ObjectiveErr, Actions, CurrentResponse, Model,
                                         PredictionState):
    lamb = 1 / (2 ** 0.5)
    # lamb=10
    nBranches, nLevels, nDim = Objective.shape

    UCB = np.zeros((nBranches, nLevels))
    UCBCSum = np.zeros((nBranches, nLevels))
    UCBSum = np.zeros((nBranches))

    # CurrentUCB=Encoding.ObjectiveFunction(CurrentResponse,[1])
    CurrentUCB = Encoding.ObjectiveFunctionY1(CurrentResponse, [1])

    for iBranch in range(nBranches):
        # print(iBranch)
        for iLevel in range(nLevels):
            UCB[iBranch, iLevel] = Objective[iBranch, iLevel]
            # UCB[iBranch,iLevel]=UCB[iBranch,iLevel]-CurrentUCB

            UCB[iBranch, iLevel] = UCB[iBranch, iLevel] + lamb * (ObjectiveErr[iBranch, iLevel])
            UCBCSum[iBranch, iLevel] = sum(UCB[iBranch, :])
        # if UCB[iBranch,iLevel] <0:
        #         UCB[iBranch,iLevel]=0

        #     print(UCB[iBranch,iLevel])
        # UCBSum[iBranch]=UCB[iBranch,-1]
        UCBSum[iBranch] = max(UCBCSum[iBranch, :])
    Q = [[], [], [], []]
    Blocks = Actions[:, 0, 0]
    for iBranch in range(nBranches):
        Q[int(Blocks[iBranch]) - 1].append(UCBSum[iBranch])
    Qavg = []
    for q in Q:
        q = np.sort(q)
        n25 = int(len(q) * 0.25)
        q = q[-n25:]
        Qavg.append(sum(q) / len(q))
        # Qavg.append(max(q))
    Qavg = np.array(Qavg)

    RecommendedInjection = np.argmax(Qavg) + 1
    iRecommendedAction = np.argmax(UCBSum)

    # fun = lambda X: -TuneXFunction(X, Model, iRecommendedAction, PredictionState)
    # Xopt = scipy.optimize.minimize(fun, [0.5, 0.5], bounds=[[0, 1], [0, 1]])

    # X1 = Xopt.x[0]
    # X2 = Xopt.x[1]


    if RecommendedInjection == 1:
        X1 = 0.11111
    elif RecommendedInjection == 2:
        X1 = 0.22222
    else:
        X1 = 0.55556

    X2 = 0.44444
    RecommendedAction = [RecommendedInjection, X1, X2]

    # RecommendedAction=[iRecommendedAction,rnd.random(),rnd.random()]
    print([RecommendedAction, Qavg])

    return RecommendedAction, iRecommendedAction

def UpperConfidenceBounds1(Objective, ObjectiveErr, Actions, CurrentResponse, Model,
                                         PredictionState,X):
    lamb = 1 / (2 ** 0.5)
    # lamb=10
    nBranches, nLevels, nDim = Objective.shape

    UCB = np.zeros((nBranches, nLevels))
    UCBCSum = np.zeros((nBranches, nLevels))
    UCBSum = np.zeros((nBranches))

    # CurrentUCB=Encoding.ObjectiveFunction(CurrentResponse,[1])
    CurrentUCB = Encoding.ObjectiveFunctionY1(CurrentResponse, [1])

    for iBranch in range(nBranches):
        # print(iBranch)
        for iLevel in range(nLevels):
            UCB[iBranch, iLevel] = Objective[iBranch, iLevel]
            # UCB[iBranch,iLevel]=UCB[iBranch,iLevel]-CurrentUCB

            UCB[iBranch, iLevel] = UCB[iBranch, iLevel] + lamb * (ObjectiveErr[iBranch, iLevel])
            UCBCSum[iBranch, iLevel] = sum(UCB[iBranch, :])
        # if UCB[iBranch,iLevel] <0:
        #         UCB[iBranch,iLevel]=0

        #     print(UCB[iBranch,iLevel])
        # UCBSum[iBranch]=UCB[iBranch,-1]
        UCBSum[iBranch] = max(UCBCSum[iBranch, :])
    Q = [[], [], [], []]
    Blocks = Actions[:, 0, 0]
    for iBranch in range(nBranches):
        Q[int(Blocks[iBranch]) - 1].append(UCBSum[iBranch])
    Qavg = []
    for q in Q:
        q = np.sort(q)
        n25 = int(len(q) * 0.25)
        q = q[-n25:]
        Qavg.append(sum(q) / len(q))
        # Qavg.append(max(q))
    Qavg = np.array(Qavg)

    N1 = 0.5 * math.sqrt(2 * math.log(sum(X[:, 0] != 0)+1) / (sum(X[:, 0] == 1)+1))
    N2 = 0.5 * math.sqrt(2 * math.log(sum(X[:, 0] != 0)+1) / (sum(X[:, 0] == 2)+1))
    N3 = 0.5 * math.sqrt(2 * math.log(sum(X[:, 0] != 0)+1) / (sum(X[:, 0] == 3)+1))
    N4 = 0.5 * math.sqrt(2 * math.log(sum(X[:, 0] != 0)+1) / (sum(X[:, 0] == 4)+1))
    N = [N1,N2,N3,N4]

    U = Qavg + N
    RecommendedInjection = np.argmax(U) + 1
    iRecommendedAction = np.argmax(UCBSum)

    # fun = lambda X: -TuneXFunction(X, Model, iRecommendedAction, PredictionState)
    # Xopt = scipy.optimize.minimize(fun, [0.5, 0.5], bounds=[[0, 1], [0, 1]])

    # X1 = Xopt.x[0]
    # X2 = Xopt.x[1]

    if RecommendedInjection == 1:
        X1 = 0.11111
    elif RecommendedInjection == 2:
        X1 = 0.22222
    else:
        X1 = 0.55556

    X2 = 0.44444
    RecommendedAction = [RecommendedInjection, X1, X2]

    # RecommendedAction=[iRecommendedAction,rnd.random(),rnd.random()]
    print([RecommendedAction, Qavg])

    return RecommendedAction, iRecommendedAction

def UpperConfidenceBounds1StateSpecific(Objective, ObjectiveErr, Actions, CurrentResponse, Model,
                                         PredictionState,X,State):
    lamb = 1 / (2 ** 0.5)
    # lamb=10
    nBranches, nLevels, nDim = Objective.shape

    UCB = np.zeros((nBranches, nLevels))
    UCBCSum = np.zeros((nBranches, nLevels))
    UCBSum = np.zeros((nBranches))

    # CurrentUCB=Encoding.ObjectiveFunction(CurrentResponse,[1])
    CurrentUCB = Encoding.ObjectiveFunctionY1(CurrentResponse, [1])

    for iBranch in range(nBranches):
        # print(iBranch)
        for iLevel in range(nLevels):
            UCB[iBranch, iLevel] = Objective[iBranch, iLevel]
            # UCB[iBranch,iLevel]=UCB[iBranch,iLevel]-CurrentUCB

            UCB[iBranch, iLevel] = UCB[iBranch, iLevel] + lamb * (ObjectiveErr[iBranch, iLevel] ** 2)
            UCBCSum[iBranch, iLevel] = sum(UCB[iBranch, :])
        # if UCB[iBranch,iLevel] <0:
        #         UCB[iBranch,iLevel]=0

        #     print(UCB[iBranch,iLevel])
        # UCBSum[iBranch]=UCB[iBranch,-1]
        UCBSum[iBranch] = max(UCBCSum[iBranch, :])
    Q = [[], [], [], []]
    Blocks = Actions[:, 0, 0]
    for iBranch in range(nBranches):
        Q[int(Blocks[iBranch]) - 1].append(UCBSum[iBranch])
    Qavg = []
    for q in Q:
        q = np.sort(q)
        n25 = int(len(q) * 0.25)
        q = q[-n25:]
        Qavg.append(sum(q) / len(q))
        # Qavg.append(max(q))
    Qavg = np.array(Qavg)

    N1 = math.sqrt(2 * math.log(sum(X[:, 0] != 0)+1) / (sum(X[:, 0] == 1)+1))
    N2 = math.sqrt(2 * math.log(sum(X[:, 0] != 0)+1) / (sum(X[:, 0] == 2)+1))
    N3 = math.sqrt(2 * math.log(sum(X[:, 0] != 0)+1) / (sum(X[:, 0] == 3)+1))
    N4 = math.sqrt(2 * math.log(sum(X[:, 0] != 0)+1) / (sum(X[:, 0] == 4)+1))
    N = [N1,N2,N3,N4]

    U = Qavg + N
    RecommendedInjection = np.argmax(U) + 1
    iRecommendedAction = np.argmax(UCBSum)

    # fun = lambda X: -TuneXFunction(X, Model, iRecommendedAction, PredictionState)
    # Xopt = scipy.optimize.minimize(fun, [0.5, 0.5], bounds=[[0, 1], [0, 1]])

    # X1 = Xopt.x[0]
    # X2 = Xopt.x[1]

    if RecommendedInjection == 1:
        X1 = 0.11111
    elif RecommendedInjection == 2:
        X1 = 0.22222
    else:
        X1 = 0.55556

    X2 = 0.44444
    RecommendedAction = [RecommendedInjection, X1, X2]

    # RecommendedAction=[iRecommendedAction,rnd.random(),rnd.random()]
    print([RecommendedAction, Qavg])

    return RecommendedAction, iRecommendedAction

def TuneXFunction(X, Model, iRecommendedAction, PredictionState):
    lam = 1 / (2 ** 0.5)
    TestState = PredictionState[iRecommendedAction, 0]
    # print(X1)
    # print(X2)
    TestState[4] = X[0]
    TestState[5] = X[1]
    # print(TestState)
    Output = BeliefModel.EnsembleMeanY(Model, [TestState]) + lam * (BeliefModel.EnsembleStdevY(Model, [TestState]) ** 2)
    return Output

def UpperConfidenceBounds1EnsembleSubSampling(Objective, ObjectiveErr, Actions, CurrentResponse, Model,
                                         PredictionState,X):
    #lamb = 2 / (2 ** 0.5)
    lamb = 1/(2**0.5)
    # lamb=10
    nBranches, nLevels = Objective.shape

    UCB = np.zeros((nBranches, nLevels))
    UCBCSum = np.zeros((nBranches, nLevels))
    UCBSum = np.zeros((nBranches))

    # CurrentUCB=Encoding.ObjectiveFunction(CurrentResponse,[1])
    CurrentUCB = Encoding.ObjectiveFunctionY1(CurrentResponse, [1])

    for iBranch in range(nBranches):
        # print(iBranch)
        for iLevel in range(nLevels):
            UCB[iBranch, iLevel] = Objective[iBranch, iLevel]
            # UCB[iBranch,iLevel]=UCB[iBranch,iLevel]-CurrentUCB

            UCB[iBranch, iLevel] = UCB[iBranch, iLevel] + lamb * (ObjectiveErr[iBranch, iLevel])
            # UCB[iBranch, iLevel] = lamb * (ObjectiveErr[iBranch, iLevel])
            # # UCBCSum[iBranch, iLevel] = sum(UCB[iBranch, :])
            UCBCSum[iBranch, iLevel] = UCB[iBranch, iLevel]

        # if UCB[iBranch,iLevel] <0:
        #         UCB[iBranch,iLevel]=0

        #     print(UCB[iBranch,iLevel])
        # UCBSum[iBranch]=UCB[iBranch,-1]

        UCBSum[iBranch] = max(UCBCSum[iBranch, :])
        # UCBSum[iBranch] = sum(UCBCSum[iBranch, :])
    Q = [[], [], [], []]
    Blocks = Actions[:, 0, 0]
    for iBranch in range(nBranches):
        Q[int(Blocks[iBranch]) - 1].append(UCBSum[iBranch])
    Qavg = []
    for q in Q:
        q = np.sort(q)
        n25 = int(len(q) * 0.25)
        q = q[-n25:]
        Qavg.append(sum(q) / len(q))
        # Qavg.append(max(q))
    Qavg = np.array(Qavg)

    N1 = math.sqrt(2 * math.log(sum(X[:, 0] != 0)+1) / (sum(X[:, 0] == 1)+1))
    N2 = math.sqrt(2 * math.log(sum(X[:, 0] != 0)+1) / (sum(X[:, 0] == 2)+1))
    N3 = math.sqrt(2 * math.log(sum(X[:, 0] != 0)+1) / (sum(X[:, 0] == 3)+1))
    N4 = math.sqrt(2 * math.log(sum(X[:, 0] != 0)+1) / (sum(X[:, 0] == 4)+1))
    N = [N1,N2,N3,N4]

    U = Qavg + N
    RecommendedInjection = np.argmax(U) + 1
    iRecommendedAction = np.argmax(UCBSum)

    # fun = lambda X: -TuneXFunction(X, Model, iRecommendedAction, PredictionState)
    # Xopt = scipy.optimize.minimize(fun, [0.5, 0.5], bounds=[[0, 1], [0, 1]])

    # X1 = Xopt.x[0]
    # X2 = Xopt.x[1]

    if RecommendedInjection == 1:
        X1 = 0.11111
    elif RecommendedInjection == 2:
        X1 = 0.22222
    else:
        X1 = 0.55556

    X2 = 0.44444
    RecommendedAction = [RecommendedInjection, X1, X2]

    # RecommendedAction=[iRecommendedAction,rnd.random(),rnd.random()]
    print([RecommendedAction, Qavg])

    return RecommendedAction, iRecommendedAction


def UpperConfidenceBounds1EnsembleSubSamplingReplicatePenalty(Objective, ObjectiveErr, Actions, CurrentResponse, Model,
                                         PredictionState,State,X):
    lamb = 1 / (2 ** 0.5)
    # lamb=10
    nBranches, nLevels = Objective.shape

    UCB = np.zeros((nBranches, nLevels))
    UCBCSum = np.zeros((nBranches, nLevels))
    UCBSum = np.zeros((nBranches))

    # CurrentUCB=Encoding.ObjectiveFunction(CurrentResponse,[1])
    CurrentUCB = Encoding.ObjectiveFunctionY1(CurrentResponse, [1])

    ReplicateCounts = Encoding.BuildReplicateList(State, PredictionState)

    for iBranch in range(nBranches):
        # print(iBranch)
        for iLevel in range(nLevels):
            UCB[iBranch, iLevel] = Objective[iBranch, iLevel]
            # UCB[iBranch,iLevel]=UCB[iBranch,iLevel]-CurrentUCB

            UCB[iBranch, iLevel] = UCB[iBranch, iLevel] + lamb * (ObjectiveErr[iBranch, iLevel])
            # UCB[iBranch, iLevel] = lamb * (ObjectiveErr[iBranch, iLevel])
            # # UCBCSum[iBranch, iLevel] = sum(UCB[iBranch, :])

            ReplicatePenaltyFactor=0.8+(0.2*rnd.random())
            UCBCSum[iBranch, iLevel] = UCB[iBranch, iLevel] * (ReplicatePenaltyFactor ** ReplicateCounts[iBranch,iLevel])

        # if UCB[iBranch,iLevel] <0:
        #         UCB[iBranch,iLevel]=0

        #     print(UCB[iBranch,iLevel])
        # UCBSum[iBranch]=UCB[iBranch,-1]

        UCBSum[iBranch] = max(UCBCSum[iBranch, :])
        # UCBSum[iBranch] = sum(UCBCSum[iBranch, :])
    Q = [[], [], [], []]
    Blocks = Actions[:, 0, 0]
    for iBranch in range(nBranches):
        Q[int(Blocks[iBranch]) - 1].append(UCBSum[iBranch])
    Qavg = []
    for q in Q:
        q = np.sort(q)
        n25 = int(len(q) * 0.25)
        q = q[-n25:]
        Qavg.append(sum(q) / len(q))
        # Qavg.append(max(q))
    Qavg = np.array(Qavg)

    N1 = math.sqrt(2 * math.log(sum(X[:, 0] != 0)+1) / sum(X[:, 0] == 1)+1)
    N2 = math.sqrt(2 * math.log(sum(X[:, 0] != 0)+1) / sum(X[:, 0] == 2)+1)
    N3 = math.sqrt(2 * math.log(sum(X[:, 0] != 0)+1) / sum(X[:, 0] == 3)+1)
    N4 = math.sqrt(2 * math.log(sum(X[:, 0] != 0)+1) / sum(X[:, 0] == 4)+1)
    N = [N1,N2,N3,N4]

    U = Qavg + N
    RecommendedInjection = np.argmax(U) + 1
    iRecommendedAction = np.argmax(UCBSum)

    # fun = lambda X: -TuneXFunction(X, Model, iRecommendedAction, PredictionState)
    # Xopt = scipy.optimize.minimize(fun, [0.5, 0.5], bounds=[[0, 1], [0, 1]])

    # X1 = Xopt.x[0]
    # X2 = Xopt.x[1]

    if RecommendedInjection == 1:
        X1 = 0.11111
    elif RecommendedInjection == 2:
        X1 = 0.22222
    else:
        X1 = 0.55556

    X2 = 0.44444
    RecommendedAction = [RecommendedInjection, X1, X2]

    # RecommendedAction=[iRecommendedAction,rnd.random(),rnd.random()]
    print([RecommendedAction, Qavg])

    return RecommendedAction, iRecommendedAction

def UCBNodeRandomBranchSum(Objective, Nodes):
    lamb = 1 / (2 ** 0.5)

    nNodes, nNodeBranches, nLevels = Objective.shape
    BranchObjectiveSum = np.zeros((nNodes, nNodeBranches, nLevels))
    BranchObjectiveSumMax = np.zeros((nNodes, nNodeBranches))
    for iNode in range(nNodes):
        for iNodeBranch in range(nNodeBranches):
            for iLevel in range(nLevels):
                BranchObjectiveSum[iNode, iNodeBranch, iLevel] = sum(Objective[iNode, iNodeBranch, 0:iLevel])
    for iNode in range(nNodes):
        for iNodeBranch in range(nNodeBranches):
            BranchObjectiveSumMax[iNode, iNodeBranch] = max(BranchObjectiveSum[iNode, iNodeBranch])


    NodeMean = np.zeros((nNodes))
    NodeStdev = np.zeros((nNodes))
    NodeUCB = np.zeros((nNodes))

    for iNode in range(nNodes):
        NodeMean[iNode] = np.average(BranchObjectiveSum[iNode,:])
        NodeStdev[iNode] = np.std(BranchObjectiveSum[iNode,:])

        NodeUCB[iNode] = NodeMean[iNode] + lamb * NodeStdev[iNode]


    iRec = np.argmax(NodeUCB)

    RecommendedAction = [1, Nodes[iRec][0], Nodes[iRec][1]]

    return RecommendedAction

def ExploitNodeRandomBranchSum(Objective, Nodes):
    lamb = 0

    nNodes, nNodeBranches, nLevels = Objective.shape
    BranchObjectiveSum = np.zeros((nNodes, nNodeBranches, nLevels))
    BranchObjectiveSumMax = np.zeros((nNodes, nNodeBranches))
    for iNode in range(nNodes):
        for iNodeBranch in range(nNodeBranches):
            for iLevel in range(nLevels):
                BranchObjectiveSum[iNode, iNodeBranch, iLevel] = sum(Objective[iNode, iNodeBranch, 0:iLevel])
    for iNode in range(nNodes):
        for iNodeBranch in range(nNodeBranches):
            BranchObjectiveSumMax[iNode, iNodeBranch] = max(BranchObjectiveSum[iNode, iNodeBranch])


    NodeMean = np.zeros((nNodes))
    NodeStdev = np.zeros((nNodes))
    NodeUCB = np.zeros((nNodes))

    for iNode in range(nNodes):
        NodeMean[iNode] = np.average(BranchObjectiveSum[iNode,:])
        NodeStdev[iNode] = np.std(BranchObjectiveSum[iNode,:])

        NodeUCB[iNode] = NodeMean[iNode] + lamb * NodeStdev[iNode]


    iRec = np.argmax(NodeUCB)

    RecommendedAction = [1, Nodes[iRec][0], Nodes[iRec][1]]

    return RecommendedAction