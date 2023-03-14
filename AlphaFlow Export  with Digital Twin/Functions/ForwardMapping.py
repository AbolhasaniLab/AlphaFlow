from Functions import Encoding
from Functions import BeliefModel

import numpy as np

import random as rnd


def GridMonteCarloObjective(Model, ClassModel, X, Y, nBranches, nLevels, MemorySteps, BlockEncoding,
                            RegressionModelStructure, ClassModelStructure):
    # Build random action grid of nBranches by nLevels and format into short term memory
    X = X[-(MemorySteps + 1):]
    Y = Y[-(MemorySteps + 1):]

    Actions = np.zeros((nBranches, nLevels, 3))
    STM = []
    ClassSTM = []
    for iBranch in range(nBranches):
        tempXAction = []
        tempXAction.append(X[:])
        for iLevel in range(nLevels):
            Actions[iBranch, iLevel, 0] = rnd.randint(1, 4)
            Actions[iBranch, iLevel, 1] = rnd.random()
            Actions[iBranch, iLevel, 2] = rnd.random()
            tempXAction.append(Actions[iBranch, iLevel, :])
        tempXAction = np.array(tempXAction, dtype=object)
        tempXAction = np.vstack(tempXAction)
        tempSTM = Encoding.ShortTermMemory(tempXAction, MemorySteps, BlockEncoding)
        tempClassSTM = Encoding.ClassShortTermMemory(tempXAction, MemorySteps, BlockEncoding)
        ClassSTM.append(tempClassSTM)
        STM.append(tempSTM)
    STM = np.array(STM)
    ClassSTM = np.array(ClassSTM)

    PredictionSTM = STM[:, (len(Y)):None]
    PredictionClassSTM = ClassSTM[:, (len(Y)):None]

    PredictionState = []
    PredictionClassState = []
    PredictionResponse = []
    PredictionResponseErr = []
    PredictionProbability = []

    Objective = []
    ObjectiveErr = []

    for iBranch in range(nBranches):
        tempState = []
        tempClassState = []
        PredictionStateBranch = []
        PredictionClassStateBranch = []
        PredictionResponseBranch = []
        PredictionResponseErrBranch = []
        PredictionProbabilityBranch = []

        ObjectiveBranch = []
        ObjectiveErrBranch = []

        tempState = PredictionSTM[iBranch, 0]
        tempState = np.concatenate((tempState, Y[-1]))
        PredictionStateBranch.append(np.array(tempState))

        tempClassState = PredictionClassSTM[iBranch, 0]
        # tempClassState=np.concatenate((tempClassState,Y[-1]))
        PredictionClassStateBranch.append(np.array(tempClassState))

        if RegressionModelStructure == 'Ensemble':
            tempY = BeliefModel.EnsembleMeanY(Model, PredictionStateBranch)
            tempYErr = BeliefModel.EnsembleStdevY(Model, PredictionStateBranch)
        elif RegressionModelStructure == 'SKLearn Ensemble':
            tempY = BeliefModel.RegressionPredictY(Model, PredictionStateBranch)
            tempYErr = BeliefModel.RegressionPredictYErr(Model, PredictionStateBranch)

        PredictionResponseBranch.append(tempY)
        PredictionResponseErrBranch.append(tempYErr)

        if ClassModelStructure == 'Ensemble':
            tempP = BeliefModel.EnsembleClassProbability(ClassModel, PredictionClassStateBranch)
        elif ClassModelStructure == 'SKLearn Ensemble':
            tempP = BeliefModel.ClassProbability(ClassModel, PredictionClassStateBranch)

        PredictionProbabilityBranch.append(np.array(tempP))

        tempObjectiveVal = Encoding.ObjectiveFunction(tempY, tempP)
        tempObjectiveErr = Encoding.ObjectiveFunctionErr(tempYErr, tempP)
        ObjectiveBranch.append(tempObjectiveVal)
        ObjectiveErrBranch.append(tempObjectiveErr)

        for iLevel in range(nLevels - 1):
            tempState = []
            tempState = PredictionSTM[iBranch, (iLevel + 1)]
            tempState = np.concatenate((tempState, tempY[0]))
            tempClassState = []
            tempClassState = PredictionClassSTM[iBranch, (iLevel + 1)]
            # tempClassState=np.concatenate((tempClassState,tempY[0]))
            tempY = []
            tempYErr = []
            tempStatemat = []
            tempStatemat.append(np.array(tempState))
            tempClassStatemat = []
            tempClassStatemat.append(np.array(tempClassState))

            if RegressionModelStructure == 'Ensemble':
                tempY = BeliefModel.EnsembleMeanY(Model, tempStatemat)
                tempYErr = BeliefModel.EnsembleStdevY(Model, tempStatemat)
            elif RegressionModelStructure == 'SKLearn Ensemble':
                tempY = BeliefModel.RegressionPredictY(Model, tempStatemat)
                tempYErr = BeliefModel.RegressionPredictYErr(Model, tempStatemat)

            if ClassModelStructure == 'Ensemble':
                tempP = BeliefModel.EnsembleClassProbability(ClassModel, tempClassStatemat)
            elif ClassModelStructure == 'SKLearn Ensemble':
                tempP = BeliefModel.ClassProbability(ClassModel, tempClassStatemat)

            tempP = tempP * PredictionProbabilityBranch[iLevel]

            PredictionStateBranch.append(tempState)
            PredictionClassStateBranch.append(tempClassState)
            PredictionResponseBranch.append(tempY)
            PredictionResponseErrBranch.append(tempYErr)
            PredictionProbabilityBranch.append(np.array(tempP))

            tempObjectiveVal = Encoding.ObjectiveFunction(tempY, tempP)
            tempObjectiveErr = Encoding.ObjectiveFunctionErr(tempYErr, tempP)
            ObjectiveBranch.append(tempObjectiveVal)
            ObjectiveErrBranch.append(tempObjectiveErr)

        PredictionState.append(PredictionStateBranch)
        PredictionClassState.append(PredictionClassStateBranch)
        PredictionResponse.append(PredictionResponseBranch)
        PredictionResponseErr.append(PredictionResponseErrBranch)
        PredictionProbability.append(PredictionProbabilityBranch)

        Objective.append(ObjectiveBranch)
        ObjectiveErr.append(ObjectiveErrBranch)
    PredictionState = np.array(PredictionState)
    PredictionClassState = np.array(PredictionClassState)
    PredictionResponse = np.array(PredictionResponse)
    PredictionResponseErr = np.array(PredictionResponseErr)
    PredictionProbability = np.array(PredictionProbability)

    Objective = np.array(Objective)
    ObjectiveErr = np.array(ObjectiveErr)

    return Objective, ObjectiveErr, Actions, PredictionState, PredictionClassState, PredictionResponse, PredictionProbability


def GridMonteCarloY1(Model, ClassModel, X, Y, nBranches, nLevels, MemorySteps, BlockEncoding, RegressionModelStructure,
                     ClassModelStructure):
    # Build random action grid of nBranches by nLevels and format into short term memory
    X = X[-(MemorySteps + 1):]
    Y = Y[-(MemorySteps + 1):]

    Actions = np.zeros((nBranches, nLevels, 3))
    STM = []
    ClassSTM = []
    for iBranch in range(nBranches):
        tempXAction = []
        tempXAction.append(X[:])
        for iLevel in range(nLevels):
            Actions[iBranch, iLevel, 0] = rnd.randint(1, 4)
            Actions[iBranch, iLevel, 1] = rnd.random()
            Actions[iBranch, iLevel, 2] = rnd.random()
            tempXAction.append(Actions[iBranch, iLevel, :])
        tempXAction = np.array(tempXAction, dtype=object)
        tempXAction = np.vstack(tempXAction)
        tempSTM = Encoding.ShortTermMemory(tempXAction, MemorySteps, BlockEncoding)
        tempClassSTM = Encoding.ClassShortTermMemory(tempXAction, MemorySteps, BlockEncoding)
        ClassSTM.append(tempClassSTM)
        STM.append(tempSTM)
    STM = np.array(STM)
    ClassSTM = np.array(ClassSTM)

    PredictionSTM = STM[:, (len(Y)):None]
    PredictionClassSTM = ClassSTM[:, (len(Y)):None]

    PredictionState = []
    PredictionClassState = []
    PredictionResponse = []
    PredictionResponseErr = []
    PredictionProbability = []

    Objective = []
    ObjectiveErr = []

    for iBranch in range(nBranches):
        tempState = []
        tempClassState = []
        PredictionStateBranch = []
        PredictionClassStateBranch = []
        PredictionResponseBranch = []
        PredictionResponseErrBranch = []
        PredictionProbabilityBranch = []

        ObjectiveBranch = []
        ObjectiveErrBranch = []

        tempState = PredictionSTM[iBranch, 0]
        # tempState=np.concatenate((tempState,Y[-1]))
        PredictionStateBranch.append(np.array(tempState))

        tempClassState = PredictionClassSTM[iBranch, 0]
        # tempClassState=np.concatenate((tempClassState,Y[-1]))
        PredictionClassStateBranch.append(np.array(tempClassState))

        if RegressionModelStructure == 'Ensemble':
            tempY = BeliefModel.EnsembleMeanY(Model, PredictionStateBranch)
            tempYErr = BeliefModel.EnsembleStdevY(Model, PredictionStateBranch)
        elif RegressionModelStructure == 'SKLearn Ensemble':
            tempY = BeliefModel.RegressionPredictY(Model, PredictionStateBranch)
            tempYErr = BeliefModel.RegressionPredictYErr(Model, PredictionStateBranch)

        PredictionResponseBranch.append(tempY)
        PredictionResponseErrBranch.append(tempYErr)

        if ClassModelStructure == 'Ensemble':
            tempP = BeliefModel.EnsembleClassProbability(ClassModel, PredictionClassStateBranch)
        elif ClassModelStructure == 'SKLearn Ensemble':
            tempP = BeliefModel.ClassProbability(ClassModel, PredictionClassStateBranch)

        PredictionProbabilityBranch.append(np.array(tempP))

        tempObjectiveVal = Encoding.ObjectiveFunctionY1(tempY, tempP)
        tempObjectiveErr = Encoding.ObjectiveFunctionErrY1(tempYErr, tempP)
        ObjectiveBranch.append(tempObjectiveVal)
        ObjectiveErrBranch.append(tempObjectiveErr)

        for iLevel in range(nLevels - 1):
            tempState = []
            tempState = PredictionSTM[iBranch, (iLevel + 1)]
            # tempState=np.concatenate((tempState,[tempY[0]]))
            tempClassState = []
            tempClassState = PredictionClassSTM[iBranch, (iLevel + 1)]
            # tempClassState=np.concatenate((tempClassState,tempY[0]))
            tempY = []
            tempYErr = []
            tempStatemat = []
            tempStatemat.append(np.array(tempState))
            tempClassStatemat = []
            tempClassStatemat.append(np.array(tempClassState))

            if RegressionModelStructure == 'Ensemble':
                tempY = BeliefModel.EnsembleMeanY(Model, tempStatemat)
                tempYErr = BeliefModel.EnsembleStdevY(Model, tempStatemat)
            elif RegressionModelStructure == 'SKLearn Ensemble':
                tempY = BeliefModel.RegressionPredictY(Model, tempStatemat)
                tempYErr = BeliefModel.RegressionPredictYErr(Model, tempStatemat)

            if ClassModelStructure == 'Ensemble':
                tempP = BeliefModel.EnsembleClassProbability(ClassModel, tempClassStatemat)
            elif ClassModelStructure == 'SKLearn Ensemble':
                tempP = BeliefModel.ClassProbability(ClassModel, tempClassStatemat)

            tempP = tempP * PredictionProbabilityBranch[iLevel]

            PredictionStateBranch.append(tempState)
            PredictionClassStateBranch.append(tempClassState)
            PredictionResponseBranch.append(tempY)
            PredictionResponseErrBranch.append(tempYErr)
            PredictionProbabilityBranch.append(np.array(tempP))

            tempObjectiveVal = Encoding.ObjectiveFunctionY1(tempY, tempP)
            tempObjectiveErr = Encoding.ObjectiveFunctionErrY1(tempYErr, tempP)
            ObjectiveBranch.append(tempObjectiveVal)
            ObjectiveErrBranch.append(tempObjectiveErr)

        PredictionState.append(PredictionStateBranch)
        PredictionClassState.append(PredictionClassStateBranch)
        PredictionResponse.append(PredictionResponseBranch)
        PredictionResponseErr.append(PredictionResponseErrBranch)
        PredictionProbability.append(PredictionProbabilityBranch)

        Objective.append(ObjectiveBranch)
        ObjectiveErr.append(ObjectiveErrBranch)
    PredictionState = np.array(PredictionState)
    PredictionClassState = np.array(PredictionClassState)
    PredictionResponse = np.array(PredictionResponse)
    PredictionResponseErr = np.array(PredictionResponseErr)
    PredictionProbability = np.array(PredictionProbability)

    Objective = np.array(Objective)
    ObjectiveErr = np.array(ObjectiveErr)

    return Objective, ObjectiveErr, Actions, PredictionState, PredictionClassState, PredictionResponse, PredictionProbability


def PermutationMonteCarloY1(Model, ClassModel, X, Y, PermDuplicates, nLevels, MemorySteps, BlockEncoding, RegressionModelStructure,
                            ClassModelStructure):
    # Build random action grid of nBranches by nLevels and format into short term memory
    X = X[-(MemorySteps + 1):]
    Y = Y[-(MemorySteps + 1):]
#?update memory steps ?

    Injections = [1, 2, 3, 4]
    InjectionPermutations = [[iia, iib, iic, iid] for dup in range(PermDuplicates)
                             for iia in Injections
                             for iib in Injections
                             for iic in Injections
                             for iid in Injections]


    nBranches=len(InjectionPermutations)

    Actions = np.zeros((nBranches, nLevels, 3))
    STM = []
    ClassSTM = []

    for iBranch in range(nBranches):
        tempXAction = []
        tempXAction.append(X[:])
        for iLevel in range(nLevels):
            Actions[iBranch, iLevel, 0] = InjectionPermutations[iBranch][iLevel]
            Actions[iBranch, iLevel, 1] = rnd.random()
            Actions[iBranch, iLevel, 2] = rnd.random()
            tempXAction.append(Actions[iBranch, iLevel, :])
        tempXAction = np.array(tempXAction, dtype=object)
        tempXAction = np.vstack(tempXAction)
        tempSTM = Encoding.ShortTermMemory(tempXAction, MemorySteps, BlockEncoding)
        tempClassSTM = Encoding.ClassShortTermMemory(tempXAction, MemorySteps, BlockEncoding)
        ClassSTM.append(tempClassSTM)
        STM.append(tempSTM)
    STM = np.array(STM)
    ClassSTM = np.array(ClassSTM)

    PredictionSTM = STM[:, (len(Y)):None]
    PredictionClassSTM = ClassSTM[:, (len(Y)):None]

    PredictionState = []
    PredictionClassState = []
    PredictionResponse = []
    PredictionResponseErr = []
    PredictionProbability = []

    Objective = []
    ObjectiveErr = []

    for iBranch in range(nBranches):
        tempState = []
        tempClassState = []
        PredictionStateBranch = []
        PredictionClassStateBranch = []
        PredictionResponseBranch = []
        PredictionResponseErrBranch = []
        PredictionProbabilityBranch = []

        ObjectiveBranch = []
        ObjectiveErrBranch = []

        tempState = PredictionSTM[iBranch, 0]
        # tempState=np.concatenate((tempState,Y[-1]))
        PredictionStateBranch.append(np.array(tempState))

        tempClassState = PredictionClassSTM[iBranch, 0]
        # tempClassState=np.concatenate((tempClassState,Y[-1]))
        PredictionClassStateBranch.append(np.array(tempClassState))

        if RegressionModelStructure == 'Ensemble':
            tempY = BeliefModel.EnsembleMeanY(Model, PredictionStateBranch)
            tempYErr = BeliefModel.EnsembleStdevY(Model, PredictionStateBranch)
        elif RegressionModelStructure == 'SKLearn Ensemble':
            tempY = BeliefModel.RegressionPredictY(Model, PredictionStateBranch)
            tempYErr = BeliefModel.RegressionPredictYErr(Model, PredictionStateBranch)

        PredictionResponseBranch.append(tempY)
        PredictionResponseErrBranch.append(tempYErr)

        if ClassModelStructure == 'Ensemble':
            tempP = BeliefModel.EnsembleClassProbability(ClassModel, PredictionClassStateBranch)
        elif ClassModelStructure == 'SKLearn Ensemble':
            tempP = BeliefModel.ClassProbability(ClassModel, PredictionClassStateBranch)

        PredictionProbabilityBranch.append(np.array(tempP))

        tempObjectiveVal = Encoding.ObjectiveFunctionY1(tempY, tempP)
        tempObjectiveErr = Encoding.ObjectiveFunctionErrY1(tempYErr, tempP)
        ObjectiveBranch.append(tempObjectiveVal)
        ObjectiveErrBranch.append(tempObjectiveErr)

        for iLevel in range(nLevels - 1):
            tempState = []
            tempState = PredictionSTM[iBranch, (iLevel + 1)]
            # tempState=np.concatenate((tempState,[tempY[0]]))
            tempClassState = []
            tempClassState = PredictionClassSTM[iBranch, (iLevel + 1)]
            # tempClassState=np.concatenate((tempClassState,tempY[0]))
            tempY = []
            tempYErr = []
            tempStatemat = []
            tempStatemat.append(np.array(tempState))
            tempClassStatemat = []
            tempClassStatemat.append(np.array(tempClassState))

            if RegressionModelStructure == 'Ensemble':
                tempY = BeliefModel.EnsembleMeanY(Model, tempStatemat)
                tempYErr = BeliefModel.EnsembleStdevY(Model, tempStatemat)
            elif RegressionModelStructure == 'SKLearn Ensemble':
                tempY = BeliefModel.RegressionPredictY(Model, tempStatemat)
                tempYErr = BeliefModel.RegressionPredictYErr(Model, tempStatemat)

            if ClassModelStructure == 'Ensemble':
                tempP = BeliefModel.EnsembleClassProbability(ClassModel, tempClassStatemat)
            elif ClassModelStructure == 'SKLearn Ensemble':
                tempP = BeliefModel.ClassProbability(ClassModel, tempClassStatemat)

            tempP = tempP * PredictionProbabilityBranch[iLevel]

            PredictionStateBranch.append(tempState)
            PredictionClassStateBranch.append(tempClassState)
            PredictionResponseBranch.append(tempY)
            PredictionResponseErrBranch.append(tempYErr)
            PredictionProbabilityBranch.append(np.array(tempP))

            tempObjectiveVal = Encoding.ObjectiveFunctionY1(tempY, tempP)
            tempObjectiveErr = Encoding.ObjectiveFunctionErrY1(tempYErr, tempP)
            ObjectiveBranch.append(tempObjectiveVal)
            ObjectiveErrBranch.append(tempObjectiveErr)

        PredictionState.append(PredictionStateBranch)
        PredictionClassState.append(PredictionClassStateBranch)
        PredictionResponse.append(PredictionResponseBranch)
        PredictionResponseErr.append(PredictionResponseErrBranch)
        PredictionProbability.append(PredictionProbabilityBranch)

        Objective.append(ObjectiveBranch)
        ObjectiveErr.append(ObjectiveErrBranch)
    PredictionState = np.array(PredictionState)
    PredictionClassState = np.array(PredictionClassState)
    PredictionResponse = np.array(PredictionResponse)
    PredictionResponseErr = np.array(PredictionResponseErr)
    PredictionProbability = np.array(PredictionProbability)

    Objective = np.array(Objective)
    ObjectiveErr = np.array(ObjectiveErr)

    return Objective, ObjectiveErr, Actions, PredictionState, PredictionClassState, PredictionResponse, PredictionProbability

def PermutationMonteCarloX1Y1(Model, ClassModel, X, Y, PermDuplicates, nLevels, MemorySteps, BlockEncoding, RegressionModelStructure,
                            ClassModelStructure):
    # Build random action grid of nBranches by nLevels and format into short term memory
    X = X[-(MemorySteps + 1):]
    Y = Y[-(MemorySteps + 1):]

    Injections = [1, 2, 3, 4]
    InjectionPermutations = [[iia, iib, iic, iid] for dup in range(PermDuplicates)
                             for iia in Injections
                             for iib in Injections
                             for iic in Injections
                             for iid in Injections]


    nBranches=len(InjectionPermutations)

    Actions = np.zeros((nBranches, nLevels, 3))
    STM = []
    ClassSTM = []

    for iBranch in range(nBranches):
        tempXAction = []
        tempXAction.append(X[:])
        for iLevel in range(nLevels):
            Actions[iBranch, iLevel, 0] = InjectionPermutations[iBranch][iLevel]
            Actions[iBranch, iLevel, 1] = 0.5
            Actions[iBranch, iLevel, 2] = 0.5
            tempXAction.append(Actions[iBranch, iLevel, :])
        tempXAction = np.array(tempXAction, dtype=object)
        tempXAction = np.vstack(tempXAction)
        tempSTM = Encoding.ShortTermMemoryX1(tempXAction, MemorySteps, BlockEncoding)
        tempClassSTM = Encoding.ClassShortTermMemoryX1(tempXAction, MemorySteps, BlockEncoding)
        ClassSTM.append(tempClassSTM)
        STM.append(tempSTM)
    STM = np.array(STM)
    ClassSTM = np.array(ClassSTM)

    PredictionSTM = STM[:, (len(Y)):None]
    PredictionClassSTM = ClassSTM[:, (len(Y)):None]

    PredictionState = []
    PredictionClassState = []
    PredictionResponse = []
    PredictionResponseErr = []
    PredictionProbability = []

    Objective = []
    ObjectiveErr = []

    for iBranch in range(nBranches):
        tempState = []
        tempClassState = []
        PredictionStateBranch = []
        PredictionClassStateBranch = []
        PredictionResponseBranch = []
        PredictionResponseErrBranch = []
        PredictionProbabilityBranch = []

        ObjectiveBranch = []
        ObjectiveErrBranch = []

        tempState = PredictionSTM[iBranch, 0]
        # tempState=np.concatenate((tempState, Y[-1]))
        PredictionStateBranch.append(np.array(tempState))

        tempClassState = PredictionClassSTM[iBranch, 0]
        # tempClassState=np.concatenate((tempClassState, Y[-1]))
        PredictionClassStateBranch.append(np.array(tempClassState))

        if RegressionModelStructure == 'Ensemble':
            tempY = BeliefModel.EnsembleMeanY(Model, PredictionStateBranch)
            tempYErr = BeliefModel.EnsembleStdevY(Model, PredictionStateBranch)
        elif RegressionModelStructure == 'SKLearn Ensemble':
            tempY = BeliefModel.RegressionPredictY(Model, PredictionStateBranch)
            tempYErr = BeliefModel.RegressionPredictYErr(Model, PredictionStateBranch)

        PredictionResponseBranch.append(tempY)
        PredictionResponseErrBranch.append(tempYErr)

        if ClassModelStructure == 'Ensemble':
            tempP = BeliefModel.EnsembleClassProbability(ClassModel, PredictionClassStateBranch)
        elif ClassModelStructure == 'SKLearn Ensemble':
            tempP = BeliefModel.ClassProbability(ClassModel, PredictionClassStateBranch)

        PredictionProbabilityBranch.append(np.array(tempP))

        tempObjectiveVal = Encoding.ObjectiveFunctionY1(tempY, tempP)
        tempObjectiveErr = Encoding.ObjectiveFunctionErrY1(tempYErr, tempP)
        ObjectiveBranch.append(tempObjectiveVal)
        ObjectiveErrBranch.append(tempObjectiveErr)

        for iLevel in range(nLevels - 1):
            tempState = []
            tempState = PredictionSTM[iBranch, (iLevel + 1)]
            # tempState=np.concatenate((tempState,[tempY[0]]))
            tempClassState = []
            tempClassState = PredictionClassSTM[iBranch, (iLevel + 1)]
            # tempClassState=np.concatenate((tempClassState,tempY))
            tempY = []
            tempYErr = []
            tempStatemat = []
            tempStatemat.append(np.array(tempState))
            tempClassStatemat = []
            tempClassStatemat.append(np.array(tempClassState))

            if RegressionModelStructure == 'Ensemble':
                tempY = BeliefModel.EnsembleMeanY(Model, tempStatemat)
                tempYErr = BeliefModel.EnsembleStdevY(Model, tempStatemat)
            elif RegressionModelStructure == 'SKLearn Ensemble':
                tempY = BeliefModel.RegressionPredictY(Model, tempStatemat)
                tempYErr = BeliefModel.RegressionPredictYErr(Model, tempStatemat)

            if ClassModelStructure == 'Ensemble':
                tempP = BeliefModel.EnsembleClassProbability(ClassModel, tempClassStatemat)
            elif ClassModelStructure == 'SKLearn Ensemble':
                tempP = BeliefModel.ClassProbability(ClassModel, tempClassStatemat)

            tempP = tempP * PredictionProbabilityBranch[iLevel]

            PredictionStateBranch.append(tempState)
            PredictionClassStateBranch.append(tempClassState)
            PredictionResponseBranch.append(tempY)
            PredictionResponseErrBranch.append(tempYErr)
            PredictionProbabilityBranch.append(np.array(tempP))

            tempObjectiveVal = Encoding.ObjectiveFunctionY1(tempY, tempP)
            tempObjectiveErr = Encoding.ObjectiveFunctionErrY1(tempYErr, tempP)
            ObjectiveBranch.append(tempObjectiveVal)
            ObjectiveErrBranch.append(tempObjectiveErr)

        PredictionState.append(PredictionStateBranch)
        PredictionClassState.append(PredictionClassStateBranch)
        PredictionResponse.append(PredictionResponseBranch)
        PredictionResponseErr.append(PredictionResponseErrBranch)
        PredictionProbability.append(PredictionProbabilityBranch)

        Objective.append(ObjectiveBranch)
        ObjectiveErr.append(ObjectiveErrBranch)
    PredictionState = np.array(PredictionState)
    PredictionClassState = np.array(PredictionClassState)
    PredictionResponse = np.array(PredictionResponse)
    PredictionResponseErr = np.array(PredictionResponseErr)
    PredictionProbability = np.array(PredictionProbability)

    Objective = np.array(Objective)
    ObjectiveErr = np.array(ObjectiveErr)


    return Objective, ObjectiveErr, Actions, PredictionState, PredictionClassState, PredictionResponse, PredictionProbability

def SubsamplePermutationX1Y1(Model, ClassModel, X, Y, PermDuplicates, nLevels, MemorySteps, BlockEncoding, RegressionModelStructure,
                            ClassModelStructure):
    # Build random action grid of nBranches by nLevels and format into short term memory
    #creating array of STM and most recent action for forward mapping (dont need all STM data)
    X = X[-(MemorySteps + 1):]
    Y = Y[-(MemorySteps + 1):]

    Injections = [1, 2, 3, 4]
    InjectionPermutations = [[iia, iib, iic, iid] for dup in range(PermDuplicates)
                             for iia in Injections
                             for iib in Injections
                             for iic in Injections
                             for iid in Injections]


    nBranches=len(InjectionPermutations)

    Actions = np.zeros((nBranches, nLevels, 3))
    STM = []
    ClassSTM = []

    for iBranch in range(nBranches):

        tempXAction = []
        tempXAction.append(X[:])
        for iLevel in range(nLevels):
            Actions[iBranch, iLevel, 0] = InjectionPermutations[iBranch][iLevel]
            Actions[iBranch, iLevel, 1] = 0.5
            Actions[iBranch, iLevel, 2] = 0.5
            tempXAction.append(Actions[iBranch, iLevel, :]) #tempXaction becomes STM, most recent action, and forward map branch in one nested array
        tempXAction = np.array(tempXAction, dtype=object)
        #array of pointers to array contents - or dataframe - double check
        tempXAction = np.vstack(tempXAction)
        tempSTM = Encoding.ShortTermMemoryX1(tempXAction, MemorySteps, BlockEncoding)
        tempClassSTM = Encoding.ClassShortTermMemoryX1(tempXAction, MemorySteps, BlockEncoding)
        ClassSTM.append(tempClassSTM)
        STM.append(tempSTM)
    STM = np.array(STM)
    ClassSTM = np.array(ClassSTM)

    PredictionSTM = STM[:, (len(Y)):None]
    PredictionClassSTM = ClassSTM[:, (len(Y)):None]

    PredictionState = []
    PredictionClassState = []
    PredictionResponse = []
    PredictionProbability = []

    Objective = []
    ObjectiveErr = []

    for iBranch in range(nBranches):
        tempState = []
        tempClassState = []
        PredictionStateBranch = []
        PredictionClassStateBranch = []
        PredictionResponseBranch = []
        PredictionProbabilityBranch = []

        ObjectiveBranch = []

        tempState = PredictionSTM[iBranch, 0]
        # tempState=np.concatenate((tempState, Y[-1]))
        PredictionStateBranch.append(np.array(tempState))

        tempClassState = PredictionClassSTM[iBranch, 0]
        # tempClassState=np.concatenate((tempClassState, Y[-1]))
        PredictionClassStateBranch.append(np.array(tempClassState))

        if RegressionModelStructure == 'Ensemble':
            tempY = BeliefModel.EnsembleSubsampleY(Model, PredictionStateBranch)
        elif RegressionModelStructure == 'SKLearn Ensemble':
            tempY = BeliefModel.EnsembleSubsampleY(Model, PredictionStateBranch)

        PredictionResponseBranch.append(tempY)

        if ClassModelStructure == 'Ensemble':
            tempP = BeliefModel.EnsembleClassProbability(ClassModel, PredictionClassStateBranch)
        elif ClassModelStructure == 'SKLearn Ensemble':
            tempP = BeliefModel.ClassProbability(ClassModel, PredictionClassStateBranch)

        PredictionProbabilityBranch.append(np.array(tempP))

        tempObjectiveVal = Encoding.ObjectiveFunctionY1(tempY, tempP)
        ObjectiveBranch.append(tempObjectiveVal)

        for iLevel in range(nLevels - 1):
            tempState = []
            tempState = PredictionSTM[iBranch, (iLevel + 1)]
            # tempState=np.concatenate((tempState,[tempY[0]]))
            tempClassState = []
            tempClassState = PredictionClassSTM[iBranch, (iLevel + 1)]
            # tempClassState=np.concatenate((tempClassState,tempY))
            tempY = []
            tempYErr = []
            tempStatemat = []
            tempStatemat.append(np.array(tempState))
            tempClassStatemat = []
            tempClassStatemat.append(np.array(tempClassState))

            if RegressionModelStructure == 'Ensemble':
                tempY = BeliefModel.EnsembleSubsampleY(Model, tempStatemat)
            elif RegressionModelStructure == 'SKLearn Ensemble':
                tempY = BeliefModel.EnsembleSubsampleY(Model, tempStatemat)

            if ClassModelStructure == 'Ensemble':
                tempP = BeliefModel.EnsembleClassProbability(ClassModel, tempClassStatemat)
            elif ClassModelStructure == 'SKLearn Ensemble':
                tempP = BeliefModel.ClassProbability(ClassModel, tempClassStatemat)

            tempP = tempP * PredictionProbabilityBranch[iLevel]

            PredictionStateBranch.append(tempState)
            PredictionClassStateBranch.append(tempClassState)
            PredictionResponseBranch.append(tempY)
            PredictionProbabilityBranch.append(np.array(tempP))

            tempObjectiveVal = Encoding.ObjectiveFunctionY1(tempY, tempP)
            ObjectiveBranch.append(tempObjectiveVal)

        PredictionState.append(PredictionStateBranch)
        PredictionClassState.append(PredictionClassStateBranch)
        PredictionResponse.append(PredictionResponseBranch)
        PredictionProbability.append(PredictionProbabilityBranch)

        Objective.append(ObjectiveBranch)
    PredictionState = np.array(PredictionState)
    PredictionClassState = np.array(PredictionClassState)
    PredictionResponse = np.array(PredictionResponse)
    PredictionProbability = np.array(PredictionProbability)

    Objective = np.array(Objective)

    ActionsNoDup=[]
    PredictionStateNoDup=[]
    PredictionClassStateNoDup = []
    PredictionProbabilityNoDup = []

    PermLength = int(nBranches / PermDuplicates)

    for iBranch in range(PermLength):
        ActionsNoDup.append(Actions[iBranch])
        PredictionStateNoDup.append(PredictionState[iBranch])
        PredictionClassStateNoDup.append(PredictionClassState[iBranch])
        PredictionProbabilityNoDup.append(PredictionProbability[iBranch])

    ActionsNoDup=np.array(ActionsNoDup)
    PredictionStateNoDup = np.array(PredictionStateNoDup)
    PredictionClassStateNoDup = np.array(PredictionClassStateNoDup)
    PredictionProbabilityNoDup = np.array(PredictionProbabilityNoDup)

    ObjectiveGroupedDup= np.empty([PermLength,nLevels,PermDuplicates])
    PredictionResponseGroupedDup = np.empty([PermLength,nLevels,PermDuplicates])
    for iDup in range(PermDuplicates):
        for iBranch in range(PermLength):
            for iLevel in range(nLevels):
                ObjectiveGroupedDup[iBranch,iLevel,iDup] = Objective[iDup*PermLength+iBranch,iLevel]
                PredictionResponseGroupedDup[iBranch, iLevel, iDup] = PredictionResponse[iDup * PermLength + iBranch, iLevel]


    ObjectiveMean=np.empty([PermLength,nLevels])
    ObjectiveErr=np.empty([PermLength,nLevels])
    PredictionResponseMean=np.empty([PermLength,nLevels])
    for iBranch in range(PermLength):
        for iLevel in range(nLevels):
            ObjectiveMean[iBranch,iLevel]=np.mean(ObjectiveGroupedDup[iBranch,iLevel,:])
            ObjectiveErr[iBranch, iLevel] = np.std(ObjectiveGroupedDup[iBranch,iLevel, :])
            PredictionResponseMean[iBranch, iLevel] = np.mean(PredictionResponseGroupedDup[iBranch, iLevel, :])

    ObjectiveMean = np.array(ObjectiveMean)
    ObjectiveErr = np.array(ObjectiveErr)
    PredictionResponseMean = np.array(PredictionResponseMean)

    return ObjectiveMean, ObjectiveErr, ActionsNoDup, PredictionStateNoDup, PredictionClassStateNoDup, PredictionResponseMean, PredictionProbabilityNoDup

def SubsamplePermutationX1Y1RState(Model, ClassModel, X, Y, PermDuplicates, nLevels, MemorySteps, BlockEncoding, RegressionModelStructure,
                            ClassModelStructure):
    # Build random action grid of nBranches by nLevels and format into short term memory
    #creating array of STM and most recent action for forward mapping (dont need all STM data)
    X = X[-(MemorySteps + 1):]
    Y = Y[-(MemorySteps + 1):]


    Injections = [1, 2, 3, 4]
    InjectionPermutations = [[iia, iib, iic, iid] for dup in range(PermDuplicates)
                             for iia in Injections
                             for iib in Injections
                             for iic in Injections
                             for iid in Injections]


    nBranches=len(InjectionPermutations)

    Actions = np.zeros((nBranches, nLevels, 3))
    STM = []
    ClassSTM = []

    for iBranch in range(nBranches):

        tempXAction = []
        tempXAction.append(X[:])
        for iLevel in range(nLevels):
            Actions[iBranch, iLevel, 0] = InjectionPermutations[iBranch][iLevel]
            Actions[iBranch, iLevel, 1] = 0.5
            Actions[iBranch, iLevel, 2] = 0.5
            tempXAction.append(Actions[iBranch, iLevel, :]) #tempXaction becomes STM, most recent action, and forward map branch in one nested array
        tempXAction = np.array(tempXAction, dtype=object)
        #array of pointers to array contents - or dataframe - double check
        tempXAction = np.vstack(tempXAction)
        tempSTM = Encoding.ShortTermMemoryX1(tempXAction, MemorySteps, BlockEncoding)
        tempClassSTM = Encoding.ClassShortTermMemoryX1(tempXAction, MemorySteps, BlockEncoding)
        ClassSTM.append(tempClassSTM)
        STM.append(tempSTM)
    STM = np.array(STM)
    ClassSTM = np.array(ClassSTM)

    PredictionSTM = STM[:, (len(Y)):None]
    PredictionClassSTM = ClassSTM[:, (len(Y)):None]

    PredictionState = []
    PredictionClassState = []
    PredictionResponse = []
    PredictionProbability = []

    Objective = []
    ObjectiveErr = []

    for iBranch in range(nBranches):
        tempState = []
        tempClassState = []
        PredictionStateBranch = []
        PredictionClassStateBranch = []
        PredictionResponseBranch = []
        PredictionProbabilityBranch = []

        ObjectiveBranch = []

        tempY=Y[-1]

        tempState = np.append(PredictionSTM[iBranch, 0],tempY)
        # tempState=np.concatenate((tempState, Y[-1]))
        PredictionStateBranch.append(np.array(tempState))

        tempClassState = np.append(PredictionClassSTM[iBranch, 0],tempY)
        # tempClassState=np.concatenate((tempClassState, Y[-1]))
        PredictionClassStateBranch.append(np.array(tempClassState))

        if RegressionModelStructure == 'Ensemble':
            tempY = BeliefModel.EnsembleSubsampleY(Model, PredictionStateBranch)
        elif RegressionModelStructure == 'SKLearn Ensemble':
            tempY = BeliefModel.EnsembleSubsampleY(Model, PredictionStateBranch)

        PredictionResponseBranch.append(tempY)

        if ClassModelStructure == 'Ensemble':
            tempP = BeliefModel.EnsembleClassProbability(ClassModel, PredictionClassStateBranch)
        elif ClassModelStructure == 'SKLearn Ensemble':
            tempP = BeliefModel.ClassProbability(ClassModel, PredictionClassStateBranch)

        PredictionProbabilityBranch.append(np.array(tempP))

        tempObjectiveVal = Encoding.ObjectiveFunctionY1(tempY, tempP)
        ObjectiveBranch.append(tempObjectiveVal)

        for iLevel in range(nLevels - 1):
            tempState = []
            tempState = np.append(PredictionSTM[iBranch, (iLevel + 1)], tempY)
            # tempState=np.concatenate((tempState,[tempY[0]]))
            tempClassState = []
            tempClassState = np.append(PredictionClassSTM[iBranch, (iLevel + 1)], tempY)
            # tempClassState=np.concatenate((tempClassState,tempY))

            tempYErr = []
            tempStatemat = []
            tempStatemat.append(np.array(tempState))
            tempClassStatemat = []
            tempClassStatemat.append(np.array(tempClassState))

            tempY = []

            if RegressionModelStructure == 'Ensemble':
                tempY = BeliefModel.EnsembleSubsampleY(Model, tempStatemat)
            elif RegressionModelStructure == 'SKLearn Ensemble':
                tempY = BeliefModel.EnsembleSubsampleY(Model, tempStatemat)

            if ClassModelStructure == 'Ensemble':
                tempP = BeliefModel.EnsembleClassProbability(ClassModel, tempClassStatemat)
            elif ClassModelStructure == 'SKLearn Ensemble':
                tempP = BeliefModel.ClassProbability(ClassModel, tempClassStatemat)

            tempP = tempP * PredictionProbabilityBranch[iLevel]

            PredictionStateBranch.append(tempState)
            PredictionClassStateBranch.append(tempClassState)
            PredictionResponseBranch.append(tempY)
            PredictionProbabilityBranch.append(np.array(tempP))

            tempObjectiveVal = Encoding.ObjectiveFunctionY1(tempY, tempP)
            ObjectiveBranch.append(tempObjectiveVal)

        PredictionState.append(PredictionStateBranch)
        PredictionClassState.append(PredictionClassStateBranch)
        PredictionResponse.append(PredictionResponseBranch)
        PredictionProbability.append(PredictionProbabilityBranch)

        Objective.append(ObjectiveBranch)
    PredictionState = np.array(PredictionState)
    PredictionClassState = np.array(PredictionClassState)
    PredictionResponse = np.array(PredictionResponse)
    PredictionProbability = np.array(PredictionProbability)

    Objective = np.array(Objective)

    ActionsNoDup=[]
    PredictionStateNoDup=[]
    PredictionClassStateNoDup = []
    PredictionProbabilityNoDup = []

    PermLength = int(nBranches / PermDuplicates)

    for iBranch in range(PermLength):
        ActionsNoDup.append(Actions[iBranch])
        PredictionStateNoDup.append(PredictionState[iBranch])
        PredictionClassStateNoDup.append(PredictionClassState[iBranch])
        PredictionProbabilityNoDup.append(PredictionProbability[iBranch])

    ActionsNoDup=np.array(ActionsNoDup)
    PredictionStateNoDup = np.array(PredictionStateNoDup)
    PredictionClassStateNoDup = np.array(PredictionClassStateNoDup)
    PredictionProbabilityNoDup = np.array(PredictionProbabilityNoDup)

    ObjectiveGroupedDup= np.empty([PermLength,nLevels,PermDuplicates])
    PredictionResponseGroupedDup = np.empty([PermLength,nLevels,PermDuplicates])
    for iDup in range(PermDuplicates):
        for iBranch in range(PermLength):
            for iLevel in range(nLevels):
                ObjectiveGroupedDup[iBranch,iLevel,iDup] = Objective[iDup*PermLength+iBranch,iLevel]
                PredictionResponseGroupedDup[iBranch, iLevel, iDup] = PredictionResponse[iDup * PermLength + iBranch, iLevel]


    ObjectiveMean=np.empty([PermLength,nLevels])
    ObjectiveErr=np.empty([PermLength,nLevels])
    PredictionResponseMean=np.empty([PermLength,nLevels])
    for iBranch in range(PermLength):
        for iLevel in range(nLevels):
            ObjectiveMean[iBranch,iLevel]=np.mean(ObjectiveGroupedDup[iBranch,iLevel,:])
            ObjectiveErr[iBranch, iLevel] = np.std(ObjectiveGroupedDup[iBranch,iLevel, :])
            PredictionResponseMean[iBranch, iLevel] = np.mean(PredictionResponseGroupedDup[iBranch, iLevel, :])

    ObjectiveMean = np.array(ObjectiveMean)
    ObjectiveErr = np.array(ObjectiveErr)
    PredictionResponseMean = np.array(PredictionResponseMean)

    return ObjectiveMean, ObjectiveErr, ActionsNoDup, PredictionStateNoDup, PredictionClassStateNoDup, PredictionResponseMean, PredictionProbabilityNoDup

def SubsamplePermutationX1Y16Level(Model, ClassModel, X, Y, PermDuplicates, nLevels, MemorySteps, BlockEncoding, RegressionModelStructure,
                            ClassModelStructure):
    # Build random action grid of nBranches by nLevels and format into short term memory
    X = X[-(MemorySteps + 1):]
    Y = Y[-(MemorySteps + 1):]

    Injections = [1, 2, 3, 4]
    InjectionPermutations = [[iia, iib, iic, iid, iie, iif] for dup in range(PermDuplicates)
                             for iia in Injections
                             for iib in Injections
                             for iic in Injections
                             for iid in Injections
                             for iie in Injections
                             for iif in Injections]


    nBranches=len(InjectionPermutations)

    Actions = np.zeros((nBranches, nLevels, 3))
    STM = []
    ClassSTM = []

    for iBranch in range(nBranches):
        tempXAction = []
        tempXAction.append(X[:])
        for iLevel in range(nLevels):
            Actions[iBranch, iLevel, 0] = InjectionPermutations[iBranch][iLevel]
            Actions[iBranch, iLevel, 1] = 0.5
            Actions[iBranch, iLevel, 2] = 0.5
            tempXAction.append(Actions[iBranch, iLevel, :])
        tempXAction = np.array(tempXAction, dtype=object)
        tempXAction = np.vstack(tempXAction)
        tempSTM = Encoding.ShortTermMemoryX1(tempXAction, MemorySteps, BlockEncoding)
        tempClassSTM = Encoding.ClassShortTermMemoryX1(tempXAction, MemorySteps, BlockEncoding)
        ClassSTM.append(tempClassSTM)
        STM.append(tempSTM)
    STM = np.array(STM)
    ClassSTM = np.array(ClassSTM)

    PredictionSTM = STM[:, (len(Y)):None]
    PredictionClassSTM = ClassSTM[:, (len(Y)):None]

    PredictionState = []
    PredictionClassState = []
    PredictionResponse = []
    PredictionProbability = []

    Objective = []
    ObjectiveErr = []

    for iBranch in range(nBranches):
        tempState = []
        tempClassState = []
        PredictionStateBranch = []
        PredictionClassStateBranch = []
        PredictionResponseBranch = []
        PredictionProbabilityBranch = []

        ObjectiveBranch = []

        tempState = PredictionSTM[iBranch, 0]
        # tempState=np.concatenate((tempState, Y[-1]))
        PredictionStateBranch.append(np.array(tempState))

        tempClassState = PredictionClassSTM[iBranch, 0]
        # tempClassState=np.concatenate((tempClassState, Y[-1]))
        PredictionClassStateBranch.append(np.array(tempClassState))

        if RegressionModelStructure == 'Ensemble':
            tempY = BeliefModel.EnsembleSubsampleY(Model, PredictionStateBranch)
        elif RegressionModelStructure == 'SKLearn Ensemble':
            tempY = BeliefModel.EnsembleSubsampleY(Model, PredictionStateBranch)

        PredictionResponseBranch.append(tempY)

        if ClassModelStructure == 'Ensemble':
            tempP = BeliefModel.EnsembleClassProbability(ClassModel, PredictionClassStateBranch)
        elif ClassModelStructure == 'SKLearn Ensemble':
            tempP = BeliefModel.ClassProbability(ClassModel, PredictionClassStateBranch)

        PredictionProbabilityBranch.append(np.array(tempP))

        tempObjectiveVal = Encoding.ObjectiveFunctionY1(tempY, tempP)
        ObjectiveBranch.append(tempObjectiveVal)

        for iLevel in range(nLevels - 1):
            tempState = []
            tempState = PredictionSTM[iBranch, (iLevel + 1)]
            # tempState=np.concatenate((tempState,[tempY[0]]))
            tempClassState = []
            tempClassState = PredictionClassSTM[iBranch, (iLevel + 1)]
            # tempClassState=np.concatenate((tempClassState,tempY))
            tempY = []
            tempYErr = []
            tempStatemat = []
            tempStatemat.append(np.array(tempState))
            tempClassStatemat = []
            tempClassStatemat.append(np.array(tempClassState))

            if RegressionModelStructure == 'Ensemble':
                tempY = BeliefModel.EnsembleSubsampleY(Model, tempStatemat)
            elif RegressionModelStructure == 'SKLearn Ensemble':
                tempY = BeliefModel.EnsembleSubsampleY(Model, tempStatemat)

            if ClassModelStructure == 'Ensemble':
                tempP = BeliefModel.EnsembleClassProbability(ClassModel, tempClassStatemat)
            elif ClassModelStructure == 'SKLearn Ensemble':
                tempP = BeliefModel.ClassProbability(ClassModel, tempClassStatemat)

            tempP = tempP * PredictionProbabilityBranch[iLevel]
            # tempP = tempP * 1/(iLevel+1)

            PredictionStateBranch.append(tempState)
            PredictionClassStateBranch.append(tempClassState)
            PredictionResponseBranch.append(tempY)
            PredictionProbabilityBranch.append(np.array(tempP))

            tempObjectiveVal = Encoding.ObjectiveFunctionY1(tempY, tempP)
            ObjectiveBranch.append(tempObjectiveVal)

        PredictionState.append(PredictionStateBranch)
        PredictionClassState.append(PredictionClassStateBranch)
        PredictionResponse.append(PredictionResponseBranch)
        PredictionProbability.append(PredictionProbabilityBranch)

        Objective.append(ObjectiveBranch)
    PredictionState = np.array(PredictionState)
    PredictionClassState = np.array(PredictionClassState)
    PredictionResponse = np.array(PredictionResponse)
    PredictionProbability = np.array(PredictionProbability)

    Objective = np.array(Objective)

    ActionsNoDup=[]
    PredictionStateNoDup=[]
    PredictionClassStateNoDup = []
    PredictionProbabilityNoDup = []

    PermLength = int(nBranches / PermDuplicates)

    for iBranch in range(PermLength):
        ActionsNoDup.append(Actions[iBranch])
        PredictionStateNoDup.append(PredictionState[iBranch])
        PredictionClassStateNoDup.append(PredictionClassState[iBranch])
        PredictionProbabilityNoDup.append(PredictionProbability[iBranch])

    ActionsNoDup=np.array(ActionsNoDup)
    PredictionStateNoDup = np.array(PredictionStateNoDup)
    PredictionClassStateNoDup = np.array(PredictionClassStateNoDup)
    PredictionProbabilityNoDup = np.array(PredictionProbabilityNoDup)

    ObjectiveGroupedDup= np.empty([PermLength,nLevels,PermDuplicates])
    PredictionResponseGroupedDup = np.empty([PermLength,nLevels,PermDuplicates])
    for iDup in range(PermDuplicates):
        for iBranch in range(PermLength):
            for iLevel in range(nLevels):
                ObjectiveGroupedDup[iBranch,iLevel,iDup] = Objective[iDup*PermLength+iBranch,iLevel]
                PredictionResponseGroupedDup[iBranch, iLevel, iDup] = PredictionResponse[iDup * PermLength + iBranch, iLevel]


    ObjectiveMean=np.empty([PermLength,nLevels])
    ObjectiveErr=np.empty([PermLength,nLevels])
    PredictionResponseMean=np.empty([PermLength,nLevels])
    for iBranch in range(PermLength):
        for iLevel in range(nLevels):
            ObjectiveMean[iBranch,iLevel]=np.mean(ObjectiveGroupedDup[iBranch,iLevel,:])
            ObjectiveErr[iBranch, iLevel] = np.std(ObjectiveGroupedDup[iBranch,iLevel, :])
            PredictionResponseMean[iBranch, iLevel] = np.mean(PredictionResponseGroupedDup[iBranch, iLevel, :])

    ObjectiveMean = np.array(ObjectiveMean)
    ObjectiveErr = np.array(ObjectiveErr)
    PredictionResponseMean = np.array(PredictionResponseMean)

    return ObjectiveMean, ObjectiveErr, ActionsNoDup, PredictionStateNoDup, PredictionClassStateNoDup, PredictionResponseMean, PredictionProbabilityNoDup

def SubsampleNodeMonteCarloCIX23Y1(Model, ClassModel, X, Y, nBranches, nLevels, MemorySteps, RegressionModelStructure,
                            ClassModelStructure, descRangesX2, descRangesX3):
    X = X[-(MemorySteps + 1):]
    Y = Y[-(MemorySteps + 1):]

    Volumes = range(np.shape(descRangesX2)[0])
    Times = range(np.shape(descRangesX2)[0])

    #every combination of descritized volumes and times
    Nodes = [[iV, iT] for iV in Volumes
        for iT in Times]
    nNodes = np.shape(Nodes)[0]
    #total number branches to forward map, evenly distributing branches across the nodes
    nNodeBranches = nBranches // nNodes

    Actions = np.zeros((nNodes,nBranches, nLevels, 3))
    ClassSTM = []
    STM = []

    for iNode in range(nNodes):
        NodeSTM = []
        NodeClassSTM = []
        for iNodeBranch in range(nNodeBranches):
            tempXAction = []
            tempXAction.append(X[:])
            for iLevel in range(nLevels):
                Actions[iNode, iNodeBranch, iLevel, 0] = 0
                if iLevel == 0:
                    TempContinuousNode = Encoding.ValueinDescretizedRangeSample(Nodes[iNode], descRangesX2, descRangesX3)
                    Actions[iNode, iNodeBranch, iLevel, 1] = TempContinuousNode[0]
                    Actions[iNode, iNodeBranch, iLevel, 2] = TempContinuousNode[1]
                else:
                    Actions[iNode, iNodeBranch, iLevel, 1] = rnd.random()
                    Actions[iNode, iNodeBranch, iLevel, 2] = rnd.randint(0, 9)/9
                tempXAction.append(Actions[iNode,iNodeBranch, iLevel, :])
            tempXAction = np.array(tempXAction, dtype=object)
            tempXAction = np.vstack(tempXAction)
            tempSTM = Encoding.ShortTermMemoryCIX23(tempXAction, MemorySteps)
            tempClassSTM = Encoding.ShortTermMemoryCIX23(tempXAction, MemorySteps)
            NodeClassSTM.append(tempClassSTM)
            NodeSTM.append(tempSTM)
        ClassSTM.append(NodeClassSTM)
        STM.append(NodeSTM)
    STM = np.array(STM)
    ClassSTM = np.array(ClassSTM)

    PredictionState = STM[:,:, (len(Y)):None]
    PredictionClassState = ClassSTM[:,:, (len(Y)):None]

    PredictionResponse = np.zeros((nNodes,nNodeBranches, nLevels))
    PredictionProbability = np.zeros((nNodes,nNodeBranches, nLevels))
    Objective = np.zeros((nNodes,nNodeBranches, nLevels))

    for iNode in range(nNodes):
        for iNodeBranch in range(nNodeBranches):
            oldP = 1
            for iLevel in range(nLevels):

                tempMat= []
                tempMat.append(PredictionState[iNode,iNodeBranch,iLevel])

                if RegressionModelStructure == 'Ensemble':
                    tempY = BeliefModel.EnsembleSubsampleY(Model, tempMat)
                elif RegressionModelStructure == 'SKLearn Ensemble':
                    tempY = BeliefModel.EnsembleSubsampleY(Model, tempMat)

                PredictionResponse[iNode,iNodeBranch,iLevel] = tempY

                tempMat = []
                tempMat.append(PredictionClassState[iNode, iNodeBranch, iLevel])

                if ClassModelStructure == 'Ensemble':
                    tempP = BeliefModel.EnsembleClassProbability(ClassModel, tempMat)
                elif ClassModelStructure == 'SKLearn Ensemble':
                    tempP = BeliefModel.ClassProbability(ClassModel, tempMat)

                PredictionProbability[iNode, iNodeBranch, iLevel] = tempP * oldP
                oldP = tempP

                tempObj = Encoding.ObjectiveFunctionY1(tempY, tempP)
                Objective[iNode, iNodeBranch, iLevel] = tempObj[0]

    return Objective, Nodes
