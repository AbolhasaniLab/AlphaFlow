import numpy as np
import matplotlib.pyplot as plt
from Functions import BeliefModel
from scipy.io import savemat, loadmat
import random as rnd
from scipy import optimize as sciopt

def SumRewardbyDroplet(State,Response):

    tempSum = 0
    CSumResponse = np.zeros((len(Response),1))
   # indexenddroplet = np.array([0])
    for iRow in range(len(Response)):
        if iRow == 0:
            tempSum = 0
            CSumResponse[iRow] = Response[iRow]
        elif all(State[iRow,0:2] == [0,1]) and not(all(State[iRow-1,0:2] == [0,1])):
            tempSum = 0
            CSumResponse[iRow] = Response[iRow]
        elif (iRow+1) >= len(Response):
            CSumResponse[iRow] = Response[iRow] + tempSum
        elif not(all(State[iRow,0:6] == State[iRow+1,0:6])):
            CSumResponse[iRow] = Response[iRow] + tempSum
            tempSum = tempSum + Response[iRow]
            #np.append(indexenddroplet, [iRow])
        else:
            CSumResponse[iRow] = Response[iRow] + tempSum

    return CSumResponse
   # return indexenddroplet

def SumRewardbyMovingWindow(State,Response):

    tempSum = 0
    CSumResponse = np.zeros((len(Response),1))
    indexenddroplet = np.array([0])
    for iRow in range(len(Response)):
        if iRow == 0:
            tempSum = 0
            CSumResponse[iRow] = Response[iRow]
        elif all(State[iRow,0:2] == [0,1]) and not(all(State[iRow-1,0:2] == [0,1])):
            tempSum = 0
            indexenddroplet=np.array([0])
            CSumResponse[iRow] = Response[iRow]
        elif (iRow+1) >= len(Response):
            CSumResponse[iRow] = Response[iRow] + tempSum
            indexenddroplet = np.append(indexenddroplet,[iRow])
            if len(indexenddroplet) >= 6:
                CSumResponse[iRow] = CSumResponse[iRow] - CSumResponse[indexenddroplet[-4]]

        elif not(all(State[iRow,0:6] == State[iRow+1,0:6])):
            CSumResponse[iRow] = Response[iRow] + tempSum
            tempSum = tempSum + Response[iRow]
            indexenddroplet = np.append(indexenddroplet, [iRow])
            if len(indexenddroplet) >= 6:
                CSumResponse[iRow] = CSumResponse[iRow]-CSumResponse[indexenddroplet[-4]]
        else:
            CSumResponse[iRow] = Response[iRow] + tempSum
            if len(indexenddroplet) >= 6:
                CSumResponse[iRow] = CSumResponse[iRow] - CSumResponse[indexenddroplet[-4]]

    return CSumResponse
   # return indexenddroplet


def PlotRewardRange(Models):
    testState = [0, 0, 0, 1, 0, 0, 0.5, 0.5, 1, 0.5, 0, 0.5, 0, 0]
    PredArray = []
    ErrArray = []
    for ii in np.arange(0,1.1,0.1):
        testState[10] = ii
        PredArray = np.append(PredArray, BeliefModel.EnsembleMeanY(Models, [testState]))
        ErrArray = np.append(ErrArray, BeliefModel.EnsembleStdevY(Models, [testState]))
    print(ErrArray)
    plt.errorbar(np.arange(0,1.1,0.1),PredArray, yerr=ErrArray, fmt="o")
    plt.show()
    return PredArray, ErrArray

def PlotRewardSurf(Models):
    testState = [0, 0, 1, 0, 0, 0, 0.5, 0.5, 0.5, 1, 0, 0, 0, 0]
    PredArray = []
    ErrArray = []
    x = np.arange(0,1.1,0.1)
    PredArray = np.zeros((len(x), len(x)))
    ErrArray = np.zeros((len(x), len(x)))
    for ii in range(len(x)):
        testState[6] = x[ii]
        for jj in range(len(x)):
            testState[8] = x[jj]
            PredArray[ii,jj] = BeliefModel.EnsembleMeanY(Models, [testState])
            ErrArray[ii,jj] = BeliefModel.EnsembleStdevY(Models, [testState])
    return PredArray, ErrArray

def PlotRewardSurftoMatlab(Models):
    testState = [0, 0, 0, 0, 1, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    x = np.arange(0,1.1,0.1)
    x0 = np.arange(0,1.1,0.5)
    PredArray = np.zeros((len(x0),len(x0),len(x), len(x)))
    ErrArray = np.zeros((len(x0),len(x0),len(x), len(x)))
    for ii0 in range(len(x0)):
        testState[6] = x0[ii0]
        for jj0 in range(len(x0)):
            testState[8] = x0[jj0]
            for ii in range(len(x)):
                testState[10] = x[ii]
                for jj in range(len(x)):
                    testState[12] = x[jj]
                    PredArray[ii0,jj0,ii,jj] = BeliefModel.EnsembleMeanY(Models, [testState])
                    ErrArray[ii0,jj0,ii,jj] = BeliefModel.EnsembleStdevY(Models, [testState])

    #savemat("C:/Users/Walnut/Documents/MATLAB/Pred Array Const Time.mat", {"array": PredArray})
    #savemat("C:/Users/Walnut/Documents/MATLAB/Err Array Const Time.mat", {"array": ErrArray})
    return PredArray, ErrArray

def PlotRewardSurftoMatlabinvYJ(Models,YJFit):
    testState = [0, 0, 0, 0, 1, 0, 0.5, 0.4444, 0.5, 0.4444, 0.5, 0.4444, 0.5, 0.4444]
    x = np.arange(0,1.1,0.1)
    x0 = np.arange(0,1.1,0.5)
    PredArray = np.zeros((len(x0),len(x0),len(x), len(x)))
    ErrArray = np.zeros((len(x0),len(x0),len(x), len(x)))
    for ii0 in range(len(x0)):
        testState[6] = x0[ii0]
        for jj0 in range(len(x0)):
            testState[8] = x0[jj0]
            for ii in range(len(x)):
                testState[10] = x[ii]
                for jj in range(len(x)):
                    testState[12] = x[jj]
                    PredArray[ii0,jj0,ii,jj] = YJFit.inverse_transform(BeliefModel.EnsembleMeanY(Models, [testState]).reshape(-1,1))
                    ErrArray[ii0,jj0,ii,jj] = YJFit.inverse_transform(BeliefModel.EnsembleStdevY(Models, [testState]).reshape(-1,1))

    #savemat("C:/Users/Walnut/Documents/MATLAB/Pred Array Const Time.mat", {"array": PredArray})
    #savemat("C:/Users/Walnut/Documents/MATLAB/Err Array Const Time.mat", {"array": ErrArray})
    return PredArray, ErrArray

def PlotSumRewardSurftoMatlabinvYJ(Models,YJFit):
    testState = [[0, 1, 0, 0, 0, 0, 0.5, 0.4444, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0.5, 0.4444, 0.5, 0.4444, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0.5, 0.4444, 0.5, 0.4444, 0.5, 0.4444, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0.5, 0.4444, 0.5, 0.4444, 0.5, 0.4444, 0.5, 0.4444]]
    x = np.arange(0,1.1,0.1)
    x0 = np.arange(0,1.1,0.5)
    PredArray = np.zeros((len(x0), len(x0), len(x), len(x)))
    ErrArray = np.zeros((len(x0), len(x0), len(x), len(x)))
    for ii0 in range(len(x0)):
        testState[3][6] = x0[ii0]
        for jj0 in range(len(x0)):
            testState[3][8] = x0[jj0]
            testState[2][6] = x0[jj0]
            for ii in range(len(x)):
                testState[3][10] = x[ii]
                testState[2][8] = x[ii]
                testState[1][6] = x[ii]
                for jj in range(len(x)):
                    testState[3][12] = x[jj]
                    testState[2][10] = x[jj]
                    testState[1][8] = x[jj]
                    testState[0][6] = x[jj]

                    PredArray[ii0,jj0,ii,jj] = sum(YJFit.inverse_transform(BeliefModel.EnsembleMeanY(Models, testState).reshape(-1,1)))
                    ErrArray[ii0,jj0,ii,jj] = sum(YJFit.inverse_transform(BeliefModel.EnsembleStdevY(Models, testState).reshape(-1,1)))

    #savemat("C:/Users/Walnut/Documents/MATLAB/Pred Array Const Time.mat", {"array": PredArray})
    #savemat("C:/Users/Walnut/Documents/MATLAB/Err Array Const Time.mat", {"array": ErrArray})
    return PredArray, ErrArray

def ForwardPotential(Models,ClassModels,YJFit):
    reps = 1000

    testState = [[0, 1, 0, 0, 0, 0, 0.5, 0.4444, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0.5, 0.4444, 0.5, 0.4444, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0.5, 0.4444, 0.5, 0.4444, 0.5, 0.4444, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0.5, 0.4444, 0.5, 0.4444, 0.5, 0.4444, 0.5, 0.4444]]
    # testState = [[0, 0, 1, 0, 0, 0, 0.5, 0.4444, 0.79211, 0.4444, 0, 0, 0, 0],
    #              [0, 0, 0, 1, 0, 0, 0.5, 0.4444, 0.5, 0.4444, 0.79211, 0.4444, 0, 0],
    #              [0, 0, 0, 0, 1, 0, 0.5, 0.4444, 0.5, 0.4444, 0.5, 0.4444, 0.79211, 0.4444],
    #              [0, 0, 0, 0, 0, 1, 0.5, 0.4444, 0.5, 0.4444, 0.5, 0.4444, 0.5, 0.4444]]

    x = np.arange(0, 1, 0.1111)
    y = np.arange(0, 1, 0.1111)

    PredArray = np.zeros((len(x), len(y), reps))
    ClassPredArray = np.zeros((len(x), len(y), reps))
    ErrArray = np.zeros((len(x), len(y), reps))

    MeanPredArray = np.zeros((len(x), len(y)))
    MeanClassPredArray = np.zeros((len(x), len(y)))
    MeanErrArray = np.zeros((len(x), len(y)))

    for ii in range(len(x)):
        for jj in range(len(y)):
            for irep in range(reps):
                v1 = x[ii]
                t1 = y[jj]

                v2 = rnd.random()
                t2 = rnd.randint(0, 9)/9

                v3 = rnd.random()
                t3 = rnd.randint(0, 9) / 9

                v4 = rnd.random()
                t4 = rnd.randint(0, 9) / 9

                testState[3][6] = v4
                testState[3][7] = t4

                testState[3][8] = v3
                testState[3][9] = t3
                testState[2][6] = v3
                testState[2][7] = t3

                testState[3][10] = v2
                testState[3][11] = t2
                testState[2][8] = v2
                testState[2][9] = t2
                testState[1][6] = v2
                testState[1][7] = t2

                testState[3][12] = v1
                testState[3][13] = t1
                testState[2][10] = v1
                testState[2][11] = t1
                testState[1][8] = v1
                testState[1][9] = t1
                testState[0][6] = v1
                testState[0][7] = t1

                ClassPredArray[ii, jj, irep] = np.prod(BeliefModel.ClassProbability(ClassModels, testState)>0.5)
                # PredArray[ii, jj, irep] = ClassPredArray[ii, jj, irep]*sum(YJFit.inverse_transform(BeliefModel.EnsembleMeanY(Models, testState).reshape(-1,1)))
                PredArray[ii, jj, irep] = max(YJFit.inverse_transform(BeliefModel.EnsembleMeanY(Models, testState).reshape(-1,1)))
                # ErrArray[ii, jj, irep] = sum(YJFit.inverse_transform(BeliefModel.EnsembleStdevY(Models, testState).reshape(-1,1)))

            # MeanPredArray[ii, jj] = sum(PredArray[ii, jj])/len(PredArray[ii, jj])
            MeanPredArray[ii, jj] = sum(PredArray[ii, jj])/len(PredArray[ii,jj])
            MeanClassPredArray[ii, jj] = sum(ClassPredArray[ii, jj])/len(ClassPredArray[ii, jj])
            # MeanErrArray[ii, jj] = np.average(ErrArray[ii, jj])

    #savemat("C:/Users/Walnut/Documents/MATLAB/AlphaFlow/Pred Forward Potential.mat", {"array": MeanPredArray})
    #savemat("C:/Users/Walnut/Documents/MATLAB/AlphaFlow/Class Pred Forward Potential.mat", {"array": MeanClassPredArray})
    # savemat("C:/Users/LBD/Documents/MATLAB/Err Forward Potential.mat", {"array": MeanErrArray})
    return MeanPredArray, MeanClassPredArray

def ForwardPotentialSeqSel(Models,ClassModels,YJFit):
    reps = 10000

    x = np.arange(0, 4)

    PredArray = np.zeros((len(x), reps))
    ClassPredArray = np.zeros((len(x), reps))

    for ii in range(len(x)):
        for irep in range(reps):
            testState = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

            i1 = x[ii]

            i2 = rnd.randint(1, 4)
            i3 = rnd.randint(1, 4)
            i4 = rnd.randint(1, 4)

            testState[0][i1] = 1
            testState[1][4 + i1] = 1
            testState[2][8 + i1] = 1
            testState[3][12 + i1] = 1

            testState[1][i2] = 1
            testState[2][4 + i2] = 1
            testState[3][8 + i2] = 1

            testState[2][i3] = 1
            testState[3][4 + i3] = 1

            testState[3][i4] = 1

            ClassPredArray[ii, irep] = np.prod(BeliefModel.ClassProbability(ClassModels, testState)>0.3)
            PredArray[ii, irep] = sum(YJFit.inverse_transform(BeliefModel.EnsembleMeanY(Models, testState).reshape(-1,1)))


    #savemat("C:/Users/Walnut/Documents/MATLAB/AlphaFlow/Pred Forward Potential Seq Select.mat", {"array": PredArray})
    #savemat("C:/Users/Walnut/Documents/MATLAB/AlphaFlow/Class Pred Forward Potential Seq Select.mat", {"array": ClassPredArray})
    # savemat("C:/Users/LBD/Documents/MATLAB/Err Forward Potential.mat", {"array": MeanErrArray})
    return PredArray, ClassPredArray

def FwdValFunc(x,v1,t1,Models,ClassModels,YJFit):
    v2 = x[0]
    t2 = x[1]
    v3 = x[2]
    t3 = x[3]
    v4 = x[4]
    t4 = x[5]

    testState = [[0, 1, 0, 0, 0, 0, 0.5, 0.4444, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0.5, 0.4444, 0.5, 0.4444, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0.5, 0.4444, 0.5, 0.4444, 0.5, 0.4444, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0.5, 0.4444, 0.5, 0.4444, 0.5, 0.4444, 0.5, 0.4444]]

    testState[3][6] = v4
    testState[3][7] = round(t4 * 9) / 9

    testState[3][8] = v3
    testState[3][9] = round(t3 * 9) / 9
    testState[2][6] = v3
    testState[2][7] = round(t3 * 9) / 9

    testState[3][10] = v2
    testState[3][11] = round(t2 * 9) / 9
    testState[2][8] = v2
    testState[2][9] = round(t2 * 9) / 9
    testState[1][6] = v2
    testState[1][7] = round(t2 * 9) / 9

    testState[3][12] = v1
    testState[3][13] = round(t1 * 9) / 9
    testState[2][10] = v1
    testState[2][11] = round(t1 * 9) / 9
    testState[1][8] = v1
    testState[1][9] = round(t1 * 9) / 9
    testState[0][6] = v1
    testState[0][7] = round(t1 * 9) / 9
    if any(BeliefModel.ClassProbability(ClassModels,testState) < 0.5):
        NegFwdVal = 1
    else:
        NegFwdVal = -sum(YJFit.inverse_transform(BeliefModel.EnsembleSubsampleY(Models, testState).reshape(-1, 1)))
    return NegFwdVal

def OptimizeFunc(func,v1,t1,Models,ClassModels,YJFit):
    result = sciopt.minimize(func,[rnd.random(),rnd.random(),rnd.random(),rnd.random(),rnd.random(),rnd.random()],args=(v1,t1,Models,ClassModels,YJFit), bounds=[(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)])
    return -result.fun

def OptimizeFuncOverGrid(Models,ClassModels,YJFit):
    reps=200

    x = np.arange(0, 1.1, 0.1)
    y = np.arange(0, 1, 0.1111)

    FwdVal = np.zeros((len(x), len(y)))

    for ii in range(len(x)):
        for jj in range(len(y)):
            OptVal = np.zeros((reps))
            for irep in range(reps):
                OptVal[irep] = OptimizeFunc(FwdValFunc, x[ii], y[jj], Models, ClassModels, YJFit)
            FwdVal[ii,jj] = max(OptVal)

    #savemat("C:/Users/Walnut/Documents/MATLAB/Pred Forward Potential.mat", {"array": FwdVal})
    return FwdVal

def reject_outliers(data, m=4):
    return data[abs(data - np.median(data)) < m * np.std(data)]

def TrimOscfromIndex(State,iTrain):
    iTrim=[]
    for ind in iTrain:
        if (ind + 1) >= len(iTrain):
            iTrim.append(ind)
        elif not (all(State[ind, 0:6] == State[ind + 1, 0:6])):
            iTrim.append(ind)
    return iTrim

def DefineOscillationRangeIndices(State,iTrain):
    iHi=[]
    for ind in iTrain:
        if (ind + 1) >= len(iTrain):
            iHi.append(ind)
        elif not (all(State[ind, 0:6] == State[ind + 1, 0:6])):
            iHi.append(ind)

    iLo=[]
    iLo.append(0)
    for ind in range(len(iHi)):
        if ind+1 < len(iHi):
            iLo.append(iHi[ind]+1)

    iGrp = np.array((iLo,iHi))
    return iGrp

def FormGroupediTrain(iGrp):
    iTrainGrp=[]
    for ind in range(len(iGrp[0])):
        iTrainGrp = np.append(iTrainGrp,np.arange(iGrp[0, ind], iGrp[1, ind] + 1, 1))

    return np.int_(iTrainGrp)

