import numpy as np
import pandas as pd
from scipy.stats import norm
import math
import random as rnd
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import PowerTransformer
# from feature_engine import transformation as vt


def OneHot(i_Block):
    i_Block = int(i_Block)
    v_Block = [0, 0, 0, 0]
    if i_Block != 0:
        v_Block[i_Block - 1] = 1
    return v_Block

def OneHotNLen(i_Block,NLength):
    i_Block = int(i_Block)
    v_Block = np.zeros(NLength)
    if i_Block != 0:
        v_Block[i_Block - 1] = 1
    return v_Block

def NInjtoCI(NInj,CycleLength):
    if NInj == 0:
        NCycle = 0
        CycleInjection = 0
    else:
        NCycle = math.floor((NInj - 1) / CycleLength)
        CycleInjection = (NInj-1) % CycleLength + 1


    CI=np.append(NCycle,OneHotNLen(CycleInjection,CycleLength))
    return CI


def ShortTermMemory(X, MemorySteps, BlockEncoding):
    X_STM = []
    for XStep in range(X.shape[0]):
        tempX_STM = []
        for MemoryStep in range(MemorySteps):
            # print(X)
            # print(XStep)
            # print(MemoryStep)
            if any(X[(XStep - MemoryStep):XStep, 0] == 0) or X[XStep, 0] == 0 or XStep - MemoryStep < 0:
                if BlockEncoding == 'One Hot':
                    tempX_STM.extend(OneHot(0))
                elif BlockEncoding == 'Direct Integer':
                    tempX_STM.extend([0])
                tempX_STM.extend([0, 0])
            else:
                if BlockEncoding == 'One Hot':
                    tempX_STM.extend(OneHot(X[XStep - MemoryStep, 0]))
                elif BlockEncoding == 'Direct Integer':
                    tempX_STM.extend([X[XStep - MemoryStep, 0]])
                tempX_STM.extend(X[XStep - MemoryStep, 1:3])
        X_STM.append(tempX_STM)
    X_STM = np.array(X_STM)
    return X_STM

def ShortTermMemoryX1(X, MemorySteps, BlockEncoding):
    X_STM = []
    for XStep in range(X.shape[0]):
        tempX_STM = []
        for MemoryStep in range(MemorySteps):
            # print(X)
            # print(XStep)
            # print(MemoryStep)
            if any(X[(XStep - MemoryStep):XStep, 0] == 0) or X[XStep, 0] == 0 or XStep - MemoryStep < 0:
                if BlockEncoding == 'One Hot':
                    tempX_STM.extend(OneHot(0))
                elif BlockEncoding == 'Direct Integer':
                    tempX_STM.extend([0])
            else:
                if BlockEncoding == 'One Hot':
                    tempX_STM.extend(OneHot(X[XStep - MemoryStep, 0]))
                elif BlockEncoding == 'Direct Integer':
                    tempX_STM.extend([X[XStep - MemoryStep, 0]])
        X_STM.append(tempX_STM)
    X_STM = np.array(X_STM)
    return X_STM

def ClassShortTermMemory(X, MemorySteps, BlockEncoding):
    X_STM = []
    for XStep in range(X.shape[0]):
        tempX_STM = []
        for MemoryStep in range(MemorySteps):
            # print(X)
            # print(XStep)
            # print(MemoryStep)
            if any(X[(XStep - MemoryStep):XStep, 0] == 0) or X[XStep, 0] == 0 or XStep - MemoryStep < 0:
                if BlockEncoding == 'One Hot':
                    tempX_STM.extend(OneHot(0))
                elif BlockEncoding == 'Direct Integer':
                    tempX_STM.extend([0])
                # if MemoryStep==0:
                tempX_STM.extend([0, 0])
            else:
                if BlockEncoding == 'One Hot':
                    tempX_STM.extend(OneHot(X[XStep - MemoryStep, 0]))
                elif BlockEncoding == 'Direct Integer':
                    tempX_STM.extend([X[XStep - MemoryStep, 0]])
                # if MemoryStep==0:
                tempX_STM.extend(X[XStep - MemoryStep, 1:3])
        X_STM.append(tempX_STM)
    X_STM = np.array(X_STM)
    return X_STM

#short term memory for classifier and regr
def ClassShortTermMemoryX1(X, MemorySteps, BlockEncoding):
    X_STM = []
    for XStep in range(X.shape[0]):
        tempX_STM = []
        for MemoryStep in range(MemorySteps):
            # print(X)
            # print(XStep)
            # print(MemoryStep)
            #if any step in STM or current step == 0 or Xstep = 0 (redund) or a baby with not enough memory steps
            #then any memory steps before cdse injection == cdse injection (0) in the STM + (else) non cdse injections after cdse in STM
            if any(X[(XStep - MemoryStep):XStep, 0] == 0) or X[XStep, 0] == 0 or XStep - MemoryStep < 0:
                if BlockEncoding == 'One Hot':
                    tempX_STM.extend(OneHot(0))
                elif BlockEncoding == 'Direct Integer':
                    tempX_STM.extend([0])
                # if MemoryStep==0:
            else:
                if BlockEncoding == 'One Hot':
                    tempX_STM.extend(OneHot(X[XStep - MemoryStep, 0]))
                elif BlockEncoding == 'Direct Integer':
                    tempX_STM.extend([X[XStep - MemoryStep, 0]])
                # if MemoryStep==0:
        X_STM.append(tempX_STM)
    X_STM = np.array(X_STM)
    return X_STM

#STM input is X_STM
#forming dataset of STM and corresponding reward arrays
def AppendYtoSTM(STM, Y):
    DataSet = []
    for STM_Step in range(len(STM)):
        DataSetRow = []
        #
        DataSetRow = STM[STM_Step, :]
        DataSetRow = np.append(DataSetRow, Y[STM_Step, :])
        DataSet.append(DataSetRow)
    DataSet = np.stack(DataSet)
    return DataSet

def AppendYEndofLastInjectiontoSTM(STM, Y):
    DataSet = []
    lastY = Y[0, :]
    ResponseLastInjection = []
    for STM_Step in range(len(STM)):
        if (STM_Step+1) >= len(Y):
            a=1
        elif not(all(STM[STM_Step,0:6] == STM[STM_Step+1,0:6])):
            lastY = Y[STM_Step, :]
        DataSetRow = []
        #
        DataSetRow = STM[STM_Step, :]
        DataSetRow = np.append(DataSetRow, Y[STM_Step, :])
        DataSet.append(DataSetRow)
        ResponseLastInjection.append(lastY)
    DataSet = np.stack(DataSet)
    return DataSet, ResponseLastInjection

def FormatDataSetforTraining(DataSet, BlockEncoding):
    State = []
    Response = []
    if BlockEncoding == 'One Hot':
        for DataSet_Step in range(len(DataSet)):
            tempState = []
            tempResponse = []
            if sum(DataSet[DataSet_Step, 0:4]) != 0:
                tempState.extend(DataSet[DataSet_Step, 0:-1])
                tempState.extend(DataSet[DataSet_Step - 1, -1:None])
                State.append(tempState)
                tempResponse.extend(DataSet[DataSet_Step, -1:None])
                Response.append(tempResponse)
    elif BlockEncoding == 'Direct Integer':
        for DataSet_Step in range(len(DataSet)):
            tempState = []
            tempResponse = []
            if DataSet[DataSet_Step, 0] != 0:
                tempState.extend(DataSet[DataSet_Step, 0:-1])
                tempState.extend(DataSet[DataSet_Step - 1, -1:None])
                State.append(tempState)
                tempResponse.extend(DataSet[DataSet_Step, -1:None])
                Response.append(tempResponse)
    return State, Response

#
#formatting data to map state (STM before most recent action and second to last reward)
# + action (most recent action in STM functions) to (most recent) reward
def FormatClassDataSetforTraining(DataSet, BlockEncoding):
    State = []
    Response = []
    if BlockEncoding == 'One Hot':
        for DataSet_Step in range(len(DataSet)):
            #for each row in dataset set tempstate to be []
            tempState = []
            tempResponse = []
            #if on a CdSe injection line, injection one hot + STM all 0s
            #so if sum != 0 then not on cdse injection line
            if sum(DataSet[DataSet_Step, 0:4]) != 0: #if not on a cdse injection line based on the stm in dataset
                tempState.extend(DataSet[DataSet_Step, 0:-1])
                #tempstate = all values in corresponding row of dataset except for the last value which is [currently] reward from most recent action
                tempState.extend(DataSet[DataSet_Step - 1, -1:None])
                #appending tempstate to include the reward of the step before the most recent state (most recent STM)
                State.append(tempState)
                tempResponse.extend(DataSet[DataSet_Step, -1:None])
                Response.append(tempResponse)
    elif BlockEncoding == 'Direct Integer':
        for DataSet_Step in range(len(DataSet)):
            tempState = []
            tempResponse = []
            if DataSet[DataSet_Step, 0] != 0:
                tempState.extend(DataSet[DataSet_Step, 0:-1])
                tempState.extend(DataSet[DataSet_Step - 1, -1:None])
                State.append(tempState)
                tempResponse.extend(DataSet[DataSet_Step, -1:None])
                Response.append(tempResponse)
    return State, Response


def ReplaceNanwith0(Data):
    for row in range(len(Data)):
        for col in range(len(Data[row, :])):
            if np.isnan(Data[row, col]):
                Data[row, col] = 0
    return Data


def RemoveNanRows(Data):
    TrimmedData = []
    for row in range(len(Data)):
        if not (any(np.isnan(Data[row, :]))):
            TrimmedData.append(Data[row, :])

    TrimmedData = np.array(TrimmedData)
    return TrimmedData


def XYtoStateResponse(X, Y, MemorySteps, BlockEncoding):
    TempSTM = ClassShortTermMemory(X, MemorySteps, BlockEncoding)
    TempData = AppendYtoSTM(TempSTM, Y)
    TempState, TempResponse = FormatDataSetforTraining(TempData, BlockEncoding)
    TempState = np.array(TempState)
    TempResponse = np.array(TempResponse)

    ClassState = TempState[:, 0:-2]
    ClassResponse = ~np.isnan(TempResponse[:, 0])

    STM = ShortTermMemory(X, MemorySteps, BlockEncoding)
    Data = AppendYtoSTM(STM, Y)
    Data = RemoveNanRows(Data)
    State, Response = FormatDataSetforTraining(Data, BlockEncoding)
    State = np.array(State)
    Response = np.array(Response)

    return State, Response, ClassState, ClassResponse


def XY1toStateResponse(X, Y, MemorySteps, BlockEncoding):
    TempSTM = ClassShortTermMemory(X, MemorySteps, BlockEncoding)
    TempData = AppendYtoSTM(TempSTM, Y)

    TempState, TempResponse = FormatClassDataSetforTraining(TempData, BlockEncoding)
    TempState = np.array(TempState)
    TempResponse = np.array(TempResponse)

    ClassState = TempState[:, 0:-1]
    ClassResponse = ~np.isnan(TempResponse)

    STM = ShortTermMemory(X, MemorySteps, BlockEncoding)
    Data = AppendYtoSTM(STM, Y)
    Data = RemoveNanRows(Data)
    State, Response = FormatDataSetforTraining(Data, BlockEncoding)
    State = np.array(State)
    Response = np.array(Response)
    Response = Response[:, 1]

    return State, Response, ClassState, ClassResponse

def X1Y1toStateResponse(X, Y, MemorySteps, BlockEncoding):
    TempSTM = ClassShortTermMemoryX1(X, MemorySteps, BlockEncoding)
    TempData = AppendYtoSTM(TempSTM, Y)

    TempState, TempResponse = FormatClassDataSetforTraining(TempData, BlockEncoding)
    TempState = np.array(TempState)
    TempResponse = np.array(TempResponse)

    ClassState = TempState[:, 0:-1]
    # ClassState = TempState
    #classifier training on t/f (notisnan/isnan)
    ClassResponse = ~np.isnan(TempResponse)

    STM = ShortTermMemoryX1(X, MemorySteps, BlockEncoding)
    Data = AppendYtoSTM(STM, Y)
    # Data = RemoveNanRows(Data)
    #turn nans into 0s so regr trained with nans as penalty
    Data[np.isnan(Data)] = 0
    TempState, TempResponse = FormatDataSetforTraining(Data, BlockEncoding)
    TempState = np.array(TempState)
    TempResponse = np.array(TempResponse)
    Response = TempResponse - TempState[:, -1:None]
    State = TempState[:, 0:-1]

    State = np.array(State)
    Response = np.array(Response)
    # print(Response)
    Response = np.ravel(Response)

    #classresponse (tempresponse and tempstate are npraveled in classifier training section)

    return State, Response, ClassState, ClassResponse


def ObjectiveFunction(Y, P):
    a_PWL = 1
    a_PV = 0
    r_PWL = 100
    r_PV = 2
    f = []
    for row in range(len(Y[:, 0])):
        tempf = a_PWL * (Y[row, 0] - 500) / r_PWL + a_PV * Y[row, 1] / r_PV
        # print(tempf)
        tempf = tempf * (P[row] > 0.5)
        # print(P[row])
        # print(tempf)
        f.append(tempf)
    return f


def ObjectiveFunctionErr(YErr, P):
    a_PWL = 1
    a_PV = 0
    r_PWL = 100
    r_PV = 2
    normP = P
    f = []
    for row in range(len(YErr[:, 0])):
        # normP[row]=norm.pdf(P[row],0.5,0.1)/norm.pdf(0.5,0.5,0.1)
        tempf = ((a_PWL * YErr[row, 0] / r_PWL) ** 2 + (a_PV * YErr[row, 1] / r_PV) ** 2) ** 0.5
        tempf = tempf * normP[row]
        f.append(tempf)
    return f


def ObjectiveFunctionY1(Y, P):
    f = []
    for row in range(len(Y)):
        tempf = Y[row]
        tempf = tempf * (P[row] > 0.3)
        if (tempf > -0.00005) and (tempf <0.00005):
            tempf = -8
        # tempf = tempf * P[row]
        f.append(tempf)
    return f


def ObjectiveFunctionErrY1(YErr, P):
    normP = P
    f = []
    for row in range(len(YErr)):
        normP[row] = norm.pdf(P[row], 0.5, 0.1) / norm.pdf(0.5, 0.5, 0.1)
        tempf = YErr[row]
        tempf = tempf * normP[row]
        f.append(tempf)
    return f


def BuildReplicateList(State, PredictionState):

    nBranches=PredictionState.shape[0]
    nLevels=PredictionState.shape[1]

    ReplicateCounts=np.zeros([nBranches,nLevels])

    iStart=[]
    for iState in range(len(State)):
        if sum(State[iState])==1:
            iStart.append(iState)

    for iState in range(max(iStart)):
        for iBranch in range(nBranches):
            for iLevel in range(nLevels):
                if np.allclose(State[iState],PredictionState[iBranch,iLevel]):
                    ReplicateCounts[iBranch,iLevel]=ReplicateCounts[iBranch,iLevel]+1

    return ReplicateCounts

# def DescretizeRangeMedian(X,Reward):
#     GroupedX=[]
#     for x in X:
#
#     return DescX

def DescretizeUniform(X,DescreteLevels):
    LevelRange = 1/(DescreteLevels)
    LowerBounds = np.arange(0,1,LevelRange)
    DescX=[]
    for x in X:
        Descx = [x[0],np.argmax(LowerBounds[x[1]>=LowerBounds]),np.argmax(LowerBounds[x[2]>=LowerBounds])]
        DescX.append(Descx)

    DescX = np.array(DescX)
    return DescX, LowerBounds, LowerBounds

def DescretizeMedian(X,Reward):

    return DescX, LowerBoundsX2, LowerBoundsX3

def ValueinDescretizedRangeSample(descX, descRangesX2, descRangesX3):

    descRangesX2 = np.append(descRangesX2, 1)
    descRangesX3 = np.append(descRangesX3, 1)

    if np.shape(descX)[0] == 3:
        X = np.zeros(3)
        X[0] = descX[0]

        templowerbound = descRangesX2[descX[1]]
        tempupperbound = descRangesX2[descX[1]+1]
        X[1] = rnd.uniform(templowerbound, tempupperbound)

        templowerbound = descRangesX3[descX[2]]
        tempupperbound = descRangesX3[descX[2]+1]
        randX2 = rnd.uniform(templowerbound, tempupperbound)
        X[2] = round(randX2 * 9) / 9
    elif np.shape(descX)[0] == 2:
        X = np.zeros(2)

        templowerbound = descRangesX2[descX[0]]
        tempupperbound = descRangesX2[descX[0] + 1]
        X[0] = rnd.uniform(templowerbound, tempupperbound)

        templowerbound = descRangesX3[descX[1]]
        tempupperbound = descRangesX3[descX[1] + 1]
        randX2 = rnd.uniform(templowerbound, tempupperbound)
        X[1] = round(randX2 * 9) / 9

    return X

def CIX23Y1toStateResponse(X, Y, MemorySteps):

    TempSTM = ShortTermMemoryBackfillCIX23(X, MemorySteps)
    TempData = AppendYtoSTM(TempSTM, Y)

    TempState, TempResponse = FormatCIX23Y1DataSetforTraining(TempData)
    TempState = np.array(TempState)
    TempResponse = np.array(TempResponse)

    TempResponse = TempResponse - TempState[:,-1:None]

    ClassState = TempState[:, 0:-1]
    State = TempState[:,0:-1]

    #classifier training on t/f (notisnan/isnan)
    ClassResponse = ~np.isnan(TempResponse)

    Response = np.array(TempResponse)
    Response[np.isnan(Response)] = -1

    pt = PowerTransformer(method='yeo-johnson')
    pt.fit(Response)
    Response = pt.transform(Response)

    Response = np.ravel(Response)

    #classresponse (tempresponse and tempstate are npraveled in classifier training section)

    return State, Response, ClassState, ClassResponse

def CIX23Y1toStateResponsewYJTrans(X, Y, MemorySteps):

    TempSTM = ShortTermMemoryBackfillCIX23(X, MemorySteps)
    TempData = AppendYtoSTM(TempSTM, Y)

    TempState, TempResponse = FormatCIX23Y1DataSetforTraining(TempData)
    TempState = np.array(TempState)
    TempResponse = np.array(TempResponse)

    TempResponse = TempResponse - TempState[:,-1:None]

    ClassState = TempState[:, 0:-1]
    State = TempState[:,0:-1]

    #classifier training on t/f (notisnan/isnan)
    ClassResponse = ~np.isnan(TempResponse)

    Response = np.array(TempResponse)
    Response[np.isnan(Response)] = -1
    Response1 = Response[0:2500]
    Response2 = Response[2500:]
    YJFit = PowerTransformer(method='yeo-johnson')
    YJFit.fit(Response1)
    Response1 = YJFit.transform(Response1)
    Response2 = YJFit.transform(Response2)

   # Response = power_transform(Response)
   # Response1 = power_transform(Response[0:2500])
   # Response2 = power_transform(Response[2500:])
    Response = np.concatenate((Response1, Response2))
    Response = np.ravel(Response)

    #classresponse (tempresponse and tempstate are npraveled in classifier training section)

    return State, Response, ClassState, ClassResponse, YJFit

def CIX23Y1toStateResponsewYJTrans_v2(X, Y, MemorySteps):

    TempSTM = ShortTermMemoryBackfillCIX23(X, MemorySteps)
    TempData, ResponseLastInjection = AppendYEndofLastInjectiontoSTM(TempSTM, Y)

    TempState, TempResponse = FormatCIX23Y1DataSetforTraining_v2(TempData, ResponseLastInjection)
    TempState = np.array(TempState)
    TempResponse = np.array(TempResponse)

    # TempResponse = TempResponse - ResponseLastInjection

    # ClassState = TempState[:, 0:-1]
    # State = TempState[:,0:-1]
    State = TempState
    ClassState = TempState

    State[np.isnan(State)] = -0.5

    #classifier training on t/f (notisnan/isnan)
    ClassResponse = ~np.isnan(TempResponse)

    Response = np.array(TempResponse)
    Response[np.isnan(Response)] = -0.5
    Response1 = Response[0:2500]
    Response2 = Response[2500:]
    YJFit = PowerTransformer(method='yeo-johnson')
    YJFit.fit(Response1)
    Response1 = YJFit.transform(Response1)
    Response2 = YJFit.transform(Response2)

   # Response = power_transform(Response)
   # Response1 = power_transform(Response[0:2500])
   # Response2 = power_transform(Response[2500:])
    Response = np.concatenate((Response1, Response2))
    Response = np.ravel(Response)

    #classresponse (tempresponse and tempstate are npraveled in classifier training section)

    return State, Response, ClassState, ClassResponse, YJFit
def CIX23Y1toStateResponsewYJTransNaNRemove_v2(X, Y, MemorySteps):

    TempSTM = ShortTermMemoryBackfillCIX23(X, MemorySteps)
    TempData, ResponseLastInjection = AppendYEndofLastInjectiontoSTM(TempSTM, Y)

    TempState, TempResponse = FormatCIX23Y1DataSetforTraining_v2(TempData, ResponseLastInjection)
    TempState = np.array(TempState)
    TempResponse = np.array(TempResponse)

    noNaNTempState=[]
    noNaNTempResponse = []
    for iRow in range(len(TempResponse)):
        if not(np.isnan(TempResponse[iRow])):
            noNaNTempState.append(TempState[iRow])
            noNaNTempResponse.append(TempResponse[iRow])

    # TempResponse = TempResponse - ResponseLastInjection
    noNaNTempResponse=np.array(noNaNTempResponse)
    noNaNTempState=np.array(noNaNTempState)

    ClassState = TempState[:, 0:-1]
    # AP
    # State = noNaNTempState[:,0:-1]

    ## PV API
    State = noNaNTempState

    #classifier training on t/f (notisnan/isnan)
    ClassResponse = ~np.isnan(TempResponse)



    Response = np.array(noNaNTempResponse)

    # # AP
    Response[np.isnan(Response)] = -1
    State[np.isnan(State)] = 460

    # # PV
    # Response[np.isnan(Response)] = -0.5
    # State[np.isnan(State)] = 1

    # API
    # Response[np.isnan(Response)] = -0.5
    # State[np.isnan(State)] = 0

    Response1 = Response[0:2500]
    Response2 = Response[2500:]
    YJFit = PowerTransformer(method='yeo-johnson')
    YJFit.fit(Response1)
    Response1 = YJFit.transform(Response1)
    Response2 = YJFit.transform(Response2)

   # Response = power_transform(Response)
   # Response1 = power_transform(Response[0:2500])
   # Response2 = power_transform(Response[2500:])
    Response = np.concatenate((Response1, Response2))
    Response = np.ravel(Response)

    #classresponse (tempresponse and tempstate are npraveled in classifier training section)

    return State, Response, ClassState, ClassResponse, YJFit

def CIX23Y1toStateResponsewYJTransNaNRemove(X, Y, MemorySteps):

    TempSTM = ShortTermMemoryBackfillCIX23(X, MemorySteps)
    TempData = AppendYtoSTM(TempSTM, Y)

    TempState, TempResponse = FormatCIX23Y1DataSetforTraining(TempData)
    TempState = np.array(TempState)
    TempResponse = np.array(TempResponse)

    TempResponse = TempResponse - TempState[:,-1:None]

    ClassState = TempState[:, 0:-1]

    noNaNTempState=[]
    noNaNTempResponse = []
    for iRow in range(len(TempResponse)):
        if not(np.isnan(TempResponse[iRow])):
            noNaNTempState.append(TempState[iRow])
            noNaNTempResponse.append(TempResponse[iRow])

    noNaNTempState = np.array(noNaNTempState)
    State = noNaNTempState[:,0:-1]

    #classifier training on t/f (notisnan/isnan)
    ClassResponse = ~np.isnan(TempResponse)

    Response = np.array(noNaNTempResponse)
    # Response[np.isnan(Response)] = -1
    Response1 = Response[0:2500]
    Response2 = Response[2500:]
    YJFit = PowerTransformer(method='yeo-johnson')
    YJFit.fit(Response1)
    Response1 = YJFit.transform(Response1)
    Response2 = YJFit.transform(Response2)

   # Response = power_transform(Response)
   # Response1 = power_transform(Response[0:2500])
   # Response2 = power_transform(Response[2500:])
    Response = np.concatenate((Response1, Response2))
    Response = np.ravel(Response)

    #classresponse (tempresponse and tempstate are npraveled in classifier training section)

    return State, Response, ClassState, ClassResponse, YJFit

def ShortTermMemoryCIX23(X, MemorySteps):
    X_STM = []
    CycleInjections = 0
    for XStep in range(X.shape[0]):
        if X[XStep,0] == 0:
            CycleInjections = 0
        else:
            CycleInjections = CycleInjections + 1

        tempX_STM = []
        tempX_STM.extend(NInjtoCI(CycleInjections, 5))
        for MemoryStep in range(MemorySteps):
            # print(X)
            # print(XStep)
            # print(MemoryStep)
            #if any step in STM or current step == 0 or Xstep = 0 (redund) or a baby with not enough memory steps
            #then any memory steps before cdse injection == cdse injection (0) in the STM + (else) non cdse injections after cdse in STM
            if any(X[(XStep - MemoryStep):XStep, 0] == 0) or X[XStep, 0] == 0 or XStep - MemoryStep < 0:
                tempX_STM.extend([0,0])

                # if MemoryStep==0:
            else:
                tempX_STM.extend(X[XStep - MemoryStep, 1:3])
                # if MemoryStep==0:
        X_STM.append(tempX_STM)
    X_STM = np.array(X_STM)
    return X_STM

def ShortTermMemoryBackfillCIX23(X, MemorySteps):
    OscLevels = 9

    X_STM = []
    CycleInjections = 0
    for XStep in range(X.shape[0]):
        if X[XStep,0] == 0:
            CycleInjections = 0
        else:
            CycleInjections = CycleInjections + 1
        tempTX = 0
        for Osc in range(round(X[XStep, 2] * OscLevels) + 1):
            tempX_STM = []
            tempX_STM.extend(NInjtoCI(CycleInjections, 5))
            for MemoryStep in range(MemorySteps):
                # print(X)
                # print(XStep)
                # print(MemoryStep)
                #if any step in STM or current step == 0 or Xstep = 0 (redund) or a baby with not enough memory steps
                #then any memory steps before cdse injection == cdse injection (0) in the STM + (else) non cdse injections after cdse in STM
                if any(X[(XStep - MemoryStep):XStep, 0] == 0) or X[XStep, 0] == 0 or XStep - MemoryStep < 0:
                    tempX_STM.extend([0,0])
                    # if MemoryStep==0:
                else:
                    if MemoryStep == 0:
                        tempX_STM.extend([X[XStep - MemoryStep, 1],tempTX])
                    else:
                        tempX_STM.extend(X[XStep - MemoryStep, 1:3])
                    # if MemoryStep==0:
            tempTX = tempTX + 1 / OscLevels
            X_STM.append(tempX_STM)
    X_STM = np.array(X_STM)
    return X_STM

def FormatCIX23Y1DataSetforTraining(DataSet):
    State = []
    Response = []
    for DataSet_Step in range(len(DataSet)):
        #for each row in dataset set tempstate to be []
        tempState = []
        tempResponse = []
        #if on a CdSe injection line, injection one hot + STM all 0s
        #so if sum != 0 then not on cdse injection line
        if sum(DataSet[DataSet_Step, 0:6]) != 0: #if not on a cdse injection line based on the stm in dataset
            tempState.extend(DataSet[DataSet_Step, 0:-1])
            #tempstate = all values in corresponding row of dataset except for the last value which is [currently] reward from most recent action
            tempState.extend(DataSet[DataSet_Step - 1, -1:None])
            #appending tempstate to include the reward of the step before the most recent state (most recent STM)
            State.append(tempState)
            tempResponse.extend(DataSet[DataSet_Step, -1:None])
            Response.append(tempResponse)
    return State, Response

def FormatCIX23Y1DataSetforTraining_v2(DataSet, ResponseLastInjection):
    State = []
    Response = []
    for DataSet_Step in range(len(DataSet)):
        #for each row in dataset set tempstate to be []
        tempState = []
        tempResponse = []
        #if on a CdSe injection line, injection one hot + STM all 0s
        #so if sum != 0 then not on cdse injection line
        if sum(DataSet[DataSet_Step, 0:6]) != 0: #if not on a cdse injection line based on the stm in dataset
            tempState.extend(DataSet[DataSet_Step, 0:-1])
            #tempstate = all values in corresponding row of dataset except for the last value which is [currently] reward from most recent action
            # tempState.extend(DataSet[DataSet_Step - 1, -1:None])
            tempState.extend(ResponseLastInjection[DataSet_Step-1])
            #appending tempstate to include the reward of the step before the most recent state (most recent STM)
            State.append(tempState)
            tempResponse.extend(DataSet[DataSet_Step, -1:None])
            ## For AP
            # Response.append(tempResponse - ResponseLastInjection[DataSet_Step-1])
            ## For PV and API
            Response.append(tempResponse)
    return State, Response

def SelectNextPrecursor(State):
    CurrentStateInjection = State[-1][1:6] == 1
    InjectionSequenceOffset = np.array([2, 4, 3, 1, 1])

    #check if any value in injectionsequenceoffset is ==1, if not then last injection was qd droplet so next inject oam
    if not(any(CurrentStateInjection)):
        InjectP = np.array([1])
    else:
        InjectP = InjectionSequenceOffset[CurrentStateInjection]

    return InjectP
