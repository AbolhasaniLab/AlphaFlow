import pickle as pkl
import numpy as np
import random as rnd
from scipy import optimize as sciopt

from Functions import Encoding
from Functions import Plotting
from Functions import BeliefModel
from Functions import ForwardMapping
from Functions import DecisionPolicy
from Functions import OtherThing
from Functions import DigitalTwinBOFunctions

def ImportModels(ModelSaveFolder):
    Models = []
    YJFit = []

    YJFit.append(pkl.load(open(ModelSaveFolder+'/AP YJFit.pkl','rb')))
    YJFit.append(pkl.load(open(ModelSaveFolder+'/PV YJFit.pkl','rb')))
    YJFit.append(pkl.load(open(ModelSaveFolder+'/API YJFit.pkl','rb')))

    Models.append(pkl.load(open(ModelSaveFolder+'/AP Model.pkl','rb')))
    Models.append(pkl.load(open(ModelSaveFolder+'/PV Model.pkl','rb')))
    Models.append(pkl.load(open(ModelSaveFolder+'/API Model.pkl','rb')))
    Models.append(pkl.load(open(ModelSaveFolder+'/Classifier Model.pkl','rb')))

    return Models, YJFit

def ImportRewardModel(ModelSaveFolder):
    Models = []
    YJFit = []

    YJFit.append(pkl.load(open(ModelSaveFolder+'/Reward YJFit.pkl','rb')))
    Models.append(pkl.load(open(ModelSaveFolder+'/Reward Model.pkl','rb')))
    Models.append(pkl.load(open(ModelSaveFolder+'/Classifier Model.pkl','rb')))

    return Models, YJFit

def SampleFromModelfminReward(x,Models,YJFit,nInject):
    X = ConvertBOxtoX(x)
    Y = np.array([[478,1.68,0.041]])

    for iInject in range(nInject-1):
        Y = np.append(Y,SampleFromModel(X[:(len(X)-(nInject-iInject-1))],Y,Models,YJFit),axis=0)

    if np.any(np.isnan(Y[-1:])):
        Reward = 0
    else:
        Reward = DigitalTwinBOFunctions.LocalReward(Y[-1:])

    xPath = 'C:/Users/RWE/Desktop/cALD Oobert Desktop 072222/cALD/Basin Hopping Save Files 081622/x.txt'
    RPath = 'C:/Users/RWE/Desktop/cALD Oobert Desktop 072222/cALD/Basin Hopping Save Files 081622/R.txt'
    savex = np.loadtxt(xPath,delimiter=',')
    saveR = np.loadtxt(RPath)
    if len(savex)==0:
        savex = x.reshape(1,-1)
        saveR = [Reward]
    else:
        savex = np.vstack((savex,x))
        saveR = np.append(saveR,Reward)

    np.savetxt(xPath,savex, fmt='%.10f',delimiter=',')
    np.savetxt(RPath,saveR, fmt='%.10f',delimiter=',')

    return -Reward

def SampleFromRewardModelBO(x,Models,YJFit,nInject):
    X = ConvertBOxtoX(x)
    Y = np.array([[0]])

    for iInject in range(nInject):
        Y = np.append(Y,SampleFromRewardModelProbability(X[:(len(X)-(nInject-iInject-1))],Y,Models,YJFit),axis=0)

# Cumulative reward for full cycles
    iRewardsX = np.array([5,10,15,20])
    iRewardsY=[]
    for ii in iRewardsX:
        iRewardsY=np.append(iRewardsY,round(np.sum(X[:ii,2])*9+len(X[:ii,2])))
    iRewardsY = np.int_(iRewardsY)
    Reward = np.nansum(Y[iRewardsY])

    return Reward, X, Y

def SampleFromRewardModelfmin(x,Models,YJFit,nInject):
    X = ConvertBOxtoX(x)
    Y = np.array([[0]])

    for iInject in range(nInject):
        Y = np.append(Y,SampleFromRewardModelProbability(X[:(len(X)-(nInject-iInject-1))],Y,Models,YJFit),axis=0)

# Cumulative reward for full cycles
    iRewardsX = np.array([5,10,15,20])
    iRewardsY=[]
    for ii in iRewardsX:
        iRewardsY=np.append(iRewardsY,round(np.sum(X[:ii,2])*9+len(X[:ii,2])))
    iRewardsY = np.int_(iRewardsY)
    try:
        Reward = np.nansum(Y[iRewardsY])
    except:
        print(X,Y)
    return -Reward


def SampleFromModelfminY(x,Models,YJFit,nInject):
    X = ConvertBOxtoX(x)
    Y = np.array([[478,1.68,0.041]])

    for iInject in range(nInject-1):
        Y = np.append(Y,SampleFromModel(X[:(len(X)-(nInject-iInject-1))],Y,Models,YJFit),axis=0)

    return Y[-1:]

def SampleFromModelProbabilityInjY(x,Models,YJFit,nInject):
    X = ConvertBOxtoX(x)
    Y = np.array([[478,1.68,0.041]])

    YInj = []

    for iInject in range(nInject-1):
        Y = np.append(Y,SampleFromModelProbability(X[:(len(X)-(nInject-iInject-1))],Y,Models,YJFit),axis=0)
        YInj = np.append(YInj,Y[-1:])

    return YInj


def ConvertBOxtoX(x):
    nrow = int(len(x)/2)
    X=[[0,0,0]]
    for irow in range(nrow):
        X = np.append(X,[[1,x[irow*2],round(x[irow*2+1]*9)/9]],axis=0)
    return X

def SampleFromModelProbability(X,Y,Models,YJFit):
    Y = AppendUnsampledY(X, Y)
    X,Y = TrimXYData(X, Y, 25)
    xOsc = round(X[-1, 2] * 9 + 1)
    Y_New = []

    if any(np.isnan(Y[-1])):
        for iOsc in range(xOsc):
            Y_New.append([np.nan, np.nan, np.nan])
    else:
        StateClass, ResponseClass = FormatStateResponse_Class(X, Y[:, 0], 4)
        Y_Viable = BeliefModel.ClassProbability(Models[3],StateClass[-xOsc:])

        StateAP, ResponseAP = FormatStateResponse_AP(X, Y[:,0], 4, YJFit[0])
        StatePV, ResponsePV = FormatStateResponse_PV(X, Y[:,1], 4, YJFit[1])
        StateAPI, ResponseAPI = FormatStateResponse_API(X, Y[:,2], 4, YJFit[2])

        Y_AP = Y[-(xOsc+1), 0] + YJFit[0].inverse_transform(
            BeliefModel.EnsembleMeanYRmvOutlier(Models[0], StateAP[-xOsc:]).reshape(-1, 1))
        # Y_AP = np.ravel(Y_AP)
        Y_PV = YJFit[1].inverse_transform(
            BeliefModel.EnsembleMeanYRmvOutlier(Models[1], StatePV[-xOsc:]).reshape(-1, 1))
        Y_API = YJFit[2].inverse_transform(
            BeliefModel.EnsembleMeanYRmvOutlier(Models[2], StateAPI[-xOsc:]).reshape(-1, 1))

        y_New = np.append(Y_AP,Y_PV,axis=1)
        y_New = np.append(y_New,Y_API,axis=1)

        for irow in range(len(y_New)):
            testrand = rnd.random()
            if Y_Viable[irow] > testrand and not(any(np.isnan(y_New[irow]))) and not(any(y_New[:,0] < 450)):
                Y_New.append(y_New[irow])
            else:
                Y_New.append([np.nan,np.nan,np.nan])

    return np.array(Y_New)

def SampleFromRewardModelProbability(X,Y,Models,YJFit):
    Y = AppendUnsampledY(X, Y)
    X,Y = TrimXYData(X, Y, 25)
    xOsc = round(X[-1, 2] * 9 + 1)
    Y_New = []

    if np.isnan(Y[-1]):
        for iOsc in range(xOsc):
            Y_New.append([np.nan])
    else:
        StateClass, ResponseClass = FormatStateResponse_Class(X, Y, 4)
        Y_Viable = BeliefModel.ClassProbability(Models[1],StateClass[-xOsc:])

        StateRew, ResponseRew = FormatStateResponse_Rew(X, Y[:,0], 4, YJFit[0])

        Y_Rew = YJFit[0].inverse_transform(
            BeliefModel.EnsembleMeanYRmvOutlier(Models[0], StateRew[-xOsc:]).reshape(-1, 1))

        y_New = np.array(Y_Rew)

        for irow in range(len(y_New)):
            testrand = rnd.random()
            if Y_Viable[irow] > testrand and not(np.isnan(y_New[irow])) and not(any(y_New > 2)):
                Y_New.append(y_New[irow])
            else:
                Y_New.append([np.nan])

    return np.array(Y_New)

def SampleFromModel(X,Y,Models,YJFit):
    Y = AppendUnsampledY(X, Y)
    X,Y = TrimXYData(X, Y, 25)
    xOsc = round(X[-1, 2] * 9 + 1)
    Y_New = []

    if any(np.isnan(Y[-1])):
        for iOsc in range(xOsc):
            Y_New.append([np.nan, np.nan, np.nan])
    else:
        StateClass, ResponseClass = FormatStateResponse_Class(X, Y[:, 0], 4)
        Y_Viable = (BeliefModel.ClassProbability(Models[3],StateClass[-xOsc:])>0.9)

        StateAP, ResponseAP = FormatStateResponse_AP(X, Y[:,0], 4, YJFit[0])
        StatePV, ResponsePV = FormatStateResponse_PV(X, Y[:,1], 4, YJFit[1])
        StateAPI, ResponseAPI = FormatStateResponse_API(X, Y[:,2], 4, YJFit[2])

        Y_AP = Y[-(xOsc+1), 0] + YJFit[0].inverse_transform(
            BeliefModel.EnsembleMeanYRmvOutlier(Models[0], StateAP[-xOsc:]).reshape(-1, 1))
        # Y_AP = np.ravel(Y_AP)
        Y_PV = YJFit[1].inverse_transform(
            BeliefModel.EnsembleMeanYRmvOutlier(Models[1], StatePV[-xOsc:]).reshape(-1, 1))
        Y_API = YJFit[2].inverse_transform(
            BeliefModel.EnsembleMeanYRmvOutlier(Models[2], StateAPI[-xOsc:]).reshape(-1, 1))

        y_New = np.append(Y_AP,Y_PV,axis=1)
        y_New = np.append(y_New,Y_API,axis=1)

        for irow in range(len(y_New)):
            if Y_Viable[irow] and not(any(np.isnan(y_New[irow]))) and not(any(y_New[:,0] < 450)):
                Y_New.append(y_New[irow])
            else:
                Y_New.append([np.nan,np.nan,np.nan])

    return np.array(Y_New)

def FormatStateResponse_Class(X, Y, MemorySteps):

    TempSTM = Encoding.ShortTermMemoryBackfillCIX23(X, MemorySteps)
    TempData, ResponseLastInjection = AppendYSTMYEndFill(TempSTM, Y)

    TempState, TempResponse = FormatDataSet_AP(TempData, ResponseLastInjection)
    State = np.array(TempState,dtype=object)
    State = State[:,0:-1]
    TempResponse = np.array(TempResponse)

    Response = np.array(TempResponse)

    return State, Response

def FormatStateResponse_AP(X, Y, MemorySteps, YJFit):

    TempSTM = Encoding.ShortTermMemoryBackfillCIX23(X, MemorySteps)
    TempData, ResponseLastInjection = AppendYSTMYEndFill(TempSTM, Y)

    TempState, TempResponse = FormatDataSet_AP(TempData, ResponseLastInjection)
    TempState = np.array(TempState)
    TempResponse = np.array(TempResponse)

    # AP
    State = TempState[:,0:-1]

    ## PV API
    # State = noNaNTempState

    Response = np.array(TempResponse)

    # # AP
    Response[np.isnan(Response)] = -1
    State[np.isnan(State)] = 460

    # # PV
    # Response[np.isnan(Response)] = -0.5
    # State[np.isnan(State)] = 1

    # API
    # Response[np.isnan(Response)] = -0.5
    # State[np.isnan(State)] = 0

    Response = YJFit.transform(Response)
    Response = np.array(Response)

    return State, Response

def FormatStateResponse_PV(X, Y, MemorySteps, YJFit):

    TempSTM = Encoding.ShortTermMemoryBackfillCIX23(X, MemorySteps)
    TempData, ResponseLastInjection = AppendYSTMYEndFill(TempSTM, Y)

    TempState, TempResponse = FormatDataSet_APIPV(TempData, ResponseLastInjection)
    TempState = np.array(TempState)
    TempResponse = np.array(TempResponse)

    # AP
    # State = noNaNTempState[:,0:-1]

    ## PV API
    State = TempState

    Response = np.array(TempResponse)

    # # AP
    # Response[np.isnan(Response)] = -1
    # State[np.isnan(State)] = 460

    # # PV
    Response[np.isnan(Response)] = -0.5
    State[np.isnan(State)] = 1

    # API
    # Response[np.isnan(Response)] = -0.5
    # State[np.isnan(State)] = 0

    Response = YJFit.transform(Response)
    Response = np.array(Response)

    return State, Response

def FormatStateResponse_PV(X, Y, MemorySteps, YJFit):

    TempSTM = Encoding.ShortTermMemoryBackfillCIX23(X, MemorySteps)
    TempData, ResponseLastInjection = AppendYSTMYEndFill(TempSTM, Y)

    TempState, TempResponse = FormatDataSet_APIPV(TempData, ResponseLastInjection)
    TempState = np.array(TempState)
    TempResponse = np.array(TempResponse)

    # AP
    # State = noNaNTempState[:,0:-1]

    ## PV API
    State = TempState

    Response = np.array(TempResponse)

    # # AP
    # Response[np.isnan(Response)] = -1
    # State[np.isnan(State)] = 460

    # # PV
    Response[np.isnan(Response)] = -0.5
    State[np.isnan(State)] = 1

    # API
    # Response[np.isnan(Response)] = -0.5
    # State[np.isnan(State)] = 0

    Response = YJFit.transform(Response)
    Response = np.array(Response)

    return State, Response

def FormatStateResponse_API(X, Y, MemorySteps, YJFit):

    TempSTM = Encoding.ShortTermMemoryBackfillCIX23(X, MemorySteps)
    TempData, ResponseLastInjection = AppendYSTMYEndFill(TempSTM, Y)

    TempState, TempResponse = FormatDataSet_APIPV(TempData, ResponseLastInjection)
    TempState = np.array(TempState)
    TempResponse = np.array(TempResponse)

    # AP
    # State = noNaNTempState[:,0:-1]

    ## PV API
    State = TempState

    Response = np.array(TempResponse)

    # # AP
    # Response[np.isnan(Response)] = -1
    # State[np.isnan(State)] = 460

    # # PV
    # Response[np.isnan(Response)] = -0.5
    # State[np.isnan(State)] = 1

    # API
    Response[np.isnan(Response)] = -0.5
    State[np.isnan(State)] = 0

    Response = YJFit.transform(Response)
    Response = np.array(Response)

    return State, Response

def FormatStateResponse_Rew(X, Y, MemorySteps, YJFit):

    TempSTM = Encoding.ShortTermMemoryBackfillCIX23(X, MemorySteps)
    TempData, ResponseLastInjection = AppendYSTMYEndFill(TempSTM, Y)

    TempState, TempResponse = FormatDataSet_APIPV(TempData, ResponseLastInjection)
    TempState = np.array(TempState)
    TempResponse = np.array(TempResponse)

    # AP
    # State = noNaNTempState[:,0:-1]

    ## PV API
    State = TempState

    Response = np.array(TempResponse)

    # # AP
    # Response[np.isnan(Response)] = -1
    # State[np.isnan(State)] = 460

    # # PV
    # Response[np.isnan(Response)] = -0.5
    # State[np.isnan(State)] = 1

    # API
    Response[np.isnan(Response)] = -0.5
    State[np.isnan(State)] = 0

    Response = YJFit.transform(Response)
    Response = np.array(Response)

    return State, Response

def FormatDataSet_AP(DataSet, ResponseLastInjection):
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
            tempState.append(ResponseLastInjection[DataSet_Step-1])
            #appending tempstate to include the reward of the step before the most recent state (most recent STM)
            State.append(tempState)
            tempResponse.extend(DataSet[DataSet_Step, -1:None])
            ## For AP
            Response.append(tempResponse - ResponseLastInjection[DataSet_Step-1])
            ## For PV and API
            # Response.append(tempResponse)
    return State, Response

def FormatDataSet_APIPV(DataSet, ResponseLastInjection):
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
            tempState.append(ResponseLastInjection[DataSet_Step-1])
            #appending tempstate to include the reward of the step before the most recent state (most recent STM)
            State.append(tempState)
            tempResponse.extend(DataSet[DataSet_Step, -1:None])
            ## For AP
            # Response.append(tempResponse - ResponseLastInjection[DataSet_Step-1])
            ## For PV and API
            Response.append(tempResponse)
    return State, Response

def AppendYSTMYEndFill(STM, Y):
    DataSet = []
    lastY = Y[0]
    ResponseLastInjection = []
    for STM_Step in range(len(STM)):
        if (STM_Step+1) >= len(Y):
            Y = np.append(Y,[0])
        if (STM_Step + 1) >= len(STM):
            a = 1
        elif not(all(STM[STM_Step,0:6] == STM[STM_Step+1,0:6])):
            lastY = Y[STM_Step]
        DataSetRow = []
        #colons unnecessary but afraid
        DataSetRow = STM[STM_Step]
        DataSetRow = np.append(DataSetRow, Y[STM_Step])
        DataSet.append(DataSetRow)
        ResponseLastInjection.append(lastY)
    DataSet = np.stack(DataSet)
    return DataSet, ResponseLastInjection

def TrimXYData(X,Y,XStep):
    X=X[-XStep:]
    Ybackstep = round(sum(X[:,2])*9+len(X))
    Y=Y[-Ybackstep:]
    return X, Y

def AppendUnsampledY(X, Y):
    xOsc = round(X[-1,2]*9+1)
    for ii in range(xOsc):
        Y = np.append(Y,np.zeros((1,Y.shape[1])),axis=0)
    return Y


def RLSelectNextInjectionExploit(X,Y):

    MemorySteps = 4
    BlockEncoding = 'One Hot'
    SubSamplingRate = 0.25
    RegressionModelStructure = 'Ensemble'
    ClassModelStructure = 'SKLearn Ensemble'

    Xdesc, descRangesX2, descRangesX3 = Encoding.DescretizeUniform(X, 7)

    State, Response, ClassState, ClassResponse = Encoding.CIX23Y1toStateResponse(X, Y, MemorySteps)

    Model = BeliefModel.TrainEnsembleSKLearnMLPRegressor(State, Response, 20, SubSamplingRate)
    ClassModel = BeliefModel.TrainGradientBoostingClassifier(ClassState, ClassResponse)

    Objective, Nodes = ForwardMapping.SubsampleNodeMonteCarloCIX23Y1(
        Model, ClassModel, X, Y, 5000, 4, MemorySteps, RegressionModelStructure, ClassModelStructure, descRangesX2,
        descRangesX3)

    CurrentResponse = Response[-1:]
    RecommendedActionDesc = DecisionPolicy.ExploitNodeRandomBranchSum(Objective, Nodes)
    RecommendedAction = Encoding.ValueinDescretizedRangeSample(RecommendedActionDesc, descRangesX2, descRangesX3)

    if X[-1, 0] == 0:
        NewInjection = [1]
    else:
        NewInjection = Encoding.SelectNextPrecursor(State)

    RecommendedAction[0] = NewInjection[0]
    return RecommendedAction

def RLSelectNextInjectionUCB(X,Y):

    MemorySteps = 4
    BlockEncoding = 'One Hot'
    SubSamplingRate = 0.25
    RegressionModelStructure = 'Ensemble'
    ClassModelStructure = 'SKLearn Ensemble'

    Xdesc, descRangesX2, descRangesX3 = Encoding.DescretizeUniform(X, 7)

    State, Response, ClassState, ClassResponse = Encoding.CIX23Y1toStateResponse(X, Y, MemorySteps)

    Model = BeliefModel.TrainEnsembleSKLearnMLPRegressor(State, Response, 20, SubSamplingRate)
    ClassModel = BeliefModel.TrainGradientBoostingClassifier(ClassState, ClassResponse)

    Objective, Nodes = ForwardMapping.SubsampleNodeMonteCarloCIX23Y1(
        Model, ClassModel, X, Y, 5000, 4, MemorySteps, RegressionModelStructure, ClassModelStructure, descRangesX2,
        descRangesX3)

    CurrentResponse = Response[-1:]
    RecommendedActionDesc = DecisionPolicy.UCBNodeRandomBranchSum(Objective, Nodes)
    RecommendedAction = Encoding.ValueinDescretizedRangeSample(RecommendedActionDesc, descRangesX2, descRangesX3)

    if X[-1, 0] == 0:
        NewInjection = [1]
    else:
        NewInjection = Encoding.SelectNextPrecursor(State)

    RecommendedAction[0] = NewInjection[0]
    return RecommendedAction

def RLSelectNextInjectionExploit(X,Y):

    MemorySteps = 4
    BlockEncoding = 'One Hot'
    SubSamplingRate = 0.25
    RegressionModelStructure = 'Ensemble'
    ClassModelStructure = 'SKLearn Ensemble'

    Xdesc, descRangesX2, descRangesX3 = Encoding.DescretizeUniform(X, 7)

    State, Response, ClassState, ClassResponse = Encoding.CIX23Y1toStateResponse(X, Y, MemorySteps)

    Model = BeliefModel.TrainEnsembleSKLearnMLPRegressor(State, Response, 20, SubSamplingRate)
    ClassModel = BeliefModel.TrainGradientBoostingClassifier(ClassState, ClassResponse)

    Objective, Nodes = ForwardMapping.SubsampleNodeMonteCarloCIX23Y1(
        Model, ClassModel, X, Y, 5000, 4, MemorySteps, RegressionModelStructure, ClassModelStructure, descRangesX2,
        descRangesX3)

    CurrentResponse = Response[-1:]
    RecommendedActionDesc = DecisionPolicy.ExploitNodeRandomBranchSum(Objective, Nodes)
    RecommendedAction = Encoding.ValueinDescretizedRangeSample(RecommendedActionDesc, descRangesX2, descRangesX3)

    if X[-1, 0] == 0:
        NewInjection = [1]
    else:
        NewInjection = Encoding.SelectNextPrecursor(State)

    RecommendedAction[0] = NewInjection[0]
    return RecommendedAction

def SlopeReward(X,Y):

    lR = DigitalTwinBOFunctions.LocalReward(Y)
    b1 = 470
    b2 = 600
    ndY = (Y[:,0]-b1)/(b2-b1)

    tempi = -1
    iOscEnd = []
    for ii in range(len(X)):
        tempi = tempi + round(X[ii, 2] * 9 + 1)
        iOscEnd.append(tempi)

    iOscEnd = np.array(iOscEnd)

    iStart = np.where(Y[:,0] == 478)[0]
    R = np.zeros((len(lR),1))

    for ii in range(len(lR)):
        iLastStart = np.max(iStart[iStart <= ii])
        if not(any(iOscEnd < ii)):
            iLastOscEnd = 0
        else:
            iLastOscEnd = np.max(iOscEnd[iOscEnd < ii])

        ndY[ii] = ndY[iLastStart] + np.abs(ndY[iLastStart] - ndY[ii])
        if ii == iLastStart:
            a=0
        elif lR[ii] < lR[iLastOscEnd]:
            lR[ii] = lR[iLastOscEnd]

        if ii == iLastStart:
            R[ii] = 0
        elif np.isnan(lR[ii]):
            R[ii] = np.nan
        else:
            iFit = iOscEnd[(iOscEnd >= iLastStart) & (iOscEnd < ii)]
            iFit = np.append(iFit, ii)
            if len(iFit) > 8:
                iFit = iFit[-8:]
            R[ii] = np.polyfit(ndY[iFit],lR[iFit],1)[0]

    return R

def TrimXYbyDropletNum(X,Y,nDroplets):
    iZero = np.where(X[:,0]==0)[0]

    if nDroplets > len(iZero):
        print('too many droplets')
    else:
        trimX = X[:iZero[nDroplets]]
        iMeas = int(np.round(sum(trimX[:,2])*9+len(trimX[:,2])))
        trimY = Y[:iMeas]

    return trimX, trimY