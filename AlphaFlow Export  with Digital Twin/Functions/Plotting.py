import matplotlib.pyplot as plt
from IPython import display
import time

from sklearn import metrics
from Functions import BeliefModel

import seaborn as sns

def PlotTFEnsembleRegression(Model,X_train,Y_train,X_test,Y_test):

    Y_test_pred=BeliefModel.EnsembleMeanY(Model,X_test)
    Y_train_pred=BeliefModel.EnsembleMeanY(Model,X_train)
    Y_test_predErr=BeliefModel.EnsembleStdevY(Model,X_test)**2
    Y_train_predErr=BeliefModel.EnsembleStdevY(Model,X_train)**2

    for variable in range(len(Y_train[0])):
        fig, ax = plt.subplots(figsize=(6,6))

        # ax.scatter(Y_train[:,variable], Y_train_pred[:,variable])
        ax.errorbar(Y_train[:,variable], Y_train_pred[:,variable], yerr=Y_train_predErr[:,variable], fmt="o")
        #ax.plot([1.8,2.6],[1.8,2.6])
        #ax.set(xlim=(1.8, 2.6), ylim=(1.8, 2.6))
        ax.set_xlabel('PWL Train (nm)')
        ax.set_ylabel('PWL Train Predicted (nm)')
        plt.show()

        fig, ax = plt.subplots(figsize=(6,6))

        # ax.scatter(Y_test[:,variable], Y_test_pred[:,variable])
        ax.errorbar(Y_test[:,variable], Y_test_pred[:,variable], yerr=Y_test_predErr[:,variable], fmt="o")
        #ax.plot([1.8,2.6],[1.8,2.6])

        #ax.set(xlim=(1.8, 2.6), ylim=(1.8, 2.6))
        ax.set_xlabel('PV Test')
        ax.set_ylabel('PV Test Predicted')
        plt.show()
    
def PlotSingleEnsembleRegression(Model,X_train,Y_train,X_test,Y_test):

    Y_test_pred=BeliefModel.EnsembleMeanY(Model,X_test)
    Y_train_pred=BeliefModel.EnsembleMeanY(Model,X_train)
    Y_test_predErr=BeliefModel.EnsembleStdevY(Model,X_test)**2
    Y_train_predErr=BeliefModel.EnsembleStdevY(Model,X_train)**2

    fig, ax = plt.subplots(figsize=(6,6))

    # ax.scatter(Y_train[:,variable], Y_train_pred[:,variable])
    ax.errorbar(Y_train, Y_train_pred, yerr=Y_train_predErr, fmt="o")
    #ax.plot([1.8,2.6],[1.8,2.6])
    #ax.set(xlim=(1.8, 2.6), ylim=(1.8, 2.6))
    ax.set_xlabel('PWL Train (nm)')
    ax.set_ylabel('PWL Train Predicted (nm)')
    plt.show()

    fig, ax = plt.subplots(figsize=(6,6))

    # ax.scatter(Y_test[:,variable], Y_test_pred[:,variable])
    ax.errorbar(Y_test, Y_test_pred, yerr=Y_test_predErr, fmt="o")
    #ax.plot([1.8,2.6],[1.8,2.6])

    #ax.set(xlim=(1.8, 2.6), ylim=(1.8, 2.6))
    ax.set_xlabel('PV Test')
    ax.set_ylabel('PV Test Predicted')
    plt.show()

def PlotTFEnsembleClassifier(Model,X_train,Y_train,X_test,Y_test):
    Y_test_pred=BeliefModel.EnsembleClassProbability(Model,X_test)
    Y_train_pred=BeliefModel.EnsembleClassProbability(Model,X_train)

    fig, ax = plt.subplots(figsize=(6,6))

    ax.scatter(Y_train, Y_train_pred, s=10)
    #ax.plot([1.8,2.6],[1.8,2.6])
    #ax.set(xlim=(1.8, 2.6), ylim=(1.8, 2.6))
    ax.set_xlabel('Class Train')
    ax.set_ylabel('Class Train Probability')
    plt.show()

    fig, ax = plt.subplots(figsize=(6,6))

    ax.scatter(Y_test, Y_test_pred, s=10)
    #ax.plot([1.8,2.6],[1.8,2.6])

    #ax.set(xlim=(1.8, 2.6), ylim=(1.8, 2.6))
    ax.set_xlabel('Class Test')
    ax.set_ylabel('Class Test Probability')
    plt.show()


    fig, ax = plt.subplots(figsize=(6,6))
    con_mat_train=metrics.confusion_matrix(Y_train,Y_train_pred>0.5)
    print(con_mat_train)
    con_mat_train=[con_mat_train[0,:]/sum(con_mat_train[0,:]),con_mat_train[1,:]/sum(con_mat_train[1,:])]
    print(con_mat_train)
    sns.heatmap(con_mat_train)
    plt.show()

    fig, ax = plt.subplots(figsize=(6,6))
    con_mat_test=metrics.confusion_matrix(Y_test,Y_test_pred>0.5)
    print(con_mat_test)
    con_mat_test=[con_mat_test[0,:]/sum(con_mat_test[0,:]),con_mat_test[1,:]/sum(con_mat_test[1,:])]
    print(con_mat_test)
    sns.heatmap(con_mat_test)
    plt.show()


def PlotTFRegression(Model,X_train,Y_train,X_test,Y_test):

    Y_test_pred=BeliefModel.RegressionPredictY(Model,X_test)
    Y_train_pred=BeliefModel.RegressionPredictY(Model,X_train)
    Y_test_predErr=BeliefModel.RegressionPredictYErr(Model,X_test)**2
    Y_train_predErr=BeliefModel.RegressionPredictYErr(Model,X_train)**2

    for variable in range(len(Y_train[0])):
        fig, ax = plt.subplots(figsize=(6,6))

        # ax.scatter(Y_train[:,variable], Y_train_pred[:,variable])
        ax.errorbar(Y_train[:,variable], Y_train_pred[:,variable], yerr=Y_train_predErr[:,variable], fmt="o")
        #ax.plot([1.8,2.6],[1.8,2.6])
        #ax.set(xlim=(1.8, 2.6), ylim=(1.8, 2.6))
        ax.set_xlabel('PWL Train (nm)')
        ax.set_ylabel('PWL Train Predicted (nm)')
        plt.show()

        fig, ax = plt.subplots(figsize=(6,6))

        # ax.scatter(Y_test[:,variable], Y_test_pred[:,variable])
        ax.errorbar(Y_test[:,variable], Y_test_pred[:,variable], yerr=Y_test_predErr[:,variable], fmt="o")
        #ax.plot([1.8,2.6],[1.8,2.6])

        #ax.set(xlim=(1.8, 2.6), ylim=(1.8, 2.6))
        ax.set_xlabel('PV Test')
        ax.set_ylabel('PV Test Predicted')
        plt.show()

def PlotSingleRegression(Model,X_train,Y_train,X_test,Y_test):

    Y_test_pred=BeliefModel.RegressionPredictY(Model,X_test)
    Y_train_pred=BeliefModel.RegressionPredictY(Model,X_train)
    Y_test_predErr=BeliefModel.RegressionPredictYErr(Model,X_test)**2
    Y_train_predErr=BeliefModel.RegressionPredictYErr(Model,X_train)**2

    fig, ax = plt.subplots(figsize=(6,6))

    # ax.scatter(Y_train[:,variable], Y_train_pred[:,variable])
    ax.errorbar(Y_train, Y_train_pred, yerr=Y_train_predErr, fmt="o")
    #ax.plot([1.8,2.6],[1.8,2.6])
    #ax.set(xlim=(1.8, 2.6), ylim=(1.8, 2.6))
    ax.set_xlabel('PWL Train (nm)')
    ax.set_ylabel('PWL Train Predicted (nm)')
    plt.show()

    fig, ax = plt.subplots(figsize=(6,6))

    # ax.scatter(Y_test[:,variable], Y_test_pred[:,variable])
    ax.errorbar(Y_test, Y_test_pred, yerr=Y_test_predErr, fmt="o")
    #ax.plot([1.8,2.6],[1.8,2.6])

    #ax.set(xlim=(1.8, 2.6), ylim=(1.8, 2.6))
    ax.set_xlabel('PV Test')
    ax.set_ylabel('PV Test Predicted')
    plt.show()


def PlotTFClassifier(Model,X_train,Y_train,X_test,Y_test):
    Y_test_pred=BeliefModel.ClassProbability(Model,X_test)
    Y_train_pred=BeliefModel.ClassProbability(Model,X_train)

    fig, ax = plt.subplots(figsize=(6,6))

    ax.scatter(Y_train, Y_train_pred, s=10)
    #ax.plot([1.8,2.6],[1.8,2.6])
    #ax.set(xlim=(1.8, 2.6), ylim=(1.8, 2.6))
    ax.set_xlabel('Class Train')
    ax.set_ylabel('Class Train Probability')
    plt.show()

    fig, ax = plt.subplots(figsize=(6,6))

    ax.scatter(Y_test, Y_test_pred, s=10)
    #ax.plot([1.8,2.6],[1.8,2.6])

    #ax.set(xlim=(1.8, 2.6), ylim=(1.8, 2.6))
    ax.set_xlabel('Class Test')
    ax.set_ylabel('Class Test Probability')
    plt.show()


    fig, ax = plt.subplots(figsize=(6,6))
    con_mat_train=metrics.confusion_matrix(Y_train,Y_train_pred>0.5)
    print(con_mat_train)
    con_mat_train=[con_mat_train[0,:]/sum(con_mat_train[0,:]),con_mat_train[1,:]/sum(con_mat_train[1,:])]
    print(con_mat_train)
    sns.heatmap(con_mat_train)
    plt.show()


    fig, ax = plt.subplots(figsize=(6,6))
    con_mat_test=metrics.confusion_matrix(Y_test,Y_test_pred>0.5)
    print(con_mat_test)
    con_mat_test=[con_mat_test[0,:]/sum(con_mat_test[0,:]),con_mat_test[1,:]/sum(con_mat_test[1,:])]
    print(con_mat_test)
    sns.heatmap(con_mat_test)
    plt.show()

def PlotXYYYLineScatter(X, Y):
    display.display(plt.gcf())
    time.sleep(0.5)
    plt.cla()
    plt.clf()
    display.clear_output(wait=True)

    fig, ax = plt.subplots(figsize=(15, 5),ncols=3)
    fig.tight_layout(pad=5.0)

    ax[0].plot(X, Y[:, 0], 'o-')
    ax[1].plot(X, Y[:, 1], 'o-')
    ax[2].plot(X, Y[:, 2], 'o-')
    ax[0].set_xlabel('Injections per Droplet')
    ax[1].set_xlabel('Injections per Droplet')
    ax[2].set_xlabel('Injections per Droplet')
    ax[0].set_ylabel('First Absorption Peak Wavelength (nm)')
    ax[1].set_ylabel('Peak to Valley Ratio')
    ax[2].set_ylabel('Absorption Peak Intensity (a.u.)')