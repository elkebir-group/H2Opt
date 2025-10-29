import numpy as np
import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from adjustText import adjust_text
import time
import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.optim import Optimizer
import scipy
from scipy.stats import pearsonr
from scipy.fft import fft, ifft

from shared import *

import seaborn as sns



################################################
###                                          ###
###                Simulated                 ###
###                                          ###
################################################



def plotEncodeCorrelation():

    #plotType = 'proj'
    #plotType = 'true'

    #plotTypes = ['proj', 'true']
    plotTypes = ['true']


    #simulationNames = ['uncorSims']

    #simulationNames = ['uncorSims']
    #simulationNames = ['random100SNP', 'uncorSims']
    simulationNames = ['seperate100SNP']
    #simulationNames = ['sameHeritSep100']
    

    for simulationName in simulationNames:
        for plotType in plotTypes:
        
            folder0 = './data/plant/simulations/encoded/' + simulationName + '/'
            

            if plotType == 'true':
                corMatrix_PCA = loadnpz(folder0 + 'PCA_corMatrix.npz')
            elif plotType == 'proj':
                corMatrix_PCA = loadnpz(folder0 + 'PCA_corMatrixProj.npz')
            corMatrix_PCA = np.abs(corMatrix_PCA)
            #heritMatrix = loadnpz(folder0 + 'PCA_heritMatrix.npz')


            if plotType == 'true':
                corMatrix_maxWave = loadnpz(folder0 + 'maxWave_corMatrix.npz')
            elif plotType == 'proj':
                corMatrix_maxWave = loadnpz(folder0 + 'maxWave_corMatrixProj.npz')
            corMatrix_maxWave = np.abs(corMatrix_maxWave)


            if plotType == 'true':
                corMatrix = loadnpz(folder0 + 'A2_corMatrix.npz')
            elif plotType == 'proj':
                corMatrix = loadnpz(folder0 + 'A2_corMatrixProj.npz')
            corMatrix = np.abs(corMatrix)

            #print (corMatrix.shape)
            #quit()



            if False:
                corMatrix = corMatrix[:, np.arange(3), np.arange(3)]
                corMatrix_PCA = corMatrix_PCA[:, np.arange(3), np.arange(3)]
                corMatrix_maxWave = corMatrix_maxWave[:, np.arange(3), np.arange(3)]
                #print (np.median(corMatrix, axis=0))
                #print (np.median(corMatrix_PCA, axis=0))
                #print (np.median(corMatrix_maxWave, axis=0))
                #quit()

            



            valueName = 'correlation with true trait'
            traitName = 'trait number'
            methodName = 'method'
            plotData = {}
            plotData[valueName] = []
            plotData[methodName] = []
            plotData[traitName] = []

            Ntrait = 3
            #if plotType == 'max':
            #    Ntrait = 5

            #print (corMatrix_PCA.shape)
            #quit()

            Nsim = 5


            #hue="alive"
            for simNum in range(Nsim):
                #plt.imshow(corMatrix[simNum])
                #plt.show()

                #if simNum == 1:
                #    print (corMatrix[simNum] )
                #    print (corMatrix_PCA[simNum] )
                #    print (corMatrix_maxWave[simNum] )
                #    quit()



                for traitNum in range(Ntrait):
                    value1 = corMatrix[simNum, traitNum, traitNum] 
                    value2 = corMatrix_PCA[simNum, traitNum, traitNum]
                    value3 = corMatrix_maxWave[simNum, traitNum, traitNum]
                    
                    plotData[valueName].append(value1 )
                    plotData[methodName].append('H2Opt')
                    plotData[traitName].append(str(traitNum+1))
                    

                    plotData[valueName].append(value2  )
                    plotData[methodName].append('PCA')
                    plotData[traitName].append(str(traitNum+1))

                    plotData[valueName].append(value3  )
                    plotData[methodName].append('single trait')
                    plotData[traitName].append(str(traitNum+1))


            if False:
                sns.boxplot(data=plotData, x=traitName, y=valueName, hue=methodName, dodge=True)
                #sns.stripplot(data=plotData, x=traitName, y=valueName, hue=methodName, dodge=True, jitter=True, alpha=0.6)

                plt.legend([],[], frameon=False)

                plt.gcf().set_size_inches(4, 4)
                plt.tight_layout()
                #plt.gcf().set_size_inches(4, 4)

                plt.savefig('./images/encodeSim/' + simulationName + '_' + plotType + '.pdf')
                plt.show()


            

            if True:

                plotData_first = {}
                plotData_first[valueName] = []
                plotData_first[methodName] = []
                for a in range(len(plotData[valueName])):
                    if plotData[traitName][a] == '1':
                        plotData_first[valueName].append( plotData[valueName][a] )
                        plotData_first[methodName].append( plotData[methodName][a] )

                
                sns.boxplot(data=plotData_first, x=methodName, hue=methodName, y=valueName, legend=True)#,  width=0.6)
                plt.axhline(y=0.0, color='black', linestyle=':')
                plt.xticks([], [])
                plt.xlabel('')
                plt.gcf().set_size_inches(3, 4)
                plt.tight_layout()
                plt.savefig('./images/grant/cor.pdf')
                plt.show()



#plotEncodeCorrelation()
#quit()


def plotEncodeHerit():

    #simulationNames = ['random100SNP', 'uncorSims']
    simulationNames = ['seperate100SNP']


    

    for simulationName in simulationNames:
        
            folder0 = './data/plant/simulations/encoded/' + simulationName + '/'
            
            herit_PCA = loadnpz(folder0 + 'PCA_heritMatrix.npz')
            herit_maxWave = loadnpz(folder0 + 'maxWave_heritMatrix.npz')
            herit = loadnpz(folder0 + 'A2_heritMatrix.npz')
            herit_true = loadnpz(folder0 + 'true_heritMatrix.npz')

            herit_PCA[herit_PCA < 0] = 0
            herit_maxWave[herit_maxWave < 0] = 0
            herit[herit < 0] = 0
            herit_true[herit_true < 0] = 0



            highCount = np.copy(herit)
            highCount_PCA = np.copy(herit_PCA)
            highCount_maxWave = np.copy(herit_maxWave)
            #cutOff = 0.1
            #cutOff = 0.15
            cutOff = 0.2
            highCount[highCount<cutOff] = 0
            highCount_PCA[highCount_PCA<cutOff] = 0
            highCount_maxWave[highCount_maxWave<cutOff] = 0
            highCount[highCount>=cutOff] = 1
            highCount_PCA[highCount_PCA>=cutOff] = 1
            highCount_maxWave[highCount_maxWave>=cutOff] = 1
            highCount = np.sum(highCount[:, 1, :], axis=1)
            highCount_PCA = np.sum(highCount_PCA[:, 1, :], axis=1)
            highCount_maxWave = np.sum(highCount_maxWave[:, 1, :], axis=1)

            highCount = highCount-3
            highCount_PCA = highCount_PCA-3
            highCount_maxWave = highCount_maxWave-3
            highCount[highCount!=0] = 1
            highCount_PCA[highCount_PCA!=0] = 1
            highCount_maxWave[highCount_maxWave!=0] = 1
            highCount = np.sum(highCount)
            highCount_PCA = np.sum(highCount_PCA)
            highCount_maxWave = np.sum(highCount_maxWave)
            

            print (herit_true[:, 1, :3].shape)

            H2Opt_absError = np.mean(np.abs(herit_true[:, 1, :3] - herit[:, 1, :3]))
            PCA_absError = np.mean(np.abs(herit_PCA[:, 1, :3] - herit[:, 1, :3]))
            maxWave_absError = np.mean(np.abs(herit_maxWave[:, 1, :3] - herit[:, 1, :3]))

            H2Opt_absError2 = np.mean(herit[:, 1, 3:])
            PCA_absError2 = np.mean(herit_PCA[:, 1, 3:])
            maxWave_absError2 = np.mean(herit_maxWave[:, 1, 3:])


            #print (H2Opt_absError2, maxWave_absError2, PCA_absError2)
            #quit()

            printNum = 4
            #print (np.median(herit_true[:, 1, printNum]))
            #print (np.median(herit[:, 1, printNum]))
            #print (np.median(herit_PCA[:, 1, printNum]))
            #print (np.median(herit_maxWave[:, 1, printNum]))
            #quit()
            

            valueName = 'test set heritability'
            traitName = 'trait number'
            methodName = 'method'
            plotData = {}
            plotData[valueName] = []
            plotData[methodName] = []
            plotData[traitName] = []

            Ntrait = 5
            #if plotType == 'max':
            #    Ntrait = 5

            #print (corMatrix_PCA.shape)
            #quit()

            Nsim = 10


            #hue="alive"
            for simNum in range(Nsim):
                #plt.imshow(corMatrix[simNum])
                #plt.show()

                for traitNum in range(Ntrait):
                    value1 = herit[simNum, 1, traitNum] 
                    value2 = herit_PCA[simNum, 1, traitNum]
                    value3 = herit_maxWave[simNum, 1, traitNum]
                    value4 = herit_true[simNum, 1, traitNum]

                    if traitNum < 3:
                        plotData[valueName].append(value4 )
                        plotData[methodName].append('ground truth')
                        plotData[traitName].append(str(traitNum))
                    
                    plotData[valueName].append(value1 )
                    plotData[methodName].append('H2Opt')
                    plotData[traitName].append(str(traitNum))
                    

                    plotData[valueName].append(value2  )
                    plotData[methodName].append('PCA')
                    plotData[traitName].append(str(traitNum))

                    plotData[valueName].append(value3  )
                    plotData[methodName].append('single trait')
                    plotData[traitName].append(str(traitNum))

            palette = ['red', 'tab:blue', 'tab:orange', 'tab:green']

            if True:
                sns.boxplot(data=plotData, x=traitName, y=valueName, hue=methodName, dodge=True, palette=palette)
                #sns.stripplot(data=plotData, x=traitName, y=valueName, hue=methodName, dodge=True, jitter=True, alpha=0.6)
                plt.axhline(y=0.0, color='black', linestyle=':')
                plt.xticks( np.arange(5), np.arange(5)+1 )

                plt.legend([],[], frameon=False)
                plt.gcf().set_size_inches(4, 4)
                plt.tight_layout()

                

                #plt.savefig('./images/encodeSim/' + simulationName + '_herit.pdf')
                plt.show()


            if True:

                plotData_first = {}
                plotData_first[valueName] = []
                plotData_first[methodName] = []
                for a in range(len(plotData[valueName])):
                    if plotData[traitName][a] == '0':
                        plotData_first[valueName].append( plotData[valueName][a] )
                        plotData_first[methodName].append( plotData[methodName][a] )

                
                sns.boxplot(data=plotData_first, x=methodName, hue=methodName, y=valueName,  palette=palette, legend=True)#,  width=0.6)

                #plt.boxplot(methodName, methodName)#  palette=palette, legend=True)

                #sns.stripplot(data=plotData, x=traitName, y=valueName, hue=methodName, dodge=True, jitter=True, alpha=0.6)
                plt.axhline(y=0.0, color='black', linestyle=':')
                plt.ylim(-0.05, 1)
                plt.xticks([], [])
                plt.xlabel('')
                #plt.legend([],[], frameon=False)
                #plt.legend()
                plt.gcf().set_size_inches(3, 4)
                plt.tight_layout()
                #plt.savefig('./images/grant/herit.pdf')
                plt.show()

plotEncodeHerit()
quit()
        



def simSigSNPs():

    methodNames = ['groundTruth', 'H2Opt', 'PCA', 'maxWave']
    for phenIndex in range(0, 3):

        print ('')

        for methodIndex in range(len(methodNames)):
            methodName = methodNames[methodIndex]

            numSig = 0

            for simIndex in range(10):    
                data = loadnpz('./data/plant/simulations/encoded/seperate100SNP/' + str(simIndex) + '/Gemma_' + methodName + '_' + str(phenIndex+1) + '.npz')
                
                argGood = np.argwhere(np.isin( data[:, 0], np.arange(100).astype(str)  ))[:, 0]
                argGood = np.concatenate(( np.zeros(1, dtype=int), argGood ), axis=0)
                data = data[argGood]
                pvals = data[1:, -1]
                pvals = pvals.astype(float)
                pvals = pvals + (1e-300)
                pvals_log = -1 * np.log(pvals) / np.log(10)

                observed_p_values = np.sort(pvals_log)

                cutOff = np.log(data.shape[0]) / np.log(10)
                cutOff = cutOff - (np.log(0.05) / np.log(10))

                argHigh = np.argwhere(pvals_log > cutOff)[:, 0]
                if argHigh.shape[0] >= 1:
                    numSig += 1

            print (numSig)


#simSigSNPs()
#quit()



def simulationGWASplot():


    
    for phenIndex in range(0, 5):
        

        simIndex = '0'
        
        #data = loadnpz('./data/plant/simulations/encoded/random100SNP/0/Gemma_H2Opt_' + str(phenIndex+1) + '.npz')
        data = loadnpz('./data/plant/simulations/encoded/seperate100SNP/' + simIndex + '/Gemma_H2Opt_' + str(phenIndex+1) + '.npz')
        #data = loadnpz('./data/plant/simulations/encoded/uncorSims/1/Gemma_PCA_' + str(phenIndex+1) + '.npz')
        #data = loadnpz('./data/plant/simulations/encoded/random100SNP/0/Gemma_H2Opt_' + str(phenIndex+1) + '.npz')
        #data = loadnpz('./data/plant/GWAS/Gemma/' + methodName + '_' + str(phenIndex+1) + '.npz')

        


        argGood = np.argwhere(np.isin( data[:, 0], np.arange(100).astype(str)  ))[:, 0]
        argGood = np.concatenate(( np.zeros(1, dtype=int), argGood ), axis=0)
        data = data[argGood]

        chr1 = data[1:, 0].astype(int)

        pos1 = data[1:, 2].astype(int)
        pos2 = np.copy(pos1)

        chr1_unique = np.unique(chr1)
        lastMax = 0
        for chr_now in chr1_unique:
            args1 = np.argwhere(chr1 == chr_now)[:, 0]
            pos2[args1] = pos1[args1] + lastMax
            lastMax = np.max(pos2[args1]) + 1
        

        pvals = data[1:, -1]
        pvals = pvals.astype(float)
        pvals = pvals + (1e-300)
        pvals_log = -1 * np.log(pvals) / np.log(10)


        # Calculate the expected p-values
        expected_p_values = np.linspace(0, 1, len(pvals_log)+1)[1:][-1::-1]
        

        expected_p_values = -1 * np.log10(expected_p_values)

        # Sort observed p-values
        observed_p_values = np.sort(pvals_log)


        cutOff = np.log(data.shape[0]) / np.log(10)
        cutOff = cutOff - (np.log(0.05) / np.log(10))

        argHigh = np.argwhere(pvals_log > cutOff)[:, 0]

        print ('argHigh', argHigh.shape)
        



        if True:
            max1 = min( np.max(observed_p_values), np.max(expected_p_values) )
            # Create the Q-Q plot
            print (observed_p_values.shape, expected_p_values.shape)
            print (max1)
            plt.figure(figsize=(8, 8))
            plt.plot(expected_p_values, observed_p_values, marker='o', linestyle='none')
            plt.plot([0, max1], [0, max1], color='red', linestyle='--')  # Reference line
            plt.xlabel('expected -log10(p)')
            plt.ylabel('observed -log10(p)')
            #plt.title('Q-Q Plot for GWAS p-values')
            #plt.title('synthetic trait ' + str(phenIndex + 1))
            #plt.gcf().set_size_inches(5, 4.5)
            #plt.gcf().set_size_inches(4, 2.2)
            plt.gcf().set_size_inches(3.8, 2)
            plt.tight_layout()
            plt.savefig('./images/encodeSim/GWAS/QQ_sim' + str(simIndex) + '_' + str(phenIndex) + '.png')
            #plt.xlim(0, 1)
            #plt.ylim(0, 1)
            plt.grid()
            plt.show()

        

        colors = ['red', 'blue']

        bestGenes = []

        for a in range(chr1_unique.shape[0]):
            args1 = np.argwhere(chr1 == a+1)[:, 0]

            argHigh = args1[pvals_log[args1] > cutOff ]  # + np.log10(10) ]

            chr_now = chr1[argHigh] - 1
            pos_now = pos1[argHigh]

            #genes_now = pastNames[chr_now, pos_now]
            #bestGenes = bestGenes + list(genes_now)





            if True:
                plt.scatter(pos2[args1], pvals_log[args1], c=colors[a%2])

        #print (bestGenes)

        #line1 = np.array([0, pvals_log.shape[0]])
        line1 = np.array([0, np.max(pos2)])

        if True:
            #plt.scatter(np.arange(pvals_log.shape[0]), pvals_log)
            if False:
                plt.plot( line1, np.zeros(2) + cutOff , color='black' , linestyle=':')
            #plt.plot( line1, np.zeros(2) + cutOff + np.log10(10)  )
            #plt.title('phenotype ' + str(phenIndex+1))
            plt.xlabel("genomic bin")
            plt.ylabel('-log10(p)')
            plt.xticks([])
            #plt.gcf().set_size_inches(5, 4.5)
            #plt.gcf().set_size_inches(4, 2)
            plt.gcf().set_size_inches(3.8, 1.9)
            plt.tight_layout()
            plt.savefig('./images/encodeSim/GWAS/Manhat_sim' + str(simIndex) + '_' + str(phenIndex) + '.png')
            plt.show()

        #quit()

        #1 bad, 2 good, 5 good, 10 good, 13 good, 15 good



#simulationGWASplot()
#quit()





def GWASdetectPlot():

    from sklearn.metrics import roc_auc_score, roc_curve
    # Calculate AUC

    simulationNames = ['seperate100SNP']

    traitNum = 2

    for simulationName in simulationNames:
        folder0 = './data/plant/simulations/encoded/' + simulationName + '/'

        perm1 = np.array([3, 0, 1, 2], dtype=int)

        #probValues = loadnpz(folder0 + 'GWAS_probValues.npz')[perm1]
        #trueValues = loadnpz(folder0 + 'GWAS_trueValues.npz')[perm1]
        probValues = loadnpz(folder0 + 'GWAS_probValues_new.npz')[:, perm1]
        trueValues = loadnpz(folder0 + 'GWAS_trueValues_new.npz')[:, perm1]

        probValues, trueValues = probValues[traitNum], trueValues[traitNum]

        

        probValues = probValues.reshape((probValues.shape[0], -1))
        trueValues = trueValues.reshape((trueValues.shape[0], -1))

        print (np.max(trueValues, axis=1))

        

        #print (probValues.shape)
        #quit()

        methodNames = ['groundTruth', 'H2Opt', 'PCA', 'maxWave']
        colorList = ['red', 'tab:blue', 'tab:orange', 'tab:green']
        
        for methodIndex in range(len(methodNames)):
            if probValues[methodIndex].shape[0] > 0:
                probValues[methodIndex] = probValues[methodIndex] / np.max(probValues[methodIndex])

        legendStr = ['ground truth', 'H2Opt', 'PCA', 'single trait']

        for methodIndex in range(len(methodNames)):
            print (trueValues[methodIndex].shape)
            auc = roc_auc_score(trueValues[methodIndex], probValues[methodIndex])
            print("AUC:", auc)

            AUC_string = ' AUC = ' + str(auc)[:5]
            legendStr[methodIndex] = legendStr[methodIndex] + AUC_string

            fpr, tpr, thresholds = roc_curve(trueValues[methodIndex], probValues[methodIndex])

            plt.plot(fpr, tpr, c=colorList[methodIndex], linewidth=3, alpha=0.6)

        plt.plot([0, 1], [0, 1], c='black', linestyle=':')
        
        #plt.legend(['ground truth', 'H2Opt', 'PCA', 'single-wave'])
        plt.legend(legendStr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.gcf().set_size_inches(4, 4)
        plt.tight_layout()
        #plt.savefig('./images/encodeSim/SNP_compare.pdf')
        plt.show()

    

#GWASdetectPlot()
#quit()



#def grantGWAS():




def individualGWASdetectPlot():

    from sklearn.metrics import roc_auc_score, roc_curve
    # Calculate AUC

    simulationNames = ['seperate100SNP']

    plotCurves = False

    traitNum = 3

    methodNames = ['groundTruth', 'H2Opt', 'PCA', 'maxWave']
    colorList = ['red', 'tab:blue', 'tab:orange', 'tab:green']

    aucValues = np.zeros(( traitNum, 10, len(methodNames)  ))

    valueName = 'AUC'
    traitName = 'trait number'
    methodNamer = 'method'
    plotData = {}
    plotData[valueName] = []
    plotData[methodNamer] = []
    plotData[traitName] = []

    
    for simulationName in simulationNames:
        for traitIndex in range(traitNum):
            folder0 = './data/plant/simulations/encoded/' + simulationName + '/'

            perm1 = np.array([3, 0, 1, 2], dtype=int)
            probValues_all = loadnpz(folder0 + 'GWAS_probValues_new.npz')[:, perm1]
            trueValues_all = loadnpz(folder0 + 'GWAS_trueValues_new.npz')[:, perm1]

            probValues_all, trueValues_all = probValues_all[traitIndex], trueValues_all[traitIndex]

            for simIndex in range(10):
                #print (probValues_all.shape)
                #quit()
                probValues = probValues_all[:, simIndex]#.reshape((probValues.shape[0], -1))
                trueValues = trueValues_all[:, simIndex]#.reshape((trueValues.shape[0], -1))

            

                #print (probValues.shape)
                #quit()

                
                
                for methodIndex in range(len(methodNames)):
                    if probValues[methodIndex].shape[0] > 0:
                        probValues[methodIndex] = probValues[methodIndex] / np.max(probValues[methodIndex])

                legendStr = ['ground truth', 'H2Opt', 'PCA', 'single trait']
                curveList = []



                for methodIndex in range(len(methodNames)):
                    #print (trueValues[methodIndex].shape)
                    auc = roc_auc_score(trueValues[methodIndex], probValues[methodIndex])
                    #print("AUC:", auc)
                    aucValues[traitIndex, simIndex, methodIndex] = auc

                    plotData[valueName].append(auc)
                    plotData[methodNamer].append(  methodNames[methodIndex])
                    plotData[traitName].append(traitIndex + 1)

                    AUC_string = ' AUC = ' + str(auc)[:5]
                    legendStr[methodIndex] = legendStr[methodIndex] + AUC_string

                    fpr, tpr, thresholds = roc_curve(trueValues[methodIndex], probValues[methodIndex])

                    curveList.append( (np.copy(fpr), np.copy(tpr) ) )

                
                if plotCurves:
                    for methodIndex in range(len(methodNames)):
                        fpr, tpr = curveList[methodIndex]
                        plt.plot(fpr, tpr, c=colorList[methodIndex], linewidth=3, alpha=0.6)

                    plt.plot([0, 1], [0, 1], c='black', linestyle=':')
                    
                    #plt.legend(['ground truth', 'H2Opt', 'PCA', 'single-wave'])
                    plt.legend(legendStr)
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.gcf().set_size_inches(4, 4)
                    plt.tight_layout()
                    #plt.savefig('./images/encodeSim/SNP_compare.pdf')
                    plt.show()

    
    print (aucValues.shape)
    aucValues_med = np.median(aucValues, axis=1)
    print (aucValues_med)


    palette = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green']

    sns.boxplot(data=plotData, x=traitName, y=valueName, hue=methodNamer, dodge=True, palette=palette)
    plt.xlabel('trait number')
    plt.legend([],[], frameon=False)
    plt.gcf().set_size_inches(3, 4)
    plt.tight_layout()
    plt.savefig('./images/encodeSim/gwas/boxAUC.pdf')
    plt.show()


    quit()



#individualGWASdetectPlot()
#quit()




def rangeGWASdetectPlot():

    from sklearn.metrics import roc_auc_score, roc_curve
    # Calculate AUC

    simulationNames = ['seperate100SNP']

    traitNum = 3
    for traitIndex in range(3):

        for simulationName in simulationNames:

            folder0 = './data/plant/simulations/encoded/' + simulationName + '/'
            perm1 = np.array([3, 0, 1, 2], dtype=int)

            probValues = loadnpz(folder0 + 'GWAS_probValues_new.npz')[:, perm1]
            trueValues = loadnpz(folder0 + 'GWAS_trueValues_new.npz')[:, perm1]

            probValues, trueValues = probValues[traitIndex], trueValues[traitIndex]

            print (probValues.shape)
            

            

            #probValues = loadnpz(folder0 + 'GWAS_probValues.npz')[perm1]
            #trueValues = loadnpz(folder0 + 'GWAS_trueValues.npz')[perm1]

            #print (probValues.shape)
            #quit()

            methodNames = ['groundTruth', 'H2Opt', 'PCA', 'maxWave']
            colorList = ['red', 'tab:blue', 'tab:orange', 'tab:green']

            #methodNames = ['H2Opt', 'PCA', 'maxWave']
            #colorList = ['tab:blue', 'tab:orange', 'tab:green']
            #probValues = probValues[1:]
            #trueValues = trueValues[1:]

            #methodNames = ['H2Opt']
            #colorList = ['tab:blue']

            #probValues = probValues.reshape((probValues.shape[0], 10, probValues.shape[1] // 10 ))
            #trueValues = trueValues.reshape((trueValues.shape[0], 10, trueValues.shape[1] // 10 ))

            

            AUC_data = np.zeros((len(methodNames), probValues.shape[1]))


            #print (trueValues.shape)
            #quit()


            xMetric = "talse positive rate"
            yMetric = "true positive rate"
            namer = 'method'
            df = {}
            df[xMetric] = []
            df[yMetric] = []
            df[namer] = []
            
            for methodIndex in range(len(methodNames)):
                for simIndex in range(probValues.shape[1]):
                    if probValues[methodIndex, simIndex].shape[0] > 0:
                        probValues[methodIndex, simIndex] = probValues[methodIndex, simIndex] / np.max(probValues[methodIndex, simIndex])

        

                

                

            legendStr = ['ground truth', 'H2Opt', 'PCA', 'single trait']

            for methodIndex in range(len(methodNames)):

                roc_data = []
                    
                for simIndex in range(probValues.shape[1]):
                    print (trueValues[methodIndex, simIndex].shape)
                    auc = roc_auc_score(trueValues[methodIndex, simIndex], probValues[methodIndex, simIndex])
                    print("AUC:", auc)

                    AUC_data[methodIndex, simIndex] = auc

                    
                    

                    fpr, tpr, thresholds = roc_curve(trueValues[methodIndex, simIndex], probValues[methodIndex, simIndex])

                    # Simulated example data: list of (fpr, tpr) for each instance
                    roc_data.append((fpr, tpr))

                    
                # Define common FPR points for interpolation
                fpr_common = np.linspace(0, 1, 100)

                # Interpolate TPRs at the common FPR points
                tpr_interp = []
                for fpr, tpr in roc_data:
                    tpr_interp.append(np.interp(fpr_common, fpr, tpr))

                tpr_interp = np.array(tpr_interp)

                # Calculate mean and confidence intervals
                tpr_mean = np.mean(tpr_interp, axis=0)
                tpr_std = np.std(tpr_interp, axis=0)
                tpr_min = np.min(tpr_interp, axis=0)
                tpr_max = np.max(tpr_interp, axis=0)

                # Plot with Seaborn and Matplotlib
                #plt.figure(figsize=(8, 6))
                color1 = colorList[methodIndex]
                #sns.lineplot(x=fpr_common, y=tpr_mean, label="Mean ROC", color=color1, legend=None)

                plt.plot(fpr_common, tpr_mean, color=color1)

                if False:# != 0:
                    plt.fill_between(
                        fpr_common,
                        tpr_min, #tpr_mean - tpr_std,
                        tpr_max, #tpr_mean + tpr_std,
                        color=color1,
                        alpha=0.4,
                        label="Error Band" #"Error Band (±1 std)"
                    )

            #print (np.median(AUC_data, axis=1))

            median_AUC = np.median(AUC_data, axis=1)

            for methodIndex in range(len(methodNames)):
                auc = median_AUC[methodIndex]
                AUC_string = 'AUC = ' + str(auc)[:5]
                stringSplit = ' '
                stringSplit = ':\n'
                legendStr[methodIndex] = legendStr[methodIndex] + stringSplit + AUC_string

                    
                    #legendStr[methodIndex] = legendStr[methodIndex] + AUC_string


            print (median_AUC)
            #quit()

            AUC_string = ' AUC = ' + str(auc)[:5]

            
            plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
            plt.xlabel("false positive rate (1 - specificity)")
            plt.ylabel("true positive rate (sensitivity)")
            #plt.title("Summarized ROC Curve with Error Bands")
            plt.legend(legendStr)
            if traitIndex == 0:
                plt.gcf().set_size_inches(3, 4)
            else:
                plt.gcf().set_size_inches(4, 3)
            plt.tight_layout()
            plt.savefig('./images/encodeSim/SNP_compare_' + str(traitIndex) + '.pdf')
            plt.show()






            #for a in range(fpr.shape[0]):
            #    df[xMetric].append(fpr[a])
            #    df[yMetric].append(tpr[a])
            #    df[namer].append(methodNames[methodIndex])


            #sns.lineplot(x=xMetric, y=yMetric,
            #     hue=namer,
            #     data=df)
            #plt.show()
            
            #quit()

            #plt.plot([0, 1], [0, 1], c='black', linestyle=':')
            
            #plt.legend(['ground truth', 'H2Opt', 'PCA', 'single-wave'])
            #plt.legend(legendStr)
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.gcf().set_size_inches(4, 4)
            #plt.tight_layout()
            #plt.savefig('./images/encodeSim/SNP_compare.pdf')
            #plt.show()



#rangeGWASdetectPlot()
#quit()




def plotWavelengthIntensityExample():


    X = loadnpz('./data/plant/processed/sor/X.npz')


    if True:
        a = 0
        wavelengths = np.arange(X.shape[1]) + +350
        plt.plot(wavelengths, X[a])
        plt.xlabel('wavelength (nm)')
        plt.ylabel("intensity")
        #plt.gcf().set_size_inches(4, 4)
        #plt.gcf().set_size_inches(3.5, 1.3)
        #plt.gcf().set_size_inches(3, 1.3)
        plt.gcf().set_size_inches(2.5, 1.3)
        plt.tight_layout()
        plt.savefig('./images/examples/sor/example_' + str(a) + '_short.pdf')
        plt.show()
        quit()

    for a in range(3):

        wavelengths = np.arange(X.shape[1]) + +350
        plt.plot(wavelengths, X[a])
        plt.xlabel('wavelength (nm)')
        plt.ylabel("intensity")
        #plt.gcf().set_size_inches(5, 3)
        plt.gcf().set_size_inches(4, 4)
        plt.tight_layout()
        plt.savefig('./images/examples/sor/example_' + str(a) + '.pdf')
        plt.show()
    quit()


#plotWavelengthIntensityExample()
#quit()


def plotWavelengthAutoEncode():

    X = loadnpz('./data/plant/processed/sor/X.npz')

    model = torch.load('./data/plant/models/autoencode_2.pt')

    X_encode = model( torch.tensor(X[:1]).float() )
    X_encode = X_encode.data.numpy()

    wavelengths = np.arange(X.shape[1]) + +350
    plt.plot(wavelengths, X[0])
    plt.xlabel('wavelength (nm)')
    plt.ylabel("intensity")
    plt.gcf().set_size_inches(2, 4)
    plt.tight_layout()
    plt.savefig('./images/examples/sor/tall_original.pdf')
    plt.show()


    plt.plot(wavelengths, X_encode[0])
    plt.xlabel('wavelength (nm)')
    plt.ylabel("intensity")
    plt.gcf().set_size_inches(2, 4)
    plt.tight_layout()
    plt.savefig('./images/examples/sor/tall_encode.pdf')
    plt.show()
    quit()

#plotWavelengthAutoEncode()
#quit()




################################################
###                                          ###
###                Sorghum                   ###
###                                          ###
################################################




def plotCoeffMulti():

    #modelName = './data/plant/models/linear_6.pt'
    #modelName = './data/plant/models/linear_8.pt'
    #modelName = './data/plant/models/linear_crossVal_' + str(0) + '_mod2.pt'

    #modelName = './data/plant/simulations/encoded/random100SNP/0/model_A2.pt'

    modelName = './data/plant/models/linear_crossVal_reg4_' + str(0) + '_mod10.pt'


    #modelName = './data/plant/models/lowReg_4.pt'
    model = torch.load(modelName)

    coef = getMultiModelCoef(model, multi=True)

    X = loadnpz('./data/plant/processed/sor/X.npz')
    Y = model(torch.tensor(X).float(), np.arange(20))
    Y, trackComputation = normalizeIndependent(Y, trackCompute=True)
    coef = np.matmul( coef.T, trackComputation.data.numpy() ).T



    if False:
        coefIndex = 0
        wavelengths = np.arange(coef[0].shape[0])  + 350
        plt.plot(coef[coefIndex], wavelengths)
        max1 = np.max(np.abs(coef[coefIndex]))
        plt.ylabel('wavelength (nm)')
        plt.xlabel("coefficient")
        plt.xlim( -max1, max1 )
        plt.gcf().set_size_inches(1.7, 3.5)
        plt.tight_layout()
        plt.savefig('./images/sor/coef/coefPlot_' + str(coefIndex) + '_sideways.pdf')
        plt.show()
        quit()

   

    print (coef.shape)

    doSmall = True

    
    wavelengths = np.arange(coef[0].shape[0])  + 350
    for coefIndex in range(10):
        plt.plot(wavelengths, coef[coefIndex])
        max1 = np.max(np.abs(coef[coefIndex]))
        plt.xlabel('wavelength (nm)')
        plt.ylabel("coefficient")
        plt.ylim( -max1, max1 )

        #plt.gca().yaxis.tick_right()  # Move the ticks to the right
        #plt.gca().yaxis.set_label_position("right")  # Move the label to the right

        #plt.axis().yaxis.set_ticks_position('right') 
        #plt.legend(['trait 1', 'trait 2', 'trait 3'])
        #plt.gcf().set_size_inches(7, 3)
        #plt.gcf().set_size_inches(4, 1.3)
        if doSmall:
            #plt.gcf().set_size_inches(3, 1.3)
            plt.gcf().set_size_inches(2.5, 1.3)
        else:
            plt.gcf().set_size_inches(3.5, 1.3)
        plt.tight_layout()

        if doSmall:
            plt.savefig('./images/sor/coef/coefPlot_' + str(coefIndex) + '_small.pdf')
        else:
            plt.savefig('./images/sor/coef/coefPlot_' + str(coefIndex) + '.pdf')
        plt.show()
    quit()



#plotCoeffMulti()
#quit()






def sorGWASplot():



    geneLoc = getGeneLocations('sor')

    #data = loadnpz('./data/plant/simulations/encoded/random100SNP/0/Gemma_H2Opt_' + str(phenIndex+1) + '.npz')
    #phenIndex = 0
    for phenIndex in range(0, 10):
    
        #methodName = 'linear_trainAll1'  #OLD  
        #methodName = 'sor_linear_7'#OLD
        #methodName = 'linear_trainAll2'#OLD 
        #methodName = 'sor_simple_1' #OLD 
        #data = loadnpz('./data/plant/GWAS/Gemma/' + methodName + '_' + str(phenIndex + 1) + '.npz') #OLD

        methodName = 'linear_crossVal_reg4_' + str(0) + '_mod10'
        data = loadnpz('./data/plant/GWAS/Gemma/' + methodName + '_' + str(phenIndex+1) + '.npz')

        #print (data[1])


        argGood = np.argwhere(np.isin( data[:, 0], np.arange(100).astype(str)  ))[:, 0]
        argGood = np.concatenate(( np.zeros(1, dtype=int), argGood ), axis=0)
        data = data[argGood]

        chr1 = data[1:, 0].astype(int)

        pos1 = data[1:, 2].astype(int)
        pos2 = np.copy(pos1)

        chr1_unique = np.unique(chr1)
        lastMax = 0
        for chr_now in chr1_unique:
            args1 = np.argwhere(chr1 == chr_now)[:, 0]
            pos2[args1] = pos1[args1] + lastMax
            lastMax = np.max(pos2[args1]) + 1
        

        pvals = data[1:, -1]
        pvals = pvals.astype(float)
        pvals = pvals + (1e-300)
        pvals_log = -1 * np.log(pvals) / np.log(10)


        # Calculate the expected p-values
        expected_p_values = np.linspace(0, 1, len(pvals_log)+1)[1:][-1::-1]
        

        expected_p_values = -1 * np.log10(expected_p_values)

        # Sort observed p-values
        observed_p_values = np.sort(pvals_log)


        cutOff = np.log(data.shape[0]) / np.log(10)
        cutOff = cutOff - (np.log(0.05) / np.log(10))

        argHigh = np.argwhere(pvals_log > cutOff)[:, 0]

        print ('argHigh', argHigh.shape)


        if argHigh.shape[0] > 0:
            closeGenes = findClosestGene(geneLoc, data[argHigh])
            #print ('closeGenes', closeGenes.shape)

            print (closeGenes)
            quit()


        



        if False:
            max1 = min( np.max(observed_p_values), np.max(expected_p_values) )
            # Create the Q-Q plot
            print (observed_p_values.shape, expected_p_values.shape)
            print (max1)
            plt.figure(figsize=(8, 8))
            plt.plot(expected_p_values, observed_p_values, marker='o', linestyle='none')
            plt.plot([0, max1], [0, max1], color='red', linestyle='--')  # Reference line
            plt.xlabel('Expected p-values')
            plt.ylabel('Observed p-values')
            #plt.title('Q-Q Plot for GWAS p-values')
            #plt.title('synthetic trait ' + str(phenIndex + 1))
            #plt.gcf().set_size_inches(5, 4.5)
            plt.gcf().set_size_inches(4, 2.2)
            plt.tight_layout()
            #plt.xlim(0, 1)
            #plt.ylim(0, 1)
            plt.grid()
            plt.savefig('./images/sor/GWAS/QQ_' + str(phenIndex) + '.png')
            plt.show()

        

        colors = ['red', 'blue']

        pos2 = pos2 / np.max(pos2) #Helps plotting

        bestGenes = []

        for a in range(chr1_unique.shape[0]):
            args1 = np.argwhere(chr1 == a+1)[:, 0]

            argHigh_local = args1[pvals_log[args1] > cutOff ]  # + np.log10(10) ]

            chr_now = chr1[argHigh_local] - 1
            pos_now = pos1[argHigh_local]

            #genes_now = pastNames[chr_now, pos_now]
            #bestGenes = bestGenes + list(genes_now)

            if False:
                plt.scatter(pos2[args1], pvals_log[args1], c=colors[a%2])

        #print (bestGenes)

        #line1 = np.array([0, pvals_log.shape[0]])
        line1 = np.array([0, np.max(pos2)])

        if False:
            #plt.scatter(np.arange(pvals_log.shape[0]), pvals_log)
            plt.plot( line1, np.zeros(2) + cutOff , color='black' , linestyle=':')
            #plt.plot( line1, np.zeros(2) + cutOff + np.log10(10)  )
            #plt.title('phenotype ' + str(phenIndex+1))
            plt.xlabel("genomic bin")
            plt.ylabel('-log10(p)')
            plt.xticks([])


            texts = []
            Xpos, YPos = pos2[argHigh], pvals_log[argHigh]

            Xpos[Xpos>0.75] = 0.75

            #Xpos = Xpos + (np.random.random(Xpos.shape[0]) / 20)
            #YPos = YPos + (np.random.random(YPos.shape[0]) / 20)
            YPos = YPos + 0.2

            if phenIndex == 1:
                YPos[2] = YPos[2] + 0.4
                YPos[3] = YPos[3] - 0.3

            print (closeGenes[:, 3])
            
            for gene_index in range(argHigh.shape[0]):
                #SNP_index = argHigh[gene_index]
                yPosNow = YPos[gene_index]
                xPosNow = Xpos[gene_index]
                
                geneName = closeGenes[gene_index][3]
                #print (geneName)
                geneName = geneName.replace('SORBI_', '')
                #print (geneName)
                #quit()
                plt.annotate(geneName, xy=(xPosNow, yPosNow), fontsize=9)
                #texts.append(plt.text(xPosNow, yPosNow, geneName, fontsize=8))
                #print (bestGenes)
                #quit()

            
            # Adjust positions to avoid overlap
            #adjust_text(texts, Xpos, YPos,
            #            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
            plt.ylim(0, np.max(pvals_log)*1.15)
            

            
            #plt.gcf().set_size_inches(5, 4.5)



            #plt.gcf().set_size_inches(5, 3)
            plt.gcf().set_size_inches(4, 2)
            plt.tight_layout()
            plt.savefig('./images/sor/GWAS/Manhat_' + str(phenIndex) + '.png')
            plt.show()




#sorGWASplot()
#quit()





def sorGWASHitsplot():

    showAll = True 

    if showAll:
        numTrait = 10
    else:
        numTrait = 4

    numHigh = []
    for phenIndex in range(0, numTrait):
        methodName = 'linear_crossVal_reg4_' + str(0) + '_mod10'
        #methodName = 'maxWave' #0 high overall
        #methodName = 'PCA' #0 high overall
        data = loadnpz('./data/plant/GWAS/Gemma/' + methodName + '_' + str(phenIndex+1) + '.npz')

        argGood = np.argwhere(np.isin( data[:, 0], np.arange(100).astype(str)  ))[:, 0]
        argGood = np.concatenate(( np.zeros(1, dtype=int), argGood ), axis=0)
        data = data[argGood]

        pvals = data[1:, -1]
        pvals = pvals.astype(float)
        pvals = pvals + (1e-300)
        pvals_log = -1 * np.log(pvals) / np.log(10)
        cutOff = np.log(data.shape[0]) / np.log(10)
        cutOff = cutOff - (np.log(0.05) / np.log(10))

        argHigh = np.argwhere(pvals_log > cutOff)[:, 0]
        numHigh.append(argHigh.shape[0])

        print (argHigh)
    
    #plt.scatter(numHigh, np.arange( len(numHigh) ))
    #plt.show()

    maxHit = int(np.max(np.array(numHigh)))

    if showAll:
        plt.barh(np.arange( len(numHigh) )+1, numHigh)
        plt.yticks(np.arange(len( numHigh))+1, np.arange( len(numHigh) )+1)
        plt.xticks(np.arange(maxHit+1), np.arange(maxHit+1) )
        plt.ylabel('synthetic trait')
        plt.xlabel('# of significant SNPs')
        plt.gcf().set_size_inches(2, 4)
        plt.tight_layout()
        #plt.savefig('./images/sor/GWAS_hits_all.pdf')
        plt.show()

    else:
        plt.bar(np.arange( len(numHigh) )+1, numHigh)
        plt.xticks(np.arange(len( numHigh))+1, np.arange( len(numHigh) )+1)
        plt.yticks(np.arange(maxHit+1), np.arange(maxHit+1) )
        plt.xlabel('synthetic trait')
        plt.ylabel('number of significant SNPs')
        plt.gcf().set_size_inches(2, 4)
        plt.tight_layout()
        #plt.savefig('./images/sor/GWAS_hits.pdf')
        plt.show()



#sorGWASHitsplot()
#quit()


def sorCompareSplitGWAS():

    
    for phenIndex in range(2 ):
        #methodName = 'linear_crossVal_reg4_' + str(0) + '_mod10'


        pval_all = []
        for split_index in range(10):
            methodName = 'linear_crossVal_reg4_' + str(split_index) + '_mod10'
            data = loadnpz('./data/plant/GWAS/Gemma/' + methodName + '_' + str(phenIndex+1) + '.npz')
            pvals = data[1:, -1]
            pvals = pvals.astype(float)
            pval_all.append(np.copy(pvals))
            #data_all.append(np.copy(data))
        pval_all = np.array(pval_all)
        pvals_log = np.log(pval_all)

        corMatrix = np.zeros((10, 10))
        for split_index1 in range(10):
            corMatrix[split_index1, split_index1] = 1
            for split_index2 in range(10):
                if split_index2 > split_index1:
                    cor = scipy.stats.pearsonr(pvals_log[split_index1], pvals_log[split_index2])
                    corMatrix[split_index1, split_index2] = cor[0]
                    corMatrix[split_index2, split_index1] = cor[0]
                    #print (cor)

        print (np.median( corMatrix[corMatrix!=1] ))
        
        Nsplit = 10
        ax = sns.heatmap(np.abs(corMatrix), annot=True)#, cmap='bwr')
        ax.set_yticklabels(  np.arange(Nsplit)+1 )
        ax.set_xticklabels(   np.arange(Nsplit)+1 )
        plt.xlabel('cross-validation split')
        plt.ylabel('cross-validation split')
        plt.gcf().set_size_inches(5.5, 4)
        plt.tight_layout()
        plt.savefig('./images/sor/crossGWAS_' + str(phenIndex) + '.pdf')
        plt.show()
        #quit()

        


#sorCompareSplitGWAS()
#quit()



def sorComparetrait():

    traitData = np.loadtxt('./data/plant/processed/traits/SorgumTraits.tsv', delimiter='\t', dtype=str)

    traitData = traitData[2:, :-4]
    traitNames = traitData[0, 1:]

    for a in range(traitNames.shape[0]):
        traitNames[a] = traitNames[a].replace(' model', '')

        traitNames[a] = traitNames[a].replace('gs', '$g_s$')
        traitNames[a] = traitNames[a].replace('AN', '$A_N$')
        traitNames[a] = traitNames[a].replace('ci/ca', '$c_i/c_a$')

    print (traitNames)
    #quit()


    corMatrix = loadnpz('./data/plant/eval/traitCompare.npz')
    corMatrix = np.median(np.abs(corMatrix), axis=0).T
    print (corMatrix.shape)
    corMatrix = np.round(corMatrix * 100) / 100

    showAll = True

    if showAll:
        Ntrait = 10
    else:
        Ntrait = 4
    
    corMatrix = corMatrix[:, :Ntrait]

    ax = sns.heatmap(np.abs(corMatrix), annot=True)#, cmap='bwr')
    ax.set_yticklabels(  traitNames , rotation=0)
    ax.set_xticklabels(   np.arange(Ntrait)+1 )
    plt.xlabel('synthetic trait')
    #plt.gcf().set_size_inches(7, 3)
    if showAll:
        plt.gcf().set_size_inches(6, 4)
    else:
        plt.gcf().set_size_inches(4, 4)

    plt.tight_layout()
    if showAll:
        plt.savefig('./images/sor/traitCorrelation_all.pdf')
    else:
        plt.savefig('./images/sor/traitCorrelation.pdf')
    plt.show()
    quit()


#sorComparetrait()
#quit()



def sorCrossValidHeritPlot():


    
    methodNames = ['H2Opt', 'PCA', 'single trait']
    trainNames = ['training set', 'test set']

    #valueName = 'heritability'
    valueName = 'top trait heritability'
    traitName = 'synthetic trait number'
    methodNamer = 'method'
    plotData = {}
    plotData[valueName] = []
    plotData[methodNamer] = []
    plotData[traitName] = []

    Ntrait = 5
    #Ntrait = 3
    Nsim = 10

    doSmall = True

    #doTrain = True

    perm1 = np.array([0, 2, 1]).astype(int)

    trainInfos = loadnpz('./data/plant/eval/herit/crossValid_train_proper.npz')[perm1]
    testInfos = loadnpz('./data/plant/eval/herit/crossValid_test_proper.npz')[perm1]
    

    #print (np.mean(trainInfos, axis=1)[:, 1])
    #print (np.mean(testInfos, axis=1)[:, 3])
    #quit()

    for methodIndex in range(len(methodNames)):
        methodName = methodNames[methodIndex]

        herits = trainInfos[methodIndex][:, :3]
        
        #if doTrain:
        #    herits = trainInfos[methodIndex][:, :3]
        #else:
        #    herits = testInfos[methodIndex][:, :3]
        

        for splitIndex in range(10):
            phenIndex = 0
            #for phenIndex in range(3):
            for trainName in trainNames:

                if trainName == 'training set':
                    plotData[valueName].append(trainInfos[methodIndex, splitIndex, 0])
                else:
                    plotData[valueName].append(testInfos[methodIndex, splitIndex, 0])
        
                
                plotData[methodNamer].append(methodName)
                plotData[traitName].append(trainName)
    

    #if doTrain:
    #    palette = ['red', 'red', 'red']
    #else:
    palette = ['tab:blue', 'tab:orange', 'tab:green']

    sns.boxplot(data=plotData, x=traitName, y=valueName, hue=methodNamer, dodge=True, palette=palette)
    #sns.stripplot(data=plotData, x=traitName, y=valueName, hue=methodName, dodge=True, jitter=True, alpha=0.6)
    #plt.axhline(y=0.0, color='black', linestyle=':')

    plt.xlabel('')
    plt.legend([],[], frameon=False)
    if doSmall:
        plt.gcf().set_size_inches(3, 2)
    else:
        plt.gcf().set_size_inches(3, 4)
    plt.tight_layout()

    
    if doSmall:
        plt.savefig('./images/sor/topTraitHerit_small.pdf')
    else:
        plt.savefig('./images/sor/topTraitHerit.pdf')
    plt.show()
    


#sorCrossValidHeritPlot()
#quit()


def sorPlotHeritability():


    if False:

        plt.plot([0, 1], c='black')
        plt.plot([0, 1], c='black', linestyle='dashed')
        plt.legend(['test set', 'training set'])
        plt.savefig('./images/sor/herit_trainTestLegend.pdf')
        plt.show()

        quit()


    doSmall = False
    showAll = True
    onlyTest = False
    

    methodNames = ['H2Opt', 'maxWave', 'PCA']

    colorList = ['tab:blue', 'tab:orange', 'tab:green']

    perm1 = np.array([0, 2, 1]).astype(int)
    trainInfos = loadnpz('./data/plant/eval/herit/crossValid_train_proper.npz')[perm1]
    testInfos = loadnpz('./data/plant/eval/herit/crossValid_test_proper.npz')[perm1]

    trainInfos = np.mean(trainInfos, axis=1)
    testInfos = np.mean(testInfos, axis=1)

    #print (trainInfos[:, 3])
    #print (testInfos)
    #quit()

    if showAll:
        numTrait = 10
    else:
        numTrait = 4
    

    for methodIndex in range(len(methodNames)):
        heritTrain = trainInfos[methodIndex][:numTrait]
        heritTest = testInfos[methodIndex][:numTrait]
        color1 = colorList[methodIndex]
        arange1 = np.arange(heritTrain.shape[0]) + 1
        if not onlyTest:
            plt.plot(arange1, heritTrain, c=color1, linestyle=':')
        plt.plot(arange1, heritTest, c=color1)
    
    for methodIndex in range(len(methodNames)):
        heritTrain = trainInfos[methodIndex][:numTrait]
        heritTest = testInfos[methodIndex][:numTrait]
        color1 = colorList[methodIndex]

        if not onlyTest:
            plt.scatter(arange1, heritTrain, c=color1)
        plt.scatter(arange1, heritTest, c=color1)
    plt.xticks(arange1)
    plt.ylim(bottom=0)
    
    plt.xlabel('sythetic trait number')
    #if onlyTest:
    #    plt.ylabel("test set heritability")
    #else:
    plt.ylabel("heritability")
    #plt.legend(['training set', 'test set'])

    

    if doSmall:
        plt.ylim(top=0.9)
        plt.gcf().set_size_inches(3, 2.25)
        if showAll:
            plt.gcf().set_size_inches(3.3, 2.2)
        
    elif showAll:
        plt.legend(['H2OPT: train', 'H2Opt:test', 'PCA: train', 'PCA:test', 'single trait: train', 'single trait: test'])
        plt.gcf().set_size_inches(7, 4)
    else:
        plt.gcf().set_size_inches(3, 4)

    plt.tight_layout()

    if doSmall:
        if showAll:
            plt.savefig('./images/sor/multiTraitHerit_smallAll.pdf')
        else:
            plt.savefig('./images/sor/multiTraitHerit_small.pdf')


    elif showAll:
        plt.savefig('./images/sor/multiTraitHerit_all.pdf')
        True
    else:
        plt.savefig('./images/sor/multiTraitHerit.pdf')
        True
    plt.show()



#sorPlotHeritability()
#quit()




def sorCompare_ANOVA_lmer():

    X = loadnpz('./data/plant/processed/sor/X.npz')
    names = loadnpz('./data/plant/processed/sor/names.npz')
    envirement = loadnpz('./data/plant/processed/sor/set1.npz').reshape((-1, 1)) 
    #traitAll = loadnpz('./data/plant/processed/sor/traits.npz')
    X = torch.tensor(X).float()
    
    heritList =  cheapHeritability(X, names, envirement )
    heritList = heritList.data.numpy()

    sorHerit = np.loadtxt("./data/software/lmeHerit/sor.csv", delimiter=',', dtype=str)
    sorHerit = sorHerit[1:, 1].astype(float)

    print (scipy.stats.pearsonr( sorHerit, heritList ))
    
    #plt.scatter( sorHerit, heritList )
    #plt.show()

    m, b = np.polyfit(sorHerit, heritList, 1)
    y_fit = m * sorHerit + b

    rValue = scipy.stats.pearsonr( sorHerit, heritList )[0]
    rValue = np.round(rValue * 1000) / 1000
    rText = '$r = ' + str(rValue) + '$'

    plt.plot(sorHerit, y_fit, color='red')
    print (np.max(heritList))
    plt.annotate(rText, xy=(0.02, 0.9 * np.max(heritList) ), fontsize=12)
    plt.scatter( sorHerit, heritList)
    plt.xlabel('lme4 heritability')
    plt.ylabel("ANOVA heritability")
    plt.gcf().set_size_inches(3.5, 4)
    plt.tight_layout()
    plt.savefig('./images/ANOVA/sor.pdf')
    plt.show()
    


#sorCompare_ANOVA_lmer()
#quit()


#def checkBaselineGWAS():





################################################
###                                          ###
###                Miscanthus                ###
###                                          ###
################################################



def miscGWASplot():

    #data = loadnpz('./data/plant/simulations/encoded/random100SNP/0/Gemma_H2Opt_' + str(phenIndex+1) + '.npz')
    #phenIndex = 0

    geneLoc = getGeneLocations('misc')

    doBig = False


    for dateIndex in range(0, 14):
        print (dateIndex)

        #phenIndex = 0
        
        #methodName = 'linear_trainAll2'

        namePart = 'central'
        #namePart = 'south'
        #methodName = 'MSI_' + namePart + '_singlepoint_' + str(dateIndex) + '_split_' + str(0) + '_1'

        #methodName = 'MSI_' + namePart + '_singlepoint_' + str(dateIndex) + '_split_' + str(0) + '_1'
        
        #methodName = 'PCA'
        #methodName = 'maxTrait'
        methodName = 'maxWave'
        methodName = methodName + '_' + namePart + '_' + str(dateIndex) + '_1'
        
        data = loadnpz('./data/plant/GWAS/Gemma/' + methodName + '.npz')



        argGood = np.argwhere(np.isin( data[:, 0], np.arange(100).astype(str)  ))[:, 0]
        argGood = np.concatenate(( np.zeros(1, dtype=int), argGood ), axis=0)
        data = data[argGood]

        chr1 = data[1:, 0].astype(int)

        pos1 = data[1:, 2].astype(int)
        pos2 = np.copy(pos1)

        chr1_unique = np.unique(chr1)
        lastMax = 0
        for chr_now in chr1_unique:
            args1 = np.argwhere(chr1 == chr_now)[:, 0]
            pos2[args1] = pos1[args1] + lastMax
            lastMax = np.max(pos2[args1]) + 1
        

        pvals = data[1:, -1]
        pvals = pvals.astype(float)
        pvals = pvals + (1e-300)
        pvals_log = -1 * np.log(pvals) / np.log(10)


        # Calculate the expected p-values
        expected_p_values = np.linspace(0, 1, len(pvals_log)+1)[1:][-1::-1]
        

        expected_p_values = -1 * np.log10(expected_p_values)

        # Sort observed p-values
        observed_p_values = np.sort(pvals_log)


        cutOff = np.log(data.shape[0]) / np.log(10)
        cutOff = cutOff - (np.log(0.05) / np.log(10))

        argHigh = np.argwhere(pvals_log > cutOff)[:, 0]

        print ('argHigh', argHigh.shape)

        argHigh_top = argHigh[np.argsort(pvals_log[argHigh] * -1)[:4]]
        
        if argHigh.shape[0] > 0:
            #closeGenes = findClosestGene(geneLoc, data[argHigh_top])
            closeGenes = findClosestGene(geneLoc, data[argHigh])
            #print (closeGenes)
            closeString = ''
            for a in range(closeGenes.shape[0]):
                closeString += ', ' + closeGenes[a, -1]
            #print (closeString)

        pos2 = pos2 / np.max(pos2)

        


        if False:
            max1 = min( np.max(observed_p_values), np.max(expected_p_values) )
            # Create the Q-Q plot
            print (observed_p_values.shape, expected_p_values.shape)
            print (max1)
            plt.figure(figsize=(8, 8))
            plt.plot(expected_p_values, observed_p_values, marker='o', linestyle='none')
            plt.plot([0, max1], [0, max1], color='red', linestyle='--')  # Reference line
            plt.xlabel('Expected p-values')
            plt.ylabel('Observed p-values')
            #plt.title('Q-Q Plot for GWAS p-values')
            #plt.title('synthetic trait ' + str(phenIndex + 1))
            #plt.gcf().set_size_inches(5, 4.5)
            #plt.gcf().set_size_inches(4, 2.2)
            plt.gcf().set_size_inches(3.5, 1.8)
            #plt.gcf().set_size_inches(4, 2.0)
            plt.tight_layout()
            #plt.xlim(0, 1)
            #plt.ylim(0, 1)
            plt.grid()
            plt.savefig('./images/misc/GWAS/QQ_' + str(dateIndex) + '.png')
            plt.show()

        

        colors = ['red', 'blue']

        bestGenes = []

        for a in range(chr1_unique.shape[0]):
            args1 = np.argwhere(chr1 == a+1)[:, 0]

            argHigh_mini = args1[pvals_log[args1] > cutOff ]  # + np.log10(10) ]

            chr_now = chr1[argHigh_mini] - 1
            pos_now = pos1[argHigh_mini]

            #genes_now = pastNames[chr_now, pos_now]
            #bestGenes = bestGenes + list(genes_now)

            if False:
                plt.scatter(pos2[args1], pvals_log[args1], c=colors[a%2])

        #print (bestGenes)

        if False:
            Xpos, YPos = pos2[argHigh_top], pvals_log[argHigh_top]
            #Xpos[Xpos>0.75] = 0.75
            Xpos = (Xpos * 0.78)
            #Xpos[Xpos>0.8] = 0.8
            #YPos = YPos + 0.2
            YPos = YPos + (0.03 * np.max(pvals_log))

            if dateIndex == 0:
                Xpos[2] = 0.35
                Xpos[3] = 0.01
            
            if dateIndex == 4:
                YPos[0] = 8.25
                #print (Xpos)
                #print (YPos)
            
            #if phenIndex == 1:
            #    YPos[2] = YPos[2] + 0.4
            #    YPos[3] = YPos[3] - 0.3
            for gene_index in range(argHigh_top.shape[0]):
                #print (gene_index)
                yPosNow = YPos[gene_index]
                xPosNow = Xpos[gene_index]
                geneName = closeGenes[gene_index][3]
                geneName = geneName.replace('Misin', '')
                plt.annotate(geneName, xy=(xPosNow, yPosNow), fontsize=9)


        #line1 = np.array([0, pvals_log.shape[0]])
        line1 = np.array([0, np.max(pos2)])

        if False:
            #plt.scatter(np.arange(pvals_log.shape[0]), pvals_log)
            plt.plot( line1, np.zeros(2) + cutOff , color='black' , linestyle=':')
            #plt.plot( line1, np.zeros(2) + cutOff + np.log10(10)  )
            #plt.title('phenotype ' + str(phenIndex+1))
            #plt.xlabel("genomic bin")
            plt.xlabel("chromosome")
            plt.ylabel('-log10(p)')
            plt.xticks([])
            plt.ylim(0, np.max(pvals_log) * 1.15 )
            plt.xlim(0  - (0.01*np.max(pos2) ) , np.max(pos2) * 1.01 )
            #plt.gcf().set_size_inches(5, 4.5)
            #plt.gcf().set_size_inches(4, 2)
            if doBig:
                plt.gcf().set_size_inches(6, 4)
            else:
                #plt.gcf().set_size_inches(3.5, 1.7)
                plt.gcf().set_size_inches(4, 2)
            #plt.gcf().set_size_inches(4, 1.8)

            plt.tight_layout()
            #plt.savefig('./images/misc/GWAS/Manhat_' + str(dateIndex) + '.png')
            plt.show()




#miscGWASplot()
#quit()




def miscGWASHitsplot():

    
    for namePart in ['central', 'south']:
        numHigh = []
        for dateIndex in range(0, 5):
            methodName = 'MSI_' + namePart + '_singlepoint_' + str(dateIndex) + '_split_' + str(0) + '_1'
            data = loadnpz('./data/plant/GWAS/Gemma/' + methodName + '.npz')

            argGood = np.argwhere(np.isin( data[:, 0], np.arange(100).astype(str)  ))[:, 0]
            argGood = np.concatenate(( np.zeros(1, dtype=int), argGood ), axis=0)
            data = data[argGood]

            pvals = data[1:, -1]
            pvals = pvals.astype(float)
            pvals = pvals + (1e-300)
            pvals_log = -1 * np.log(pvals) / np.log(10)
            cutOff = np.log(data.shape[0]) / np.log(10)
            cutOff = cutOff - (np.log(0.05) / np.log(10))

            argHigh = np.argwhere(pvals_log > cutOff)[:, 0]
            numHigh.append(argHigh.shape[0])
    

        dates = ['MSI_09042020_processed_ALLstack.tif', 'MSI_05062020_processed_ALLstack.tif', 'MSI_05222020_processed_ALLstack.tif', 'MSI_09192020_processed_ALLstack.tif', 'MSI_07022020_processed_ALLstack.tif', 'MSI_11282020_processed_ALLstack.tif', 'MSI_07242020_processed_ALLstack.tif', 'MSI_10052020_processed_ALLstack.tif', 'MSI_08182020_processed_ALLstack.tif', 'MSI_06182020_processed_ALLstack.tif', 'MSI_07102020_processed_ALLstack.tif', 'MSI_06082020_processed_ALLstack.tif', 'MSI_08082020_processed_ALLstack.tif', 'MSI_11062020_processed_ALLstack.tif']
        dates = np.array(dates)
        for a in range(dates.shape[0]):
            dates[a] = dates[a].split('_')[1]
            dates[a] = dates[a][:2] + '/' + dates[a][2:4]
        dates = np.sort(dates)

        maxHit = int(np.max(np.array(numHigh)))

        plt.bar(np.arange( len(numHigh) )+1, numHigh)
        plt.xticks(np.arange(5)+1,  dates[:5], rotation=45 )
        
        if maxHit <= 4:
            plt.yticks(np.arange(maxHit+1), np.arange(maxHit+1) )
        plt.xlabel('date')
        plt.ylabel('# of significant SNPS')
        plt.gcf().set_size_inches(2.5, 2.5)
        plt.tight_layout()
        plt.savefig('./images/misc/GWAS_hits_' + namePart + '.pdf')
        plt.show()



#miscGWASHitsplot()
#quit()



def miscCompare_ANOVA_lmer():

    miscNames = ['central', 'south']
    for dataIndex in range(2):
        miscName = miscNames[dataIndex]


        phenotypes = np.loadtxt("./data/software/lmeHerit/input_misc_" + miscName + ".csv", delimiter=',', dtype=str)
        phenotypes_names = phenotypes[0]
        phenotypes = phenotypes[1:]
        names, row, col = phenotypes[:, 0], phenotypes[:, 1], phenotypes[:, 2]
        phenotypes = phenotypes[:, 3:].astype(float)
        envirement = np.array([row, col]).T

        print (np.unique(row))
        print (np.unique(col))

        lmerHerit =  np.loadtxt("./data/software/lmeHerit/output_misc_" +  miscName + ".csv", delimiter=',', dtype=str)
        lmerHerit = lmerHerit[1:, 1].astype(float)
        
        heritList = cheapHeritability(torch.tensor(phenotypes).float(), names, envirement)
        heritList = heritList.data.numpy()

        channelInfo = np.arange(heritList.shape[0]) // (  heritList.shape[0] // 6 )


        lmerHerit = lmerHerit * 4
        heritList = heritList * 4

        #heritList = heritList.reshape((6, heritList.shape[0]//6))
        #lmerHerit = lmerHerit.reshape((6, lmerHerit.shape[0]//6))
            
        #print (lmerHerit.shape)
        #print (heritList.shape)
        #print (scipy.stats.pearsonr( lmerHerit, heritList ))

        #for channel in range(6):
        #    print (scipy.stats.pearsonr( lmerHerit[channelInfo==channel], heritList[channelInfo==channel] ))
    
        #print (scipy.stats.pearsonr( lmerHerit[channelInfo==0], heritList[channelInfo==0] ))
        print ('')
        #print (scipy.stats.pearsonr( lmerHerit[channelInfo!=0], heritList[channelInfo!=0] ))
        print (scipy.stats.pearsonr( lmerHerit, heritList ))
        #print (scipy.stats.spearmanr( lmerHerit, heritList ))


        m, b = np.polyfit(lmerHerit, heritList, 1)
        y_fit = m * lmerHerit + b

        rValue = scipy.stats.pearsonr( lmerHerit, heritList )[0]
        rValue = np.round(rValue * 1000) / 1000
        rText = '$r = ' + str(rValue) + '$'

        plt.plot(lmerHerit, y_fit, color='red')
        print (np.max(heritList))
        plt.annotate(rText, xy=(0.02, 0.9 * np.max(heritList) ), fontsize=12)
        plt.scatter( lmerHerit, heritList)
        plt.xlabel('lme4 heritability')
        plt.ylabel("ANOVA heritability")
        plt.gcf().set_size_inches(3.5, 4)
        plt.tight_layout()
        plt.savefig('./images/ANOVA/misc_' + miscName + '.pdf')
        plt.show()
        #quit()

        #plt.scatter( lmerHerit[channelInfo!=0], heritList[channelInfo!=0] )
        #plt.scatter( lmerHerit[channelInfo==0], heritList[channelInfo==0] )
        #plt.show()



#miscCompare_ANOVA_lmer()
#quit()


def plotMiscExample():

    from matplotlib.colors import LogNorm

    miscName = 'south'
    
    if miscName == 'south':
        imagesAll = loadnpz('./data/miscPlant/inputs/MSI_combine/south_all.npz')
    else:
        imagesAll = loadnpz('./data/miscPlant/inputs/MSI_combine/all.npz')


    for a in range(3):
        

        sns.heatmap(imagesAll[2, a, 1] , cbar=False, linewidths=0) 
        
        #plt.ylabel('m/z (Da)')
        #plt.xticks( np.arange(4)*2*5, np.arange(4)*5 )
        #plt.yticks( np.arange(15) * 100, np.arange(15) * 100  )
        #plt.xlabel('retention time (minutes)')
        #plt.gcf().set_size_inches(4, 4)
        plt.gcf().set_size_inches(4, 4)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.tight_layout()
        if miscName == 'south':
            plt.savefig('./images/examples/misc/exampleHeatmap_' + str(a) + '_south.pdf')
        else:
            plt.savefig('./images/examples/misc/exampleHeatmap_' + str(a) + '.pdf')
        plt.show()


#plotMiscExample()
#quit()

def miscCompareTrait():


    miscName = 'south'

    if miscName == 'south':
        traitList = ['Survival (Oct 04 - Oct 07 2019)', 'Survival (spring 2020)', 'Hardiness score (%) ( July 2020)', 'Basal Circumference (BCirc; cm)   2020', 
        'Autumn Dormancy Date 2020', 'Spring Re-growth Date 2020', 'First Heading Date 2020 (HD)', '50% Heading Date 2020 (HD 50%)']
        #traitList = [ 'Survival (spring 2020)', 'Hardiness score (%) ( July 2020)', 'Basal Circumference (BCirc; cm)   2020', 
        #'Autumn Dormancy Date 2020', 'Spring Re-growth Date 2020', 'First Heading Date 2020 (HD)', '50% Heading Date 2020 (HD 50%)']
    else:
        traitList = ['biomass', 'Average Internode Length', 'Culm Density', 'Culm Dry Weight', 'Culm node num', 'Culm Outer Diameter at the Basal Internode', 'Calm Outer Diameter at last internode',
                 'Height', 'Basal circumference', 'Dormancy date', 'First Heading Date', '50% Heading Date', 'Hardiness score', 'Spring Regrowth Date', 'Survival October', 'Survival Spring']
    traitList = np.array(traitList)

    dates = ['MSI_09042020_processed_ALLstack.tif', 'MSI_05062020_processed_ALLstack.tif', 'MSI_05222020_processed_ALLstack.tif', 'MSI_09192020_processed_ALLstack.tif', 'MSI_07022020_processed_ALLstack.tif', 'MSI_11282020_processed_ALLstack.tif', 'MSI_07242020_processed_ALLstack.tif', 'MSI_10052020_processed_ALLstack.tif', 'MSI_08182020_processed_ALLstack.tif', 'MSI_06182020_processed_ALLstack.tif', 'MSI_07102020_processed_ALLstack.tif', 'MSI_06082020_processed_ALLstack.tif', 'MSI_08082020_processed_ALLstack.tif', 'MSI_11062020_processed_ALLstack.tif']
    dates = np.array(dates)
    for a in range(dates.shape[0]):
        dates[a] = dates[a].split('_')[1]
        dates[a] = dates[a][:2] + '/' + dates[a][2:4]
    dates = np.sort(dates)


    simplePearsonCor = False
    allTraits = True

    if miscName == 'south':
        corList2 = loadnpz('./data/miscPlant/eval/corTrait_south.npz')
    else:
        corList2 = loadnpz('./data/miscPlant/eval/corTrait.npz')

    if simplePearsonCor:
        corList2 = corList2[0]
    else:
        #corList2 = corList2[1]
        corList2 = corList2[2]
    #print (corList2.shape)
    #quit()
    corList2 = np.abs(corList2)
    corList2 = np.mean(corList2, axis=2)

    
    if not (simplePearsonCor or allTraits):
        traitList[traitList ==  'Culm Dry Weight'] =  'Culm\n Dry Weight'
        traitList[traitList ==  'Culm node num'] =  'Culm\n Node\n Number'
        traitList[traitList == 'Dormancy date'] = 'Dormancy\n Date'
        traitList[traitList == 'Culm Outer Diameter at the Basal Internode'] = 'Basal\n Diameter'
        traitList[traitList == 'Basal circumference'] = 'Basal\n Circumference'
        traitList[traitList == 'biomass'] = 'Biomass'
        traitList[traitList == 'Survival October'] = 'Survival\n October'
        traitList[traitList == 'Calm Outer Diameter at last internode'] = 'Calm Outer Diameter\n at last internode'



    
    if False:
        ax = sns.heatmap(corList2, annot=True)
        ax.set_yticklabels(  traitList , rotation=0)
        ax.set_xticklabels(  dates )
        
        plt.xlabel('date')
        plt.show()


    

    #print (corList2.shape)
    #print (dates.shape)
    #quit()

    traitCounts = loadnpz('./data/miscPlant/eval/corTrait_traitCount.npz')

    

    maxAbs = np.max(np.abs(corList2), axis=1)

    #cutOff = 0.15
    #cutOff = 0.05
    #argGood = np.argwhere(maxAbs > cutOff)[:, 0]
    if simplePearsonCor or allTraits:
        argGood = np.argsort(maxAbs)
    else:
        corList2 = corList2[:, :5]
        dates = dates[:5]
        argGood = np.argsort(maxAbs)[-6:]

        

    corList2 = corList2[argGood]
    #print (traitList)
    traitList = traitList[argGood]
    traitCounts = traitCounts[argGood]


    

    #print (argGood)
    #print (traitList)
    #quit()


    #print (maxAbs.shape)
    #print (traitList.shape)
    #quit()
    

    #plt.imshow(corList2)
    #plt.colorbar()

    print (corList2.shape)
    print (traitList.shape)

    print (traitCounts)

    corList2 = np.round(corList2 * 100) / 100

    #ax = sns.heatmap(np.abs(corList2), annot=True)#, cmap='bwr')
    ax = sns.heatmap(corList2, annot=True)
    ax.set_yticklabels(  traitList , rotation=0)
    #ax.set_xticklabels(  dates, rotation=45)  # np.arange(len(dates))+1 )
    ax.set_xticklabels(  dates )
    

    #plt.xticks(np.arange(corList2.shape[1]), dates , rotation=45)
    #plt.yticks(np.arange(corList2.shape[0]), traitList , rotation=90)
    #plt.gcf().set_size_inches(8, 4)
    plt.xlabel('date')
    if simplePearsonCor or allTraits:
        plt.gcf().set_size_inches(13, 4)
    else:
        plt.gcf().set_size_inches(5, 4)
    plt.tight_layout()
    if False:
        if simplePearsonCor:
            plt.savefig('./images/misc/traitCorrelation_simplePearson.pdf')
        elif allTraits:
            plt.savefig('./images/misc/traitCorrelation_allTraits.pdf')
        else:
            plt.savefig('./images/misc/traitCorrelation.pdf')
        
    plt.show()


#miscCompareTrait()
#quit()



def miscHeritPlot():

    for dataIndex in range(2):

        #dataIndex = 1

        includeVeg = False

        heritArray_trait = loadnpz('./data/miscPlant/eval/herits_maxTrait.npz')[:, dataIndex][:, :, :, 0]
        heritArray_wave = loadnpz('./data/miscPlant/eval/herits_maxWave.npz')[:, dataIndex][:, :, :, 0]
        heritArray_PCA = loadnpz('./data/miscPlant/eval/herits_PCA.npz')[:, dataIndex][:, :, :, 0]
        heritArray = loadnpz('./data/miscPlant/eval/herits_H2Opt.npz')[:, dataIndex][:, :, :, 0]

        heritVeg = loadnpz('./data/miscPlant/eval/herits_vegIndex.npz')[:, dataIndex][:, :, :, 1]


        #heritArray_trait_all = loadnpz('./data/miscPlant/eval/herits_maxTrait.npz')[:, dataIndex]
        #print (heritArray_trait_all.shape)
        #quit()



        heritArray[heritArray>0.25] = 0.25
        heritArray_wave[heritArray_wave>0.25] = 0.25
        heritArray_PCA[heritArray_PCA>0.25] = 0.25
        heritArray_trait[heritArray_trait>0.25] = 0.25
        heritVeg[heritVeg>0.25] = 0.25

        heritArray[heritArray<0.0] = 0.0
        heritArray_wave[heritArray_wave<0.0] = 0.0
        heritArray_PCA[heritArray_PCA<0.0] = 0.0
        heritArray_trait[heritArray_trait<0.0] = 0.0
        heritVeg[heritVeg<0.0] = 0.0



        if True:
            heritArray_trait = np.mean(heritArray_trait, axis=1)
            heritArray_wave = np.mean(heritArray_wave, axis=1)
            heritArray_PCA = np.mean(heritArray_PCA, axis=1)
            heritArray = np.mean(heritArray, axis=1)
            heritVeg = np.mean(heritVeg, axis=1)

        else:
            heritArray_trait = np.median(heritArray_trait, axis=1)
            heritArray_wave = np.median(heritArray_wave, axis=1)
            heritArray_PCA = np.median(heritArray_PCA, axis=1)
            heritArray = np.median(heritArray, axis=1)
        #print (heritArray)
        #quit()

        heritArray_trait = heritArray_trait * 4
        heritArray_wave = heritArray_wave * 4
        heritArray_PCA = heritArray_PCA * 4
        heritArray = heritArray * 4
        heritVeg = heritVeg * 4

        
        #print (heritArray.shape)
        #print (heritArray[1, 2:5])
        #print (heritArray[1, :2])

        #print (np.max(heritArray_wave[1]))
        #print (np.max(heritArray_trait[1]))
        #print (np.max(heritArray_PCA[1]))
        

        dates = ['MSI_09042020_processed_ALLstack.tif', 'MSI_05062020_processed_ALLstack.tif', 'MSI_05222020_processed_ALLstack.tif', 'MSI_09192020_processed_ALLstack.tif', 'MSI_07022020_processed_ALLstack.tif', 'MSI_11282020_processed_ALLstack.tif', 'MSI_07242020_processed_ALLstack.tif', 'MSI_10052020_processed_ALLstack.tif', 'MSI_08182020_processed_ALLstack.tif', 'MSI_06182020_processed_ALLstack.tif', 'MSI_07102020_processed_ALLstack.tif', 'MSI_06082020_processed_ALLstack.tif', 'MSI_08082020_processed_ALLstack.tif', 'MSI_11062020_processed_ALLstack.tif']
        dates = np.array(dates)
        for a in range(dates.shape[0]):
            dates[a] = dates[a].split('_')[1]
            dates[a] = dates[a][:2] + '/' + dates[a][2:4]
        dates = np.sort(dates)


        


        #lineStyle = ':'
        #if dataIndex == 1:
        #    lineStyle = '--'

        #

        lineStyle = '-'
        
        arange1 = np.arange(14)+1

        doLegend = False

        doTrain = False
        trainTestIndex = 1
        if doTrain:
            trainTestIndex = 0


        plt.plot(arange1, heritArray[trainTestIndex], c='tab:blue', linestyle=lineStyle, alpha=0.5)
        plt.plot(arange1, heritArray_PCA[trainTestIndex], c='tab:orange', linestyle=lineStyle, alpha=0.5)
        plt.plot(arange1, heritArray_trait[trainTestIndex], c='tab:green', linestyle=lineStyle, alpha=0.5)
        plt.plot(arange1, heritArray_wave[trainTestIndex], c='lime', linestyle=lineStyle, alpha=0.5)

        if includeVeg:
            plt.plot(arange1, heritVeg[trainTestIndex], c='tab:red')

        plt.scatter(arange1, heritArray[trainTestIndex], c='tab:blue')
        plt.scatter(arange1, heritArray_PCA[trainTestIndex], c='tab:orange')
        plt.scatter(arange1, heritArray_trait[trainTestIndex], c='tab:green')
        plt.scatter(arange1, heritArray_wave[trainTestIndex], c='lime')
        

        if includeVeg:
            plt.scatter(arange1, heritVeg[trainTestIndex], c='tab:red')

        
        #plt.xticks(arange1)
        plt.xticks(arange1, dates , rotation=45)
        plt.ylabel('test set heritability')

        #plt.gcf().set_size_inches(4, 2)
        plt.gcf().set_size_inches(4, 1.8)
        #plt.gcf().set_size_inches(3.8, 1.8)
        if doLegend:
            plt.gcf().set_size_inches(4, 5)
        if doTrain:
            plt.gcf().set_size_inches(5, 3)
            plt.ylabel('training set heritability')
        if includeVeg:
            plt.gcf().set_size_inches(5, 3)


        plt.xlabel('date')
        
        plt.tight_layout()
        

        if includeVeg:
            
            plt.legend(['H2Opt', 'PCA', 'single trait', 'single wavelength', 'NDVI'])
            if dataIndex == 0:
                plt.savefig('./images/misc/heritability_central_veg.pdf')
            else:
                plt.savefig('./images/misc/heritability_south_veg.pdf')
            plt.show()
            

        elif doTrain:
            plt.legend(['H2Opt', 'PCA', 'single trait', 'single wavelength'])
            if dataIndex == 0:
                plt.savefig('./images/misc/heritability_central_train.pdf')
            else:
                plt.savefig('./images/misc/heritability_south_train.pdf')
            plt.show()


        else:

            if doLegend:
                plt.legend(['H2Opt', 'PCA', 'single\ntrait', 'single\nband'])
                plt.savefig('./images/misc/heritability_legend.pdf')
            
            else:
                #plt.ylabel('')
                if dataIndex == 0:
                    plt.yticks([0, 0.25, 0.5], [0, 0.25, 0.5])
                    plt.savefig('./images/misc/heritability_central.pdf')
                    True
                else:
                    plt.yticks([0, 0.25, 0.5, 0.75], [0, 0.25, 0.5, 0.75])
                    plt.savefig('./images/misc/heritability_south.pdf')
                    True
            plt.show()


#miscHeritPlot()
#quit()


def miscSecondTraitHerit():

    for dataIndex in range(2):
        heritArray = loadnpz('./data/miscPlant/eval/herits_H2Opt.npz')[:, dataIndex][:, :, :, 1]
        
        heritArray[heritArray>0.25] = 0.25
        heritArray[heritArray<0.0] = 0.0



        if True:
            heritArray = np.mean(heritArray, axis=1)
        heritArray = heritArray * 4
        

        dates = ['MSI_09042020_processed_ALLstack.tif', 'MSI_05062020_processed_ALLstack.tif', 'MSI_05222020_processed_ALLstack.tif', 'MSI_09192020_processed_ALLstack.tif', 'MSI_07022020_processed_ALLstack.tif', 'MSI_11282020_processed_ALLstack.tif', 'MSI_07242020_processed_ALLstack.tif', 'MSI_10052020_processed_ALLstack.tif', 'MSI_08182020_processed_ALLstack.tif', 'MSI_06182020_processed_ALLstack.tif', 'MSI_07102020_processed_ALLstack.tif', 'MSI_06082020_processed_ALLstack.tif', 'MSI_08082020_processed_ALLstack.tif', 'MSI_11062020_processed_ALLstack.tif']
        dates = np.array(dates)
        for a in range(dates.shape[0]):
            dates[a] = dates[a].split('_')[1]
            dates[a] = dates[a][:2] + '/' + dates[a][2:4]
        dates = np.sort(dates)


        lineStyle = '-'
        
        arange1 = np.arange(14)+1

        doLegend = False

        doTrain = False
        trainTestIndex = 1
        if doTrain:
            trainTestIndex = 0


        plt.plot(arange1, heritArray[trainTestIndex], c='tab:blue', linestyle=lineStyle, alpha=0.5)

        plt.scatter(arange1, heritArray[trainTestIndex], c='tab:blue')
        
        #plt.xticks(arange1)
        plt.xticks(arange1, dates , rotation=45)
        plt.gcf().set_size_inches(5, 3)
        plt.xlabel('date')
        plt.ylabel('test set heritability')
        plt.tight_layout()
        if dataIndex == 0:
            plt.savefig('./images/misc/heritability_central_secondTrait.pdf')
        else:
            plt.savefig('./images/misc/heritability_south_secondTrait.pdf')
        
        
        
        #plt.legend(['H2Opt', 'PCA', 'single trait', 'single wavelength'])
        #if dataIndex == 0:
        #    plt.savefig('./images/misc/heritability_central_train.pdf')
        #else:
        #    plt.savefig('./images/misc/heritability_south_train.pdf')
        plt.show()



#miscSecondTraitHerit()
#quit()



def miscCanopy():

    

    for dataIndex in range(2):

        if dataIndex == 0:
            imagesAll = loadnpz('./data/miscPlant/inputs/MSI_combine/south_all.npz')
            namePart = 'central'
        else:
            imagesAll = loadnpz('./data/miscPlant/inputs/MSI_combine/all.npz')
            namePart = 'south'

        dates = ['MSI_09042020_processed_ALLstack.tif', 'MSI_05062020_processed_ALLstack.tif', 'MSI_05222020_processed_ALLstack.tif', 'MSI_09192020_processed_ALLstack.tif', 'MSI_07022020_processed_ALLstack.tif', 'MSI_11282020_processed_ALLstack.tif', 'MSI_07242020_processed_ALLstack.tif', 'MSI_10052020_processed_ALLstack.tif', 'MSI_08182020_processed_ALLstack.tif', 'MSI_06182020_processed_ALLstack.tif', 'MSI_07102020_processed_ALLstack.tif', 'MSI_06082020_processed_ALLstack.tif', 'MSI_08082020_processed_ALLstack.tif', 'MSI_11062020_processed_ALLstack.tif']
        dates = np.array(dates)
        for a in range(dates.shape[0]):
            dates[a] = dates[a].split('_')[1]
            dates[a] = dates[a][:2] + '/' + dates[a][2:4]
        dates = np.sort(dates)

        waveSum = np.mean(imagesAll, axis=(3, 4))
        red_value = waveSum[:, :, 3]
        redEdge_value = waveSum[:, :, 4]
        NIR_value = waveSum[:, :, 5]

        #NDRE = (NIR_value - redEdge_value) /  (NIR_value + redEdge_value)
        NDVI = (NIR_value - red_value) /  (NIR_value + red_value)
        #vegIndices = torch.stack([NDVI, NDRE]).T

        NDVI_mean = np.mean(NDVI, axis=0)

        arange1 = np.arange(len(dates))

        if dataIndex == 0:
            NDVI_central = np.copy(NDVI_mean)
        else:
            NDVI_south = np.copy(NDVI_mean)

        print (namePart)
    plt.plot(NDVI_central)
    plt.plot(NDVI_south)
    plt.scatter(arange1, NDVI_central)
    plt.scatter(arange1, NDVI_south)
    
    plt.xticks(arange1, dates, rotation=45)
    plt.gcf().set_size_inches(5, 2.5)
    plt.xlabel('date')
    plt.ylabel('NDVI')
    plt.legend(['central', 'south'])
    plt.tight_layout()
    plt.savefig('./images/misc/NDVI_canopy.pdf')
    plt.show()


#miscCanopy()
#quit()



def miscTimeCor():




    dates = ['MSI_09042020_processed_ALLstack.tif', 'MSI_05062020_processed_ALLstack.tif', 'MSI_05222020_processed_ALLstack.tif', 'MSI_09192020_processed_ALLstack.tif', 'MSI_07022020_processed_ALLstack.tif', 'MSI_11282020_processed_ALLstack.tif', 'MSI_07242020_processed_ALLstack.tif', 'MSI_10052020_processed_ALLstack.tif', 'MSI_08182020_processed_ALLstack.tif', 'MSI_06182020_processed_ALLstack.tif', 'MSI_07102020_processed_ALLstack.tif', 'MSI_06082020_processed_ALLstack.tif', 'MSI_08082020_processed_ALLstack.tif', 'MSI_11062020_processed_ALLstack.tif']
    dates = np.array(dates)
    for a in range(dates.shape[0]):
        dates[a] = dates[a].split('_')[1]
        dates[a] = dates[a][:2] + '/' + dates[a][2:4]
    dates = np.sort(dates)

    showAll = True
    simplePearsonCor = True

    if not showAll:
        dates = dates[:5]

    for dataIndex in range(2):

        if dataIndex == 0:
            namePart = 'central'
        if dataIndex == 1:
            namePart = 'south'

        if simplePearsonCor:
            #corMatrix = loadnpz('./data/miscPlant/eval/timeGeneticCor_synth.npz')[dataIndex]
            corMatrix = loadnpz('./data/miscPlant/eval/timeCor_synth.npz')[dataIndex]
        else:
            corMatrix = loadnpz('./data/miscPlant/eval/timeHalfSibCor_synth.npz')[dataIndex]

        

        print (corMatrix.shape)

        corMatrix = np.mean(np.abs(corMatrix), axis=2)
        if not showAll:
            corMatrix = corMatrix[:5, :5]

        corMatrix = np.round(corMatrix * 100) / 100



        ax = sns.heatmap( corMatrix, annot=True, vmin=0, vmax=1)#, fmt='',annot_kws={'size':12})#, cmap='bwr')
        ax.set_yticklabels(  dates , rotation=0)
        ax.set_xticklabels(  dates, rotation=90)
        #ax.set_xticklabels(  dates, rotation=45)  # np.arange(len(dates))+1 )

        #plt.xticks(np.arange(corList2.shape[1]), dates , rotation=45)
        #plt.yticks(np.arange(corList2.shape[0]), traitList , rotation=90)
        plt.xlabel('date')
        plt.ylabel('date')
        #plt.gcf().set_size_inches(5, 4)
        #plt.gcf().set_size_inches(4, 4)
        if not showAll:
            plt.gcf().set_size_inches(3.9, 1.8)
        else:
            plt.gcf().set_size_inches(7, 5)
        plt.tight_layout()

        if showAll:
            if simplePearsonCor:
                plt.savefig('./images/misc/timeCorrelation_halfsib_' + namePart + '_simplePearson.pdf')
            else:
                plt.savefig('./images/misc/timeCorrelation_halfsib_' + namePart + '_allTrait.pdf')
        else:   
            #plt.savefig('./images/misc/timeCorrelation_' + namePart + '.pdf')
            #plt.savefig('./images/misc/timeCorrelation_halfsib_' + namePart + '.pdf')
            True
        plt.show()



    #plt.imshow(corMatrix)
    #plt.show()

#miscTimeCor()
#quit()






################################################
###                                          ###
###                Arabidopsis               ###
###                                          ###
################################################




def plotMetabPathway():

    from matplotlib.colors import LogNorm


    showAll = True

    splitIndex = 0
    #phen_index = 1

    #envChoice = '0d'
    phen_index_counter = -1
    for envChoice in ['0d', '6d']:
        for phen_index in range(10):
            phen_index_counter += 1
        
            
            #df = pd.read_csv('./data/metab/metabEnrich/split' + str(splitIndex) + '_phen' + str(phen_index) + '_' + envChoice + '.csv')
            df = pd.read_csv('./data/metab/metabEnrich/withSep_split' + str(splitIndex) + '_phen' + str(phen_index) + '_' + envChoice + '.csv')
            data = df.to_numpy()


            if phen_index_counter == 0:
                data_cat = np.copy(data)
                phenList = np.zeros(data.shape[0], dtype=int)
            else:
                data_cat = np.concatenate((data_cat, data), axis=0)
                phenList = np.concatenate((phenList, np.zeros(data.shape[0], dtype=int)  + phen_index_counter ))


    metabName = data_cat[:, 0]
    pval = data_cat[:, 5].astype(float)


    maxWordLength = 0 
    for a in range(metabName.shape[0]):
        nameSplit = metabName[a].split(' ')
        for b in range(len(nameSplit)):
            maxWordLength = max( maxWordLength, len(nameSplit[b]))

    if not showAll:

        for a in range(metabName.shape[0]):
            nameSplit = metabName[a].split(' ')
            shortest = []
            for b in range(len(nameSplit)-1):
                shortest.append( len(nameSplit[b]) + len(nameSplit[b+1]) )
            shortest = np.array(shortest)
            if np.min(shortest) < maxWordLength:
                minPart = np.argmin(shortest)
                nameSplit = nameSplit[:minPart] + [str(nameSplit[minPart]) + ' ' + str(nameSplit[minPart+1])] + nameSplit[minPart+2:]
            
            nameNew = '\n'.join(nameSplit)
            metabName[a] = nameNew



        #metabName[a] = metabName[a].replace(' ', '\n')


    

    if showAll:
        metabName_sig = metabName[pval < 0.05]
    else:
        metabName_sig = metabName[pval < 0.03]
    metabName_sig_unique = np.unique(metabName_sig)
    argInclude = np.argwhere( np.isin(metabName, metabName_sig_unique) )[:, 0]

    data_cat, phenList, pval, metabName = data_cat[argInclude], phenList[argInclude], pval[argInclude], metabName[argInclude]

    #log_pval = -1 * np.log10(pval)
    metabName_unique, metabName_inverse = np.unique(metabName, return_inverse=True)

    Nphen = int(np.max(phenList))+1
    heatmap = np.ones(( metabName_unique.shape[0],  Nphen  ))
    heatmap[metabName_inverse, phenList] = pval

    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
    'red_white', 
    [(0, 'blue'), (1, 'white')]
    )


    cutoff = np.log(0.05)
    maxValue = np.log(np.min(heatmap))
    ratio1 = cutoff / maxValue

    print (cutoff, maxValue)

    #cmap = mcolors.LinearSegmentedColormap.from_list(
    #'red_white', 
    #[(0, 'red'), (ratio1, 'yellow'), (1, 'white')]
    #)

    #print (metabName_unique)
    #quit()

    vmin = np.min(heatmap)

    for envChoice in ['0d', '6d']:
        if envChoice == '0d':
            heatmap_now = heatmap[:, :10]
        if envChoice == '6d':
            heatmap_now = heatmap[:, 10:]

        #sns.heatmap(heatmap_now, norm=LogNorm(), cmap=cmap, vmin=vmin, vmax=1)
        ax = sns.heatmap(heatmap_now, norm=LogNorm(vmin=vmin, vmax=1))
        plt.yticks(np.arange(heatmap.shape[0]) + 0.5,  metabName_unique, rotation=0 )
        plt.xticks(np.arange(heatmap_now.shape[1]) + 0.5,  np.arange(heatmap_now.shape[1]) + 1)

        for a in range(heatmap_now.shape[0]):
            for b in range(heatmap_now.shape[1]):
                if heatmap_now[a, b] <= 0.05:
                    print (heatmap_now[a, b], metabName_unique[a], b )
                    ax.text(b + 0.5, a + 0.5, '*', ha='center', va='center', color='white', fontsize=10)

        plt.xlabel('synthetic trait')
        if showAll:
            plt.gcf().set_size_inches(7.5, 4)
        else:
            plt.gcf().set_size_inches(4, 5)
        #plt.gcf().set_size_inches(5.5, 7)
        plt.tight_layout()
        if showAll:
            plt.savefig('./images/metab/pathway/heatmapSep_' + envChoice + '_all.pdf' )
        else:
            plt.savefig('./images/metab/pathway/heatmapSep_' + envChoice + '.pdf' )
        plt.show()

    print (np.unique(metabName, return_counts=True))
    quit()



    #plt.scatter(enrichment, log_pval)

    
    print (xPos)
    print (yPos)

    for index1 in range(argSig.shape[0]):
        xPosNow, yPosNow = xPos[index1], yPos[index1]
        pathName = metabName[argSig[index1]]

        pathName = pathName.replace(' ', '\n')

        plt.annotate(pathName, xy=(xPosNow, yPosNow), fontsize=9)


    min1, max1 = np.min(enrichment)-0.2, np.max(enrichment)+0.2
    plt.plot(  [min1, max1] , np.zeros(2) - np.log10(0.05) , color='black' , linestyle=':')
    plt.ylabel('-log10(p)')
    plt.xlabel('enrichment')
    plt.ylim(0,  np.max(log_pval)*1.3 )
    plt.xlim(min1, max1)
    #plt.gcf().set_size_inches(3.5, 3)
    plt.gcf().set_size_inches(3.5, 4)
    plt.tight_layout()
    plt.savefig('./images/metab/pathway/split' + str(splitIndex) + '_phen' + str(phen_index) + '_' + envChoice + '.pdf')
    plt.show()

    print (data)
    quit()

    

#plotMetabPathway()
#quit()


def correlatateMetabEnv():


    
    genotype = loadnpz('./data/metab/metabData_Fernie/processed/names.npz')
    envirement = loadnpz('./data/metab/metabData_Fernie/processed/env.npz')
    yearList = loadnpz('./data/metab/metabData_Fernie/processed/year.npz')
    envirement = np.array([envirement, yearList]).T

    genotype_0d = genotype[envirement[:, 0] == '0d']
    genotype_6d = genotype[envirement[:, 0] == '6d']
    
    genotype_unique_0d = np.unique(genotype_0d)
    genotype_unique_6d = np.unique(genotype_6d)



    Y_0d = loadnpz('./data/metab/metabData_Fernie/pred/linear_primary_many_split' + str(0) + '.npz')[:, :10]
    Y_6d = loadnpz('./data/metab/metabData_Fernie/pred/linear_secondary_many_split' + str(0) + '.npz')[:, :10]

    print (genotype_unique_0d.shape, Y_0d.shape)

    Y_6d = Y_6d[np.isin( genotype_unique_6d, genotype_unique_0d )]
    genotype_unique_6d = genotype_unique_6d[np.isin( genotype_unique_6d, genotype_unique_0d )]

    Y_0d = Y_0d[np.isin( genotype_unique_0d, genotype_unique_6d )]
    genotype_unique_0d = genotype_unique_0d[np.isin( genotype_unique_0d, genotype_unique_6d )]

    Y_0d = Y_0d[np.argsort(genotype_unique_0d)]
    Y_6d = Y_6d[np.argsort(genotype_unique_6d)]

    print (Y_0d.shape)
    print (Y_6d.shape)

    corAll = np.zeros((Y_0d.shape[1], Y_6d.shape[1]))
    for a in range(Y_0d.shape[1]):
        for b in range(Y_6d.shape[1]):
            cor1 = scipy.stats.pearsonr( Y_0d[:, a],  Y_6d[:, b] )[0]
            print (cor1, a, b)
            corAll[a, b] = cor1
    corAll = np.abs(corAll)

    sns.heatmap(corAll[:10, :10], annot=True, fmt=".2f", annot_kws={"color": "white"})
    plt.ylabel('0 day trait')
    plt.xlabel('6 day trait')
    plt.xticks(np.arange(10)+0.5, np.arange(10)+1)
    plt.yticks(np.arange(10)+0.5, np.arange(10)+1)
    #for a in range(Y_0d.shape[1]):
    #    for b in range(Y_6d.shape[1]):
    #        plt.annotate( str(corAll[a, b])[:4], (a, b) )
    plt.savefig('./images/metab/env_corelation.pdf')
    plt.show()
    


    print (genotype_unique_0d.shape)
    print (genotype_unique_6d.shape)


#correlatateMetabEnv()
#quit()



def correlatateGWASMetabEnv():



    data_0d = loadnpz('./data/metab/metabData_Fernie/GWAS/linear_primary_many_split0_'  + str(1) + '.npz' )
    data_6d = loadnpz('./data/metab/metabData_Fernie/GWAS/linear_secondary_many_split0_'  + str(1) + '.npz' )
    argGood_0d = np.argwhere(np.isin( data_0d[:, 1], data_6d[:, 1] ))[:, 0]
    argGood_6d = np.argwhere(np.isin( data_6d[:, 1], data_0d[:, 1] ))[:, 0]

    
   
    pval_all = [[], []]
    for phenIndex  in range(10):
        for envChoice_index in range(2):
            envChoice = ['0d', '6d'][envChoice_index]
            if envChoice == '0d':
                data = loadnpz('./data/metab/metabData_Fernie/GWAS/linear_primary_many_split0_'  + str(phenIndex+1) + '.npz' )[argGood_0d]
            if envChoice == '6d':
                data = loadnpz('./data/metab/metabData_Fernie/GWAS/linear_secondary_many_split0_'  + str(phenIndex+1) + '.npz' )[argGood_6d]

            pvals = data[1:, -1]
            pvals = pvals.astype(float)
            #print (pvals.shape)
            #print (data[:5])
            pval_all[envChoice_index].append(np.copy(pvals))

        #quit()
    
    pval_all = np.array(pval_all)
    pvals_log = np.log(pval_all)

    #pvals_log[pvals_log > -5] = -5

    #plt.plot(pvals_log[0, 0])
    #plt.plot(pvals_log[1, 0])
    #plt.show()

    #print (pvals_log.shape)
    #quit()

    corMatrix = np.zeros((10, 10))
    for phen_index1 in range(10):
        for phen_index2 in range(10):
            cor = scipy.stats.pearsonr(pvals_log[0, phen_index1], pvals_log[1, phen_index2])
            print (cor)
            corMatrix[phen_index1, phen_index2] = cor[0]

    print (np.median( corMatrix[corMatrix!=1] ))

    sns.heatmap(corMatrix[:10, :10], annot=True, fmt=".2f", annot_kws={"color": "white"})
    plt.ylabel('0 day trait')
    plt.xlabel('6 day trait')
    plt.xticks(np.arange(10)+0.5, np.arange(10)+1)
    plt.yticks(np.arange(10)+0.5, np.arange(10)+1)
    #for a in range(Y_0d.shape[1]):
    #    for b in range(Y_6d.shape[1]):
    #        plt.annotate( str(corAll[a, b])[:4], (a, b) )
    plt.savefig('./images/metab/envGWAS_corelation.pdf')
    plt.show()
    

#correlatateGWASMetabEnv()
#quit()


def plotMetabHerit():

    for envChoice in ['0d', '6d']:

        onlyTest = False

        #envChoice = '0d'
        #envChoice = '6d'
        
        synthUsed = 10
        #synthUsed = 20

        heritArray_trait = loadnpz('./data/metab/metabData_Fernie/eval/herits_maxTrait_' + envChoice + '.npz')[:, :, :synthUsed]
        heritArray = loadnpz('./data/metab/metabData_Fernie/eval/herits_H2Opt_' + envChoice + '.npz')[:, :, :synthUsed]
        heritArray_PCA = loadnpz('./data/metab/metabData_Fernie/eval/herits_PCA_' + envChoice + '.npz')[:, :, :synthUsed]
        #quit()

        #print (np.mean(heritArray[1, :, :] , axis=0 ))
        #quit()

        print (np.mean(heritArray[1, :, :10]  ))
        print (np.mean(heritArray_PCA[1, :, :10]  ))
        print (np.mean(heritArray_trait[1, :, :10]  ))

        print (np.min(np.mean(heritArray[1, :, :10] , axis=0 )))


        #quit()

        traitCount = np.arange(synthUsed) + 1
        plt.plot(traitCount, np.mean(heritArray[1, :, :], axis=0), color='tab:blue')
        if not onlyTest:
            plt.plot(traitCount, np.mean(heritArray[0, :, :], axis=0) , color='tab:blue' , linestyle='dashed')
        plt.plot(traitCount, np.mean(heritArray_PCA[1, :, :], axis=0), color='tab:orange')
        if not onlyTest:
            plt.plot(traitCount, np.mean(heritArray_PCA[0, :, :], axis=0) , color='tab:orange', linestyle='dashed')
        plt.plot(traitCount, np.mean(heritArray_trait[1, :, :], axis=0), color='tab:green')
        if not onlyTest:
            plt.plot(traitCount, np.mean(heritArray_trait[0, :, :], axis=0) , color='tab:green', linestyle='dashed')
        plt.xlabel('trait number')
        plt.ylabel('heritability')
        plt.xticks( traitCount, traitCount )
        
        #plt.gcf().set_size_inches(3.5, 4)
        if synthUsed == 10 and onlyTest:
            plt.yticks([0, 0.5, 1], [0, 0.5, 1])
            plt.ylim(-0.02, 1.02)
            plt.gcf().set_size_inches(3.5, 2)
        else:
            plt.gcf().set_size_inches(5, 3)
        plt.tight_layout()
        if synthUsed == 10 and onlyTest:
            plt.savefig('./images/metab/herit_' + envChoice + '.pdf')
        else:
            plt.savefig('./images/metab/herit_' + envChoice + '_' + str(synthUsed) + '.pdf')
        #plt.legend(['H2Opt: test', 'H2Opt: train', 'PCA: test', 'PCA: train',  'single trait: test', 'single trait: train'  ])

        plt.show()


#plotMetabHerit()
#quit()



def plotMetabCoef():


    envPart = '0d'

    split_index = 0
    if envPart == '0d':
        modelName = './data/metab/metabData_Fernie/models/linear_primary_many_split' + str(split_index) + '.pt'
    if envPart == '6d':
        modelName = './data/metab/metabData_Fernie/models/linear_secondary_many_split' + str(split_index) + '.pt'
    
    model = torch.load(modelName)
    coef = getMultiModelCoef(model, multi=True)

    splitIndex = 0
    metabolites = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab.npz')[:, :, :1500, :40]
        

    coef = getMultiModelCoef(model, multi=True)    
    Y = model(torch.tensor(metabolites.reshape((metabolites.shape[0], metabolites.shape[1]*metabolites.shape[2]* metabolites.shape[3] ))).float(), np.arange(20))
    coef = coef[:20]
    Y, trackComputation = normalizeIndependent(Y, trackCompute=True)
    print (coef.shape, trackComputation.shape)

    #print (np.max(np.abs(coef[0])))

    coef = np.matmul( coef.T, trackComputation.data.numpy() ).T

    coef = coef / np.max(np.abs(coef), axis=1).reshape((-1, 1))

    
    coef = coef.reshape((coef.shape[0], 2, 1500, 40))[:, 0]

    #markers = ['o', 's', '^', 'v', 'D', '+', 'x', 'P', '*', 'X']
    markers = ['s', 'X', '+']
    colors = ['cyan', 'magenta', 'lime']

    for a in range(3):
        coef_now = coef[a]
        print (np.mean(np.abs(coef_now)), np.max(np.abs(coef_now)))
        argHigh = np.argwhere(np.abs(coef_now) > 0.02)
        values = np.abs(coef_now[argHigh[:, 0], argHigh[:, 1]])

        plt.scatter(argHigh[:, 1], argHigh[:, 0], alpha=0.5, marker = markers[a], s=100*values, c=colors[a])

    quit()
    plt.legend(['synthetic trait 1', 'synthetic trait 2', 'synthetic trait 3'])
    plt.ylabel('m/z (Da)')
    plt.xticks( np.arange(4)*2*5, np.arange(4)*5 )
    plt.xlabel('retention time (minutes)')
    #plt.gcf().set_size_inches(4, 4)
    plt.gcf().set_size_inches(3.5, 4)
    plt.tight_layout()
    plt.savefig('./images/metab/coefHeatmap.pdf')
    plt.show()



#plotMetabCoef()
#quit()


def multi_plotMetabCoef():

    #envPart = '0d'

    for envPart in ['0d', '6d']:

        split_index = 0
        if envPart == '0d':
            modelName = './data/metab/metabData_Fernie/models/linear_primary_many_split' + str(split_index) + '.pt'
        if envPart == '6d':
            modelName = './data/metab/metabData_Fernie/models/linear_secondary_many_split' + str(split_index) + '.pt'
        
        model = torch.load(modelName)
        coef = getMultiModelCoef(model, multi=True)

        splitIndex = 0
        metabolites = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab.npz')[:, :, :1500, :40]
            

        coef = getMultiModelCoef(model, multi=True)    
        Y = model(torch.tensor(metabolites.reshape((metabolites.shape[0], metabolites.shape[1]*metabolites.shape[2]* metabolites.shape[3] ))).float(), np.arange(20))
        coef = coef[:20]
        Y, trackComputation = normalizeIndependent(Y, trackCompute=True)
        print (coef.shape, trackComputation.shape)

        #print (np.max(np.abs(coef[0])))

        coef = np.matmul( coef.T, trackComputation.data.numpy() ).T
        coef = coef / np.max(np.abs(coef), axis=1).reshape((-1, 1))
        coef = coef.reshape((coef.shape[0], 2, 1500, 40))[:, 0]

        
        for phenIndex in range(10):
            coef_now = coef[phenIndex]
            argHigh = np.argwhere(np.abs(coef_now) > 0.02)
            values = np.abs(coef_now[argHigh[:, 0], argHigh[:, 1]])

            plt.scatter(argHigh[:, 1], argHigh[:, 0], alpha=0.5, s=100*values)
            plt.ylabel('m/z (Da)')
            plt.xticks( np.arange(4)*2*5, np.arange(4)*5 )
            plt.xlabel('retention time (minutes)')
            #plt.gcf().set_size_inches(4, 4)
            plt.gcf().set_size_inches(3, 3.5)
            plt.tight_layout()
            plt.savefig('./images/metab/coefHeatmap/' + envPart + '_' + str(phenIndex) + '.pdf')
            plt.show()



#multi_plotMetabCoef()
#quit()

def arabiGWASplot():

    #data = loadnpz('./data/plant/simulations/encoded/random100SNP/0/Gemma_H2Opt_' + str(phenIndex+1) + '.npz')
    #phenIndex = 0

    geneLoc = getGeneLocations('arabi')

    #envChoice = '6d'
    envChoice = '0d'



    for phenIndex in range(0, 10):

        if True:
            if envChoice == '0d':
                #data = loadnpz('./data/metab/metabData_Fernie/GWAS/linear_primary_many_split0_'  + str(phenIndex+1) + '.npz' )
                data = loadnpz('./data/metab/metabData_Fernie/GWAS/kPCA5_linear_primary_many_split0_'  + str(phenIndex+1) + '.npz' )
            if envChoice == '6d':
                #data = loadnpz('./data/metab/metabData_Fernie/GWAS/linear_secondary_many_split0_'  + str(phenIndex+1) + '.npz' )
                data = loadnpz('./data/metab/metabData_Fernie/GWAS/kPCA5_linear_secondary_many_split0_'  + str(phenIndex+1) + '.npz' )
        else:
            #methodName = 'PCA'
            methodName = 'maxTrait'
            data = loadnpz('./data/metab/metabData_Fernie/GWAS/kPCA5_'  + methodName + '_' + envChoice + '_' + str(phenIndex+1) + '.npz' )


        

        argGood = np.argwhere(np.isin( data[:, 0], np.arange(100).astype(str)  ))[:, 0]
        argGood = np.concatenate(( np.zeros(1, dtype=int), argGood ), axis=0)
        data = data[argGood]

        chr1 = data[1:, 0].astype(int)

        pos1 = data[1:, 2].astype(int)
        pos2 = np.copy(pos1)

        #print (pos1.shape)
        #quit()

        chr1_unique = np.unique(chr1)
        lastMax = 0
        for chr_now in chr1_unique:
            args1 = np.argwhere(chr1 == chr_now)[:, 0]
            pos2[args1] = pos1[args1] + lastMax
            lastMax = np.max(pos2[args1]) + 1
        

        pvals = data[1:, -1]
        pvals = pvals.astype(float)
        pvals = pvals + (1e-300)
        pvals_log = -1 * np.log(pvals) / np.log(10)


        # Calculate the expected p-values
        expected_p_values = np.linspace(0, 1, len(pvals_log)+1)[1:][-1::-1]
        

        expected_p_values = -1 * np.log10(expected_p_values)

        # Sort observed p-values
        observed_p_values = np.sort(pvals_log)


        cutOff = np.log(data.shape[0]) / np.log(10)
        cutOff = cutOff - (np.log(0.05) / np.log(10))

        argHigh = np.argwhere(pvals_log > cutOff)[:, 0]
        print ('trait', phenIndex+1)
        print ('argHigh', argHigh.shape)

        #argHigh_top = argHigh[np.argsort(pvals_log[argHigh] * -1)[:4]]
        argHigh_top = argHigh[np.argsort(pvals_log[argHigh] * -1)[:1]]

        if argHigh_top.shape[0] >= 1:
        
            if argHigh.shape[0] > 0:
                #closeGenes = findClosestGene(geneLoc, data[argHigh_top])
                closeGenes = findClosestGene(geneLoc, data[argHigh])

                closeString = ''
                for a in range(closeGenes.shape[0]):
                    closeString += ', ' + closeGenes[a, -1]
                print (closeString)


            pos2 = pos2 / np.max(pos2)


            if True:
                max1 = min( np.max(observed_p_values), np.max(expected_p_values) )
                # Create the Q-Q plot
                print (observed_p_values.shape, expected_p_values.shape)
                print (max1)
                plt.figure(figsize=(8, 8))
                plt.plot(expected_p_values, observed_p_values, marker='o', linestyle='none')
                plt.plot([0, max1], [0, max1], color='red', linestyle='--')  # Reference line
                plt.xlabel('expected -log10(p)')
                plt.ylabel('observed -log10(p)')
                #plt.title('Q-Q Plot for GWAS p-values')
                #plt.title('synthetic trait ' + str(phenIndex + 1))
                
                #plt.gcf().set_size_inches(4, 2.0)
                if envChoice == '0d':
                    plt.gcf().set_size_inches(4, 2.0)
                else:
                    plt.gcf().set_size_inches(3, 2.0)
                plt.tight_layout()
                #plt.xlim(0, 1)
                #plt.ylim(0, 1)
                plt.grid()
                plt.savefig('./images/metab/GWAS_small/QQ_' + str(phenIndex) + '_' + envChoice + '.png')
                plt.show()

            

            colors = ['red', 'blue']

            bestGenes = []

            for a in range(chr1_unique.shape[0]):
                args1 = np.argwhere(chr1 == a+1)[:, 0]

                argHigh_mini = args1[pvals_log[args1] > cutOff ]  # + np.log10(10) ]

                chr_now = chr1[argHigh_mini] - 1
                pos_now = pos1[argHigh_mini]

                #genes_now = pastNames[chr_now, pos_now]
                #bestGenes = bestGenes + list(genes_now)

                if False:
                    plt.scatter(pos2[args1], pvals_log[args1], c=colors[a%2])

            #print (bestGenes)

            if False:
                Xpos, YPos = pos2[argHigh_top], pvals_log[argHigh_top]
                #Xpos[Xpos>0.75] = 0.75
                Xpos = (Xpos * 0.78)
                #Xpos[Xpos>0.8] = 0.8
                #YPos = YPos + 0.2
                YPos = YPos + (0.03 * np.max(pvals_log))

                #if dateIndex == 0:
                #    Xpos[2] = 0.35
                #    Xpos[3] = 0.01
                
                #if dateIndex == 4:
                #    YPos[0] = 8.25
                #    #print (Xpos)
                #    #print (YPos)
                
                #if phenIndex == 1:
                #    YPos[2] = YPos[2] + 0.4
                #    YPos[3] = YPos[3] - 0.3
                for gene_index in range(argHigh_top.shape[0]):
                    #print (gene_index)
                    yPosNow = YPos[gene_index]
                    xPosNow = Xpos[gene_index]
                    geneName = closeGenes[gene_index][3]
                    geneName = geneName.replace('Misin', '')
                    plt.annotate(geneName, xy=(xPosNow, yPosNow), fontsize=9)


            #line1 = np.array([0, pvals_log.shape[0]])
            line1 = np.array([0, np.max(pos2)])

            if False:
                #plt.scatter(np.arange(pvals_log.shape[0]), pvals_log)
                plt.plot( line1, np.zeros(2) + cutOff , color='black' , linestyle=':')
                #plt.plot( line1, np.zeros(2) + cutOff + np.log10(10)  )
                #plt.title('phenotype ' + str(phenIndex+1))
                #plt.xlabel("genomic bin")
                plt.xlabel("chromosome")
                plt.ylabel('-log10(p)')
                plt.xticks([])
                plt.ylim(0, np.max(pvals_log) * 1.15 )
                plt.xlim(0  - (0.01*np.max(pos2) ) , np.max(pos2) * 1.01 )
                #plt.gcf().set_size_inches(5, 4.5)
                #plt.gcf().set_size_inches(4, 2)
                if True:
                    #plt.gcf().set_size_inches(3.5, 1.7)
                    if envChoice == '0d':
                        plt.gcf().set_size_inches(4, 2.0)
                    else:
                        plt.gcf().set_size_inches(3, 2)
                #plt.gcf().set_size_inches(4, 1.8)

                plt.tight_layout() #GWAS
                plt.savefig('./images/metab/GWAS_small/Manhat_' + str(phenIndex) + '_' + envChoice + '.png')
                plt.show()




arabiGWASplot()
quit()




def arabiGWASHits():

    #envChoice = '6d'
    for envChoice in ['0d', '6d']:
    
        numHigh = []
        for phenIndex in range(0, 10):
            if envChoice == '0d':
                #data = loadnpz('./data/metab/metabData_Fernie/GWAS/linear_primary_many_split0_'  + str(phenIndex+1) + '.npz' )
                data = loadnpz('./data/metab/metabData_Fernie/GWAS/kPCA5_linear_primary_many_split0_'  + str(phenIndex+1) + '.npz' )
            if envChoice == '6d':
                #data = loadnpz('./data/metab/metabData_Fernie/GWAS/linear_secondary_many_split0_'  + str(phenIndex+1) + '.npz' )
                data = loadnpz('./data/metab/metabData_Fernie/GWAS/kPCA5_linear_secondary_many_split0_'  + str(phenIndex+1) + '.npz' )

            argGood = np.argwhere(np.isin( data[:, 0], np.arange(100).astype(str)  ))[:, 0]
            argGood = np.concatenate(( np.zeros(1, dtype=int), argGood ), axis=0)
            data = data[argGood]

            pvals = data[1:, -1]
            pvals = pvals.astype(float)
            pvals = pvals + (1e-300)
            pvals_log = -1 * np.log(pvals) / np.log(10)
            cutOff = np.log(data.shape[0]) / np.log(10)
            cutOff = cutOff - (np.log(0.05) / np.log(10))

            argHigh = np.argwhere(pvals_log > cutOff)[:, 0]
            numHigh.append(argHigh.shape[0])
        
        #plt.scatter(numHigh, np.arange( len(numHigh) ))
        #plt.show()

        maxHit = int(np.max(np.array(numHigh)))

        plt.barh(np.arange( len(numHigh) )+1, numHigh)
        plt.yticks(np.arange(10)+1, np.arange(10)+1)
        if maxHit <= 4:
            plt.xticks(np.arange(maxHit+1), np.arange(maxHit+1) )
        else:
            maxHit_5 = maxHit // 5
            #plt.xticks(np.arange(maxHit_5+1)*5, np.arange(maxHit_5+1)*5 ) 
            plt.xticks([0, 5, 10], [0, 5, 10])
            #plt.xticks(np.arange((maxHit + 2) // 2) * 2, np.arange( (maxHit + 2) // 2) * 2 ) )
            True
        plt.ylabel('synthetic trait')
        plt.xlabel('# of significant SNPS')
        plt.gcf().set_size_inches(2, 4)
        plt.tight_layout()
        plt.savefig('./images/metab/GWAS_hits_' + envChoice + '.pdf')
        plt.show()



#arabiGWASHits()
#quit()





def plotMetabExample():

    from matplotlib.colors import LogNorm

    metabolites = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab.npz')[:, :, :1500, :40]
    metabolites[metabolites == 0] = np.min(metabolites[metabolites != 0])

    envList = loadnpz('./data/metab/metabData_Fernie/processed/env.npz')

    for envChoice in ['0d', '6d']:
        metabolites_now = np.copy(metabolites[envList == envChoice][:3])


        for a in range(1):
            sns.heatmap(metabolites_now[a, 0, 100:, :35 ], norm=LogNorm(), linewidths=0, cbar=False)
            
            plt.ylabel('m/z (Da)')
            plt.xticks( np.arange(4)*2*5, np.arange(4)*5 )
            #plt.yticks( np.arange(15-1) * 100, np.arange(1, 15) * 100  )
            #plt.yticks( ((np.arange(6)*2)) * 100,  ((np.arange(6)*2)+1) * 100   )
            yValues = np.array([100, 500, 1000, 1500])
            plt.yticks( yValues - 100 , yValues  )

            #plt.yticks( np.arange(15-1) * 100, np.arange(1, 15) * 100  )

            plt.xlabel('retention time (minutes)')
            
            #plt.gcf().set_size_inches(4, 4)
            #plt.gcf().set_size_inches(3.5, 4)
            #plt.gcf().set_size_inches(3, 3)
            plt.gcf().set_size_inches(2, 1.5)
            plt.tight_layout()
            plt.savefig('./images/examples/metab/exampleHeatmap_' + str(a) + '_' + envChoice + '.png')
            plt.show()


#plotMetabExample()
#quit()





def metabCompare_ANOVA_lmer():

    miscNames = ['0d', '6d']
    for dataIndex in range(2):
        miscName = miscNames[dataIndex]


        phenotypes = np.loadtxt("./data/software/lmeHerit/input_metab_" + miscName + ".csv", delimiter=',', dtype=str)
        phenotypes_names = phenotypes[0]
        phenotypes = phenotypes[1:]
        names = phenotypes[:, 0]
        phenotypes = phenotypes[:, 1:].astype(float)
        
        lmerHerit =  np.loadtxt("./data/software/lmeHerit/output_metab_" +  miscName + ".csv", delimiter=',', dtype=str)
        lmerHerit = lmerHerit[1:, 1]
        

        envirement = np.zeros((names.shape[0], 0), dtype=int)
        
        heritList = cheapHeritability(torch.tensor(phenotypes).float(), names, envirement)
        heritList = heritList.data.numpy()
        heritList[heritList<0] = 0


        argGood = np.argwhere(lmerHerit != 'NA')[:, 0]
        heritList = heritList[argGood]
        lmerHerit = lmerHerit[argGood].astype(float)



        

        
            
        print (lmerHerit.shape)
        print (heritList.shape)
        #print (scipy.stats.pearsonr( lmerHerit, heritList ))

        #for channel in range(6):
    
        print (scipy.stats.pearsonr( lmerHerit, heritList ))
        rValue = scipy.stats.pearsonr( lmerHerit, heritList )[0]
        rValue = np.round(rValue * 1000) / 1000
        rText = '$r = ' + str(rValue) + '$'

        #print (scipy.stats.pearsonr( lmerHerit[channelInfo!=0], heritList[channelInfo!=0] ))

        #plt.scatter( lmerHerit, heritList )
        #plt.show()

        m, b = np.polyfit(lmerHerit, heritList, 1)
        y_fit = m * lmerHerit + b

        plt.plot(lmerHerit, y_fit, color='red')

        plt.annotate(rText, xy=(0.0, 0.9), fontsize=12)

        plt.scatter( lmerHerit, heritList)
        plt.xlabel('lme4 heritability')
        plt.ylabel("ANOVA heritability")
        plt.gcf().set_size_inches(3.5, 4)
        plt.tight_layout()
        plt.savefig('./images/ANOVA/arabi_' + miscName + '.pdf')
        plt.show()



#metabCompare_ANOVA_lmer()
#quit()










################################################
###                                          ###
###              Not Charecterized           ###
###                                          ###
################################################





def miscGSPlot():


    #heritValue_C = loadnpz('./data/miscPlant/eval/GS_central.npz')
    #heritValue_S = loadnpz('./data/miscPlant/eval/GS_south.npz')

    #namePart = 'central'
    namePart = 'south'

    heritValue = loadnpz('./data/miscPlant/eval/GS_' + namePart + '.npz').astype(float)
    heritValue_PCA = loadnpz('./data/miscPlant/eval/GS_' + namePart + '_PCA.npz').astype(float)
    heritValue_trait = loadnpz('./data/miscPlant/eval/GS_' + namePart + '_maxTrait.npz').astype(float)
    heritValue_wave = loadnpz('./data/miscPlant/eval/GS_' + namePart + '_maxWave.npz').astype(float)
    
    print (heritValue)

    plt.plot(heritValue)
    plt.plot(heritValue_PCA)
    plt.plot(heritValue_trait)
    plt.plot(heritValue_wave)
    plt.show()


#miscGSPlot()
#quit()


def crossValidOnlyH2Opt():

    trainInfos = loadnpz('./data/plant/eval/herit/crossValid_train.npz')[0]
    testInfos = loadnpz('./data/plant/eval/herit/crossValid_test.npz')[0]

    trainInfos = np.mean(trainInfos, axis=0)
    testInfos = np.mean(testInfos, axis=0)
    
    arange1 = np.arange(trainInfos.shape[0]) + 1

    plt.plot(arange1, trainInfos)
    plt.plot(arange1, testInfos,  c='tab:cyan')
    plt.scatter(arange1, trainInfos)
    plt.scatter(arange1, testInfos, c='tab:cyan')
    plt.legend(['training set', 'test set'])
    plt.xlabel('synthetic trait')
    plt.ylabel('heritability')
    plt.gcf().set_size_inches(4, 4)
    plt.tight_layout()
    plt.savefig('./images/sor/crossValidH2Opt.pdf')
    plt.show()

#crossValidOnlyH2Opt()
#quit()



def manual_plotHeritability():


    #modelName = './data/plant/models/poly_2.pt'
    #modelName = './data/plant/models/simple_1.pt'
    #modelName = './data/plant/models/linear_6.pt'
    #modelName = './data/plant/models/linear_8.pt'
    modelName = './data/plant/models/linear_trainAll2.pt'
    
    #modelName = './data/plant/models/A_4.pt'
    #modelName = './data/plant/models/conv_1.pt'

    model = torch.load(modelName)

    X = loadnpz('./data/plant/processed/sor/X.npz')
    names = loadnpz('./data/plant/processed/sor/names.npz')

    X = torch.tensor(X).float()

    rand1 = torch.rand(size=X.shape)
    #Y = model(X  + (rand1 * 0.001) )
    

    try:
        Y = model(X)#, np.arange(10))
    except:
        Y = model(X, np.arange(10))
    Y = normalizeIndependent(Y)

    


    envirement = loadnpz('./data/plant/processed/sor/set1.npz')

    envirement = envirement.reshape((-1, 1))
    name_unique, name_inverse, name_counts = np.unique(names, return_inverse=True, return_counts=True)




    np.random.seed(2)
    trainTest2 = np.random.randint(3, size=name_unique.shape[0])
    trainTest2 = trainTest2[name_inverse]


    trainTest3 = np.copy(trainTest2)
    trainTest3[trainTest3 == 0] = 100
    trainTest3[trainTest3!=100] = 0
    trainTest3[trainTest3 == 100] = 1

    #heritability_train = cheapHeritability(Y[trainTest3 == 0], names[trainTest3 == 0], envirement[trainTest3 == 0] )
    #heritability_train = cheapHeritability(Y, names, envirement )
    #print (heritability_train[:])
    heritability_train = cheapHeritability(Y[trainTest3 == 0], names[trainTest3 == 0], envirement[trainTest3 == 0] )
    #print (heritability_train[:3])
    #quit()
    heritability_test = cheapHeritability(Y[trainTest3 == 1], names[trainTest3 == 1], envirement[trainTest3 == 1] )


    #print (heritability_train[0], heritability_test[0])
    #quit()
    
    
    arange1 = np.arange(Y.shape[1]) + 1
    plt.plot(arange1, heritability_train.data.numpy())
    plt.plot(arange1, heritability_test.data.numpy())
    plt.scatter(arange1, heritability_train.data.numpy())
    plt.scatter(arange1, heritability_test.data.numpy())
    plt.xticks(arange1)
    plt.ylim(bottom=0)
    plt.xlabel('sythetic trait number')
    plt.ylabel("heritability")
    plt.legend(['training set', 'test set'])
    #plt.gcf().set_size_inches(7, 3)
    plt.gcf().set_size_inches(7, 2)
    plt.tight_layout()
    plt.show()
    quit()



#manual_plotHeritability()
#quit()


def sorAllHerit():


    if False:
        #modelName = './data/plant/models/poly_2.pt'
        #modelName = './data/plant/models/simple_1.pt'
        #modelName = './data/plant/models/linear_6.pt'
        #modelName = './data/plant/models/linear_8.pt'
        modelName = './data/plant/models/linear_trainAll2.pt'
        
        #modelName = './data/plant/models/A_4.pt'
        #modelName = './data/plant/models/conv_1.pt'

        model = torch.load(modelName)

        X = loadnpz('./data/plant/processed/sor/X.npz')

        X = torch.tensor(X).float()

        rand1 = torch.rand(size=X.shape)
        #Y = model(X  + (rand1 * 0.001) )
        

        try:
            Y = model(X)#, np.arange(10))
        except:
            Y = model(X, np.arange(10))
        Y = normalizeIndependent(Y)

    names = loadnpz('./data/plant/processed/sor/names.npz')

    name_unique, name_inverse = np.unique(names, return_inverse=True)
    np.random.seed(0)
    trainTest2 = np.random.randint(10, size=name_unique.shape[0])
    trainTest2 = trainTest2[name_inverse]
    trainTest3 = np.copy(trainTest2)
    splitIndex = 0
    trainTest3[trainTest3 == splitIndex] = 100
    trainTest3[trainTest3!=100] = 0
    trainTest3[trainTest3 == 100] = 1

    Y = loadnpz('./data/plant/syntheticTraits/linear_crossVal_reg4_' + str(0) + '_mod10.npz')
    Y = torch.tensor(Y).float()

    Y2 = loadnpz('./data/plant/syntheticTraits/linear_crossVal_reg4_' + str(0) + '_fourier.npz')
    Y2 = torch.tensor(Y2).float()
    
    envirement = loadnpz('./data/plant/processed/sor/set1.npz').reshape((-1, 1))
    heritability_test = cheapHeritability(Y[trainTest3==1], names[trainTest3==1], envirement[trainTest3==1] )
    heritability_test = heritability_test.data.numpy()

    heritability_test2 = cheapHeritability(Y2[trainTest3==1], names[trainTest3==1], envirement[trainTest3==1] )
    heritability_test2 = heritability_test2.data.numpy()
    
    
    
    arange1 = np.arange(Y.shape[1]) + 1

    
    plt.scatter(arange1, heritability_test)
    plt.scatter(arange1, heritability_test2)
    plt.xlabel('sythetic trait number')
    plt.ylabel("test set heritability")
    plt.xticks(arange1)
    plt.legend(['original', 'fourier transform'])
    plt.gcf().set_size_inches(4, 4)
    plt.tight_layout()
    
    #plt.savefig('./images/sor/allTrainHerits.pdf')
    plt.show()
    quit()

    plt.plot(arange1, heritability_train.data.numpy())
    plt.scatter(arange1, heritability_train.data.numpy())
    plt.xticks(arange1)
    plt.ylim(bottom=0)
    plt.xlabel('sythetic trait number')
    plt.ylabel("heritability")
    #plt.gcf().set_size_inches(7, 3)
    plt.gcf().set_size_inches(7, 2)
    plt.tight_layout()
    plt.show()
    quit()



#sorAllHerit()
#quit()






def plotWavePair():


    matrix1 = loadnpz('./data/plant/pairWave/originalHerit/matrix.npz')
    heritMatrix = loadnpz('./data/plant/pairWave/fastHerit/matrix.npz')   


    print (np.max(heritMatrix))


#plotWavePair()
#quit()



def plotArabiGWAS():

    def find_arghigh(data):
        pvals = data[1:, -1]
        pvals = pvals.astype(float)
        pvals = pvals + (1e-300)
        pvals_log = -1 * np.log(pvals) / np.log(10)
        cutOff = np.log(data.shape[0]) / np.log(10)
        cutOff = cutOff - (np.log(0.05) / np.log(10))
        argHigh = np.argwhere(pvals_log > cutOff)[:, 0]
        return argHigh

    
    numSig = []
    numSig_single = []
    arange1 = np.arange(36)
    for clusterNum in arange1:
        data = loadnpz('./data/metab/metabData_Alex2/GWAS/' + str(0) + '_cluster_' + str(clusterNum) + '_' + str(0+1) + '.npz' )
        argHigh = find_arghigh(data)
        numSig.append(argHigh.shape[0])

        data_single = loadnpz('./data/metab/metabData_Alex2/GWAS/maxTrait_' + str(0) + '_cluster_' + str(clusterNum) + '_' + str(0+1) + '.npz' )
        argHigh_single = find_arghigh(data_single)
        numSig_single.append(argHigh_single.shape[0])


    #bar, scatter
    plt.scatter(arange1, numSig, alpha=0.9)
    plt.scatter(arange1, numSig_single, alpha=0.9)
    plt.xlabel('cluster')
    plt.ylabel('number of significant SNPs')
    plt.legend(['H2Opt', 'best single trait'])
    plt.gcf().set_size_inches(8, 3)
    plt.tight_layout()
    plt.show()
        



#plotArabiGWAS()
#quit()


def plotMetabHerit():

    genotype = loadnpz('./data/metab/metabData_Fernie/processed/names.npz')
    genotype_unique0, genotype_inverse = np.unique(genotype, return_inverse=True)
    Nsplit = 5
    np.random.seed(2)
    trainTest = np.random.randint(Nsplit, size=genotype_unique0.shape[0])
    trainTest2 = np.zeros(trainTest.shape[0], dtype=int)
    trainTest2[trainTest == 0] = 1


    Y_mean = loadnpz('./data/metab/metabData_Fernie/pred/linear_primary_many.npz')#[:, :50]
    #Y_mean = loadnpz('./data/metab/metabData_Fernie/pred/PCA_primary.npz')

    std = np.mean(Y_mean ** 2, axis=0) ** 0.5
    Y_mean = Y_mean / std.reshape((1, -1))

    envirement = loadnpz('./data/metab/metabData_Fernie/processed/env.npz')
    
    genotype_0d = genotype[envirement == '0d']
    #print (genotype_0d.shape)
    genotype_unique = np.unique(genotype_0d)
    trainTest2 = trainTest2[np.isin(genotype_unique0, genotype_unique)]

    
    #print (np.mean(Y_mean, axis=0))
    #print (np.mean(Y_mean**2, axis=0))
    #quit()

    fileName_6d = './data/metab/metabData_Fernie/6d-Table-1.tsv'
    fileName_0d = './data/metab/metabData_Fernie/0d-Table-1.tsv'
    genotype_known_6d, metNames_6d, metValues_6d = processKnownMet(fileName_6d)
    genotype_known_0d, metNames_0d, metValues_0d = processKnownMet(fileName_0d)

    assert np.array_equal(genotype_known_6d, genotype_known_0d)
    genotype_known = genotype_known_6d


    #genotype_known, metNames, metValues = genotype_known_6d, metNames_6d, metValues_6d


    #arg_primary = np.argwhere(np.isin(metNames_0d, metNames_6d) == False)[:, 0]
    #arg_secondary = np.argwhere(np.isin(metNames_6d, metNames_0d) == False)[:, 0]

    #metValues_primary = metValues_0d[:, arg_primary]
    #metValues_secondary = metValues_6d[:, arg_secondary]

    #geno_inter = np.intersect1d(genotype_unique, genotype_known)
    arg_known = np.zeros(genotype_known.shape[0], dtype=int)
    for a in range(genotype_known.shape[0]):
        arg1 = np.argwhere(genotype_unique == genotype_known[a])[0, 0]
        arg_known[a] = arg1
    Y_mean = Y_mean[arg_known]
    trainTest2 = trainTest2[arg_known]

    corList = []
    #from sklearn.linear_model import LinearRegression

    metValues = metValues_0d

    from sklearn.linear_model import Lasso
    for met_index in range(metValues.shape[1]):
        reg = Lasso(alpha=1e-2).fit(Y_mean[trainTest2 == 0], metValues[trainTest2 == 0, met_index] )        
        pred = reg.predict(Y_mean)
        cor_train = scipy.stats.pearsonr(pred[trainTest2 == 0], metValues[trainTest2 == 0, met_index])
        cor_test = scipy.stats.pearsonr(pred[trainTest2 == 1], metValues[trainTest2 == 1, met_index])

        #print (cor_train[0], cor_test[0])
        corList.append([cor_train[0], cor_test[0]])
    corList = np.array(corList)
    
    plt.scatter(corList[:, 0], corList[:, 1])
    plt.show()







def OLD_plotMetabPathway():

    splitIndex = 0
    #phen_index = 1
    for phen_index in range(10):
    
        envChoice = '0d'
        df = pd.read_csv('./data/metab/metabEnrich/split' + str(splitIndex) + '_phen' + str(phen_index) + '_' + envChoice + '.csv')
        data = df.to_numpy()

        metabName = data[:, 0]
        pval = data[:, 5].astype(float)
        hits = data[:, 3].astype(float) #2 is total hits, 3 is signifciant
        expectedHits = data[:, 4].astype(float)
        log_pval = -1 * np.log10(pval)

        enrichment = hits.astype(float) / expectedHits.astype(float)

        argSig = np.argwhere(pval < 0.05)[:, 0]

        plt.scatter(enrichment, log_pval)

        xPos = enrichment[argSig]
        yPos = log_pval[argSig]
        if (phen_index == 1) and (envChoice == '0d'):
            yPos[0] -= 0.1
            yPos[1] = 1.35
            xPos[2] = 1.5
            yPos[2] += 0.1
            xPos[0] += 0.1
            xPos[1] += 0.1
        print (xPos)
        print (yPos)

        for index1 in range(argSig.shape[0]):
            xPosNow, yPosNow = xPos[index1], yPos[index1]
            pathName = metabName[argSig[index1]]

            pathName = pathName.replace(' ', '\n')

            plt.annotate(pathName, xy=(xPosNow, yPosNow), fontsize=9)


        min1, max1 = np.min(enrichment)-0.2, np.max(enrichment)+0.2
        plt.plot(  [min1, max1] , np.zeros(2) - np.log10(0.05) , color='black' , linestyle=':')
        plt.ylabel('-log10(p)')
        plt.xlabel('enrichment')
        plt.ylim(0,  np.max(log_pval)*1.3 )
        plt.xlim(min1, max1)
        #plt.gcf().set_size_inches(3.5, 3)
        plt.gcf().set_size_inches(3.5, 4)
        plt.tight_layout()
        plt.savefig('./images/metab/pathway/split' + str(splitIndex) + '_phen' + str(phen_index) + '_' + envChoice + '.pdf')
        plt.show()

        print (data)
        quit()



def OLD_plotCoeff():

    modelName = './data/plant/models/simple_1.pt'

    #modelName = './data/plant/models/lowReg_4.pt'
    model = torch.load(modelName)

    coef = getModelCoef(model, multi=True)

    
    wavelengths = np.arange(coef[0].shape[0]) + +350
    plt.plot(wavelengths, coef[0])
    plt.plot(wavelengths, coef[1])
    plt.plot(wavelengths, coef[2])
    #plt.plot(wavelengths, coef[1])
    #plt.plot(wavelengths, coef[2])
    plt.xlabel('wavelength (nm)')
    plt.ylabel("coefficient")
    plt.legend(['trait 1', 'trait 2', 'trait 3'])
    plt.gcf().set_size_inches(7, 3)
    plt.tight_layout()
    plt.show()
    quit()

#plotCoeff()
#quit()


def oldMetabCoef():

    


    coef = coef[:20]
    

    from matplotlib.colors import LinearSegmentedColormap

    bright_diverging = LinearSegmentedColormap.from_list(
    "black_blue_red",
    [
        (0.0, '#00008B'),     # min
        (0.15, '#00008B'),
        (0.35, 'blue'),
        (0.45, '#00BFFF'),
        (0.5, 'white'),    # zero
        (0.55, '#FF0080'),
        (0.65, 'red'),
        (0.85, '#8B0000'),
        (1.0, '#8B0000')       # max
    ]
    )



    for phenIndex in range(1):
        print (phenIndex)

        print (np.max(coef[phenIndex]), np.min(coef[phenIndex]))

        print (np.mean(np.abs(coef[phenIndex])))
        quit()

        vmax = np.max(np.abs(coef[phenIndex]))
        vmin = -1 * vmax

        coef_plot = np.copy(coef[phenIndex])
        #coef_plot[np.abs(coef_plot) > 0.01] = 1

        #coefMask =  (np.abs(coef_plot) < 0.01)

        ax = sns.heatmap(coef_plot, cmap=bright_diverging, vmin=vmin, vmax=vmax , 
            cbar=True, square=False, linewidths=0, linecolor='white')
        # Set background color for masked (zero) regions to black
        ax.set_facecolor("black")
        plt.xlabel('retention time (minutes)')
        plt.ylabel('mass (Da)')
        plt.xticks(  np.arange(coef_plot.shape[1] // 10) * 10, np.arange(coef_plot.shape[1] // 10) * 5   )
        plt.yticks(  np.arange(coef_plot.shape[0] // 100) * 100, np.arange(coef_plot.shape[0] // 100) * 100   )
        plt.gcf().set_size_inches(3, 4)
        plt.tight_layout()
        
        plt.savefig('./images/metab/coefHeatmap/heatmap_' + str(phenIndex) + '_' + envPart + '.png')
        plt.show()

        #quit()


        #sns.heatmap(coef_plot, cmap='grey')
        #plt.xlabel('retention time')
        #plt.ylabel('mass')
        #plt.show()

        
        #sns.heatmap(coef[a], cmap='bwr', vmin=vmin, vmax=vmax)
        #plt.xlabel('retention time')
        #plt.ylabel('mass')
        #plt.show()






def conv_plotEncodeCorrelation():


    plotTypes = ['true']


    #simulationNames = ['uncorSims']

    #simulationNames = ['uncorSims']
    #simulationNames = ['random100SNP', 'uncorSims']
    #simulationNames = ['seperate100SNP']
    simulationNames = ['random100SNP']

    for simulationName in simulationNames:
        for plotType in plotTypes:
        
            folder0 = './data/plant/simulations/encoded/' + simulationName + '/'
            

            if plotType == 'true':
                corMatrix_PCA = loadnpz(folder0 + 'PCA_corMatrix.npz')
            elif plotType == 'proj':
                corMatrix_PCA = loadnpz(folder0 + 'PCA_corMatrixProj.npz')
            corMatrix_PCA = np.abs(corMatrix_PCA)
            #heritMatrix = loadnpz(folder0 + 'PCA_heritMatrix.npz')


            if plotType == 'true':
                corMatrix_maxWave = loadnpz(folder0 + 'maxWave_corMatrix.npz')
            elif plotType == 'proj':
                corMatrix_maxWave = loadnpz(folder0 + 'maxWave_corMatrixProj.npz')
            corMatrix_maxWave = np.abs(corMatrix_maxWave)


            if plotType == 'true':
                corMatrix = loadnpz(folder0 + 'A2_corMatrix.npz')
            elif plotType == 'proj':
                corMatrix = loadnpz(folder0 + 'A2_corMatrixProj.npz')
            corMatrix = np.abs(corMatrix)

            #./data/plant/simulations/encoded/random100SNP/H2Opt-Conv_corMatrix.npz

            if True:
                if plotType == 'true':
                    corMatrix_conv = loadnpz(folder0 + 'H2Opt-Conv_corMatrix.npz')
                elif plotType == 'proj':
                    corMatrix_conv = loadnpz(folder0 + 'H2Opt-Conv_corMatrixProj.npz')
                corMatrix_conv = np.abs(corMatrix_conv)

            if plotType == 'true':
                corMatrix_factor = loadnpz(folder0 + 'factorAnalysis_corMatrix.npz')
            elif plotType == 'proj':
                corMatrix_factor = loadnpz(folder0 + 'factorAnalysis_corMatrixProj.npz')
            corMatrix_factor = np.abs(corMatrix_factor)


            if False:
                corMatrix = corMatrix[:, np.arange(3), np.arange(3)]
                corMatrix_PCA = corMatrix_PCA[:, np.arange(3), np.arange(3)]
                corMatrix_maxWave = corMatrix_maxWave[:, np.arange(3), np.arange(3)]
                corMatrix_conv = corMatrix_conv[:, np.arange(3), np.arange(3)]

                #print (corMatrix.shape)
                #print (corMatrix_conv.shape)
                #quit()
            



            valueName = 'correlation with true trait'
            traitName = 'trait number'
            methodName = 'method'
            plotData = {}
            plotData[valueName] = []
            plotData[methodName] = []
            plotData[traitName] = []

            Ntrait = 3
            #if plotType == 'max':
            #    Ntrait = 5

            #print (corMatrix_PCA.shape)
            #quit()

            Nsim = 5


            #hue="alive"
            for simNum in range(Nsim):
                #plt.imshow(corMatrix[simNum])
                #plt.show()

                #if simNum == 1:
                #    print (corMatrix[simNum] )
                #    print (corMatrix_PCA[simNum] )
                #    print (corMatrix_maxWave[simNum] )
                #    quit()



                for traitNum in range(Ntrait):
                    value1 = corMatrix[simNum, traitNum, traitNum] 
                    value2 = corMatrix_PCA[simNum, traitNum, traitNum]
                    value3 = corMatrix_maxWave[simNum, traitNum, traitNum]
                    value4 = corMatrix_conv[simNum, traitNum, traitNum] 
                    value5 = corMatrix_factor[simNum, traitNum, traitNum] 
                    
                    

                    plotData[valueName].append(value1 )
                    plotData[methodName].append('H2Opt')
                    plotData[traitName].append(str(traitNum+1))

                    #plotData[valueName].append(value4 )
                    #plotData[methodName].append('H2Opt-conv')
                    #plotData[traitName].append(str(traitNum+1))
                    

                    plotData[valueName].append(value2  )
                    plotData[methodName].append('PCA')
                    plotData[traitName].append(str(traitNum+1))

                    plotData[valueName].append(value5  )
                    plotData[methodName].append('factor analysis')
                    plotData[traitName].append(str(traitNum+1))

                    plotData[valueName].append(value3  )
                    plotData[methodName].append('single wavelength')
                    plotData[traitName].append(str(traitNum+1))

                    


            sns.boxplot(data=plotData, x=traitName, y=valueName, hue=methodName, dodge=True)
            #sns.stripplot(data=plotData, x=traitName, y=valueName, hue=methodName, dodge=True, jitter=True, alpha=0.6)

            #plt.legend([],[], frameon=False)

            plt.gcf().set_size_inches(4, 4)
            plt.tight_layout()
            #plt.gcf().set_size_inches(4, 4)

            #plt.savefig('./images/encodeSim/' + simulationName + '_conv_' + plotType + '.pdf')
            plt.savefig('./images/encodeSim/' + simulationName + '_factorAnalysis_' + plotType + '.pdf')
            plt.show()



#conv_plotEncodeCorrelation()
#quit()


def plotEncodeGS():

    simulationNames = ['random100SNP']

    for simulationName in simulationNames:
        
        folder0 = './data/plant/simulations/encoded/' + simulationName + '/'

        #heritMatrix = loadnpz(folder0 + 'GSherit_all.npz')
        heritMatrix = loadnpz(folder0 + 'GSherit_all_noOutlier3.npz')

    
        valueName = 'V(G)/Vp'
        traitName = 'trait number'
        methodName = 'method'
        plotData = {}
        plotData[valueName] = []
        plotData[methodName] = []
        plotData[traitName] = []

        Ntrait = 5
        #Ntrait = 3
        Nsim = 10


        #hue="alive"
        for simNum in range(Nsim):
            #plt.imshow(corMatrix[simNum])
            #plt.show()

            for traitNum in range(Ntrait):
                value1 = heritMatrix[0, simNum, traitNum] 
                value2 = heritMatrix[1, simNum, traitNum]
                value3 = heritMatrix[2, simNum, traitNum]
                value4 = heritMatrix[3, simNum, traitNum]

                if traitNum < 3:
                    plotData[valueName].append(value4 )
                    plotData[methodName].append('ground truth')
                    plotData[traitName].append(str(traitNum))
                
                plotData[valueName].append(value1 )
                plotData[methodName].append('H2Opt')
                plotData[traitName].append(str(traitNum))
                

                plotData[valueName].append(value2  )
                plotData[methodName].append('PCA')
                plotData[traitName].append(str(traitNum))

                plotData[valueName].append(value3  )
                plotData[methodName].append('single wavelength')
                plotData[traitName].append(str(traitNum))

        palette = ['red', 'tab:blue', 'tab:orange', 'tab:green']

        sns.boxplot(data=plotData, x=traitName, y=valueName, hue=methodName, dodge=True, palette=palette)
        #sns.stripplot(data=plotData, x=traitName, y=valueName, hue=methodName, dodge=True, jitter=True, alpha=0.6)
        #plt.axhline(y=0.0, color='black', linestyle=':')

        plt.legend([],[], frameon=False)
        plt.gcf().set_size_inches(4, 4)
        plt.tight_layout()

        

        plt.savefig('./images/encodeSim/' + simulationName + '_GSherit.pdf')
        plt.show()



#plotEncodeGS()
#quit()
