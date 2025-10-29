import numpy as np
import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
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



def loadSNPs():

    #data = np.loadtxt('./data/plant/simulations/SNP/maize.csv', dtype=str, delimiter=',')
    data = np.loadtxt('./data/plant/simulations/SNP/sorgum.csv', dtype=str, delimiter=',')

    

    SNP_names = data[1:, 1]
    plant_names = data[0, 6:]

    SNPs = data[1:, 6:]
    for a in range(SNP_names.shape[0]):
        SNP_names[a] = SNP_names[a].replace('"', '')
    for a in range(plant_names.shape[0]):
        plant_names[a] = plant_names[a].replace('"', '')

    return SNPs, plant_names, SNP_names

def saveRelevantSNPs():

    #X = np.loadtxt('./data/plant/simulations/sim10/Simulated_Data_10_Reps_Herit_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1.txt', dtype=str)
    #X = np.loadtxt('./data/plant/simulations/simPhen/sim12/Simulated_Data_10_Reps_Herit_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1.txt', dtype=str)
    #X = np.loadtxt('./data/plant/simulations/simPhen/sim15/Simulated_Data_10_Reps_Herit_0.1...0.1.txt', dtype=str)

    dataFolder = './data/plant/simulations/simPhen/uncor1'
    dataFile = dataFolder + '/'
    files1 = os.listdir(dataFolder)
    for file1 in files1:
        if 'Simulated_Data_' in file1:
            dataFile = dataFile + file1
    
    X = np.loadtxt(dataFile, dtype=str)


    names = X[1:, 0]

    SNPs, plant_names, SNP_names = loadSNPs()

    SNPinfo = np.loadtxt('./data/plant/simulations/simPhen/uncor1/Additive_Selected_QTNs.txt', dtype=str)
    #SNPinfo = np.loadtxt('./data/plant/simulations/simPhen/uncor1/Dominance_Selected_QTNs.txt', dtype=str)

    

    #print (SNPinfo)

    SNPused = SNPinfo[1:, 1]

    indices = np.zeros(SNPused.shape[0], dtype=int)
    for a in range(SNPused.shape[0]):
        arg1 = np.argwhere(SNP_names == SNPused[a])[0, 0]
        indices[a] = arg1 

    print (indices)
    quit()
    SNPs = SNPs[indices].T
    
    SNPs_paste = np.zeros((names.shape[0], SNPs.shape[1]), dtype=int)
    for a in range(names.shape[0]):
        name1 = names[a]
        arg1 = np.argwhere(plant_names == name1)[0, 0]
        SNPs_paste[a] = SNPs[arg1]

    print (SNPs_paste.shape)
    print (scipy.stats.pearsonr(SNPs_paste[:, 0], SNPs_paste[:, 1]))
    quit()

    np.savez_compressed('./data/plant/simulations/SNP/sim15_importantSNP.npz', SNPs_paste)

#saveRelevantSNPs()
#quit()


def savePCA():

    X = np.loadtxt('./data/plant/simulations/sim4/Simulated_Data_10_Reps_Herit_0.4_0.4_0.4_0.4.txt', dtype=str)

    names = X[1:, 0]

    SNPs, plant_names, SNP_names = loadSNPs()

    for a in range(SNP_names.shape[0]):
        SNP_names[a] = SNP_names[a].replace('"', '')
    for a in range(plant_names.shape[0]):
        plant_names[a] = plant_names[a].replace('"', '')


    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    PCAvals = pca.fit_transform(SNPs)


    PCApaste = np.zeros((names.shape[0], PCAvals.shape[1]), dtype=int)
    for a in range(names.shape[0]):
        name1 = names[a]
        arg1 = np.argwhere(plant_names == name1)[0, 0]
        PCApaste[a] = PCAvals[arg1]

    np.savez_compressed('./data/plant/simulations/SNP/maize_PCA.npz', PCApaste)



#savePCA()
#quit()


def OLD_remove_projections(v, u):

    for a in range(u.shape[1]):
        return v - ((np.dot(v, u[:, a]) / np.dot(u[:, a], u[:, a])) * u[:, a])
    





def encodeSim():

    
    #saveFolder =  paste('./data/plant/simulations/simPhen/uncorSims/', toString(a), sep="") 
    #saveFolder =  paste('./data/plant/simulations/simPhen/random3SNP/', toString(a), sep="") 
    #saveFolder =  paste('./data/plant/simulations/simPhen/random100SNP/', toString(a), sep="") 

    np.random.seed(0)

    modelName = './data/plant/models/autoencode_2.pt'
    model = torch.load(modelName)
    X = loadnpz('./data/plant/processed/sor/X.npz')
    X = torch.tensor(X).float()
    Y = model.encode(X)
    Y = Y.data.numpy()
    Y_copy = np.copy(Y)

    means1 = np.mean(Y, axis=0)
    Y = Y - means1.reshape((1, -1))
    stds1 = np.mean(Y ** 2, axis=0) ** 0.5

    #print (stds1)

    #simulationName = 'uncorSims'
    #simulationName = 'random3SNP'
    #simulationName = 'random100SNP'
    #simulationName = 'seperate100SNP'
    simulationName = 'sameHeritSep100'


    for simIndex in range(10):

        #dataFolder = './data/plant/simulations/simPhen/simA3'
        #dataFolder = './data/plant/simulations/simPhen/uncor4'
        #dataFolder = './data/plant/simulations/simPhen/cor1'

        dataFolder =  './data/plant/simulations/simPhen/' + simulationName + '/' + str(simIndex) 

        dataFile = dataFolder + '/'
        files1 = os.listdir(dataFolder)
        for file1 in files1:
            if 'Simulated_Data_' in file1:
                dataFile = dataFile + file1
        
        X = np.loadtxt(dataFile, dtype=str)

        names = X[1:, 0]

        print (np.unique(names).shape)
        quit()
        X = X[1:, 1:-1].astype(float)

        X = X - np.mean(X, axis=0).reshape((1, -1))
        X = X / (np.mean(X**2, axis=0).reshape((1, -1)) ** 0.5)


        X_latent = np.random.normal(size= X.shape[0]*5 ).reshape((  X.shape[0], 5 ))
        perm1 = np.random.permutation(5)
        perm1 = perm1[:X.shape[1]]
        X_latent[:, perm1] = X


        X_latent = X_latent * stds1.reshape((1, -1))
        X_latent = X_latent + means1.reshape((1, -1))

        #X = X * stds1

        if False:
            for a in range(5):

                print (a)
                print (np.mean(X_latent[:, a]))
                print (np.mean(Y_copy[:, a]))
                print (np.mean(X_latent[:, a] ** 2))
                print (np.mean(Y_copy[:, a] ** 2))

                plt.hist(X_latent[:, a], bins=100, alpha=0.5)
                plt.hist(Y_copy[:, a], bins=100, alpha=0.5)
                plt.show()

            print (X.shape)
            quit()


        modelName = './data/plant/models/autoencode_2.pt'
        model = torch.load(modelName)

        X_final = model.decode(torch.tensor(X_latent[:, :5]).float())
        X_final = X_final.data.numpy()
        



        #folder1 = './data/plant/simulations/encoded/simA3/'
        #folder1 = './data/plant/simulations/encoded/uncor4/'

        folder0 = './data/plant/simulations/encoded/' + simulationName + '/'
        folder1 = folder0 + str(simIndex)
        try:
            command1 = 'mkdir ' + folder1
            os.system(command1)
        except:
            True 

        folder1 = folder1 + '/'

        #print (folder1)
        #quit()
        quit()

        np.savez_compressed(folder1 + 'names.npz', names)
        np.savez_compressed(folder1 + 'X.npz', X_final)
        np.savez_compressed(folder1 + 'LatentPermuation.npz', perm1)


#encodeSim()
#quit()
#   



def OLD_trainModel(X, names, envirement, trainTest2, modelName, Niter = 10000, doPrint=True, regScale=1e-8, learningRate=1e-4, noiseLevel=0.1, Nphen=1, NphenStart=0):

    X = torch.tensor(X).float()


    numWavelengths = X.shape[1]
    model = simpleModel(numWavelengths, Nphen)
    #model = neuralNet(numWavelengths, Nphen)
    #model = convModel2(numWavelengths, Nphen)

    



    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate, betas=(0.9, 0.99))

    if not isinstance(corVar, bool):
        corVar = corVar.reshape((-1, 1))

    #Niter = 10000

    error_train = []
    error_test = []

    for a in range(Niter):

        rand1 = torch.randn(size=X.shape)
        Y = model(X  + (rand1 * noiseLevel) )
        
        
        Y = normalizeIndependent(Y, includeRandom=True)
        

        heritability_train = cheapHeritability(Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0].copy() )

        heritability_test = cheapHeritability(Y[trainTest2 == 1], names[trainTest2 == 1], envirement[trainTest2 == 1].copy()  )


        loss = -1 * torch.mean(heritability_train)


        reg1 = 0
        for param in model.parameters():
            reg1 += torch.mean(torch.abs(param) ** 2)
            #reg1 += torch.mean(torch.abs(param))
        reg1 = reg1 ** 0.5


        if doPrint:
            if a % 100 == 0:
                if 1 in trainTest2:
                    print ("")
                    print (heritability_train.data.numpy())
                    print (heritability_test.data.numpy())
                else:
                    print (heritability_train)

                torch.save(model, modelName)

        regLoss = torch.mean(reg1 / Y_abs)
        loss = loss + (regLoss * regScale)



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        

        



    if 1 in trainTest2:
        #print ("E1")
        if not isinstance(corVar, bool):
            return (coherit_train, coherit_test)
        else:
            return (heritability_train, heritability_test)
    else:
        #print ("E2")
        #print (heritability_train)
        return (heritability_train)
    



def trainModel(model, X, names, envirement, trainTest2, modelName, Niter = 10000, doPrint=True, regScale=1e-8, learningRate=1e-4, NphenStart=0, Nphen=1,  noiseLevel=0.1):

    

    X = torch.tensor(X).float()


    numWavelengths = X.shape[1]
    
    argTrain = np.argwhere(trainTest2 == 0)[:, 0]


    for phenNow in range(NphenStart, Nphen):

        print ('X shape', X.shape)


        if phenNow > 0:
            subset1 = np.arange(phenNow)

            Y_background = model(X, subset1)
            Y_background = Y_background.detach()
            Y_background = normalizeIndependent(Y_background)


        else:
            Y_background = torch.zeros((X.shape[0]), 0)

        subset_phen = np.zeros(1, dtype=int) + phenNow

        optimizer = torch.optim.RMSprop(model.parameters(), lr = learningRate)
        

        for a in range(Niter):
            
            X_train = X[trainTest2 == 0]
            
            rand1 = torch.randn(X_train.shape) * noiseLevel
            X_train = X_train + rand1
        
            
            Y = model(X_train, subset_phen)
            


            Y = removeIndependence(Y, Y_background[trainTest2 == 0])

            
            heritability_now = cheapHeritability(Y, names[trainTest2 == 0], envirement[trainTest2 == 0])#, device=mps_device )
            loss = -1 * torch.mean(heritability_now)
            

            if a % 100 == 0:
                
                print ('iter:', a)

                with torch.no_grad():
                    Y = model(X, subset_phen)
                    Y = removeIndependence(Y, Y_background)

                    heritability_train = cheapHeritability(Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0] )
                    if 1 in trainTest2:
                        heritability_test = cheapHeritability(Y[trainTest2 == 1], names[trainTest2 == 1], envirement[trainTest2 == 1] )
                

                print ('subset_phen', subset_phen)
                print (heritability_train.data.numpy())
                if 1 in trainTest2:
                    print (heritability_test.data.numpy())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if a % 10 == 0:
                torch.save(model, modelName)
        
    



def trainStandardSim():


    #simulationName = 'uncorSims'
    #simulationName = 'random3SNP'
    #simulationName = 'random100SNP'

    np.random.seed(0)

    #simulationNames = ['uncorSims']#:, 'random3SNP', 'random100SNP']
    #simulationNames = ['random3SNP']#:, 'random3SNP', 'random100SNP']
    #simulationNames = ['random100SNP']
    #simulationNames = ['seperate100SNP']
    simulationNames = ['sameHeritSep100']

    for simulationName in simulationNames:

        for a in range(10):
            print ('')
        print ('simulationName: ', simulationName)
        for a in range(10):
            print ('')


        folder0 = './data/plant/simulations/encoded/' + simulationName + '/'

        #for simIndex in range(0, 10):
        for simIndex in range(0, 10):

            for a in range(3):
                print ('')
            print ('simIndex: ', simIndex)
            for a in range(3):
                print ('')


            folder1 = folder0 + str(simIndex)
    

            names = loadnpz(folder1 + '/names.npz')
            X = loadnpz(folder1 + '/X.npz')
            
            envirement = np.zeros((names.shape[0], 0))
            name_unique, name_inverse, name_counts = np.unique(names, return_inverse=True, return_counts=True)

            

            #np.random.seed(0)
            trainTest2 = np.random.randint(3, size=name_unique.shape[0])
            trainTest2 = trainTest2[name_inverse]


            trainTest3 = np.copy(trainTest2)
            trainTest3[trainTest3 == 0] = 100
            trainTest3[trainTest3!=100] = 0
            trainTest3[trainTest3 == 100] = 1




            trainSplitFile = folder1 + '/trainSplit.npz'
            np.savez_compressed(trainSplitFile, trainTest3)
            
            
            Nphen = 10
            #modelName = folder1 + '/model_4.pt'
            #modelName = folder1 + '/model_3.pt'
            #modelName = folder1 + '/model_5.pt'
            #modelName = folder1 + '/model_8.pt'
            #modelName = folder1 + '/model_2.pt'
            #modelName = folder1 + '/convModel.pt'

            #modelName = folder1 + '/model_A2.pt'
            #modelName = folder1 + '/model_A1.pt'
            modelName = folder1 + '/model_A1.pt'

            #modelName = folder1 + '/model_conv1.pt'


            args = [X.shape[1], 1]
            #model = torch.load(folder1 + '/model_A1.pt')
            model = multiConv(Nphen, args, simpleModel)
            #model = multiConv(Nphen, args, convModel)
            


            regScale = 0.0

            
            noiseLevel = 0.02 #Good

            #noiseLevel = 0.1 #Noisy


            #learningRate=3e-6
            learningRate=1e-5 #Good
            #learningRate=1e-3
            #learningRate=1e-4
            
            #Niter = 10000
            Niter = 10000
            Nphen = 5
            NphenStart = 0
            trainModel(model, X, names, envirement, trainTest3, modelName, Niter=Niter, doPrint=True, regScale=regScale, NphenStart=NphenStart, Nphen=Nphen, learningRate=learningRate, noiseLevel=noiseLevel)
            quit()


#trainStandardSim()
#quit()



def OLD_evaluateHerits():
    

    np.random.seed(0)

    #simulationNames = ['uncorSims', 'random3SNP', 'random100SNP']

    simulationNames = ['uncorSims']
    for simulationName in simulationNames:

        
        print ('simulationName: ', simulationName)


        folder0 = './data/plant/simulations/encoded/' + simulationName + '/'

        for simIndex in range(0, 5):

            for a in range(3):
                print ('')
            print ('simIndex: ', simIndex)
            for a in range(3):
                print ('')


            dataFolder =  './data/plant/simulations/simPhen/' + simulationName + '/' + str(simIndex) 
            dataFile = dataFolder + '/'
            files1 = os.listdir(dataFolder)
            for file1 in files1:
                if 'Simulated_Data_' in file1:
                    dataFile = dataFile + file1
            X_original = np.loadtxt(dataFile, dtype=str)
            #names = X[1:, 0]
            X_original = X_original[1:, 1:-1].astype(float)


            folder1 = folder0 + str(simIndex)
    

            names = loadnpz(folder1 + '/names.npz')
            X = loadnpz(folder1 + '/X.npz')
            X = torch.tensor(X).float()
            
            envirement = np.zeros((names.shape[0], 0))

            trainSplitFile = folder1 + '/trainSplit.npz'
            trainTest2 = loadnpz(trainSplitFile)
            
            
            #Nphen = 10
            #modelName = folder1 + '/model_2.pt'
            #modelName = folder1 + '/model_3.pt'
            modelName = folder1 + '/model_A2.pt'
            #modelName = folder1 + '/convModel.pt'
            model = torch.load(modelName)
            #print (modelName)

            subset1 = np.arange(3)

            Y = model(X, subset1)
            Y = normalizeIndependent(Y)
            Y_np = Y.data.numpy()

            X_original_proj = find_projections(X_original,  Y_np[:, :X_original.shape[1]] )

            if simIndex == 0:
                corMatrix = np.zeros((  10,  X_original.shape[1], Y_np.shape[1] ))
                corMatrix_proj = np.zeros((  10,  X_original.shape[1], Y_np.shape[1] ))
                heritMatrix = np.zeros( (10, 2, Y_np.shape[1]) )

            for a in range(X_original.shape[1]):
                for b in range(Y_np.shape[1]):
                    cor1 = scipy.stats.pearsonr(Y_np[:, b],  X_original[:, a] )
                    cor2 = scipy.stats.pearsonr(Y_np[:, b],  X_original_proj[:, a] )
                    corMatrix[simIndex, a, b] = cor1[0]
                    corMatrix_proj[simIndex, a, b] = cor2[0]

            

            #print (corMatrix)

            heritability_train = cheapHeritability(Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0] )
            heritability_test = cheapHeritability(Y[trainTest2 == 1], names[trainTest2 == 1], envirement[trainTest2 == 1] )

            heritMatrix[simIndex, 0] = heritability_train.data.numpy()
            heritMatrix[simIndex, 1] = heritability_test.data.numpy()
            

        np.savez_compressed(folder0 + 'A2_corMatrix.npz', corMatrix)
        np.savez_compressed(folder0 + 'A2_corMatrixProj.npz', corMatrix_proj)
        np.savez_compressed(folder0 + 'A2_heritMatrix.npz', heritMatrix)





def savePreds():

    np.random.seed(0)

    #simulationNames = ['uncorSims', 'random3SNP', 'random100SNP']

    #simulationNames = ['uncorSims']
    #simulationNames = ['seperate100SNP']
    simulationNames = ['random100SNP']
    #simulationNames = ['sameHeritSep100']
    
    

    #simulationNames = ['uncorSims', 'random100SNP']

    #methodNames = ['H2Opt', 'maxWave', 'PCA']
    #methodNames = ['H2Opt']
    methodNames = ['maxWave', 'PCA']

    #methodNames = ['H2Opt-Conv']
    #methodNames = ['factorAnalysis']

    

    synthUsed = 5


    for simulationName in simulationNames:

        for methodName in methodNames:

            

        
        
            print ('simulationName: ', simulationName)


            folder0 = './data/plant/simulations/encoded/' + simulationName + '/'

            for simIndex in range(0, 10):
                #for simIndex in range(0, 4):

                for a in range(3):
                    print ('')
                print ('simIndex: ', simIndex)
                for a in range(3):
                    print ('')

                folder1 = folder0 + str(simIndex)
        

                names = loadnpz(folder1 + '/names.npz')
                print (names.shape)
                quit()
                X = loadnpz(folder1 + '/X.npz')
                #X = torch.tensor(X).float()

                envirement = np.zeros((names.shape[0], 0))
                

                trainSplitFile = folder1 + '/trainSplit.npz'
                trainTest2 = loadnpz(trainSplitFile)

                if 'H2Opt' in methodName:
                    if simulationName == 'random3SNP':
                        modelName = folder1 + '/model_A1.pt'
                        #modelName = folder1 + '/model_A1_noisy.pt'
                    elif simulationName == 'random100SNP':
                        modelName = folder1 + '/model_A1.pt'
                    elif simulationName == 'seperate100SNP':
                        modelName = folder1 + '/model_A1.pt'
                    elif simulationName == 'sameHeritSep100':
                        modelName = folder1 + '/model_A1.pt'
                    else:
                        modelName = folder1 + '/model_A2.pt'

                    if methodName == 'H2Opt-Conv':
                        modelName = folder1 + '/model_conv1.pt'

                    model = torch.load(modelName)
                    subset1 = np.arange(5)

                    Y = model(torch.tensor(X).float(), subset1)
                    Y = normalizeIndependent(Y)
                    Y_np = Y.data.numpy()

                if methodName == 'PCA':
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=synthUsed, random_state=0)
                    pca.fit(X)
                    Y_np = pca.transform(X)

                if methodName == 'factorAnalysis':
                    from sklearn.decomposition import FactorAnalysis
                    transformer = FactorAnalysis(n_components=synthUsed, random_state=0)
                    Y_np = transformer.fit_transform(X)

                if methodName == 'maxWave':

                    #print ("A")
                    #quit()
                    X_copy = np.copy(X[trainTest2 == 0])
                    X_copy = X_copy - np.mean(X_copy, axis=0).reshape((1, -1))
                    argBest_list = []
                    for a in range(synthUsed):
                        heritability_wave = cheapHeritability(torch.tensor(X_copy).float() , names[trainTest2 == 0], envirement[trainTest2 == 0] )
                        heritability_wave = heritability_wave.data.numpy()
                        if a > 0:
                            heritability_wave[np.array(argBest_list)] = 0
                        #print (np.max(heritability_wave))
                        argBest = np.argmax(heritability_wave)
                        argBest_list.append(argBest)
                        X_copy = remove_projections(X_copy, np.copy(X_copy[:, argBest:argBest+1]))
                        X_copy[:, argBest] = 1

                        #plt.plot(X_copy[:, 0])
                        #plt.plot(X_copy[:, 10])
                        #plt.show()
                        #quit()


                    argBest_list = np.array(argBest_list)
                    Y = np.copy(X[:, argBest_list])

                    Y = normalizeIndependent( torch.tensor(Y).float() )
                    Y_np = Y.data.numpy()


                if methodName == 'factorAnalysis':
                    np.savez_compressed(folder1 + '/factorAnalysis_predValues.npz', Y_np)
                if methodName == 'H2Opt-Conv':
                    np.savez_compressed(folder1 + '/H2Opt-Conv_predValues.npz', Y_np)
                if methodName == 'H2Opt':
                    np.savez_compressed(folder1 + '/H2Opt_predValues.npz', Y_np)
                if methodName == 'PCA':
                    np.savez_compressed(folder1 + '/PCA_predValues.npz', Y_np)
                if methodName == 'maxWave':
                    np.savez_compressed(folder1 + '/maxWave_predValues.npz', Y_np)




#savePreds()
#quit()


def evaluateSimSNV():

    np.random.seed(0)

    #simulationNames = ['uncorSims', 'random3SNP', 'random100SNP']

    #simulationNames = ['uncorSims']
    #simulationNames = ['seperate100SNP']
    #simulationNames = ['random100SNP']
    simulationNames = ['sameHeritSep100']

    methodNames = ['H2Opt', 'maxWave', 'PCA']
    #methodNames = ['H2Opt']
    #methodNames = ['H2Opt-Conv']
    #methodNames = ['factorAnalysis']

    

    

    synthUsed = 5


    for simulationName in simulationNames:

        for methodName in methodNames:

            

        
        
            print ('simulationName: ', simulationName)


            folder0 = './data/plant/simulations/encoded/' + simulationName + '/'

            for simIndex in range(0, 10):
                #for simIndex in range(0, 2):

                folder1 = folder0 + str(simIndex) 

                for a in range(3):
                    print ('')
                print ('simIndex: ', simIndex)
                for a in range(3):
                    print ('')


                dataFolder =  './data/plant/simulations/simPhen/' + simulationName + '/' + str(simIndex) 
                dataFile = dataFolder + '/'
                files1 = os.listdir(dataFolder)
                for file1 in files1:
                    if 'Simulated_Data_' in file1:
                        dataFile = dataFile + file1
                X_original = np.loadtxt(dataFile, dtype=str)
                #print (X_original[:10])
                #quit()
                X_original = X_original[1:, 1:-1].astype(float)


                

                #print (X_original.shape)
                #quit()


                
        

                names = loadnpz(folder1 + '/names.npz')
                X = loadnpz(folder1 + '/X.npz')
                #X = torch.tensor(X).float()

                envirement = np.zeros((names.shape[0], 0))
                

                trainSplitFile = folder1 + '/trainSplit.npz'
                trainTest2 = loadnpz(trainSplitFile)

                if methodName == 'factorAnalysis':
                    Y_np = loadnpz(folder1 + '/factorAnalysis_predValues.npz')
                if methodName == 'H2Opt-Conv':
                    Y_np = loadnpz(folder1 + '/H2Opt-Conv_predValues.npz')
                if methodName == 'H2Opt':
                    Y_np = loadnpz(folder1 + '/H2Opt_predValues.npz')
                if methodName == 'PCA':
                    Y_np = loadnpz(folder1 + '/PCA_predValues.npz')
                if methodName == 'maxWave':
                    Y_np = loadnpz(folder1 + '/maxWave_predValues.npz')
                Y = torch.tensor(Y_np).float()


                X_original_proj = find_projections(X_original,  Y_np[:, :X_original.shape[1]] )
                #X_original_proj = find_projections(X_original,  Y_np[:, :] )

                Y_proj = find_projections(Y_np[:, :X_original.shape[1]], X_original ) #New reverse projections useful for grant proposal 



                if simIndex == 0:
                    corMatrix = np.zeros((  10,  X_original.shape[1], Y_np.shape[1] ))
                    corMatrix_proj = np.zeros((  10,  X_original.shape[1], Y_np.shape[1] ))
                    corMatrix_proj_mod = np.zeros((  10,  X_original.shape[1], Y_np.shape[1] ))
                    heritMatrix = np.zeros( (10, 2, Y_np.shape[1]) )
                    heritMatrix_true = np.zeros( (10, 2, Y_np.shape[1]) )
                

                for a in range(X_original.shape[1]):
                    for b in range(Y_np.shape[1]):
                        cor1 = scipy.stats.pearsonr(Y_np[:, b],  X_original[:, a] )
                        corMatrix[simIndex, a, b] = cor1[0]

                        if b < X_original.shape[1]:
                            cor2 = scipy.stats.pearsonr(X_original_proj[:, b],  X_original[:, a] )
                            corMatrix_proj[simIndex, a, b] = cor2[0]

                            cor_mod = scipy.stats.pearsonr(Y_proj[:, b],  Y_np[:, a] )
                            corMatrix_proj_mod[simIndex, a, b] = cor_mod[0]
                        


                #print (corMatrix[simIndex])
                #print (corMatrix_proj[simIndex, np.arange(3), np.arange(3)])
                #quit()

                #for a in range(3):
                #    plt.scatter(X_original[:, a], Y_np[:, a])
                #    plt.show()

                heritability_train = cheapHeritability(Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0] )
                heritability_test = cheapHeritability(Y[trainTest2 == 1], names[trainTest2 == 1], envirement[trainTest2 == 1] )

                if methodName == 'H2Opt':
                    heritability_train_true = cheapHeritability(torch.tensor(X_original[trainTest2 == 0]).float() , names[trainTest2 == 0], envirement[trainTest2 == 0] )
                    heritability_test_true = cheapHeritability(torch.tensor(X_original[trainTest2 == 1]).float() , names[trainTest2 == 1], envirement[trainTest2 == 1] )
                    heritMatrix_true[simIndex, 0, :heritability_train_true.shape[0]] = heritability_train_true.data.numpy()
                    heritMatrix_true[simIndex, 1, :heritability_test_true.shape[0]] = heritability_test_true.data.numpy()


                
                heritMatrix[simIndex, 0] = heritability_train.data.numpy()
                heritMatrix[simIndex, 1] = heritability_test.data.numpy()

            #print (methodName)

            #print (heritMatrix[2])
            #quit()  

            if True:
                #factorAnalysis
                if methodName == 'factorAnalysis':
                    #np.savez_compressed(folder0 + 'factorAnalysis_corMatrix.npz', corMatrix)
                    #np.savez_compressed(folder0 + 'factorAnalysis_corMatrixProj.npz', corMatrix_proj)
                    np.savez_compressed(folder0 + 'factorAnalysis_corMatrixProj_mod.npz', corMatrix_proj_mod)
                    #np.savez_compressed(folder0 + 'factorAnalysis_heritMatrix.npz', heritMatrix)

                
                if methodName == 'H2Opt-Conv':
                    #np.savez_compressed(folder0 + 'H2Opt-Conv_corMatrix.npz', corMatrix)
                    #np.savez_compressed(folder0 + 'H2Opt-Conv_corMatrixProj.npz', corMatrix_proj)
                    #np.savez_compressed(folder0 + 'H2Opt-Conv_heritMatrix.npz', heritMatrix)
                    np.savez_compressed(folder0 + 'H2Opt-Conv_corMatrixProj_mod.npz', corMatrix_proj_mod)


                
                if methodName == 'H2Opt':
                    #np.savez_compressed(folder0 + 'A2_corMatrix.npz', corMatrix)
                    #np.savez_compressed(folder0 + 'A2_corMatrixProj.npz', corMatrix_proj)
                    #np.savez_compressed(folder0 + 'A2_heritMatrix.npz', heritMatrix)

                    #np.savez_compressed(folder0 + 'true_heritMatrix.npz', heritMatrix_true)

                    np.savez_compressed(folder0 + 'A2_corMatrixProj_mod.npz', corMatrix_proj_mod)

                if methodName == 'PCA':
                    #np.savez_compressed(folder0 + 'PCA_corMatrix.npz', corMatrix)
                    #np.savez_compressed(folder0 + 'PCA_corMatrixProj.npz', corMatrix_proj)
                    #np.savez_compressed(folder0 + 'PCA_heritMatrix.npz', heritMatrix)

                    np.savez_compressed(folder0 + 'PCA_corMatrixProj_mod.npz', corMatrix_proj_mod)


                if methodName == 'maxWave':
                    #np.savez_compressed(folder0 + 'maxWave_corMatrix.npz', corMatrix)
                    #np.savez_compressed(folder0 + 'maxWave_corMatrixProj.npz', corMatrix_proj)
                    #np.savez_compressed(folder0 + 'maxWave_heritMatrix.npz', heritMatrix)

                    np.savez_compressed(folder0 + 'maxWave_corMatrixProj_mod.npz', corMatrix_proj)







evaluateSimSNV()
quit()



def trainMultiplePhenotypes():

    #X = np.loadtxt('./data/plant/simulations/sim3/Simulated_Data_10_Reps_Herit_0.2_0.4_0.4.txt', dtype=str)
    #X = np.loadtxt('./data/plant/simulations/sim4/Simulated_Data_10_Reps_Herit_0.4_0.4_0.4_0.4.txt', dtype=str)
    #X = np.loadtxt('./data/plant/simulations/sim6/Simulated_Data_10_Reps_Herit_0.5_0.4_0.3_0.2.txt', dtype=str)
    #X = np.loadtxt('./data/plant/simulations/sim7/Simulated_Data_10_Reps_Herit_0.4_0.4_0.4_0.4.txt', dtype=str)
    #X = np.loadtxt('./data/plant/simulations/sim9/Simulated_Data_10_Reps_Herit_0.4_0.4_0.4_0.4_0.4_0.4_0.4_0.4.txt', dtype=str)
    #X = np.loadtxt('./data/plant/simulations/simPhen/sim13/Simulated_Data_10_Reps_Herit_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1.txt', dtype=str)
    #X = np.loadtxt('./data/plant/simulations/simPhen/sim16/Simulated_Data_10_Reps_Herit_0.1...0.1.txt', dtype=str)

    #./data/plant/simulations/uncor1


    if False:
        #dataFolder = './data/plant/simulations/simPhen/simA6'
        #dataFolder = './data/plant/simulations/simPhen/cor1'
        dataFolder = './data/plant/simulations/simPhen/uncor3'
        dataFile = dataFolder + '/'
        files1 = os.listdir(dataFolder)
        for file1 in files1:
            if 'Simulated_Data_' in file1:
                dataFile = dataFile + file1
        
        X = np.loadtxt(dataFile, dtype=str)
        

        
        names = X[1:, 0]
        X = X[1:, 1:-1].astype(float)

    else:

        folder1 = './data/plant/simulations/encoded/uncor3/'
        #folder1 = './data/plant/simulations/encoded/simA3/'

        names = loadnpz(folder1 + 'names.npz')
        X = loadnpz(folder1 + 'X.npz')


    if False:
        X = X - np.mean(X, axis=0).reshape((1,-1))
        X = np.concatenate((X, np.abs(X)), axis=1)
    

    
    envirement = np.zeros((names.shape[0], 0))
    name_unique, name_inverse, name_counts = np.unique(names, return_inverse=True, return_counts=True)

    

    np.random.seed(0)
    #np.random.seed(1)
    trainTest2 = np.random.randint(3, size=name_unique.shape[0])
    trainTest2 = trainTest2[name_inverse]


    trainTest3 = np.copy(trainTest2)
    trainTest3[trainTest3 == 0] = 100
    trainTest3[trainTest3!=100] = 0
    trainTest3[trainTest3 == 100] = 1
    
    
    Nphen = 10
    #modelName = './data/plant/simulations/models/conv_9_reg2.pt'
    #modelName = './data/plant/simulations/models/conv_9_fromSquare.pt'
    #modelName = './data/plant/simulations/models/linear_9_mod.pt'
    #modelName = './data/plant/simulations/models/conv_9_modNoise_overfit.pt'
    #modelName = './data/plant/simulations/models/linear_9_modNoise_overfit.pt'

    #modelName = './data/plant/simulations/models/linear_16.pt'
    #modelName = './data/plant/simulations/models/conv_16.pt'
    #modelName = './data/plant/simulations/models/simA6_linear.pt'
    #modelName = './data/plant/simulations/models/cor1_linear_SVD.pt'
    modelName = './data/plant/simulations/models/uncor3_encode.pt'
    #modelName = './data/plant/simulations/models/simA4_conv2.pt'



    #regScale = 3e0
    #regScale = 1e0
    #regScale = 1e-1
    #regScale = 5e-2
    #regScale = 1e-2
    #regScale = 1e-3
    regScale = 1e-20
    
    Niter = 1000000
    #trainModel(X, names, envirement, trainTest3, modelName, Niter=Niter, doPrint=True, regScale=regScale, Nphen=Nphen, learningRate=1e-6)
    trainModel(X, names, envirement, trainTest3, modelName, Niter=Niter, doPrint=True, regScale=regScale, Nphen=Nphen, learningRate=1e-5, noiseLevel=0.2)
    #trainModel(X, names, envirement, trainTest3, modelName, Niter=Niter, doPrint=True, regScale=regScale, Nphen=Nphen, learningRate=1e-3)
    quit()



#trainMultiplePhenotypes()
#quit()


def showCoef():

    #[ 0.6015115   0.01866098 -0.00765308 -0.01051495 -0.00231613 -0.01821068 -0.00625738  0.00245286 -0.01117931  0.02155356]
    modelName = './data/plant/simulations/models/cor1_linear.pt'
    #modelName = './data/plant/simulations/models/cor1_linear.pt'
    model = torch.load(modelName)
    coef = getModelCoef(model, multi=True)
    print (coef)
    #quit()


    dataFolder = './data/plant/simulations/simPhen/cor1'
    dataFile = dataFolder + '/'
    files1 = os.listdir(dataFolder)
    for file1 in files1:
        if 'Simulated_Data_' in file1:
            dataFile = dataFile + file1
    
    X = np.loadtxt(dataFile, dtype=str)
    names = X[1:, 0]
    X = X[1:, 1:-1].astype(float)

    Y = model(torch.tensor(X).float())

    Y, trackComputation = normalizeIndependent(Y, trackCompute=True)

    corMatrix = np.zeros((3, 3))
    for a in range(X.shape[1]):
        for b in range(3):
            corMatrix[a, b] = scipy.stats.pearsonr(  X[:, a], Y[:, b].data.numpy() )[0]

    coef_indep = np.matmul( coef.T, trackComputation.data.numpy() ).T

    coef_indep = coef_indep / (np.mean(coef_indep**2, axis=1) ** 0.5).reshape((-1, 1))

    for a in range(coef_indep.shape[0]):
        coef_indep[a] = coef_indep[a] * np.sign(np.max(coef_indep[a]) + np.min(coef_indep[a]))
    for a in range(3):
        corMatrix[a] = corMatrix[a] * np.sign(np.max(corMatrix[a]) + np.min(corMatrix[a]))



    #plt.imshow(corMatrix)
    sns.heatmap(corMatrix, annot=True)
    plt.title('correlation of synthetic trait with original trait')
    plt.ylabel('Synthetic trait number')
    plt.xlabel('Original trait number')
    #plt.colorbar()
    plt.show()

    #print (coef_indep.shape)
    print (coef_indep)

    #plt.imshow(coef_indep[:3])
    sns.heatmap(coef_indep[:3], annot=True)
    plt.title('coefficients of synthetic traits (normalized)')
    plt.ylabel('Synthetic trait number')
    plt.xlabel('Original trait number')
    #plt.colorbar()
    plt.show()
    quit()

    herit = cheapHeritability(Y[:, :1], names, np.zeros((names.shape[0], 0)))
    print (herit)
    quit()

    #[ 0.56696635  0.56293535  0.01146309 -0.00122653 -0.00961547  0.00879283 -0.00483341 -0.02457267 -0.01677736 -0.00280625]


#showCoef()
#quit()


def plotHerit():


    modelName = './data/plant/simulations/models/cor1_linear.pt'
    #modelName = './data/plant/simulations/models/cor1_linear.pt'
    model = torch.load(modelName)
    coef = getModelCoef(model, multi=True)
    print (coef)
    #quit()


    dataFolder = './data/plant/simulations/simPhen/cor1'
    dataFile = dataFolder + '/'
    files1 = os.listdir(dataFolder)
    for file1 in files1:
        if 'Simulated_Data_' in file1:
            dataFile = dataFile + file1
    
    X = np.loadtxt(dataFile, dtype=str)
    names = X[1:, 0]
    name_unique, name_inverse, name_counts = np.unique(names, return_inverse=True, return_counts=True)
    X = X[1:, 1:-1].astype(float)

    Y = model(torch.tensor(X).float())

    Y, trackComputation = normalizeIndependent(Y, trackCompute=True)

    np.random.seed(0)
    trainTest2 = np.random.randint(3, size=name_unique.shape[0])
    trainTest2 = trainTest2[name_inverse]


    trainTest3 = np.copy(trainTest2)
    trainTest3[trainTest3 == 0] = 100
    trainTest3[trainTest3!=100] = 0
    trainTest3[trainTest3 == 100] = 1

    herit_train = cheapHeritability(Y[trainTest3 == 0, :3], names[trainTest3 == 0], np.zeros((names.shape[0], 0)))
    herit_test = cheapHeritability(Y[trainTest3 == 1, :3], names[trainTest3 == 1], np.zeros((names.shape[0], 0)))
    herit_train, herit_test = herit_train.data.numpy(), herit_test.data.numpy()


    traitNum = np.arange(3) + 1
    groundTruthHerit = [0.6, 0.4, 0.2]

    plt.plot(traitNum, herit_train)
    plt.plot(traitNum, herit_test)
    plt.plot(traitNum, groundTruthHerit, c='black', linestyle='dashed')
    plt.legend(['training set heritability', 'test set heritability', 'heritability of original trait'] )
    plt.show()


#plotHerit()
#quit()



def analyzeEncoded():

    #simA3
    folder1 = './data/plant/simulations/encoded/uncor3/'

    names = loadnpz(folder1 + 'names.npz')
    X = loadnpz(folder1 + 'X.npz')


    modelName = './data/plant/simulations/models/uncor3_encode.pt'
    model = torch.load(modelName)
    X = torch.tensor(X).float()
    Y = model(X)
    Y = normalizeIndependent(Y)
    Y = Y.data.numpy()

    Y = Y[:, :5]

    

    dataFolder = './data/plant/simulations/simPhen/uncor3'
    
    dataFile = dataFolder + '/'
    files1 = os.listdir(dataFolder)
    for file1 in files1:
        if 'Simulated_Data_' in file1:
            dataFile = dataFile + file1
    
    X = np.loadtxt(dataFile, dtype=str)
    

    #print (X.shape)
    #quit()
    

    names = X[1:, 0]
    X = X[1:, 1:-1].astype(float)
    
    
    X = X[:, :5]

    corMatrix = np.zeros((X.shape[1], Y.shape[1]))
    for a in range(X.shape[1]):
        for b in range(Y.shape[1]):
            cor1 = scipy.stats.pearsonr( X[:, a], Y[:, b])[0]
            corMatrix[a, b] = cor1
        

        cor1 = scipy.stats.pearsonr( X[:, a], Y[:, a])[0]
        if cor1 < 0:
            Y[:, a] = Y[:, a] * -1
        cor1 = scipy.stats.pearsonr( X[:, a], Y[:, a])[0]
        cor1_string = 'Trait ' + str(a+1) + ' Pearson correlation ' + str(cor1)[:5]
        plt.scatter( X[:, a], Y[:, a]  )
        plt.title(cor1_string)
        plt.xlabel('original trait')
        plt.ylabel('synthetic trait')
        plt.show()
    quit()

    pred1 = np.matmul(Y, corMatrix.T)

    print (Y.shape)
    print (pred1.shape)

    for a in range(pred1.shape[1]):
        cor1 =  scipy.stats.pearsonr(  pred1[:, a], X[:, a] )

        pearsonCor = str(cor1[0])[:5]


        plt.scatter( X[:, a], pred1[:, a]  )
        plt.xlabel('original trait')
        plt.ylabel('synthetic trait')
        plt.show()


    

    plt.imshow(corMatrix)
    plt.show()
    quit()


#analyzeEncoded()
#quit()



def comparePCAencode():


    folder1 = './data/plant/simulations/encoded/uncor3/'

    names = loadnpz(folder1 + 'names.npz')
    X = loadnpz(folder1 + 'X.npz')

    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    Y = pca.fit_transform(X)
    

    dataFolder = './data/plant/simulations/simPhen/uncor3'
    
    dataFile = dataFolder + '/'
    files1 = os.listdir(dataFolder)
    for file1 in files1:
        if 'Simulated_Data_' in file1:
            dataFile = dataFile + file1
    
    X = np.loadtxt(dataFile, dtype=str)
    

    

    names = X[1:, 0]
    X = X[1:, 1:-1].astype(float)
    
    
    X = X[:, :5]

    corMatrix = np.zeros((X.shape[1], Y.shape[1]))
    for a in range(X.shape[1]):
        for b in range(Y.shape[1]):
            cor1 = scipy.stats.pearsonr( X[:, a], Y[:, b])[0]
            corMatrix[a, b] = cor1

    pred1 = np.matmul(Y, corMatrix.T)

    for a in range(pred1.shape[1]):
        print (scipy.stats.pearsonr(  pred1[:, a], X[:, a] ))


    

    plt.imshow(corMatrix)
    plt.show()
    quit()



#comparePCAencode()
#quit()




def analyzeUncor():

    dataFolder = './data/plant/simulations/simPhen/cor1'
    
    dataFile = dataFolder + '/'
    files1 = os.listdir(dataFolder)
    for file1 in files1:
        if 'Simulated_Data_' in file1:
            dataFile = dataFile + file1
    
    X = np.loadtxt(dataFile, dtype=str)
    

    modelName = './data/plant/simulations/models/cor1_linear_SVD.pt'

    model = torch.load(modelName)

    names = X[1:, 0]
    X = X[1:, 1:-1].astype(float)
    X = torch.tensor(X).float()
    Y = model(X)

    #Y = normalizeIndependent(Y)
    Y, _, _ = torch.svd(Y)

    Y = Y.data.numpy()

    

    #snpUncorFile = './data/plant/SNP/simulation/uncorrelatedSNP.csv'
    snpUncorFile = './data/plant/SNP/simulation/correlatedSNP.csv'
    SNPs = np.loadtxt(snpUncorFile, delimiter=',', dtype=str )
    names_SNPs = SNPs[0, 5:]
    SNP_values = SNPs[1:, 5:].astype(int).astype(float)
    SNP_values = SNP_values.T


    Y_new = np.zeros( (names_SNPs.shape[0], Y.shape[1]) )
    for a in range(names_SNPs.shape[0]):
        name1 = names_SNPs[a]
        args1 = np.argwhere(names == name1)[:, 0]
        mean1 = np.mean(Y[args1], axis=0)
        Y_new[a] = mean1


    corMatrix = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            cor1 = scipy.stats.pearsonr( Y_new[:, a], SNP_values[:, b])[0]
            corMatrix[a, b] = cor1
    
    print (corMatrix)
    corMatrix = np.abs(corMatrix)
    plt.imshow(corMatrix)
    plt.ylabel('sythetic trait number')
    plt.xlabel('SNP number')
    plt.show()
    

#analyzeUncor()
#quit()



def testHeritSim():

    X = np.loadtxt('./data/plant/simulations/sim5/Simulated_Data_10_Reps_Herit_0.4_0.4_0.4_0.4.txt', dtype=str)

    names = X[1:, 0]
    X = X[1:, 1:-1].astype(float)

    envirement = np.zeros((names.shape[0], 0))
    name_unique, name_inverse, name_counts = np.unique(names, return_inverse=True, return_counts=True)


    X = torch.tensor(X).float()

    herit = cheapHeritability(X[:, 0:1] + X[:, 1:2], names, np.zeros((names.shape[0], 0)))
    print (herit)
    herit = cheapHeritability(X[:, 0:1] + X[:, 1:2]+ X[:, 2:3] + X[:, 3:4], names, np.zeros((names.shape[0], 0)))
    print (herit)
    herit = cheapHeritability(X[:, 2:3] + X[:, 3:4], names, np.zeros((names.shape[0], 0)))
    print (herit)
    quit()
    
    
    Nphen = 10
    modelName = './data/plant/simulations/models/linear_3.pt'
    regScale = 1e-20


#testHeritSim()
#quit()


def analyzeSNPs():

    #SNPs_paste = loadnpz('./data/plant/simulations/SNP/maize_importantSNP.npz')
    SNPs_paste = loadnpz('./data/plant/simulations/SNP/sim9_importantSNP.npz')

    #SNPs_paste[SNPs_paste == 1] = 0
    

    #modelName = './data/plant/simulations/models/linear_9.pt'
    #modelName = './data/plant/simulations/models/linear_9_fromSquare.pt'
    #modelName = './data/plant/simulations/models/conv_9_fromSquare.pt'
    #modelName = './data/plant/simulations/models/linear_9_modNoise_overfit.pt'
    #modelName = './data/plant/simulations/models/conv_9_modNoise_overfit.pt'
    #modelName = './data/plant/simulations/models/conv_12.pt'
    modelName = './data/plant/simulations/models/linear_9.pt'
    #modelName = './data/plant/simulations/models/linear_10.pt'
    #modelName = './data/plant/simulations/models/linear_9_mod.pt'
    model = torch.load(modelName)
    coef = getModelCoef(model, multi=True)
    #print (coef)
    #quit()

    #X = np.loadtxt('./data/plant/simulations/simPhen/sim12/Simulated_Data_10_Reps_Herit_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1_0.1.txt', dtype=str)
    #X = np.loadtxt('./data/plant/simulations/simPhen/sim15/Simulated_Data_10_Reps_Herit_0.1...0.1.txt', dtype=str)
    dataFolder = './data/plant/simulations/simPhen/sim9'
    dataFile = dataFolder + '/'
    files1 = os.listdir(dataFolder)
    for file1 in files1:
        if 'Simulated_Data_' in file1:
            dataFile = dataFile + file1
    
    X = np.loadtxt(dataFile, dtype=str)
    names = X[1:, 0]
    X = X[1:, 1:-1].astype(float)
    #X = X ** 2

    #X = X[:, np.array([2, 3, 4, 7])]

    #np.random.seed(0)
    #X = X + np.random.normal(size=X.size).reshape(X.shape)

    Y = model(torch.tensor(X).float())


    Y, trackComputation = normalizeIndependent(Y, trackCompute=True)

    Y = Y.data.numpy()


    

    #print (scipy.stats.pearsonr( X[:, 0], Y[:, 0] ))
    #print (scipy.stats.pearsonr( X[:, 1], Y[:, 0] ))
    #print (scipy.stats.pearsonr( X[:, 2], Y[:, 0] ))
    #print (scipy.stats.pearsonr( X[:, 3], Y[:, 0] ))

    #plt.scatter(X[:, 1], X[:, 3], c=Y[:, 0] )
    #plt.show()
    #quit()
    #print (scipy.stats.pearsonr(  X[:, -3], Y[:, 0] ))
    #print (scipy.stats.pearsonr(  SNPs_paste[:, -3], Y[:, 0] ))


    if False:
        range1 = [np.min(Y[:, 0]), np.max(Y[:, 0])]
        plt.hist(Y[  SNPs_paste[:, -3] == -1 , 0] ,range=range1, alpha=0.5 )
        plt.hist(Y[  SNPs_paste[:, -3] == 0 , 0] ,range=range1 , alpha=0.5)
        plt.hist(Y[  SNPs_paste[:, -3] == 1 , 0] ,range=range1 , alpha=0.5)
        #plt.scatter(SNPs_paste[:, 2], Y[:, 0] )
        plt.show()

    
    #quit()

    #for a in range(0, X.shape[1]):
    #    print (a)
    #    for b in range(SNPs_paste.shape[1]):
    #        #b = a
    #        print (scipy.stats.pearsonr( X[:, a], SNPs_paste[:, b]))
    #quit()

    for a in range(0, Y.shape[1]):
        print (a)
        for b in range(SNPs_paste.shape[1]):

            print (scipy.stats.pearsonr( Y[:, a], SNPs_paste[:, b]))

        if a == 5:
            quit()

    print (Y.shape)
    print (SNPs_paste.shape)


#analyzeSNPs()
#quit()




def predictPhenotypes():

    #modelName = './data/plant/simulations/models/linear_9.pt'
    #modelName = './data/plant/simulations/models/conv_9_mod.pt'
    #modelName = './data/plant/simulations/models/linear_9.pt'
    modelName = './data/plant/simulations/models/simA3_linear.pt'
    model = torch.load(modelName)

    #X = np.loadtxt('./data/plant/simulations/simPhen/sim16/Simulated_Data_10_Reps_Herit_0.1...0.1.txt', dtype=str)
    dataFolder = './data/plant/simulations/simPhen/simA3'
    dataFile = dataFolder + '/'
    files1 = os.listdir(dataFolder)
    for file1 in files1:
        if 'Simulated_Data_' in file1:
            dataFile = dataFile + file1
    
    X = np.loadtxt(dataFile, dtype=str)
    names = X[1:, 0]
    X = X[1:, 1:-1].astype(float)

    #X = X ** 2
    #X = X[:, np.array([2, 3, 4, 7])]

    Y = model(torch.tensor(X).float())

    Y = Y.data.numpy()

    print (Y.shape)

    np.savez_compressed('./data/plant/syntheticTraits/simA3_linear.npz', Y)

    np.savez_compressed('./data/plant/syntheticTraits/simA3_originalTraits.npz', X)


predictPhenotypes()
quit()