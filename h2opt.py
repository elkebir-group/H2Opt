import numpy as np
#import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#import matplotlib.pyplot as plt
import time
import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.optim import Optimizer
#import scipy
#from scipy.stats import pearsonr
#from scipy.fft import fft, ifft




def getGeneLocations(plantName):

    misc_names = ['MSI_C', 'MSI', 'miscanthus', 'Miscanthus', 'misc', 'Misc']
    sor_names =  ['sor', 'Sor', 'Sorghum', 'sorghum']
    arabi_names = ['arabi','Arabi', 'arabidopsis', 'Arabidopsis', 'arab', 'metab', 'Metab']

    if plantName in misc_names:
        geneloc_file1 = './data/plant/GWAS/genes/misc/genes_manual.bed'
    elif plantName in sor_names:
        geneloc_file1 = './data/plant/GWAS/genes/sor/genes_manual.bed'
    elif plantName in arabi_names:
        geneloc_file1 = './data/plant/GWAS/genes/arabi/genes_manual.bed'
    geneLoc = np.loadtxt(geneloc_file1, delimiter='\t', dtype=str)
    return geneLoc


def findClosestGene(geneLoc, SNPloc):

    bestGenes = []

    for snp_index in range(SNPloc.shape[0]):
        chrNow = SNPloc[snp_index, 0]
        posNow = int(SNPloc[snp_index, 2])

        argChr = np.argwhere(geneLoc[:, 0] == chrNow)[:, 0]
        geneLoc_now = geneLoc[argChr]
        diff1 = np.abs(geneLoc_now[:, 1:3].astype(int) - posNow)
        min1 = np.min(diff1, axis=0)
        #print (min1)
        diff1 = diff1[:, np.argmin(min1)]
        argGene = np.argmin(diff1)
        bestGenes.append(argChr[argGene])
        
    bestGenes = np.array(bestGenes)
    geneLoc_best = geneLoc[bestGenes]

    
    return geneLoc_best



class autoEncoder(nn.Module):
    def __init__(self, length1, Nlatent):
        super(autoEncoder, self).__init__()

        self.nonlin = torch.tanh
        #self.nonlin = nn.ReLU()

        hidden1 = 100
        #hidden1 = Nlatent


        self.linE1 = torch.nn.Linear(length1, hidden1)
        self.linE2 = torch.nn.Linear(hidden1, Nlatent)

        self.linD1 = torch.nn.Linear(Nlatent, hidden1)
        self.linD2 = torch.nn.Linear(hidden1, length1)

    def forward(self, x):

        
        x = self.encode(x)

        rand1 = torch.randn(x.shape).to(x.device)
        x = x + (0.005 * rand1)

        x = self.decode(x)


        

        return x

    def encode(self, x):

        x = self.linE1(x)
        x = self.nonlin(x)
        x = self.linE2(x)
        x = self.nonlin(x)

        return x 
    
    def decode(self, x):

        x = self.linD1(x)
        x = self.nonlin(x)
        x = self.linD2(x)

        return x


class neuralNet(nn.Module):
    def __init__(self, info):
        super(neuralNet, self).__init__()

        length1, Nphen, Nhidden = info[0], info[1], info[2]

        #self.nonlin = torch.tanh
        #self.nonlin = nn.ReLU()
        self.nonlin = nn.LeakyReLU(0.1)



        self.lin0 = torch.nn.Linear(length1, Nphen)

        #self.lin1 = torch.nn.Linear(length1, Nphen)

        self.lin1 = torch.nn.Linear(length1, Nhidden)
        self.linM = torch.nn.Linear(Nhidden, Nhidden)
        self.lin2 = torch.nn.Linear(Nhidden, Nphen)

        #self.dropout1 = nn.Dropout(0.2)



    def forward(self, x):

        shape1 = x.shape

        #x = x / torch.sum(x, axis=1).reshape((-1, 1))

        #shift1 = self.lin0(x)

        x = self.lin1(x)
        x = self.nonlin(x)


        x = self.linM(x)
        x = self.nonlin(x)

        #x = self.dropout1(x)
        x = self.lin2(x)
        #x = self.nonlin(x)
        #x = self.lin3(x)

        #x = x + shift1
        


        return x
    


class convModel(nn.Module):
    def __init__(self, info):
        super(convModel, self).__init__()


        length1, Nphen = info[0], info[1]

        #self.nonlin = torch.tanh
        self.nonlin = nn.ReLU()


        self.dropout = nn.Dropout(p=0.5)

        #self.conv1 = torch.nn.Conv1d(1, 5, 20, 20)

        self.conv1 = torch.nn.Conv1d(1, 5, 20, 5)
        self.conv2 = torch.nn.Conv1d(5, 10, 20, 5)

        #self.lin1 = torch.nn.Linear(535, 1)
        #self.lin1 = torch.nn.Linear(535, Nphen)
        #self.lin1 = torch.nn.Linear(410, Nphen)
        self.lin1 = torch.nn.Linear(820, Nphen)

        #self.lin2 = torch.nn.Linear(100, Nphen)

        #1920x535



    def forward(self, x):

        shape1 = x.shape



        x = x.reshape((x.shape[0], 1, x.shape[1]))

        #print (x.shape)
        #quit()


        #print (x.shape)
        #quit()

        x = self.conv1(x)
        #x = self.nonlin(x)
        #x = self.conv2(x)

        x = self.nonlin(x)

        x = self.conv2(x)
        x = self.nonlin(x)

        #print (x.shape)
        #quit()


        x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))

        x = self.lin1(x)

        #x = self.nonlin(x)

        #x = self.dropout(x)

        #x = self.lin2(x)

        #print (x[0])
        #x = self.lin2(x)


        return x
    




class convModel2(nn.Module):
    def __init__(self, length1, Nphen):
        super(convModel2, self).__init__()

        #self.nonlin = torch.tanh
        #self.nonlin = nn.ReLU()
        self.nonlin = nn.LeakyReLU(0.1)


        self.dropout = nn.Dropout(p=0.5)

        #self.conv1 = torch.nn.Conv1d(1, 5, 20, 20)

        self.conv1 = torch.nn.Conv1d(1, 5, 20, 5)
        self.conv2 = torch.nn.Conv1d(5, 10, 20, 5)

        #self.lin1 = torch.nn.Linear(535, 1)
        #self.lin1 = torch.nn.Linear(535, Nphen)
        #self.lin1 = torch.nn.Linear(410, Nphen)
        self.lin1 = torch.nn.Linear(820, 40)

        self.lin2 = torch.nn.Linear(40, Nphen)

        #self.lin2 = torch.nn.Linear(100, Nphen)

        #1920x535



    def forward(self, x):

        shape1 = x.shape

        x = x.reshape((x.shape[0], 1, x.shape[1]))

        #print (x.shape)
        #quit()


        #print (x.shape)
        #quit()

        x = self.conv1(x)
        #x = self.nonlin(x)
        #x = self.conv2(x)

        x = self.nonlin(x)

        x = self.conv2(x)
        x = self.nonlin(x)

        #print (x.shape)
        #quit()


        x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))

        x = self.lin1(x)

        x = self.nonlin(x)

        x = self.lin2(x)

        #x = self.nonlin(x)

        #x = self.dropout(x)

        #x = self.lin2(x)

        #print (x[0])
        #x = self.lin2(x)


        return x



class simpleModel(nn.Module):
    def __init__(self, info):
        super(simpleModel, self).__init__()

        length1, Nphen = info[0], info[1]

        #self.nonlin = torch.tanh
        self.nonlin = nn.ReLU()


        self.lin1 = torch.nn.Linear(length1, Nphen)



    def forward(self, x):

        shape1 = x.shape

        #x = x / torch.sum(x, axis=1).reshape((-1, 1))

        x = self.lin1(x)


        return x





class multiConv(nn.Module):
    def __init__(self, Nphen, args, modelUse):
        super(multiConv, self).__init__()

        self.Nphen = Nphen

        self.modelList = nn.ModuleList([  modelUse(args) for _ in range(Nphen)])

    def forward(self, x, subset1):


        predVector = torch.zeros((x.shape[0], subset1.shape[0] )).to(x.device)

        for a in range(len(subset1)):
            predVector[:, a] = self.modelList[subset1[a]](x)[:, 0]
        
        return predVector






def loadnpz(name, allow_pickle=False):

    #This simple function more easily loads in compressed numpy files.

    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data


def uniqueValMaker(X):

    _, vals1 = np.unique(X[:, 0], return_inverse=True)

    for a in range(1, X.shape[1]):

        #vals2 = np.copy(X[:, a])
        #vals2_unique, vals2 = np.unique(vals2, return_inverse=True)
        vals2_unique, vals2 = np.unique(X[:, a], return_inverse=True)

        vals1 = (vals1 * vals2_unique.shape[0]) + vals2
        _, vals1 = np.unique(vals1, return_inverse=True)

    return vals1


def fastAllArgwhere(ar):
    ar_argsort = np.argsort(ar)
    ar1 = ar[ar_argsort]
    _, indicesStart = np.unique(ar1, return_index=True)
    _, indicesEnd = np.unique(ar1[-1::-1], return_index=True) #This is probably needless and can be found from indicesStart
    indicesEnd = ar1.shape[0] - indicesEnd - 1
    return ar_argsort, indicesStart, indicesEnd



def remove_projections(v, u):

    #print ('min', np.min(np.mean(np.abs(u), axis=0)))

    for a in range(v.shape[1]):
        for b in range(u.shape[1]):
            fracPart = np.dot(u[:, b], u[:, b])
            v[:, a] =  v[:, a] - ((np.dot(v[:, a], u[:, b]) /  fracPart ) * u[:, b])
    return v


def find_projections(v, u):

    v_proj = np.zeros(v.shape)

    for a in range(v.shape[1]):
        for b in range(u.shape[1]):
            v_proj[:, a] = v_proj[:, a] + ((np.dot(v[:, a], u[:, b]) / np.dot(u[:, b], u[:, b])) * u[:, b])

    return v_proj


def getModelCoef(model, multi=False):

    a = 0
    for param in model.parameters():

        if a == 0:
            if multi:
                coef = param.data.numpy()
            else:
                coef = param[0].data.numpy()
        a += 1
    return coef

def getMultiModelCoef(model, multi=False):

    coef_list = []

    a = 0
    for param in model.parameters():


        if multi:
            if a % 2 == 0:
                coef = param[0].data.numpy()
                coef_list.append(np.copy(coef))
        else:
            if a == 0:
                coef = param[0].data.numpy()
        a += 1


    if multi:
        coef = np.array(coef_list)
    
    return coef



def batch_cheapHeritability(Y, names, envirement, returnVariance=False):


    varienceName = np.zeros(Y.shape[1])

    time1 = time.time()

    Y2 = np.copy(Y)
    
    if not returnVariance:
        Y2 = Y2 - np.mean(Y2, axis=0).reshape((1, -1))
        Y2 = Y2 / np.mean(np.abs(Y2), axis=0).reshape((1, -1))

    mean2 = np.mean(Y2, axis=0)
    varienceTotal = np.sum(  ( Y2 - mean2.reshape((1, -1)) ) ** 2  , axis=0 )




    #print (time.time() - time1)
    time1 = time.time()

    #varienceEnv = 0.0
    ar_argsort, indicesStart, indicesEnd = fastAllArgwhere(envirement)

    for a in range(indicesStart.shape[0]):
        args1 = ar_argsort[indicesStart[a]: indicesEnd[a]+1]

        mean1 = np.mean(Y2[args1], axis=0)
        #var1 = np.sum(  (Y[args1] - mean1.reshape((1, -1))  ) ** 2, axis=0 ) * args1.shape[0] / (args1.shape[0] - 1)
        #varienceEnv += var1


        Y2[args1] = Y2[args1] - mean1


    mean2 = np.mean(Y2, axis=0)
    varienceEnv = np.sum(  ( Y2 - mean2.reshape((1, -1)) ) ** 2  , axis=0 )


    #print (time.time() - time1)
    time1 = time.time()


    varienceName = 0.0
    ar_argsort, indicesStart, indicesEnd = fastAllArgwhere(names)
    for a in range(indicesStart.shape[0]):
        args1 = ar_argsort[indicesStart[a]: indicesEnd[a]+1]

        mean1 = np.mean(Y2[args1], axis=0)
        var1 = np.sum(  (Y2[args1] - mean1.reshape((1, -1))  ) ** 2, axis=0 ) * args1.shape[0] / (args1.shape[0] - 1)

        varienceName += var1
    
    


    if returnVariance:

        varienceGenetic = varienceEnv - varienceName

        varienceGenetic = varienceGenetic / Y2.shape[0]
        varienceTotal = varienceTotal / Y2.shape[0]

        return varienceGenetic, varienceTotal

    else:

        #print (varienceEnv , varienceName, varienceTotal)

        #heritability = 1 - (varienceName/varienceTotal)
        heritability = (varienceEnv - varienceName) / varienceTotal

        return heritability


def groupedVarience(Y, names):


    #Y = Y[:, 0]


    ar_argsort, indicesStart, indicesEnd = fastAllArgwhere(names)
    #print (names)


    Y_sorted = Y[ar_argsort]
    if len(Y_sorted.shape) == 2:
        #Y_sorted = torch.cat(( torch.zeros((1, Y.shape[1])), Y_sorted ))
        Y_sorted = F.pad(Y_sorted, (0, 0, 1, 0), "constant", 0) #Better than above line for GPU
    else:
        #Y_sorted = torch.cat(( torch.zeros(1), Y_sorted ))
        Y_sorted = F.pad(Y_sorted, (1, 0), "constant", 0) #Better than above line for GPU
    
    


    Y_sorted_sq = Y_sorted ** 2

    Y_sorted_cumsum = torch.cumsum(Y_sorted, dim=0)
    Y_sorted_sq_cumsum = torch.cumsum(Y_sorted_sq, dim=0)

    sumList = Y_sorted_cumsum[indicesEnd+1] - Y_sorted_cumsum[indicesStart]
    sumSqList = Y_sorted_sq_cumsum[indicesEnd+1] - Y_sorted_sq_cumsum[indicesStart]
    sizeList = (indicesEnd+1 - indicesStart)
    sizeList = torch.tensor(sizeList).float()

    sizeList = sizeList.to(Y.device)

    varList = sumSqList - ((sumList ** 2) / sizeList.reshape((-1, 1)))


    #print (sumList.shape)
    #print (sizeList.shape)
    #print (((sumList ** 2) / sizeList).shape )
    #print (varList.shape)
    #quit()

    #print (Y_sorted[1:][indicesStart[0]:indicesEnd[0]+1])
    #print (varList[0])
    #print (sizeList[0])
    #print (sumList[0] / sizeList[0])

    sizeCorrection = (sizeList / (sizeList - 1))

    #print (sizeCorrection[0])
    #quit()


    varList = varList * sizeCorrection.reshape((-1, 1))

    varience = torch.sum(varList, axis=0)

    #print (varience)
    #quit()

    return varience


def removeEnvirement(Y, envirement):

    if len(envirement.shape) == 1:
        envirement = envirement.reshape((-1, 1))

    #print (Y.shape)
    #quit()

    for a in range(envirement.shape[1]):

        ar_argsort, indicesStart, indicesEnd = fastAllArgwhere(envirement[:, a])

        #print (ar_argsort.shape)
        #quit()

        Y_sorted = Y[ar_argsort]

        #print (ar_argsort.shape)
        Y_sorted = torch.cat(( torch.zeros((1, Y.shape[1])).to( Y_sorted.device )   , Y_sorted ))

        Y_sorted_cumsum = torch.cumsum(Y_sorted, dim=0)

        #print (np.max(indicesEnd))
        #print (Y_sorted_cumsum.shape)
        sumList = Y_sorted_cumsum[indicesEnd+1] - Y_sorted_cumsum[indicesStart]

        #print (Y_sorted_cumsum[:indicesEnd[0]+2])

        #print (sumList[0])

        sizeList_np = (indicesEnd+1 - indicesStart)
        sizeList = torch.tensor(sizeList_np).float().to(Y_sorted.device)


        meanList = sumList / sizeList.reshape((-1, 1))

       

        

        #print (meanList[0])

        #print (meanList[:10])


        #meanNows = torch.zeros(ar_argsort.shape[0])
        #if False:
        #for a in range(indicesStart.shape[0]):
        #    args1 = ar_argsort[indicesStart[a]: indicesEnd[a]+1]

            #print (torch.sum(Y[args1]))

        #    mean1 = torch.mean(Y[args1])

        #    meanNows[args1] = mean1






        inverse1 = np.zeros(ar_argsort.shape[0])
        inverse1[np.cumsum(sizeList_np)[:-1]] = 1
        inverse1 = np.cumsum(inverse1)


        #print ((meanList[inverse1]  - meanNows[ar_argsort] )[:10])
        #quit()


        #Y[ar_argsort] = Y[ar_argsort] - meanList[inverse1]

        #Y = Y - meanNows
        #Y[ar_argsort] = Y[ar_argsort] - meanNows[ar_argsort]
        Y[ar_argsort] = Y[ar_argsort] - meanList[inverse1]


    return Y




def removeEnvirement_regression(Y, envirement):

    if len(envirement.shape) == 1:
        envirement = envirement.reshape((-1, 1))

    

    for a in range(envirement.shape[1]):

        envirement_now = envirement[:, a]
        envirement_now = torch.tensor(envirement_now.astype(float)).float()

        envirement_now = envirement_now - torch.mean(envirement_now)
        envirement_now = envirement_now / (torch.mean(envirement_now ** 2) ** 0.5)

        envirement_now = envirement_now.reshape((-1, 1))

        cor1 = torch.mean(Y * envirement_now, axis=0)

        

        Y = Y - (cor1.reshape(1, -1) * envirement_now)




        


    return Y



def cheapHeritability(Y, names, envirement, returnVariance=False, doMod=False):


    _, count1 = np.unique(names, return_counts=True)
    if np.min(count1) == 1:

        _, inverse1 = np.unique(names, return_inverse=True)
        count1 = count1[inverse1]
        argGood = np.argwhere(count1 >= 2)[:, 0]
        if envirement.shape[0] > 0:
            envirement_good = envirement[argGood]
        else:
            envirement_good = envirement
        return cheapHeritability(Y[argGood], names[argGood], envirement_good, returnVariance=returnVariance, doMod=doMod)




    else:

        #varienceName = 0.0

        time1 = time.time()

        #Y2 = Y
        if not returnVariance:
            Y2 = Y - torch.mean(Y, axis=0).reshape((1, -1))
            Y2 = Y2 / (torch.mean(torch.abs(Y2), axis=0)).reshape((1, -1))
        else:
            Y2 = Y


        #plt.plot(Y2[0])
        #plt.show()

        mean2 = torch.mean(Y2, axis=0)
        varienceTotal = torch.sum(  ( Y2 - mean2 ) ** 2 , axis=0)


        time1 = time.time()

        time2 = time.time()

        if envirement.shape[0] == 0:
            envirement = torch.zeros((0, 0))

        if envirement.shape[1]  == 0:
            varienceEnv = varienceTotal
        else:
            Y2 = removeEnvirement(Y2, envirement)

            #print (time.time() - time2)
            mean3 = torch.mean(Y2 , axis=0).reshape((1, -1))
            varienceEnv = torch.sum(  ( Y2 - mean3 ) ** 2 , axis=0)


        
        #_, count1 = np.unique(names, return_counts=True)
        #_, inverse1 = np.unique(names, return_inverse=True)
        #count1 = count1[inverse1]
        #varienceName = groupedVarience(Y2[count1>=2], names[count1>=2])
    
    
        varienceName = groupedVarience(Y2, names)


        if doMod:
            varienceTotal = varienceEnv


        





        #print (varienceEnv, varienceName, varienceTotal)

        #heritability = 1 - (varienceName/varienceTotal)

        #print ('')
        #print (varienceEnv[0] , varienceName[0],  varienceTotal[0])
        #quit()
        #print (varienceEnv , varienceName,  varienceTotal)

        #quit()

        heritability = (varienceEnv - varienceName) / varienceTotal

        #print (varienceEnv[0], varienceName[0], varienceTotal[0])
        #print (heritability[0])
        #quit()
        


        if returnVariance:

            varienceGenetic = varienceEnv - varienceName

            varienceGenetic = varienceGenetic / Y.shape[0]
            varienceTotal = varienceTotal / Y.shape[0]

            return varienceGenetic, varienceTotal
        else:

            return heritability



def ANOVAHeritability(Y, names, envirement, returnVariance=False, doMod=False):
    return cheapHeritability(Y, names, envirement, returnVariance=returnVariance, doMod=doMod)



def cheapHeritability_related(Y, names, envirement, returnVariance=False):


    time1 = time.time()

    #Y2 = Y
    if not returnVariance:
        Y2 = Y - torch.mean(Y)
        Y2 = Y2 / (torch.mean(torch.abs(Y2)))
    else:
        Y2 = Y


    mean2 = torch.mean(Y2, axis=0)
    varienceTotal = torch.sum(  ( Y2 - mean2 ) ** 2 , axis=0)


    time1 = time.time()

    time2 = time.time()

    if envirement.shape[1]  == 0:
        varienceEnv = varienceTotal
    else:

        #print (Y2.shape, envirement.shape)

        #Y2 = removeEnvirement_regression(Y2, envirement[:, 1:]  )

        

        Y2 = removeEnvirement_regression(Y2, envirement[:, 1:]  )
        
        mean2 = torch.mean(Y2, axis=0)
        varienceTotal = torch.sum(  ( Y2 - mean2 ) ** 2 , axis=0)

        
        Y2 = removeEnvirement(Y2, envirement[:, :1])

        mean3 = torch.mean(Y2 , axis=0)
        varienceEnv = torch.sum(  ( Y2 - mean3 ) ** 2 , axis=0)

        #perm1 = np.random.permutation(envirement.shape[1] - 1)[:100]


    

        


    varienceName = groupedVarience(Y2, names)
    #varienceEnv = groupedVarience(Y2, envirement)







    #print (varienceEnv, varienceName, varienceTotal)

    #heritability = 1 - (varienceName/varienceTotal)

    

    heritability = (varienceEnv - varienceName) / varienceTotal

    #print ('')
    #print (varienceEnv[0] , varienceName[0],  varienceTotal[0])
    #quit()

    #print (time.time() - time1)
    #quit()
    if returnVariance:

        varienceGenetic = varienceEnv - varienceName

        varienceGenetic = varienceGenetic / Y.shape[0]
        varienceTotal = varienceTotal / Y.shape[0]

        return varienceGenetic, varienceTotal
    else:

        return heritability




def normalizeIndependent(Y, trackCompute=False, includeRandom=False, cutOff=-1 ):




    if trackCompute:
        trackComputation = np.eye(Y.shape[1])
        trackComputation = torch.tensor(trackComputation).float()

    


    mean1 = torch.mean(Y, axis=0)
    Y = Y - mean1.reshape((1, -1))

    Y_copy = Y.detach().clone()

    std1 = torch.mean(Y ** 2, axis=0) ** 0.5

    if includeRandom:
        rand1 = torch.randn(Y.shape).to(Y.device)
        rand1 = rand1 * std1.reshape((1, -1))
        rand1 = rand1 * 1e-5
        Y = Y + rand1

    Y_norm = Y / std1.reshape((1, -1))

    if trackCompute:
        trackComputation = trackComputation/ std1.reshape((1, -1))
    #Y = Y / std1.reshape((1, -1))
    #Y_norm = Y
    Y = Y_norm


    

    if Y.shape[1] > 1:

        #trackComputation = np.eye(Y.shape[1])
        #trackComputation = torch.tensor(trackComputation).float()

        Y_detach = Y_norm.detach().clone()
        #cor_all = torch.matmul(Y_detach.T, Y_detach) / Y_detach.shape[0]
        for a in range(Y_detach.shape[1]):
            if a > 0:
                for b in range(a):
                    cor_now = torch.mean(Y_detach[:, a] * Y_detach[:, b])
                    Y_detach[:, a] = Y_detach[:, a] - (cor_now * Y_detach[:, b])
                    std2 = torch.mean(Y_detach[:, a] ** 2) ** 0.5
                    Y_detach[:, a] = Y_detach[:, a] / std2

                    if trackCompute:
                        trackComputation[:, a] = trackComputation[:, a] - (cor_now * trackComputation[:, b])
                        trackComputation[:, a] = trackComputation[:, a] / std2


        
        #Y_copy = torch.matmul( Y_copy, trackComputation )



        cor_all = torch.matmul(Y_detach.T, Y) / Y.shape[0]
        N = Y.shape[1]
        matrix2_0 = np.eye(N, N)
        matrix2 = np.cumsum(matrix2_0, axis=1)
        matrix2 = matrix2 - matrix2_0
        matrix2 = torch.tensor(matrix2).float()
        matrix2 = matrix2.to(Y_detach.device)


        projection = torch.matmul(Y_detach, cor_all * matrix2 )

        #print ( torch.mean(Y ** 2, axis=0) ** 0.5)

        #print (scipy.stats.pearsonr(   Y[:, 0].data.numpy(),  Y[:, 1].data.numpy() ))


        Y = Y - projection

        #print ( torch.mean(Y ** 2, axis=0) ** 0.5)
        #quit()

        



    mean1 = torch.mean(Y, axis=0)
    Y = Y - mean1.reshape((1, -1))
    std1 = torch.mean(Y ** 2, axis=0) ** 0.5

    #print (std1)
    #quit()

    Y = Y / std1.reshape((1, -1))


    if cutOff > 0:
        cutOffN = cutOff * -1
        Y[Y > cutOff] = cutOff
        Y[Y < cutOffN] = cutOffN
    
    
    if trackCompute:

        return Y, trackComputation
    else:
        return Y





def removeIndependence(Y, background):


    Y = Y - torch.mean(Y, axis=0).reshape((1, -1))

    if background.shape[1] > 0:
        
        
        background = background - torch.mean(background, axis=0).reshape((1, -1))
        background = background / (torch.mean(background**2, axis=0).reshape((1, -1)) ** 0.5)

        means1 = torch.mean(Y * background, axis=0)
        projection = torch.sum(means1.reshape((1, -1)) * background, axis=1)
        Y = Y - projection.reshape((-1, 1))

        #print (torch.mean(Y[:, 0] * background[:, 0]))
        #print (torch.mean(background, axis=0))

        #print (scipy.stats.pearsonr(   Y[:, 0].cpu().detach().data.numpy(),  background[:, 0].cpu().detach().data.numpy()  ))
        
        #quit()

    Y = Y - torch.mean(Y, axis=0).reshape((1, -1))
    Y = Y / (torch.mean(Y ** 2, axis=0).reshape((1, -1)) ** 0.5)

    return Y







def doCorelate(ar1, ar2):

    cor = torch.mean(ar1 * ar2)

    scale = (torch.mean(ar1 ** 2) ** 0.5) * (torch.mean(ar2 ** 2) ** 0.5)

    cor = cor / scale

    return cor




def makeCatBoolVariables(variables):

    def makeSingleCatBool(variable):
        X_now = np.zeros((variable.shape[0],  np.unique(variable).shape[0] ), dtype=int)
        X_now[np.arange(variable.shape[0]),  variable] = 1

        return X_now
        
    spreader1 = []
    
    for a in range(variables.shape[1]):

        X_now = makeSingleCatBool(variables[:, a])

        spreader1 = spreader1 + list(np.zeros(X_now.shape[1], dtype=int) + a )
        if a == 0:
            X_all = np.copy(X_now)
        else:
            X_all = np.concatenate((X_all, X_now), axis=1) 

    spreader1 = np.array(spreader1).astype(int)


    return X_all, spreader1



def singleUpdate( Y, prior_variances, prior_residual, Z, Z_cat, spreader ):


    Ngroups = np.unique(spreader).shape[0]

    ZTZ = torch.matmul(Z.T, Z)

    sigma = prior_variances[spreader]

    sigma_inverse = 1.0 / (sigma + 1e-5)


    #print (ZTZ.shape)
    #print (sigma_inverse.shape)

    ZTZ_sig = ZTZ + (torch.tensor(np.diag(sigma_inverse)) * prior_residual)

    ZTZ_sig_inverse = torch.linalg.inv(ZTZ_sig)

    ZTY = torch.matmul(Z.T, Y)

    optimal_coef = torch.matmul(ZTZ_sig_inverse, ZTY)


    coef_estimateVariance = ZTZ_sig_inverse * prior_residual

    coef_estimateVariance_diag = coef_estimateVariance[np.arange(coef_estimateVariance.shape[0]), np.arange(coef_estimateVariance.shape[0])]

    #ZTZ_sig_inverse_diag = ZTZ_sig_inverse[np.arange(ZTZ_sig_inverse.shape[0]), np.arange(ZTZ_sig_inverse.shape[0])]

    #coef_estimateVariance = ZTZ_sig_inverse_diag * prior_residual


    optimal_coef_grouped = torch.zeros(Ngroups)
    coef_estimateVariance_grouped = torch.zeros(Ngroups)

    for a in range(Ngroups):
        args1 = np.argwhere(spreader == a)[:, 0]

        optimal_coef_grouped[a] = torch.mean(optimal_coef[args1] ** 2)
        coef_estimateVariance_grouped[a] = torch.mean(coef_estimateVariance_diag[args1])


    optimalResiduals = Y - torch.matmul(Z, optimal_coef )

    residual_estimateVariance = torch.matmul(torch.matmul( Z,  coef_estimateVariance ), Z.T)

    residual_estimateVariance_diag = residual_estimateVariance[np.arange(residual_estimateVariance.shape[0]), np.arange(residual_estimateVariance.shape[0])]

    
    optimal_residual_grouped = torch.mean(optimalResiduals ** 2)
    estimateVariance_residual_grouped = torch.mean(residual_estimateVariance_diag)


    new_variances = optimal_coef_grouped + coef_estimateVariance_grouped

    new_residual = optimal_residual_grouped + estimateVariance_residual_grouped


    return new_variances, new_residual





def maxWaveMethod(X, synthUsed, names, envirement, trainTest2):


    X_copy = np.copy(X[trainTest2 == 0])
    X_copy = X_copy - np.mean(X_copy, axis=0).reshape((1, -1))

    #print (np.min(np.mean(np.abs(X_copy), axis=0)))
    #quit()
    argBest_list = []
    for a in range(synthUsed):
        heritability_wave = cheapHeritability(torch.tensor(X_copy).float() , names[trainTest2 == 0], envirement[trainTest2 == 0] )
        heritability_wave = heritability_wave.data.numpy()
        heritability_wave[np.isnan(heritability_wave)] = 0

        if a > 0:
            heritability_wave[np.array(argBest_list)] = 0
            
        #print (np.max(heritability_wave))
        argBest = np.argmax(heritability_wave)

        #print (heritability_wave[argBest])
        #quit()
        

        argBest_list.append(argBest)
        X_copy = remove_projections(X_copy, np.copy(X_copy[:, argBest:argBest+1]))
        
        scale1 = np.sum(np.abs(X_copy), axis=0)
        

        X_copy[:, np.isnan(scale1)] = 1
        X_copy[:, scale1 < 1e-10] = 1
        X_copy[:, argBest] = 1


        #print ('min', np.min(np.mean(np.abs(X_copy), axis=0)))

        #plt.plot(X_copy[:, 0])
        #plt.plot(X_copy[:, 10])
        #plt.show()
        #quit()


    argBest_list = np.array(argBest_list)
    Y = np.copy(X[:, argBest_list])

    #print (Y[:10, 0])

    Y = normalizeIndependent( torch.tensor(Y).float() )
    Y_np = Y.data.numpy()

    #print (Y_np[:10, 0])
    #quit()

    return Y_np




def trainModel(model, X, names, envirement, trainTest2, modelName, Niter = 10000, doPrint=True, regScale=1e-8, learningRate=1e-4, NphenStart=0, Nphen=1,  noiseLevel=0.1):

    

    X = torch.tensor(X).float()


    numWavelengths = X.shape[1]
    
    argTrain = np.argwhere(trainTest2 == 0)[:, 0]


    for phenNow in range(NphenStart, Nphen):

        #print ('X shape', X.shape)


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

            #print (a)
            
            X_train = X[trainTest2 == 0]
            
            #rand1 = torch.randn(X_train.shape) * noiseLevel
            rand1 = torch.rand(size=X_train.shape) * noiseLevel
            X_train = X_train + rand1   

            #print ("B")
        
            
            Y = model(X_train, subset_phen)

            #Y_abs = torch.mean(torch.abs(Y -  torch.mean(Y, axis=0).reshape((1, -1))   ))

            Y = removeIndependence(Y, Y_background[trainTest2 == 0])

            

            Y = normalizeIndependent(Y, cutOff=2) #Include for now

            

            
            heritability_now = cheapHeritability(Y, names[trainTest2 == 0], envirement[trainTest2 == 0])#, device=mps_device )
            loss = -1 * torch.mean(heritability_now)
            
            
            if a % 100 == 0:
                
                print ('iter:', a)

                with torch.no_grad():
                    Y = model(X, subset_phen)
                    Y = removeIndependence(Y, Y_background)

                    Y = normalizeIndependent(Y, cutOff=2) #Include for now

                    heritability_train = cheapHeritability(Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0] )
                    if 1 in trainTest2:
                        heritability_test = cheapHeritability(Y[trainTest2 == 1], names[trainTest2 == 1], envirement[trainTest2 == 1] )
                

                print ('subset_phen', subset_phen)
                print ('training set heritability', heritability_train.data.numpy())
                if 1 in trainTest2:
                    print ('test set heritability', heritability_test.data.numpy())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if a % 10 == 0:
                torch.save(model, modelName)












