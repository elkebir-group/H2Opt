if True:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    #import statsmodels.api as sm
    #import statsmodels.formula.api as smf
    import time
    import os

    from shared import *

    import torch
    from torch.autograd import Variable
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import grad
    from torch.optim import Optimizer
    import scipy
    from scipy.stats import pearsonr
    from scipy.fft import fft, ifft

    from datetime import datetime



def loadnpz(name, allow_pickle=False):

    #This simple function more easily loads in compressed numpy files.

    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data




def convertTIF():

    import numpy as np
    from tifffile import tifffile
    import os

    folder1 = './data/miscPlant/inputs/MSI/'
    folder2 = './data/miscPlant/inputs/MSI_np/'

    files1 = os.listdir(folder1)


    for a in range(len(files1)):

        if files1[a] != '.DS_Store':

            folder_now = folder1 + files1[a] + '/'
            files_now = os.listdir(folder_now)
            
            folder_save = folder2 + files1[a] + '/' 
            
            try:
                os.system('mkdir ' + folder_save)
            except:
                True
            

            for b in range(len(files_now)):

                if files_now[a] != '.DS_Store':

                    file_now = folder_now + files_now[b]
                    print (file_now)
                    image = tifffile.imread(file_now)
                    print (image.shape)

                    file_save = folder_save + files_now[b]


                    np.savez_compressed(file_save, image)


#convertTIF()
#quit()



def OLD_convertTIF():


    import numpy as np
    from tifffile import tifffile

    for index1 in range(1, 2001):
        index1_str = str(index1)

        print (index1_str)

        image = tifffile.imread('./data/imageDataset/IMAGES_ALL/plot_' + index1_str + '.tif')
        np.savez_compressed('./data/imageDataset/images_npz/plot_' + index1_str + '.npz', image)
    



class BadConvModel(nn.Module):
    def __init__(self, Nphen, NchannelInput):

        
        super(BadConvModel, self).__init__()

        self.Nphen = Nphen

        #self.nonlin = torch.tanh
        self.nonlin = torch.nn.LeakyReLU()
        #self.nonlin = nn.ReLU()

        self.dropout1 = nn.Dropout(0.2)
        #Nchannel1 = 5
        Nchannel1 = 10
        #Nchannel1 = 20
        #Nchannel1 = 40

        #self.conv1 = torch.nn.Conv2d(NchannelInput, Nchannel1, 9, stride=(7, 7))

        self.conv1 = torch.nn.Conv2d(NchannelInput, Nchannel1, 11, stride=(9, 9))
        #self.conv1 = torch.nn.Conv2d(NchannelInput, Nchannel1, 9, stride=(3, 3))

        #self.conv1 = torch.nn.Conv2d(NchannelInput, Nchannel1, 4, stride=(4, 4))

        #self.conv1 = torch.nn.Conv2d(NchannelInput, Nchannel1, 9, stride=(5, 5))

        #Nchannel2 = 10

        #self.lin1 = torch.nn.Linear(Nchannel1 * 8 * 8, Nphen) 
        #self.lin1 = torch.nn.Linear(Nchannel1 * 11 * 11, Nphen) 
        #self.lin1 = torch.nn.Linear(Nchannel1 * 5 * 5, Nphen) 
        self.lin1 = torch.nn.Linear(Nchannel1 * 15 * 15, Nphen) 



    def forward(self, x):

        if True:

            shape1 = x.shape


            #print (x.shape)
            #quit()

            #x = x[:, :, 20:-20, 20:-20]
            #x = x[:, :, 40:-40, 40:-40]

            #print (x.shape)


            x = self.conv1(x)

            x = self.nonlin(x)

            #print (x.shape)
            #quit()

            print (x.shape)
            quit()

            x = x.reshape(( x.shape[0],  x.shape[1]*x.shape[2]*x.shape[3] ))

            

            #x = self.dropout1(x)

            x = self.lin1(x)


        else:
            
            x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3] ))
            x = self.badlin(x)
        
        return x
    


class ConvModel_CSM(nn.Module):
    def __init__(self, Nphen, NchannelInput):

        
        super(ConvModel_CSM, self).__init__()

        self.Nphen = Nphen

        #self.nonlin = torch.tanh
        self.nonlin = torch.nn.LeakyReLU()
        #self.nonlin = nn.ReLU()

        self.dropout1 = nn.Dropout(0.2)
        #Nchannel1 = 5
        Nchannel1 = 10

        self.conv1_0 = torch.nn.Conv2d(1, 8, 6, stride=(3, 3))
        self.conv1_1 = torch.nn.Conv2d(5, 2, 6, stride=(3, 3))
        

        Nchannel2 = 10
        self.conv2 = torch.nn.Conv2d(Nchannel1, Nchannel2, 4, stride=(3, 3))  

        #self.max2 = nn.MaxPool2d(3, stride=3) 
        #self.max2 = nn.MaxPool2d(2, stride=2) 



        #self.lin1 = torch.nn.Linear(Nchannel2 * 4 * 4, Nphen)
        #self.lin1 = torch.nn.Linear(Nchannel2 * 5 * 5, Nphen) 
        self.lin1 = torch.nn.Linear(Nchannel2 * 6 * 6, Nphen) #good
        #self.lin1 = torch.nn.Linear(Nchannel2 * 8 * 8, Nphen)
        #self.lin1 = torch.nn.Linear(Nchannel2 * 12 * 12, Nphen)

    

    def forward(self, x):

        if True:

            shape1 = x.shape

            #print (x.shape)


            
            x_1 = self.conv1_1(x[:, 1:])
            x_0 = self.conv1_0(x[:, :1])

            #x[:, :2] = x[:, :2] + x_1

            x = torch.cat((x_0, x_1), axis=1)

            #print (x.shape)
            #quit()

            #x = self.max1(x)

            x = self.nonlin(x)

            x = self.conv2(x)

            x = self.nonlin(x)

            #x = self.max2(x)

            #print (x.shape)
            #quit()

            x = x.reshape(( x.shape[0],  x.shape[1]*x.shape[2]*x.shape[3] ))

            #x = self.dropout1(x)

            x = self.lin1(x)


            #x = torch.tanh(x) #+ 

            #x = self.nonlin(x)
            #x = self.lin2(x)


        else:
            
            x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3] ))
            x = self.badlin(x)
        
        return x
    


class ConvModel(nn.Module):
    def __init__(self, Nphen, NchannelInput):

        
        super(ConvModel, self).__init__()

        self.Nphen = Nphen

        #self.nonlin = torch.tanh
        self.nonlin = torch.nn.LeakyReLU()
        #self.nonlin = nn.ReLU()

        self.dropout1 = nn.Dropout(0.2)
        #Nchannel1 = 5
        Nchannel1 = 10
        #Nchannel1 = 20
        #Nchannel1 = 40

        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0
        #self.conv1 = torch.nn.Conv2d(6, Nchannel1, 7, stride=(1, 1)) #5, 5


        #self.conv1 = torch.nn.Conv2d(6, Nchannel1, 4, stride=(2, 2)) #5, 5
        #self.conv1 = torch.nn.Conv2d(6, Nchannel1, 6, stride=(3, 3))
        self.conv1 = torch.nn.Conv2d(NchannelInput, Nchannel1, 6, stride=(3, 3))
        #self.conv1 = torch.nn.Conv2d(18, Nchannel1, 6, stride=(3, 3))
        #Nchannel2 = 10
        #Nchannel2 = 5
        Nchannel2 = 10
        #Nchannel2 = 40
        #Nchannel2 = 20
        #Nchannel2 = 100
        #self.conv2 = torch.nn.Conv2d(Nchannel1, Nchannel2, 4, stride=(2, 2)) #good
        self.conv2 = torch.nn.Conv2d(Nchannel1, Nchannel2, 4, stride=(3, 3))  

        #self.max2 = nn.MaxPool2d(3, stride=3) 
        #self.max2 = nn.MaxPool2d(2, stride=2) 



        #self.lin1 = torch.nn.Linear(Nchannel2 * 4 * 4, Nphen)
        #self.lin1 = torch.nn.Linear(Nchannel2 * 5 * 5, Nphen) 
        self.lin1 = torch.nn.Linear(Nchannel2 * 6 * 6, Nphen) #good
        #self.lin1 = torch.nn.Linear(Nchannel2 * 8 * 8, Nphen)
        #self.lin1 = torch.nn.Linear(Nchannel2 * 12 * 12, Nphen)

    

    def forward(self, x):

        if True:

            shape1 = x.shape


            x = self.conv1(x)

            #x = self.max1(x)

            x = self.nonlin(x)

            x = self.conv2(x)

            x = self.nonlin(x)

            #x = self.max2(x)

            #print (x.shape)
            #quit()

            x = x.reshape(( x.shape[0],  x.shape[1]*x.shape[2]*x.shape[3] ))

            #x = self.dropout1(x)

            x = self.lin1(x)


            #x = torch.tanh(x) #+ 

            #x = self.nonlin(x)
            #x = self.lin2(x)


        else:
            
            x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3] ))
            x = self.badlin(x)
        
        return x
    


class MultiModal(nn.Module):
    def __init__(self, Nphen, NchannelInput, Ntrait):

        
        super(MultiModal, self).__init__()

        self.nonlin = torch.nn.LeakyReLU()

        self.dropout1 = nn.Dropout(0.2)
        Nchannel1 = 10
        self.conv1 = torch.nn.Conv2d(NchannelInput, Nchannel1, 6, stride=(3, 3))
        Nchannel2 = 10
        self.conv2 = torch.nn.Conv2d(Nchannel1, Nchannel2, 4, stride=(3, 3))  

        
        self.lin1 = torch.nn.Linear(Nchannel2 * 6 * 6, Nphen) 


        #self.linP1 = torch.nn.Linear(Ntrait, Nchannel2) 
        self.linP1 = torch.nn.Linear(Ntrait, Nphen) 

    

    def forward(self, x, xPhen):

       
        shape1 = x.shape


        x = self.conv1(x)

        x = self.nonlin(x)

        x = self.conv2(x)

        #print (x.shape)
        #print (self.linP1(xPhen).shape)
        #quit()

        xPhen = self.linP1(xPhen)
        #xPhen = xPhen.reshape((xPhen.shape[0], xPhen.shape[1], 1, 1))

        

        x = self.nonlin(x)

        x = x.reshape(( x.shape[0],  x.shape[1]*x.shape[2]*x.shape[3] ))



        x = self.lin1(x)

        x = x + xPhen

            
        #x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3] ))
        #x = self.badlin(x)
        
        return x



class OLDconvModel(nn.Module):
    def __init__(self, Nphen, NchannelInput):

        
        super(OLDconvModel, self).__init__()

        self.Nphen = Nphen

        #self.nonlin = torch.tanh
        self.nonlin = torch.nn.LeakyReLU()
        #self.nonlin = nn.ReLU()

        self.dropout1 = nn.Dropout(0.2)
        #Nchannel1 = 5
        Nchannel1 = 10
        #Nchannel1 = 20
        #Nchannel1 = 40

        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0
        #self.conv1 = torch.nn.Conv2d(6, Nchannel1, 7, stride=(1, 1)) #5, 5


        #self.conv1 = torch.nn.Conv2d(6, Nchannel1, 4, stride=(2, 2)) #5, 5
        #self.conv1 = torch.nn.Conv2d(6, Nchannel1, 6, stride=(3, 3))
        self.conv1 = torch.nn.Conv2d(NchannelInput, Nchannel1, 6, stride=(3, 3))
        #self.conv1 = torch.nn.Conv2d(18, Nchannel1, 6, stride=(3, 3))
        #Nchannel2 = 10
        #Nchannel2 = 5
        Nchannel2 = 10
        #Nchannel2 = 40
        #Nchannel2 = 20
        #Nchannel2 = 100
        #self.conv2 = torch.nn.Conv2d(Nchannel1, Nchannel2, 4, stride=(2, 2)) #good
        self.conv2 = torch.nn.Conv2d(Nchannel1, Nchannel2, 4, stride=(3, 3))  

        #self.max2 = nn.MaxPool2d(3, stride=3) 
        #self.max2 = nn.MaxPool2d(2, stride=2) 



        #self.lin1 = torch.nn.Linear(Nchannel2 * 4 * 4, Nphen)
        #self.lin1 = torch.nn.Linear(Nchannel2 * 5 * 5, Nphen) 
        self.lin1 = torch.nn.Linear(Nchannel2 * 6 * 6, Nphen) #good
        #self.lin1 = torch.nn.Linear(Nchannel2 * 8 * 8, Nphen)
        #self.lin1 = torch.nn.Linear(Nchannel2 * 12 * 12, Nphen)

    

    def forward(self, x):

        if True:

            shape1 = x.shape


            x = self.conv1(x)

            #x = self.max1(x)

            x = self.nonlin(x)

            x = self.conv2(x)

            x = self.nonlin(x)

            #x = self.max2(x)

            #print (x.shape)


            x = x.reshape(( x.shape[0],  x.shape[1]*x.shape[2]*x.shape[3] ))

            #x = self.dropout1(x)

            x = self.lin1(x)


            #x = torch.tanh(x) #+ 

            #x = self.nonlin(x)
            #x = self.lin2(x)


        else:
            
            x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3] ))
            x = self.badlin(x)
        
        return x
    




class multiConv(nn.Module):
    def __init__(self, Nphen, Nchannel, modelUse):
        super(multiConv, self).__init__()

        self.Nphen = Nphen

        self.modelList = nn.ModuleList([  modelUse(1, Nchannel) for _ in range(Nphen)])

    def forward(self, x, subset1):


        predVector = torch.zeros((x.shape[0], subset1.shape[0] )).to(x.device)

        for a in range(len(subset1)):
            predVector[:, a] = self.modelList[subset1[a]](x)[:, 0]
        
        return predVector



        
    


class convTime(nn.Module):
    def __init__(self, Nphen, NchannelInput):

        
        super(convTime, self).__init__()

        self.Nphen = Nphen

        #self.nonlin = torch.tanh
        self.nonlin = torch.nn.LeakyReLU()
        #self.nonlin = nn.ReLU()

        self.dropout1 = nn.Dropout(0.2)
        #Nchannel1 = 5
        #Nchannel1 = 10
        Nchannel1 = 2
        #Nchannel1 = 20

        self.conv1 = torch.nn.Conv2d(NchannelInput, Nchannel1, 6, stride=(3, 3))


        #Nchannel2 = 10
        Nchannel2 = 2
        #Nchannel2 = 5
        #Nchannel2 = 20
        #Nchannel2 = 50
        #Nchannel2 = 100
        #self.conv2 = torch.nn.Conv2d(Nchannel1, Nchannel2, 4, stride=(3, 3))
        self.conv2 = torch.nn.Conv2d(Nchannel1, Nchannel2, 4, stride=(2, 2))

        #self.max2 = nn.MaxPool2d(3, stride=3) 
        #self.max2 = nn.MaxPool2d(2, stride=2) 



        #self.lin1 = torch.nn.Linear(Nchannel2 * 4 * 4, Nphen)
        #self.lin1 = torch.nn.Linear(Nchannel2 * 5 * 5, Nphen)
        self.lin1 = torch.nn.Linear(7 * Nchannel2 * 7 * 7, Nphen)
        #self.lin1 = torch.nn.Linear(Nchannel2 * 8 * 8, Nphen)
        #self.lin1 = torch.nn.Linear(Nchannel2 * 12 * 12, Nphen)


        #self.lin1 = torch.nn.Linear(Nchannel2 * 7 * 7, 10)
        #self.lin2 = torch.nn.Linear(10, Nphen)



        #self.badlin = torch.nn.Linear(61*61*6, Nphen)
        #self.badlin = torch.nn.Linear(63*63*6, Nphen)
        self.badlin = torch.nn.Linear(6, Nphen)





    def forward(self, x):

        if True:

            x = x.reshape((x.shape[0], 7, 6, x.shape[2], x.shape[3] ))
            
            shape1 = x.shape


            x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4] ))

            x = self.conv1(x)

            x = self.nonlin(x)

            #x = self.dropout1(x)

            x = self.conv2(x)

            x = self.nonlin(x)

            #print (x.shape)
            
            x = x.reshape(( shape1[0],  shape1[1]*x.shape[1]*x.shape[2]*x.shape[3] ))

            #x = self.dropout1(x)

            x = self.lin1(x)


        
        
        return x
    




class multiConv(nn.Module):
    def __init__(self, Nphen, Nchannel, modelUse):
        super(multiConv, self).__init__()

        self.Nphen = Nphen

        self.modelList = nn.ModuleList([  modelUse(1, Nchannel) for _ in range(Nphen)])

    def forward(self, x, subset1):


        predVector = torch.zeros((x.shape[0], subset1.shape[0] )).to(x.device)

        for a in range(len(subset1)):
            predVector[:, a] = self.modelList[subset1[a]](x)[:, 0]
        
        return predVector



        






class OLD_convModel(nn.Module):
    def __init__(self):


        #TODO Try max pooling instead of stride!


        super(OLD_convModel, self).__init__()

        #self.nonlin = torch.tanh
        self.nonlin = torch.nn.LeakyReLU()
        #self.nonlin = nn.ReLU()

        #self.dropout0 = nn.Dropout(0.05)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        #self.dropout2 = nn.Dropout(0.5)


        #Nchannel1 = 5
        Nchannel1 = 10
        #Nchannel1 = 20

        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0
        self.conv1 = torch.nn.Conv2d(6, Nchannel1, 7, stride=(1, 1)) #5, 5
        #self.conv1 = torch.nn.Conv2d(6, Nchannel1, 9, stride=(7, 7))

        #self.conv1 = torch.nn.Conv2d(6, Nchannel1, 11, stride=(5, 5))

        self.max1 = nn.MaxPool2d(5, stride=5)



        Nchannel2 = 10
        #Nchannel2 = 20
        #Nchannel2 = 50
        #Nchannel2 = 100
        self.conv2 = torch.nn.Conv2d(Nchannel1, Nchannel2, 5)#, stride=(3, 3))

        self.max2 = nn.MaxPool2d(3, stride=3)

        
        

        #Nhidden1 = 100

        #self.lin1 = torch.nn.Linear(Nchannel1 * 20 * 20, 1)

        #Nhidden1 = 100
        #self.lin1 = torch.nn.Linear(Nchannel2 * 4 * 4, 1)
        self.lin1 = torch.nn.Linear(Nchannel2 * 5 * 5, 1)
        #self.lin1 = torch.nn.Linear(Nchannel2 * 6 * 6, Nhidden1)

        #self.lin1 = torch.nn.Linear(Nchannel1 * 15 * 15, 1)

        #self.lin2 = torch.nn.Linear(Nhidden1, 1)



    def forward(self, x):

        shape1 = x.shape


        #print (x.shape)


        x = self.conv1(x)

        x = self.max1(x)

        x = self.nonlin(x)

        #print (x.shape)
        #quit()

        #x = self.dropout0(x)

        x = self.conv2(x)

        x = self.nonlin(x)

        x = self.max2(x)

        #print (x.shape)
        #quit()

        x = x.reshape(( x.shape[0],  x.shape[1]*x.shape[2]*x.shape[3] ))

        x = self.dropout1(x)

        #print (x.shape)
        #quit()

        #x = torch.mean(x, axis=(2, 3))

        x = self.lin1(x)

        #x = self.nonlin(x)
        #x = self.dropout2(x)


        #x = self.lin2(x)

        #print (x.shape)
        #quit()

        
        
        return x
    



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



def MSIC_loader(doSouth=False):

    if doSouth == True:
        imagesAll = loadnpz('./data/miscPlant/inputs/MSI_combine/south_all.npz')
    else:
        imagesAll = loadnpz('./data/miscPlant/inputs/MSI_combine/all.npz')
    plotNum_C, accession_C, plotNum_S, accession_S, _ = loadMSInames(doSouth = doSouth)

    return imagesAll, accession_C



def loadMSI_all():

    imagesAll = loadnpz('./data/miscPlant/inputs/MSI_combine/all.npz')
    imagesAll2 = loadnpz('./data/miscPlant/inputs/MSI_combine/south_all.npz')
    imagesAll = np.concatenate((imagesAll, imagesAll2), axis=0)

    plotNum_C, accession_C, _, _ = loadMSInames(doSouth = False)
    plotNum_S, accession_S, _, _ = loadMSInames(doSouth = True)

    accession_both = np.concatenate((accession_C, accession_S  + np.max(accession_C) + 1  ))

    plantType = np.concatenate(( np.zeros(accession_C.shape[0], dtype=int),  np.ones(accession_C.shape[0], dtype=int)  ))

    return imagesAll, accession_both, plantType

    




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

        #print (Y_sorted[:10])
        #Y_sorted = F.pad(Y_sorted, (1, 0), "constant", 0)
        #print (Y_sorted[:10])
        #quit()


        Y_sorted = torch.cat(( torch.zeros((1, Y.shape[1])), Y_sorted ))

        Y_sorted_cumsum = torch.cumsum(Y_sorted, dim=0)

        #print (np.max(indicesEnd))
        #print (Y_sorted_cumsum.shape)
        sumList = Y_sorted_cumsum[indicesEnd+1] - Y_sorted_cumsum[indicesStart]

        #print (Y_sorted_cumsum[:indicesEnd[0]+2])

        #print (sumList[0])

        sizeList_np = (indicesEnd+1 - indicesStart)
        sizeList = torch.tensor(sizeList_np).float()


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




def TEMP_cheapHeritability(Y, names, envirement, returnVariance=False, device=''):


    if envirement.shape[0] == 0:
        envirement = np.zeros((names.shape[0], 0))


    _, count1 = np.unique(names, return_counts=True)
    _, inverse1 = np.unique(names, return_inverse=True)
    count1_inverse = count1[inverse1]

    Y, names, envirement = Y[count1_inverse >= 2], names[count1_inverse >= 2], envirement[count1_inverse >= 2]

    #varienceName = 0.0

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


    #print (envirement.shape)

    if envirement.shape[1]  == 0:
        varienceEnv = varienceTotal
    else:
        Y2 = removeEnvirement(Y2, envirement)

        #print (time.time() - time2)
        mean3 = torch.mean(Y2 , axis=0)
        varienceEnv = torch.sum(  ( Y2 - mean3 ) ** 2 , axis=0)


    varienceName = groupedVarience(Y2, names)
    #varienceEnv = groupedVarience(Y2, envirement)


    heritability = (varienceEnv - varienceName) / varienceTotal

    #print (time.time() - time1)
    #quit()
    if returnVariance:

        varienceGenetic = varienceEnv - varienceName

        varienceGenetic = varienceGenetic / Y.shape[0]
        varienceTotal = varienceTotal / Y.shape[0]

        return varienceGenetic, varienceTotal
    else:

        return heritability
    




def subsetHeritabilities(Y, names, envirement, Nsubset):


    heritList = torch.zeros(Nsubset)

    names_unique, names_inverse = np.unique(names, return_inverse=True)

    for a in range(Nsubset):

        rand1 = np.random.randint(2, size=names_unique.shape[0])
        while np.sum(rand1) < 2:
            rand1 = np.random.randint(2, size=names_unique.shape[0])
        
        rand1 = rand1[names_inverse]

        heritability = cheapHeritability(Y[rand1==1], names[rand1==1], envirement[rand1==1])

        heritList[a] = torch.mean(heritability)

    return heritList



def quick_subsetHeritabilities(Y, names, envirement, Nsubset):


    names_unique, names_inverse = np.unique(names, return_inverse=True)

    ar_argsort, indicesStart, indicesEnd = fastAllArgwhere(names)

    Y_sorted = Y[ar_argsort]

    Y = Y - torch.mean(Y)
    Y = Y / torch.mean(Y ** 2)

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

    #varList = sumSqList - ((sumList ** 2) / sizeList.reshape((-1, 1)))

    randMatrix = np.zeros(( names_unique.shape[0], Nsubset ), dtype=int)
    for a in range(Nsubset):

        rand1 = np.random.randint(2, size=names_unique.shape[0])
        while np.sum(rand1) < 2:
            rand1 = np.random.randint(2, size=names_unique.shape[0])

        randMatrix[:, a] = rand1 

    randMatrix = torch.tensor(randMatrix).to(Y.device)

    
    sumList_all = sumList.reshape((-1, 1)) * randMatrix
    sumSqList_all = sumSqList.reshape((-1, 1)) * randMatrix
    sizeList_all = sizeList.reshape((-1, 1)) * randMatrix

    varianceSum = torch.sum(sumSqList_all, axis=0)
    totalSum = torch.sum(sumList_all, axis=0)
    totalSize = torch.sum(sizeList_all, axis=0)

    totalGenes = torch.sum(randMatrix, axis=0)



    totalVariance = (varianceSum / totalSize) - (( totalSum / totalSize ) ** 2)


    varList = sumSqList_all - ( sumList_all ** 2) / (sizeList.reshape((-1, 1)) + 1e-7 )
    residual_variance = torch.sum(varList, axis=0)
    residual_variance = residual_variance / (totalSize - totalGenes)



    heritability = (totalVariance - residual_variance) / totalVariance


    return heritability



def loadMSInames(doSouth = False):

    if doSouth:
        names_C = np.loadtxt('./data/miscPlant/inputs/data_process/Miscanthus_S.tsv', delimiter='\t', dtype=str, skiprows=1)
    else:
        names_C = np.loadtxt('./data/miscPlant/inputs/data_process/Miscanthus_C.tsv', delimiter='\t', dtype=str, skiprows=1)
    #names_S = np.loadtxt('./data/miscPlant/inputs/data_process/Miscanthus_S.tsv', delimiter='\t', dtype=str, skiprows=1)

    

    plotNum_C = names_C[:, 0]
    halfsib_C = names_C[:, 7]

    plotRow_C = names_C[:, 2]
    plantName_C = names_C[:, 3]

    plotRep_C = names_C[:, 4]




    return plotNum_C, halfsib_C, plotRow_C, plantName_C, plotRep_C


#loadMSInames()
#quit()

def MSIC_saveImages(doSouth=False):

    doSouth = True

    #MSA
    #torch.Size([2000, 6, 107, 107])
    #MSI_C
    #(64, 65, 6)

    plotNum_C, accession_C, plotNum_S, accession_S = loadMSInames(doSouth=doSouth)

    imageFolder = './data/miscPlant/inputs/MSI_np/'

    imagesAll = np.zeros(( 3900, 14, 6, 63, 63 ))

    files1 = os.listdir(imageFolder)
    files1.remove('.DS_Store')
    #print (files1)
    files1 = np.array(files1)
    files1 = np.sort(files1)
    #print (np.sort(files1))
    #print (len(files1))
    #quit()
    for a in range(len(files1)):
        print (a)
        dateFolder = imageFolder + files1[a] + '/'
        #imageFiles = os.listdir(dateFolder)
        for b in range(plotNum_C.shape[0]):# range(len(imageFiles)):
            #for b in [2214]:
            #print (b)
            imageFile = 'plot_' + plotNum_C[b] + '.tif.npz'
            imageFileFull = dateFolder + imageFile
            image1 = loadnpz(imageFileFull)
            image1 = np.swapaxes(image1, 1, 2)
            image1 = np.swapaxes(image1, 0, 1)

            rowMeans = np.mean(image1, axis=(0, 1))
            colMeans = np.mean(image1, axis=(0, 2))

            #print (image1.shape)

            #plt.imshow(image1[0])
            #plt.show()

            #argGood1 = np.argwhere(rowMeans < np.median(rowMeans) * 5 )[:, 0]
            #argGood2 = np.argwhere(colMeans < np.median(colMeans) * 5 )[:, 0]
            argGood1 = np.argwhere(rowMeans != 65535 )[:, 0]
            argGood2 = np.argwhere(colMeans != 65535 )[:, 0]
            image1 = image1[:, argGood2][:, :, argGood1]


            if 65535 in image1:
                print ("Hi")

                image1[image1 == 65535] = 0

            
            

            imagesAll[b, a] = image1[:, :63, :63]


    if doSouth:
        np.savez_compressed('./data/miscPlant/inputs/MSI_combine/south_all.npz', imagesAll)
    else:
        np.savez_compressed('./data/miscPlant/inputs/MSI_combine/all.npz', imagesAll)

    

    quit()
    

#MSIC_saveImages()
#quit()


def loadImagesNames():
    

    True

    print (plotNum_C[:10])

    print (plotNum_C.shape)



    #print (plotNum_C[:10])
    #print (accession_C[:10])
    #quit()

    #print (names_S[:10])
    quit()


#loadImagesNames()
#quit()

def cutoffNow(Y, cutoff):

    Y = nn.ReLU()(Y + cutoff) - cutoff
    Y = cutoff - nn.ReLU()(cutoff - Y)

    return Y







def coherit_both_trainModel(model, trait1, imageData_C, imageData_S, names_S, names_C, trainTest_C, trainTest_S, modelName, Niter = 10000, doPrint=True, regScale=1e-8, learningRate=1e-4, Nphen=1, corVar=False, NphenStart=0):

    


    import torchvision.transforms as transforms
    transformCrop = transforms.RandomCrop(size=(55, 55)) 

    losses = []

    error_train = []
    error_test = []

    mps_device = torch.device("mps")

    model.to(mps_device)

    imageData_C = torch.tensor(imageData_C).float()
    imageData_C = imageData_C.to(mps_device)


    imageData_S = torch.tensor(imageData_S).float()
    imageData_S = imageData_S.to(mps_device)

    



    for phenNow in range(NphenStart, Nphen):

        


        if phenNow > 0:
            subset1 = np.arange(phenNow)
            
            with torch.no_grad():
                Y_background_C = model(imageData_C[:, :, 4:4+55, 4:4+55], subset1)
                Y_background_C = normalizeIndependent(Y_background_C)

                Y_background_S = model(imageData_S[:, :, 4:4+55, 4:4+55], subset1)
                Y_background_S = normalizeIndependent(Y_background_S)

        else:
            Y_background_C = torch.zeros((imageData_C.shape[0]), 0)
            Y_background_S = torch.zeros((imageData_S.shape[0]), 0)

        
        subset_phen = np.zeros(1, dtype=int) + phenNow

        optimizer = torch.optim.RMSprop(model.parameters(), lr = learningRate)
        

        for a in range(Niter):



                X_C = imageData_C[trainTest_C == 0]
                X_S = imageData_S[trainTest_S == 0]

                if True:
                    
                    rand_coef = 0.05
                    rand_C = torch.rand(X_C.shape).to(X_C.device) * rand_coef
                    rand_S = torch.rand(X_S.shape).to(X_S.device) * rand_coef
                    
                    X_C = X_C + rand_C
                    X_S = X_S + rand_S


                if True:
                    X_C = torch.stack([transformCrop(image ) for image in X_C])
                    X_S = torch.stack([transformCrop(image ) for image in X_S])


                
                Y_C = model(X_C, subset_phen)
                Y_S = model(X_S, subset_phen)


                if True:
                    Y_C = removeIndependence(Y_C, Y_background_C[trainTest_C == 0])
                    Y_S = removeIndependence(Y_S, Y_background_S[trainTest_S == 0])

                Y_C = normalizeIndependent(Y_C)
                Y_S = normalizeIndependent(Y_S)

                #Y_C[Y_C<-2] = -2
                #Y_C[Y_C>2] = 2
                #Y_S[Y_S<-2] = -2
                #Y_S[Y_S>2] = 2

                
                envirement = torch.zeros(0)
                
                heritability_C = cheapHeritability(Y_C, names_C[trainTest_C == 0], envirement)
                heritability_S = cheapHeritability(Y_S, names_S[trainTest_S == 0], envirement)

                trait1_train = trait1[trainTest_C == 0]
                argTrait = np.argwhere(trait1_train != '')[:, 0]
                trait1_train = trait1_train[argTrait].astype(float)
                trait1_train = torch.tensor(trait1_train).float().to(mps_device)



                heritMean = (torch.mean(heritability_C) + torch.mean(heritability_S)) / 2.0
                coherit_train, _, _ = coherit(Y_C[argTrait],  trait1_train, names_C[argTrait], envirement )

                loss = -1 * (  (heritMean * 1.0) + (coherit_train * 1.0) )
                
                
                regLoss = 0
                for param in model.parameters():
                    regLoss += torch.sum(param ** 2)
                regLoss = regLoss ** 0.5
                

                if a % 10 == 0:
                    
                    print ('iter:', a)

                    with torch.no_grad():
                        X_pred_C = torch.stack([transformCrop(image ) for image in imageData_C])
                        Y_C = model(X_pred_C, subset_phen)
                        Y_C = removeIndependence(Y_C, Y_background_C)

                    argTraitTrain = np.argwhere( np.logical_and( trainTest_C == 0, trait1 != '' ) )[:, 0]
                    argTraitTest = np.argwhere( np.logical_and( trainTest_C == 1, trait1 != '' ) )[:, 0]

                    trait1_train = trait1[argTraitTrain].astype(float)
                    trait1_test = trait1[argTraitTest].astype(float)
                    trait1_train, trait1_test = torch.tensor(trait1_train).float().to(mps_device), torch.tensor(trait1_test).float().to(mps_device)


                    coherit_train, _, _ = coherit(Y_C[argTraitTrain],  trait1_train, names_C[argTraitTrain], envirement )
                    coherit_test, _, _ = coherit(Y_C[argTraitTest],  trait1_test, names_C[argTraitTest], envirement )

                    
                    
                    
                    print ('subset_phen', subset_phen)
                    print ("coherit", coherit_train, coherit_test)

                    #quit()
                


                #loss = loss + (regLoss * regScale)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if a % 10 == 0:
                    torch.save(model, modelName)
            



    #if 1 in trainTest2:
    #    return (heritability_train, heritability_test)
    #else:
    #    return (heritability_train)
    True






def coheritAllSimpleTrain():



    imageAll_C, names_C = MSIC_loader(doSouth=False)
    imageAll_S, names_S = MSIC_loader(doSouth=True)


    file1 = './data/miscPlant/inputs/data_process/Msi_C_Japan_2019-URB-Plant_Data.tsv'

    data_trait = np.loadtxt(file1, delimiter='\t', dtype=str, skiprows=1)

    with open(file1, 'r') as file:
        lines = file.readlines()
    row1 = np.array(lines[0].split('\t'))
    
    argBiomass = np.argwhere(data_trait[0] == '268.7')[0, 0]
    trait1 = data_trait[:, argBiomass]


    #for predNum in range(14):
    for predNum in range(0, 1):

        for a in range(5):
            print ('')

        print ("#####################################")
        print ('time point: ', predNum)
        print ("#####################################")
        
        for a in range(5):
            print ('')

        imageData_C = imageAll_C[:, predNum:predNum+2, :, :, :]
        imageData_S = imageAll_S[:, predNum:predNum+2, :, :, :]

        imageData_C = imageData_C.reshape((imageData_C.shape[0], imageData_C.shape[1]*imageData_C.shape[2], imageData_C.shape[3], imageData_C.shape[4]))
        imageData_S = imageData_S.reshape((imageData_S.shape[0], imageData_S.shape[1]*imageData_S.shape[2], imageData_S.shape[3], imageData_S.shape[4]))
        #means1 = (np.mean(imageData_C, axis=(0, 2, 3)) + np.mean(imageData_S, axis=(0, 2, 3))) / 2.0

        means1 = np.mean(imageData_C, axis=(0, 2, 3))
                  
        means1 = means1.reshape((1, -1, 1, 1)) * 3

        imageData_C = imageData_C / means1
        #imageData_S = imageData_S / means1






        #np.random.seed(0)
        np.random.seed(1)

        names_unique_C, names_inverse_C = np.unique(names_C, return_inverse=True)
        trainTest_C = np.random.randint(3, size=names_unique_C.shape[0])
        trainTest_C[trainTest_C == 2] = 0
        trainTest_C = trainTest_C[names_inverse_C]

        

        names_unique_S, names_inverse_S = np.unique(names_S, return_inverse=True)
        trainTest_S = np.random.randint(3, size=names_unique_S.shape[0])
        trainTest_S[trainTest_S == 2] = 0
        trainTest_S = trainTest_S[names_inverse_S]

        #modelName = './data/miscPlant/models/MSI_all_singlepoint_' + str(predNum) + '_split5.pt'
        modelName = './data/miscPlant/models/MSI_all_singlepoint_' + str(predNum) + '_coherit1.pt'

        loadModel = ''

        regScale = 1e-20


        #learningRate=5e-4

        learningRate=1e-4
        #learningRate=3e-5
        #learningRate=1e-5 #
        #learningRate=5e-6 
        #learningRate=2e-6
        #learningRate=1e-6 #
        #learningRate=1e-7

        #Niter = 100
        #Niter = 200
        Niter = 500
        
        if loadModel != '':
            model = torch.load(loadModel)
        else:
            model = multiConv(20, 6, convTime ) #Max phenotypes = 20
            #model = multiConv(20, 12, convModel ) 


        #model = torch.load('./data/miscPlant/models/MSI_all_singlepoint_' + str(predNum) + '_split3.pt')

        #Nphen = 1
        Nphen = 1

        NphenStart = 0

        coherit_both_trainModel(model, trait1, imageData_C, imageData_S, names_S, names_C, trainTest_C, trainTest_S, modelName, Niter = Niter, doPrint=True, regScale=regScale, learningRate=learningRate, Nphen=Nphen, NphenStart=NphenStart)
    
        #quit()

#coheritAllSimpleTrain()
#quit()


def rowSplitter(plotRow):


    plotRow_unique, plotRow_counts = np.unique(plotRow, return_counts=True)
    perm2 = np.zeros(plotRow.shape[0], dtype=int)

    for a in range(plotRow_unique.shape[0]):
        args1 = np.argwhere(plotRow == plotRow_unique[a])[:, 0]
        perm2[args1] = np.random.permutation(args1.shape[0])

    return perm2


def rowHerit(plotRow_now, Y, names, envirement, lossStyle='mean'):


    #heritability_now = cheapHeritability(Y, names, envirement)
    #print (heritability_now)
    #quit()


    perm2 = rowSplitter(plotRow_now)


    #perm2 = perm2 % 2 


    #heritability_fancy = 0.0
    perm2_unique = np.unique(perm2)

    heritability_matrix = torch.zeros((perm2_unique.shape[0], Y.shape[1]))

    

    
    for a in range(perm2_unique.shape[0]):
        argPlot = np.argwhere( perm2_unique[a] == perm2 )[:, 0]  

        #heritability_now = cheapHeritability(Y[argPlot], names[argPlot], envirement[argPlot])
        #print (heritability_now)
        #quit()

        #_, counts1 = np.unique(names[argPlot], return_counts=True)
        #print (np.min(counts1))
        #quit()
        #print (envirement.shape)
        #print (Y.shape)
        #print (names.shape)

        #Y = Y[:, :1] #TODO REMOVE!
        #Y = Y / torch.mean(torch.abs(Y))

        #plt.hist(Y[:, 0].data.numpy())
        #plt.show()
        if envirement.shape[0] > 0:
            envirement_now = envirement[argPlot]
        else:
            envirement_now = envirement

        heritability_now = cheapHeritability(Y[argPlot], names[argPlot], envirement_now)
        
        
        #print (heritability_now)
        
        #quit()
        #heritability_fancy = heritability_fancy + heritability_now
        #print (heritability_now)
        heritability_matrix[a] = heritability_now
    #quit()

    #print ("m")
    #print (heritability_matrix)
    #quit()

    if lossStyle == 'mean':
        heritability_fancy = torch.mean(heritability_matrix, axis=0)
    if lossStyle == 'min':
        heritability_fancy, _ = torch.min(heritability_matrix, axis=0)
        #for a in range(heritability_matrix.shape)

        #argsort1 = np.argsort(heritability_matrix[:, 0].data.numpy())[:2]
        #heritability_fancy = torch.mean( heritability_matrix[argsort1], axis=0  )




    #heritability_fancy = heritability_fancy / perm2_unique.shape[0]

    return heritability_fancy



def mod_rowHerit(plotRow_now, Y, names, envirement, lossStyle='mean'):



    heritability_matrix = torch.zeros((1000, Y.shape[1]))


    count1 = -1
    #for b in range(100):
    for b in range(10):
        perm2 = rowSplitter(plotRow_now)
        perm2_unique = np.unique(perm2)

        

        for a in range(perm2_unique.shape[0]):
            count1 += 1
            argPlot = np.argwhere( perm2_unique[a] == perm2 )[:, 0]  

            
            if envirement.shape[0] > 0:
                envirement_now = envirement[argPlot]
            else:
                envirement_now = envirement

            heritability_now = cheapHeritability(Y[argPlot], names[argPlot], envirement_now)
            
            heritability_matrix[count1] = heritability_now
    
    heritability_matrix = heritability_matrix[:count1+1]


    #plt.hist(heritability_matrix[:, 0].data.numpy(), bins=100)
    #plt.show()
    #quit()

    if lossStyle == 'mean':
        heritability_fancy = torch.mean(heritability_matrix, axis=0)
    if lossStyle == 'min':
        heritability_matrix = heritability_matrix[:, 0]

        #argsort1 = np.argsort(heritability_matrix.data.numpy())[ :heritability_matrix.shape[0] // 20 ] 
        #argsort1 = np.argsort(heritability_matrix.data.numpy())[ :heritability_matrix.shape[0] // 40 ] 
        #argsort1 = np.argsort(heritability_matrix.data.numpy())[ :2 ] 

        #heritability_fancy = torch.mean(heritability_matrix[argsort1])

        heritability_fancy, _ = torch.min(heritability_matrix, axis=0)
        #for a in range(heritability_matrix.shape)

        #argsort1 = np.argsort(heritability_matrix[:, 0].data.numpy())[:2]
        #heritability_fancy = torch.mean( heritability_matrix[argsort1], axis=0  )

        #print (heritability_fancy)
        #quit()


    return heritability_fancy


def trainModel(model, X, names, plotRow, envirement, trainTest2, modelName, Niter = 10000, doPrint=True, regScale=1e-8, learningRate=1e-4, Nphen=1, corVar=False, NphenStart=0, plantType=[]):

    

    
    import torchvision.transforms as transforms
    transformCrop = transforms.RandomCrop(size=(55, 55)) 

    losses = []

    error_train = []
    error_test = []

    mps_device = torch.device("mps")

    model.to(mps_device)

    X = torch.tensor(X).float()
        
    X = X.to(mps_device)

    if type(False) != type(corVar):
        corVar = corVar.to(X.device)

    batch_size = -1#300 #-1
    #batch_size = 500

    argTrain = np.argwhere(trainTest2 == 0)[:, 0]
    argTest = np.argwhere(trainTest2 == 1)[:, 0]

    if batch_size < 0:
        batch_size = argTrain.shape[0]



    for phenNow in range(NphenStart, Nphen):

        print ('X shape', X.shape)


        if phenNow > 0:
            subset1 = np.arange(phenNow)

            #Y_background = model(X[:, 0, :, 4:4+55, 4:4+55], subset1)
            Y_background = model(X[:, 0], subset1)
            Y_background = Y_background.detach()
            Y_background = normalizeIndependent(Y_background)

            #heritability_now = cheapHeritability(Y_background, names[:], envirement[:])


        else:
            Y_background = torch.zeros((X.shape[0]), 0)

        

        subset_phen = np.zeros(1, dtype=int) + phenNow

        #

        if not isinstance(corVar, bool):
            corVar = corVar.reshape((-1, 1))

        optimizer = torch.optim.RMSprop(model.parameters(), lr = learningRate)
        

        for iterIndex in range(Niter):

            

            

            for batch_index in range(argTrain.shape[0] // batch_size):

                argNow = argTrain[batch_index*batch_size:(batch_index+1)*batch_size]

                timeList1 = []
                time1 = time.time()

                #print (argNow, argNow.shape)


                timeList1.append(time.time() - time1)
                time1 = time.time()

                X_tensor = X[argNow, 0]

                if True:
                    
                    rand1 = torch.rand(X_tensor.shape).to(X_tensor.device)


                    #X_tensor = X_tensor + (rand1 * 0.2)
                    X_tensor = X_tensor + (rand1 * 0.05) #typical
                    #X_tensor = X_tensor + (rand1 * 0.4)
                    #X_tensor = X_tensor + (rand1 * 1.0)
                    #X_tensor = X_tensor + (rand1 * 0.01)
                    #X_tensor = X_tensor #+ (rand1 * 0.01)



                if True:
                    #X_tensor = torch.stack([transformCrop(image ) for image in X_tensor])
                    #X_tensor = X_tensor[:, :, 4:4+55, 4:4+55]
                    True

                
                timeList1.append(time.time() - time1)
                time1 = time.time()


                #print ('nanCheck', model(X_tensor, np.arange(5))[0])
                
                Y = model(X_tensor, subset_phen)

                #print (Y[:10])
                #quit()


                Y = Y + (torch.rand(Y.shape).to(Y.device) * 1e-5) #Avoid a NAN issue

                
                #quit()
                

                if True:
                    Y = removeIndependence(Y, Y_background[argNow])


                #print (Y[:2])
                #quit()
                

                

                Y_abs = torch.mean(torch.abs(Y  - torch.mean(Y) ))

                

                #print (Y[:2])
                #quit()

                #Y = Y - torch.mean(Y)
                #Y = Y / (torch.mean(Y**2)**0.5)
                #Y[Y<-2] = -2
                #Y[Y>2] = 2


                #plotRow_now = plotRow[argNow]
                #perm2 = rowSplitter(plotRow_now)

                #heritability_fancy = 0.0
                #perm2_unique = np.unique(perm2)
                #for a in range(perm2_unique.shape[0]):
                #    argPlot = np.argwhere( perm2_unique[a] == perm2 )[:, 0]  
                #    heritability_now = cheapHeritability(Y[argPlot], names[argNow][argPlot], envirement[argNow][argPlot])
                #    heritability_fancy = heritability_fancy + heritability_now


                #heritability_fancy = mod_rowHerit(plotRow[argNow], Y, names[argNow], envirement[argNow], lossStyle='min')
                #heritability_fancy = rowHerit(plotRow[argNow], Y, names[argNow], envirement[argNow], lossStyle='min')




                heritability_fancy = cheapHeritability(Y, names[argNow], envirement[argNow])

                #print (heritability_fancy)
                #quit()

                



                loss = -1 * torch.mean(heritability_fancy)
                

                
                
                        
                    
                
                regLoss = 0
                #for param in model.parameters():
                #    regLoss += torch.sum(param ** 2)
                for param in model.modelList[subset_phen[0]].parameters():
                    regLoss += torch.sum(param ** 2)
                    #regLoss += torch.sum(torch.abs(param))
                

                #quit()

                #regLoss = regLoss / Y_abs
                regLoss = regLoss / (Y_abs ** 2)

                
                #losses.append(  list(heritability_now[:5]) )   #+ list(heritability_test[:5]) )

                if iterIndex % 10 == 0:
                    
                    #if a > 20:
                    #print (rand1)
                    if batch_index == 0:
                        print ('iter:', iterIndex)

                        with torch.no_grad():
                            X_pred = X[:, 0]
                            #X_pred = torch.stack([transformCrop(image ) for image in X_pred])
                            #X_pred = X_pred[:, :, 4:4+55, 4:4+55]
                            Y = model(X_pred, subset_phen)

                            if True:
                                Y = removeIndependence(Y, Y_background)

                        
                        #Y = Y - torch.mean(Y)
                        #Y = Y / (torch.mean(Y**2)**0.5)
                        #Y[Y<-2] = -2
                        #Y[Y>2] = 2

                        #envirement_row = np.zeros((envirement.shape[0], 0), dtype=int)
                        #heritability_train_row = rowHerit(plotRow[trainTest2 == 0], Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0])
                        #heritability_test_row = rowHerit(plotRow[trainTest2 == 1], Y[trainTest2 == 1], names[trainTest2 == 1], envirement[trainTest2 == 1])

                        heritability_train = cheapHeritability(Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0])
                        if 1 in trainTest2:
                            heritability_test = cheapHeritability(Y[trainTest2 == 1], names[trainTest2 == 1], envirement[trainTest2 == 1])
                        


                        print ('subset_phen', subset_phen)
                        print ('heritability_fancy', heritability_fancy)
                        print (heritability_train.cpu().data.numpy())
                        if 1 in trainTest2:
                            print (heritability_test.cpu().data.numpy())
                        #print ('row')
                        #print (heritability_train_row.cpu().data.numpy())
                        #print (heritability_test_row.cpu().data.numpy())
                        
                        #quit()
                
                
                
                loss = loss + (regLoss * regScale)


                optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-6)
                optimizer.step()

                #timeList1.append(time.time() - time1)
                #time1 = time.time()

                if iterIndex % 10 == 0:
                    torch.save(model, modelName)
            



    if 1 in trainTest2:
        return (heritability_train, heritability_test)
    else:
        return (heritability_train)
    




def trainNoSplit():

    
    doSouth = False #Second

    namePart = 'central'
    if doSouth:
        namePart = 'south'

    print ('paused')
    quit()


    imageDataAll, names = MSIC_loader(doSouth=doSouth)

    plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=doSouth)


    np.random.seed(1)
    crossValid = np.random.randint(5, size=names.shape[0])


    for predNum in [4]:

        print ("#####################################")
        print ('simple time point: ', predNum)
        print ("#####################################")

        trainTest2 = np.zeros(crossValid.shape[0], dtype=int)

        rowGroup = plotRow.astype(int) % 130

        imageData = imageDataAll[:, predNum:predNum+1, :, :, :]


        imageData = imageData.reshape((imageData.shape[0], 1, imageData.shape[1]*imageData.shape[2],  imageData.shape[3], imageData.shape[4]  ))


        means1 = np.mean(imageData, axis=(0, 1, 3, 4))
        imageData = imageData / means1.reshape((1, 1, -1, 1, 1))

        imageData = imageData / 3



        envirement = np.array([plotRep_C, rowGroup]).T

        loadModel = ''
        modelName = './data/miscPlant/models/MSI_' + namePart + '_singlepoint_' + str(predNum) + '_noSplit.pt' 

        regScale = 5e-5 
        learningRate=5e-5
        Niter = 1000
        Nchannel = imageData.shape[2]


        
        if loadModel != '':
            model = torch.load(loadModel)
        else:
            #model = multiConv(20, 6, convTime ) #Max phenotypes = 20
            model = multiConv(20, 6, ConvModel ) 
        Nphen = 1
        NphenStart = 0

        trainModel(model, imageData, names, plotRow, envirement, trainTest2, modelName, Niter = Niter, doPrint=True, regScale=regScale, learningRate=learningRate, Nphen=Nphen, NphenStart=NphenStart)
        #quit()

        

#trainNoSplit()
#quit()



def simpleTrain():


    #plotNum_C, accession_C, plotNum_S, accession_S = loadMSInames()
    #print (np.unique(accession_C).shape)

    #_, counts = np.unique(accession_C, return_counts=True)

    #plt.hist(counts, bins=100)
    #plt.show()
    #quit()
    #doSouth = True #First
    doSouth = False #Second

    namePart = 'central'
    if doSouth:
        namePart = 'south'


    #for predNum in range(14):
    #for predNum in [3]: #8, 11
    #for predNum in [11]:
    #for predNum in [2]: #2 Central is ok, around 0.08 to 0.09, 
    #[3] central ok around 0.07 to 0.08
    #for predNum in [1]:

    #print ('paused')
    #quit()

    imageDataAll, names = MSIC_loader(doSouth=doSouth)

    plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=doSouth)


    np.random.seed(1)
    crossValid = np.random.randint(5, size=names.shape[0])



    #2 south extremely good. ~0.25
    for predNum in [5]:# range(14): #[4]:

        print ("#####################################")
        print ('simple time point: ', predNum)
        print ("#####################################")

        for splitIndex in range(np.unique(crossValid).shape[0]):

            print ("#####################################")
            print ('splitIndex: ', splitIndex)
            print ("#####################################")

            trainTest2 = np.zeros(crossValid.shape[0], dtype=int)
            trainTest2[crossValid == splitIndex] = 1

            #print (np.unique(trainTest2, return_counts=True))
            #quit()
        
            rowGroup = plotRow.astype(int) % 130
            #rowGroup = plotRow.astype(int) // 5 
            
            
            

            imageData = imageDataAll[:, predNum:predNum+1, :, :, :]

            imageData = imageData[:, :, 0:1] 

            #Central
            #6 ) 0, 10 ) 0, 12 ) 0, 13 ) 0
            #South
            #6 ~ 0, 9 ~ 0, 


            #central, 6, 
            #split 0 [0.08521222], split 1 [0.0569227], split 2 0.06844827, split 3 0.11215476, split 4 0.07345121
            #central, 10,
            #split 0 [0.07455641], 
            


            #print (imageData.shape)
            #quit()


            imageData = imageData.reshape((imageData.shape[0], 1, imageData.shape[1]*imageData.shape[2],  imageData.shape[3], imageData.shape[4]  ))


            means1 = np.mean(imageData, axis=(0, 1, 3, 4))
            imageData = imageData / means1.reshape((1, 1, -1, 1, 1))

            imageData = imageData / 3




            


            #np.random.seed(0)
            #np.random.seed(1)

            #plotRow


            #envirement = np.zeros((imageData.shape[0], 0))
            #envirement = plotRep_C.reshape((-1, 1))
            #envirement = rowGroup.reshape((-1, 1))


            envirement = np.array([plotRep_C, rowGroup]).T


            loadModel = ''
            
            
            
            #modelName = './data/miscPlant/models/MSI_' + namePart + '_singlepoint_' + str(predNum) + '_split_' + str(splitIndex) + '.pt' 
            modelName = './data/miscPlant/models/MSI_' + namePart + '_singlepoint_' + str(predNum) + '_split_' + str(splitIndex) + '_CSM.pt' 


            

            ##regScale = 1e-4 , learningRate=5e-5, iter 1000: [0.52009064] [0.24857934]
            #L2 regScale = 1e-4 , learningRate=5e-5, iter 1000:  [0.63486165] [0.2962179] 
            #Again: [0.47144413] [0.24673106]
            #L2 regScale = 5e-5 , learningRate=5e-5, iter 1000: [0.3616723] [0.2913387] Again [0.24482106][0.11949822]



            #[3] L2 regScale = 5e-5 , learningRate=5e-5, iter 1000: [0.42214903] [0.16777433]  [0.35391468] [0.1605571]



            regScale = 5e-5 #good
            #regScale = 1e-3 #good

            learningRate=5e-5 #good


            #Niter = 10000
            #Niter = 300

            #Niter = 300
            #Niter = 100
            Niter = 1000
            #Niter = 2000
            #Niter = 4000
            #Niter = 10

            #print (imageData.shape)
            #quit()
            Nchannel = imageData.shape[2]


            
            if loadModel != '':
                model = torch.load(loadModel)
            else:
                #model = multiConv(20, 6, convTime ) #Max phenotypes = 20
                #model = multiConv(20, 6, ConvModel ) 
                model = multiConv(20, 1, ConvModel ) 
                #model = multiConv(20, 1, ConvModel_CSM ) 


            #model = torch.load('./data/miscPlant/models/MSI_all_singlepoint_' + str(predNum) + '_split3.pt')

            Nphen = 1
            #Nphen = 5

            NphenStart = 0

            trainModel(model, imageData, names, plotRow, envirement, trainTest2, modelName, Niter = Niter, doPrint=True, regScale=regScale, learningRate=learningRate, Nphen=Nphen, NphenStart=NphenStart)
            #quit()

            if False:
                model = torch.load(modelName)

                X = torch.tensor(imageData[:, 0, :, 4:4+55, 4:4+55]).float()
                mps_device = torch.device("mps")
                X = X.to(mps_device)
                print (X.shape)
                Y = model(  X, np.arange(1))
                envirement = np.zeros((envirement.shape[0], 0))

                #heritability_train = rowHerit(plotRow[trainTest2 == 0], Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0])
                heritability_train = cheapHeritability(Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0])

                print (heritability_train)

                quit()
    quit()

    #0
    #[0.12793887 0.07193341 0.01645868 0.00587667 0.01844362]
    #[0.11331183 0.08526807 0.02000231 0.02440868 0.00873814]

    #2
    #[0.23275156 0.12234952 0.06279992 0.04856788 0.03429307]
    #[0.21625131 0.11829969 0.11321502 0.07261565 0.05629748]


    imageData = torch.tensor(imageData[:, 0]).float()
    imageData = imageData[:, :, 4:4+55, 4:4+55]
    model = torch.load(modelName).to('cpu')
    Y = model(imageData, np.arange(Nphen))

    #Y = model(imageData, np.zeros(1, dtype=int)+3)

    #print (Y.shape)
    #quit()

    Y = normalizeIndependent(Y)

    Y[Y<-2] = -2
    Y[Y>2] = 2

    #plt.hist(Y[:, 1].data.numpy())
    #plt.show()


    heritability_train = cheapHeritability(Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0])
    heritability_train = heritability_train.detach().data.numpy()
    print (heritability_train)
    heritability_test = cheapHeritability(Y[trainTest2 == 1], names[trainTest2 == 1], envirement[trainTest2 == 1])
    heritability_test = heritability_test.detach().data.numpy()
    print (heritability_test)


    plt.plot(heritability_train * 4)
    plt.plot(heritability_test * 4)
    plt.xlabel('trait')
    plt.ylabel('heritability')
    plt.legend(['training set', 'test set'])
    plt.show()

    quit()


    quit()


#simpleTrain()
#quit()



def multiModal_trainModel(model, X, xPhen, names, plotRow, envirement, trainTest2, modelName, Niter = 10000, doPrint=True, regScale=1e-8, learningRate=1e-4, Nphen=1, corVar=False, NphenStart=0, plantType=[]):

    

    
    import torchvision.transforms as transforms
    transformCrop = transforms.RandomCrop(size=(55, 55)) 

    losses = []

    error_train = []
    error_test = []

    mps_device = torch.device("mps")

    model.to(mps_device)

    X = torch.tensor(X).float()
        
    X = X.to(mps_device)
    xPhen = xPhen.to(mps_device)

    if type(False) != type(corVar):
        corVar = corVar.to(X.device)

    batch_size = -1#300 #-1
    #batch_size = 500

    argTrain = np.argwhere(trainTest2 == 0)[:, 0]
    argTest = np.argwhere(trainTest2 == 1)[:, 0]

    if batch_size < 0:
        batch_size = argTrain.shape[0]



    for phenNow in range(NphenStart, Nphen):

        print ('X shape', X.shape)


        if phenNow > 0:
            subset1 = np.arange(phenNow)

            #Y_background = model(X[:, 0, :, 4:4+55, 4:4+55], subset1)
            Y_background = model(X[:, 0], subset1)
            Y_background = Y_background.detach()
            Y_background = normalizeIndependent(Y_background)

            #heritability_now = cheapHeritability(Y_background, names[:], envirement[:])


        else:
            Y_background = torch.zeros((X.shape[0]), 0)

        

        subset_phen = np.zeros(1, dtype=int) + phenNow

        #

        if not isinstance(corVar, bool):
            corVar = corVar.reshape((-1, 1))

        optimizer = torch.optim.RMSprop(model.parameters(), lr = learningRate)
        

        for iterIndex in range(Niter):

            

            for batch_index in range(argTrain.shape[0] // batch_size):

                argNow = argTrain[batch_index*batch_size:(batch_index+1)*batch_size]

                timeList1 = []
                time1 = time.time()

                #print (argNow, argNow.shape)


                timeList1.append(time.time() - time1)
                time1 = time.time()

                X_tensor = X[argNow, 0]
                xPhen_now = xPhen[argNow]

                if True:
                    
                    rand1 = torch.rand(X_tensor.shape).to(X_tensor.device)
                    X_tensor = X_tensor + (rand1 * 0.05) #typical

                timeList1.append(time.time() - time1)
                time1 = time.time()

                #print (torch.mean(X_tensor))
                #print (torch.mean(xPhen_now))
                
                
                Y = model(X_tensor, xPhen_now)

                #print (Y)
                #quit()

                Y = Y + (torch.rand(Y.shape).to(Y.device) * 1e-5) #Avoid a NAN issue


                if True:
                    Y = removeIndependence(Y, Y_background[argNow])

                
                Y_abs = torch.mean(torch.abs(Y  - torch.mean(Y) ))


                heritability_fancy = cheapHeritability(Y, names[argNow], envirement[argNow])


                loss = -1 * torch.mean(heritability_fancy)
                
                regLoss = 0
                for param in model.parameters():
                    regLoss += torch.sum(param ** 2)
                
                regLoss = regLoss / (Y_abs ** 2)


                if iterIndex % 10 == 0:
                    
                    #if a > 20:
                    #print (rand1)
                    if batch_index == 0:
                        print ('iter:', iterIndex)

                        with torch.no_grad():
                            X_pred = X[:, 0]
                            Y = model(X_pred, xPhen)

                            if True:
                                Y = removeIndependence(Y, Y_background)

                        

                        heritability_train = cheapHeritability(Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0])
                        if 1 in trainTest2:
                            heritability_test = cheapHeritability(Y[trainTest2 == 1], names[trainTest2 == 1], envirement[trainTest2 == 1])
                        


                        print ('subset_phen', subset_phen)
                        print ('heritability_fancy', heritability_fancy)
                        print (heritability_train.cpu().data.numpy())
                        if 1 in trainTest2:
                            print (heritability_test.cpu().data.numpy())
                
                loss = loss + (regLoss * regScale)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iterIndex % 10 == 0:
                    torch.save(model, modelName)
            



    if 1 in trainTest2:
        return (heritability_train, heritability_test)
    else:
        return (heritability_train)
    





def multiModalTrain():


    file1 = './data/miscPlant/inputs/data_process/Msi_C_Japan_2019-URB-Plant_Data.tsv'

    data_trait = np.loadtxt(file1, delimiter='\t', dtype=str, skiprows=1)

    with open(file1, 'r') as file:
        lines = file.readlines()
    row1 = np.array(lines[0].split('\t'))


    plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=False)


    #print (data_trait[0])
    argBiomass = np.argwhere(data_trait[0] == '268.7')[0, 0]
    argAIL = np.argwhere(data_trait[0] == '16.2')[0, 0]
    argCulmDensity = np.argwhere(data_trait[0] == '0.0055')[0, 0]
    argCulmDryWeight = np.argwhere(data_trait[0] == '18.5')[0, 0]
    argCulmNodeNum = np.argwhere(data_trait[0] == '12')[0, 0]
    argDiamBasal = np.argwhere(data_trait[0] == '6.08')[0, 0]
    argDiamInter = np.argwhere(data_trait[0] == '3.17')[0, 0]
    argCULM = np.argwhere(data_trait[0] == '194.6')[0, 0] #3.17, 194.6
    argBasalCirc = np.argwhere(data_trait[0] == '118.0')[0, 0] 
    argDormancy = np.argwhere(data_trait[0] == '17-Oct-20')[0, 0] 
    argHeadDate = np.argwhere(data_trait[0] == '25-Aug-2020')[0, 0] 
    arg50HeadDate = np.argwhere(data_trait[0] == '1-Sep-2020')[0, 0] 
    argHardiness = np.argwhere(data_trait[0] == '100')[0, 0] 
    argSpringRegrowth = np.argwhere(data_trait[0] == '17-Apr-2020')[0, 0] 

    argSurvival = np.argwhere(row1 == 'Survival (Oct __07-_ 2019)')[0, 0] 
    argLeftCollect = np.argwhere(row1 == 'Leaf Sample collected(__Oct 2019)')[0, 0] 
    argDNAextract = np.argwhere(row1 == 'Select for DNA Extraction ')[0, 0] 
    argSurvivalSpring = np.argwhere(row1 == 'Survival (spring 2020)')[0, 0] 
    

    
    traitList = ['biomass', 'Average Internode Length', 'Culm Density', 'Culm Dry Weight', 'Culm node num', 'Culm Outer Diameter at the Basal Internode', 'Calm Outer Diameter at last internode',
                 'Height', 'Basal circumference', 'Dormancy date', 'First Heading Date', '50% Heading Date', 'Hardiness score', 'Spring Regrowth Date', 'Survival October', 'Survival Spring']
    

    
    #trait1 = data_trait[:, argCULM]
    trait1 = data_trait[:, argBiomass]


    argGood = np.argwhere(trait1 != '')[:, 0]
    badOptions = np.array(['Missing', 'no stem', 'no stems', 'NA'])
    argGood = argGood[np.isin(trait1[argGood], badOptions) == False]
    #trait1 = trait1[argGood]


    namePart = 'central'
    

    imageDataAll, names = MSIC_loader(doSouth=False)

    names_subset = names[argGood]
    _, names_counts = np.unique(names_subset, return_counts=True)
    _, names_inverse = np.unique(names_subset, return_inverse=True)
    names_counts = names_counts[names_inverse]
    argGood = argGood[names_counts >= 2]

    names, imageDataAll, trait1 = names[argGood], imageDataAll[argGood], trait1[argGood].astype(float)

    trait1 = trait1 - np.mean(trait1)
    trait1 = trait1 / np.mean(np.abs(trait1))


    #TODO REMVOE ZEROS
    trait1 = torch.tensor(trait1.reshape((-1, 1))  ).float() #* 0
    

    #print (torch.mean(trait1))
    #quit()

    

    plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=False)


    np.random.seed(1)
    crossValid = np.random.randint(5, size=names.shape[0])



    #2 south extremely good. ~0.25
    for predNum in [2]:# range(14): #[4]:

        print ("#####################################")
        print ('simple time point: ', predNum)
        print ("#####################################")

        for splitIndex in range(np.unique(crossValid).shape[0]):

            print ("#####################################")
            print ('splitIndex: ', splitIndex)
            print ("#####################################")

            trainTest2 = np.zeros(crossValid.shape[0], dtype=int)
            trainTest2[crossValid == splitIndex] = 1

            
            rowGroup = plotRow.astype(int) % 130
            
            

            imageData = imageDataAll[:, predNum:predNum+1, :, :, :]
            #imageData = imageData[:, :, 0:1] 
            imageData = imageData.reshape((imageData.shape[0], 1, imageData.shape[1]*imageData.shape[2],  imageData.shape[3], imageData.shape[4]  ))


            means1 = np.mean(imageData, axis=(0, 1, 3, 4))
            imageData = imageData / means1.reshape((1, 1, -1, 1, 1))
            imageData = imageData / 3


            #imageData = imageData * 0 #TODO REMOVE ZEROS


            envirement = np.array([plotRep_C, rowGroup]).T
            envirement = envirement[argGood]
            loadModel = ''


            modelName = './data/miscPlant/models/MSI_' + namePart + '_singlepoint_' + str(predNum) + '_split_' + str(splitIndex) + '_JointModel.pt' 

            #regScale = 5e-5 #good
            regScale = 2e-4 
            learningRate=5e-5 #good

            Niter = 1000
            
            Nchannel = imageData.shape[2]


            
            if loadModel != '':
                model = torch.load(loadModel)
            else:
                model = MultiModal(1, 6, 1)

            Nphen = 1

            NphenStart = 0

            multiModal_trainModel(model, imageData, trait1, names, plotRow, envirement, trainTest2, modelName, Niter = Niter, doPrint=True, regScale=regScale, learningRate=learningRate, Nphen=Nphen, NphenStart=NphenStart)
            quit()
    quit()

#multiModalTrain()
#quit()




def both_trainModel(model, imageData_C, imageData_S, names_S, names_C, plotRow_C, plotRow_S, trainTest_C, trainTest_S, modelName, Niter = 10000, doPrint=True, regScale=1e-8, learningRate=1e-4, Nphen=1, corVar=False, NphenStart=0):

    


    import torchvision.transforms as transforms
    transformCrop = transforms.RandomCrop(size=(55, 55)) 

    losses = []

    error_train = []
    error_test = []

    mps_device = torch.device("mps")

    model.to(mps_device)

    imageData_C = torch.tensor(imageData_C).float()
    imageData_C = imageData_C.to(mps_device)


    imageData_S = torch.tensor(imageData_S).float()
    imageData_S = imageData_S.to(mps_device)

    



    for phenNow in range(NphenStart, Nphen):

        


        if phenNow > 0:
            subset1 = np.arange(phenNow)
            
            with torch.no_grad():
                Y_background_C = model(imageData_C[:, :, 4:4+55, 4:4+55], subset1)
                Y_background_C = normalizeIndependent(Y_background_C)

                Y_background_S = model(imageData_S[:, :, 4:4+55, 4:4+55], subset1)
                Y_background_S = normalizeIndependent(Y_background_S)

        else:
            Y_background_C = torch.zeros((imageData_C.shape[0]), 0)
            Y_background_S = torch.zeros((imageData_S.shape[0]), 0)

        
        subset_phen = np.zeros(1, dtype=int) + phenNow

        optimizer = torch.optim.RMSprop(model.parameters(), lr = learningRate)
        

        for a in range(Niter):



                X_C = imageData_C[trainTest_C == 0]
                X_S = imageData_S[trainTest_S == 0]

                if True:
                    
                    rand_coef = 0.05
                    rand_C = torch.rand(X_C.shape).to(X_C.device) * rand_coef
                    rand_S = torch.rand(X_S.shape).to(X_S.device) * rand_coef
                    
                    X_C = X_C + rand_C
                    X_S = X_S + rand_S


                if True:
                    X_C = torch.stack([transformCrop(image ) for image in X_C])
                    X_S = torch.stack([transformCrop(image ) for image in X_S])


                
                Y_C = model(X_C, subset_phen)
                Y_S = model(X_S, subset_phen)

                Y_C_abs = torch.mean(torch.abs(Y_C  - torch.mean(Y_C, axis=0).reshape((-1, 1)) ))
                Y_S_abs = torch.mean(torch.abs(Y_S  - torch.mean(Y_S, axis=0).reshape((-1, 1)) ))


                if True:
                    Y_C = removeIndependence(Y_C, Y_background_C[trainTest_C == 0])
                    Y_S = removeIndependence(Y_S, Y_background_S[trainTest_S == 0])

                Y_C = normalizeIndependent(Y_C)
                Y_S = normalizeIndependent(Y_S)

                Y_C[Y_C<-2] = -2
                Y_C[Y_C>2] = 2
                Y_S[Y_S<-2] = -2
                Y_S[Y_S>2] = 2

                
                envirement = torch.zeros(0)
                
                #heritability_C = cheapHeritability(Y_C, names_C[trainTest_C == 0], envirement)
                #heritability_S = cheapHeritability(Y_S, names_S[trainTest_S == 0], envirement)
                heritability_C = rowHerit(plotRow_C[trainTest_C == 0], Y_C, names_C[trainTest_C == 0], envirement)
                heritability_S = rowHerit(plotRow_S[trainTest_S == 0], Y_S, names_S[trainTest_S == 0], envirement)

                

                
                heritMean = (torch.mean(heritability_C) + torch.mean(heritability_S)) / 2.0

                #heritMean = ((torch.mean(heritability_C) * 2.0) + torch.mean(heritability_S)) / 3.0
                #heritMean = torch.mean(heritability_C)
                loss = -1 * heritMean

                #print (heritMean.data.numpy(), heritMean_base.data.numpy())
                
                
                regLoss = 0
                for param in model.parameters():
                    regLoss += torch.sum(param ** 2)
                regLoss = regLoss ** 0.5

                regLoss = regLoss / (Y_C_abs + Y_S_abs)
                

                if a % 10 == 0:
                    
                    print ('iter:', a)

                    with torch.no_grad():
                        X_pred_C = torch.stack([transformCrop(image ) for image in imageData_C])
                        X_pred_S = torch.stack([transformCrop(image ) for image in imageData_S])

                        #X_pred_C = imageData_C[:, :, 4:4+55, 4:4+55]
                        #X_pred_S = imageData_S[:, :, 4:4+55, 4:4+55]

                        Y_C = model(X_pred_C, subset_phen)
                        Y_S = model(X_pred_S, subset_phen)

                        
                        if True:
                            Y_C = removeIndependence(Y_C, Y_background_C)
                            Y_S = removeIndependence(Y_S, Y_background_S)

                        Y_C = normalizeIndependent(Y_C)
                        Y_S = normalizeIndependent(Y_S)

                        Y_C[Y_C<-2] = -2
                        Y_C[Y_C>2] = 2
                        Y_S[Y_S<-2] = -2
                        Y_S[Y_S>2] = 2


                    #print (Y_C[trainTest_C == 0].shape, names_C[trainTest_C == 0].shape)
                    #print (Y_C[trainTest_C == 0][:5], names_C[trainTest_C == 0][:5])

                    heritability_train_C = rowHerit(plotRow_C[trainTest_C == 0], Y_C[trainTest_C == 0], names_C[trainTest_C == 0], envirement)
                    heritability_test_C = rowHerit(plotRow_C[trainTest_C == 1], Y_C[trainTest_C == 1], names_C[trainTest_C == 1], envirement)
                    heritability_train_S = rowHerit(plotRow_C[trainTest_S == 0], Y_S[trainTest_S == 0], names_S[trainTest_S == 0], envirement)
                    heritability_test_S = rowHerit(plotRow_C[trainTest_S == 1], Y_S[trainTest_S == 1], names_S[trainTest_S == 1], envirement)

                    #heritability_train_C = cheapHeritability(Y_C[trainTest_C == 0], names_C[trainTest_C == 0], envirement )
                    #heritability_test_C = cheapHeritability(Y_C[trainTest_C == 1], names_C[trainTest_C == 1], envirement )
                    #heritability_train_S = cheapHeritability(Y_S[trainTest_S == 0], names_S[trainTest_S == 0], envirement )
                    #heritability_test_S = cheapHeritability(Y_S[trainTest_S == 1], names_S[trainTest_S == 1], envirement )

                    heritability_train_C = heritability_train_C.cpu().data.numpy()[0]
                    heritability_test_C = heritability_test_C.cpu().data.numpy()[0]
                    heritability_train_S = heritability_train_S.cpu().data.numpy()[0]
                    heritability_test_S = heritability_test_S.cpu().data.numpy()[0]

                    
                    
                    print ('subset_phen', subset_phen)
                    print ("central", heritability_train_C, heritability_test_C)
                    print ("south", heritability_train_S, heritability_test_S)

                    #quit()
                


                loss = loss + (regLoss * regScale)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if a % 10 == 0:
                    torch.save(model, modelName)
            



    #if 1 in trainTest2:
    #    return (heritability_train, heritability_test)
    #else:
    #    return (heritability_train)
    True


def saveLastImage():

    time1 = time.time()
    image = loadnpz('./data/imageDataset/images_npz/plot_' + '1' + '.npz')

    Ninclude = 16
    imageData = np.zeros((2000, 107, 107, Ninclude, 6))#, image.shape[2]))
    for index1 in range(1, 2001):
        index1_str = str(index1)
        print (index1_str)
        image = loadnpz('./data/imageDataset/images_npz/plot_' + index1_str + '.npz')
        image = image[:107, :107]
        image = image[:, :, -(Ninclude * 6):]
        image = image.reshape((image.shape[0], image.shape[1], Ninclude, 6 ))
        #image = image.reshape((  image.shape[0],  ))
        #print (image.shape)
        #quit()
        imageData[index1-1] = image #[:107, :107, -6:]

    print (time.time() - time1)

    print (imageData.shape)
    #np.savez_compressed('./data/imageDataset/image_subsets/full_images_last' + str(Ninclude) + '.npz', imageData)
    np.savez_compressed('./data/imageDataset/image_subsets/full_images_all.npz', imageData)


#saveLastImage()
#quit()


def MSA_loadImages():


    #modelName = './data/models/image/9.pt'

    #model = torch.load(modelName).cpu()


    location_file = './data/miscPlant/inputs_MSA/MSA_referenceIDs.csv'
    measure_file = './data/miscPlant/inputs_MSA/MSA_GT.csv'

    index1_str = '1'

    mps_device = torch.device("mps")


    measurements = np.loadtxt(measure_file, delimiter=',', dtype=str)

    arg1 = np.argwhere(measurements[0] == 'autum_dorm')[0, 0]
    arg2 = np.argwhere(measurements[0] == 'biomass')[0, 0]

    measurements_phenotypes = measurements[1:, arg1:arg2+1]

    #biomass = measurements_phenotypes[:, -1]


    measurements = np.loadtxt(measure_file, delimiter=',', dtype=str)


    if False:
        envirement = np.loadtxt(location_file, delimiter=',', dtype=str)


        #envirement = envirement[1:, 1]
        envirement = envirement[1:, 1:4]

        for a in range(envirement.shape[0]):
            str1 = envirement[a, -1]
            str1 = str1.replace('A', '')
            str1 = str1.replace('B', '')
            str1 = str1.replace('C', '')
            str1 = str1.replace('D', '')
            envirement[a, -1] = str1




    names = measurements[1:, -1]
    unique_names, inverse1 = np.unique(names, return_inverse=True)

    #imageData = loadnpz('./data/imageDataset/image_subsets/full_images_last5.npz')
    imageData = loadnpz('./data/miscPlant/inputs_MSA/image_subsets/full_images_all.npz')

    print (imageData.shape)



    #print (names.shape)
    #print (unique_names)
    #quit()

    #print (imageData.shape)


    imageData = np.swapaxes(imageData, 2, 3)
    imageData = np.swapaxes(imageData, 1, 2)

    imageData = np.swapaxes(imageData, 2+1, 3+1)
    imageData = np.swapaxes(imageData, 1+1, 2+1)

    imageData[imageData < -100] = 0
    imageData[imageData > 100] = 0


    imageData = imageData - np.mean(imageData)
    imageData = imageData / np.mean(np.abs(imageData))

    print (imageData.shape)
    quit()







def matchTrain():

    modelName = './data/miscPlant/models/MSI_singlepoint_5_matchBiomass.pt'

    data_trait = np.loadtxt('./data/miscPlant/inputs/data_process/Msi_C_Japan_2019-URB-Plant_Data.tsv', delimiter='\t', dtype=str, skiprows=1)

    argBiomass = np.argwhere(data_trait[0] == '268.7')[0, 0]
    argAIL = np.argwhere(data_trait[0] == '16.2')[0, 0]
    argCULM = np.argwhere(data_trait[0] == '118.0')[0, 0] #3.17, 194.6

    
    biomass = data_trait[:, argBiomass]
    AIL = data_trait[:, argAIL]
    culm = data_trait[:, argCULM]


    trait1 = biomass
    

    argGood = np.argwhere(trait1 != '')[:, 0]


    imageData, names = MSIC_loader()

    _, counts1 = np.unique(names[argGood], return_counts=True)
    _, inverse1 =  np.unique(names[argGood], return_inverse=True)
    argGood = argGood[ counts1[inverse1] >= 2 ]

    imageData, names, trait1 = imageData[argGood], names[argGood], trait1[argGood]

    
    
    trait1 = trait1.astype(float)
    trait1 = trait1 / np.mean(np.abs(trait1))
    #print (np.mean(trait1))
    

    envirement = np.zeros((imageData.shape[0], 0))

    trait1 = torch.tensor(trait1.astype(float)  ).float()
    trait1 = trait1.reshape((-1, 1))


    imageData = imageData[:, 5:6, :, :, :]
    

    imageData = imageData.reshape((imageData.shape[0], 1, imageData.shape[1]*imageData.shape[2],  imageData.shape[3], imageData.shape[4]  ))

    means1 = np.mean(imageData, axis=(0, 1, 3, 4))
    imageData = imageData / means1.reshape((1, 1, -1, 1, 1))

    imageData = imageData / 3

    X = imageData

    
    #np.random.seed(0)
    np.random.seed(1)

    names_unique, names_inverse = np.unique(names, return_inverse=True)
    trainTest2 = np.random.randint(3, size=names_unique.shape[0])
    trainTest2[trainTest2 == 2] = 0
    trainTest2 = trainTest2[names_inverse]

    #learningRate=1e-3
    learningRate=1e-4
    #learningRate=1e-5
    #learningRate=5e-6
    #learningRate=5e-6
    #learningRate=1e-7

    Nphen = 1
    #Niter = 10000
    Niter = 1000


    print (torch.mean(trait1[trainTest2==0]))
    print (torch.mean(trait1[trainTest2==1]))




    numWavelengths = X.shape[1]

    
    model = convModel(Nphen)

    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate, betas=(0.9, 0.98)) #TODO REMOVE, trying it for coherit
    #optimizer = torch.optim.RMSprop(model.parameters(), lr = learningRate) #Typical
    #optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)

    #if not isinstance(corVar, bool):
    #    corVar = corVar.reshape((-1, 1))

    #Niter = 10000

    import torchvision.transforms as transforms 
    transformFlip1 = transforms.RandomHorizontalFlip() 
    transformFlip2 = transforms.RandomVerticalFlip() 
    transformCrop = transforms.RandomCrop(size=(55, 55)) 

    losses = []

    error_train = []
    error_test = []

    mps_device = torch.device("mps")

    model.to(mps_device)

    X = torch.tensor(X).float()
        
    X = X.to(mps_device)

    trait1 = trait1.to(X.device)

    #batch_size = -1#300 #-1
    batch_size = 100

    argTrain = np.argwhere(trainTest2 == 0)[:, 0]
    argTest = np.argwhere(trainTest2 == 1)[:, 0]

    #names_unique, counts_unique = np.unique(names, return_counts=True)
    #print (counts_unique.shape)
    #quit()

    

    

    if batch_size < 0:
        batch_size = argTrain.shape[0]
    

    for a in range(Niter):

        print ('iter:', a)

        for batch_index in range(argTrain.shape[0] // batch_size):

            argNow = argTrain[batch_index*batch_size:(batch_index+1)*batch_size]

            

            if True:
                X_tensor = X[argNow, 0]
                rand1 = torch.rand(X_tensor.shape).to(X_tensor.device)
                X_tensor = torch.stack([transformCrop(image ) for image in X_tensor])

            
            Y = model(X_tensor)
            

            loss = torch.mean( (Y - trait1[argNow]) ** 2 )

            print (scipy.stats.pearsonr( Y.cpu().detach().data.numpy() , trait1[argNow].cpu().detach().data.numpy() ))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        torch.save(model, modelName)






#matchTrain()
#quit()




def OLD_allSimpleTrain():



    imageAll_C, names_C = MSIC_loader(doSouth=False)
    imageAll_S, names_S = MSIC_loader(doSouth=True)

    plotNum_C, halfsib_C, plotRow_C, plantName_C = loadMSInames(doSouth=False)
    plotNum_C, halfsib_C, plotRow_S, plantName_C = loadMSInames(doSouth=True)

    #imageAll_S = imageAll_S * 0
    #names_S = names_S[np.arange(names_S.shape[0]) % 2]

    for predNum in range(14):

        for a in range(5):
            print ('')

        print ("#####################################")
        print ('time point: ', predNum)
        print ("#####################################")
        
        for a in range(5):
            print ('')

        imageData_C = imageAll_C[:, predNum, :, :, :]
        imageData_S = imageAll_S[:, predNum, :, :, :]

        #means1 = (np.mean(imageData_C, axis=(0, 2, 3)) + np.mean(imageData_S, axis=(0, 2, 3))) / 2.0

        means1 = np.mean(imageData_C, axis=(0, 2, 3))
                  
        means1 = means1.reshape((1, -1, 1, 1)) * 3

        imageData_C = imageData_C / means1
        #imageData_S = imageData_S / means1






        #np.random.seed(0)
        np.random.seed(1)

        #names_unique_C, names_inverse_C = np.unique(names_C, return_inverse=True)
        #trainTest_C = np.random.randint(3, size=names_C.shape[0])
        #trainTest_C[trainTest_C == 2] = 0
        #trainTest_C = trainTest_C[names_inverse_C]        

        #names_unique_S, names_inverse_S = np.unique(names_S, return_inverse=True)
        #trainTest_S = np.random.randint(3, size=names_S.shape[0])
        #trainTest_S[trainTest_S == 2] = 0
        #trainTest_S = trainTest_S[names_inverse_S]

        trainTest_C = rowSplitter(plotRow_C)
        trainTest_C[trainTest_C<=6] = 0
        trainTest_C[trainTest_C>=7] = 1

        trainTest_S = rowSplitter(plotRow_S)
        trainTest_S[trainTest_S<=6] = 0
        trainTest_S[trainTest_S>=7] = 1



        #trainTest_C[:] = 0
        #trainTest_S[:] = 0


        #modelName = './data/miscPlant/models/MSI_all_singlepoint_' + str(predNum) + '_split5.pt'
        modelName = './data/miscPlant/models/MSI_all_singlepoint_' + str(predNum) + '_3.pt'

        loadModel = ''



        regScale = 0.0
        #regScale = 1e-5
        #regScale = 1e-4
        #regScale = 1e-2

        learningRate=1e-3
        #learningRate=1e-4
        #learningRate=1e-5 #
        #learningRate=5e-6 
        #learningRate=2e-6
        #learningRate=1e-6 #
        #learningRate=1e-7

        Niter = 1000
        
        if loadModel != '':
            model = torch.load(loadModel)
        else:
            #model = multiConv(20, 6, convTime ) #Max phenotypes = 20
            model = multiConv(20, 6, convModel ) 


        #model = torch.load('./data/miscPlant/models/MSI_all_singlepoint_' + str(predNum) + '_split3.pt')

        #Nphen = 1
        Nphen = 1

        NphenStart = 0

        

        both_trainModel(model, imageData_C, imageData_S, names_S, names_C, plotRow_C, plotRow_S, trainTest_C, trainTest_S, modelName, Niter = Niter, doPrint=True, regScale=regScale, learningRate=learningRate, Nphen=Nphen, NphenStart=NphenStart)
    
        #quit()

#allSimpleTrain()
#quit()






def row_maxWaveMethod(X, plotRow, synthUsed, names, envirement, trainTest2):


    X_copy = np.copy(X[trainTest2 == 0])
    X_copy = X_copy - np.mean(X_copy, axis=0).reshape((1, -1))
    argBest_list = []
    for a in range(synthUsed):
        heritability_wave = rowHerit(plotRow[trainTest2 == 0], torch.tensor(X_copy).float() , names[trainTest2 == 0], envirement[trainTest2 == 0] )
        heritability_wave = heritability_wave.data.numpy()

        #print (heritability_wave)
        #quit()
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

    #print (Y[:10, 0])

    Y = normalizeIndependent( torch.tensor(Y).float() )
    Y_np = Y.data.numpy()

    #print (Y_np[:10, 0])
    #quit()

    return Y_np




def saveExample():

    import seaborn as sns

    miscNames = ['central', 'south']
    for dataIndex in range(2):
        miscName = miscNames[dataIndex]

        if dataIndex == 0:
            imageAll_C, names_C = MSIC_loader(doSouth=False)
            plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=False)
            namePart = 'central'
        else:
            imageAll_C, names_C = MSIC_loader(doSouth=True)
            plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=True)
            namePart = 'south'


        rowGroup = plotRow.astype(int) % 130
        #rowGroup = rowGroup #// 5 

        envirement = np.array([plotRep_C, rowGroup]).T
        
        for predNum in [2]:# range(14):

            #print ("#####################################")
            print ('time point: ', predNum)
            #print ("#####################################")
            
            imageData_C = imageAll_C[:, predNum, :, :, :]

            for imageIndex in range(100):
                for channel in range(6):
                    print (imageIndex, channel)
            
                    sns.heatmap(imageData_C[imageIndex, channel])
                    plt.show()



#saveExample()
#quit()

def saveForLMER():


    miscNames = ['central', 'south']
    for dataIndex in range(2):
        miscName = miscNames[dataIndex]

        if dataIndex == 0:
            imageAll_C, names_C = MSIC_loader(doSouth=False)
            plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=False)
            namePart = 'central'
        else:
            imageAll_C, names_C = MSIC_loader(doSouth=True)
            plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=True)
            namePart = 'south'


        rowGroup = plotRow.astype(int) % 130
        #rowGroup = rowGroup #// 5 

        envirement = np.array([plotRep_C, rowGroup]).T
        
        for predNum in [2]:# range(14):

            #print ("#####################################")
            print ('time point: ', predNum)
            #print ("#####################################")
            
            imageData_C = imageAll_C[:, predNum, :, :, :]
            means1 = np.mean(imageData_C, axis=(0, 2, 3))
            means1 = means1.reshape((1, -1, 1, 1)) * 3

            #for channel in range(6):
            #    print (channel)
            #    plt.hist( imageData_C[:, channel].reshape((-1,)), bins=100 )
            #    plt.show()

            #print (means1)
            #quit()
            imageData_C = imageData_C / means1

            imageData_C_subset = imageData_C[:, :, 1::3, 1::3]
            imageData_C_subset = imageData_C_subset.reshape((imageData_C_subset.shape[0],  imageData_C_subset.shape[1]*imageData_C_subset.shape[2]*imageData_C_subset.shape[3] ))

            imageData_C_subset = imageData_C_subset - np.mean(imageData_C_subset, axis=0).reshape((1, -1))
            imageData_C_subset = imageData_C_subset / (np.mean(imageData_C_subset**2, axis=0).reshape((1, -1)) ** 0.5)


            print (np.mean(imageData_C_subset**2, axis=0))

            traitNames = ['name2', 'col', 'row']

            values = np.array([names_C, plotRep_C, rowGroup ]).T.astype(str)

            print (values.shape)
            print (imageData_C_subset.shape)
            values = np.concatenate((  values, imageData_C_subset.astype(str)  ) , axis=1)


            #phenoNames = []
            for channel in range(6):
                for pixel_1 in range(21):
                    for pixel_2 in range(21):
                        traitNames.append('c_' + str(channel) + '_p_' + str(pixel_1) + '_' + str(pixel_2) )

                
                #Y = imageData_C[:, channel].reshape((imageData_C.shape[0], imageData_C.shape[2]*imageData_C.shape[3]))
                #Y = torch.tensor(Y).float()
                #heritList_train = cheapHeritability(Y, names_C, envirement)
                #heritList_train = heritList_train.data.numpy()
                #heritList_train = heritList_train.reshape((63, 63)) * 4
                #plt.imshow(heritList_train)
                #plt.show()

            traitNames = np.array(traitNames).reshape((1, -1))
            values = np.concatenate(( traitNames, values  ), axis=0)


            np.savetxt("./data/software/lmeHerit/input_misc_"  + miscName +  ".csv", values, delimiter=',', fmt='%s')

            print (values.shape)
            


#saveForLMER()
#quit()

def topSingleTrait():



    heritArray_trait = np.zeros((2, 2, 5, 14, 10))
    heritArray_PCA = np.zeros((2, 2, 5, 14, 10))
    heritArray_wave = np.zeros((2, 2, 5, 14, 10))

    heritArray_vegIndices = np.zeros((2, 2, 5, 14, 2))

    synthUsed = 10

    for dataIndex in range(2):

        if dataIndex == 0:
            imageAll_C, names_C = MSIC_loader(doSouth=False)
            plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=False)
            namePart = 'central'
        else:
            imageAll_C, names_C = MSIC_loader(doSouth=True)
            plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=True)
            namePart = 'south'

        


        rowGroup = plotRow.astype(int) % 130
        rowGroup = rowGroup #// 5 

        
        #plotUnique, counts1 = np.unique(plotRow, return_counts=True)
        #perm1 = np.random.permutation(plotUnique.shape[0])
        #perm2 = np.zeros(names_C.shape[0], dtype=int)
        #for a in range(perm1.shape[0]):
        #    args1 = np.argwhere(plotRow == plotUnique[a])[:, 0]
        #    args2 = np.argwhere(plotRow == plotUnique[perm1[a]])[:, 0]
        #    perm2[args1] = args2 
        #imageAll_C, names_C, plotRow = imageAll_C[perm2], names_C[perm2], plotRow[perm2]


        
        np.random.seed(1)
        crossValid = np.random.randint(5, size=names_C.shape[0])





        for predNum in range(14):

            #print ("#####################################")
            print ('time point: ', predNum)
            #print ("#####################################")
            
            for splitIndex in range(5):

            
                trainTest2 = np.zeros(crossValid.shape[0], dtype=int)
                trainTest2[crossValid == splitIndex] = 1
                

                imageData_C = imageAll_C[:, predNum, :, :, :]
                #imageData_S = imageAll_S[:, predNum, :, :, :]

                means1 = np.mean(imageData_C, axis=(0, 2, 3))
                        
                means1 = means1.reshape((1, -1, 1, 1)) * 3

                imageData_C = imageData_C / means1
                #imageData_S = imageData_S / means1


                waveSum = np.mean(imageData_C, axis=(2, 3))
                waveSum = torch.tensor(waveSum).float()

                envirement = np.array([plotRep_C, rowGroup]).T


                if True:

                    #nd1=CSM, band2=blue, band3=green, band4=red, band5=rededge, band6=nir). 

                    red_value = waveSum[:, 3]
                    redEdge_value = waveSum[:, 4]
                    NIR_value = waveSum[:, 5]

                    NDRE = (NIR_value - redEdge_value) /  (NIR_value + redEdge_value)
                    NDVI = (NIR_value - red_value) /  (NIR_value + red_value)
                    vegIndices = torch.stack([NDVI, NDRE]).T

                    heritList_train = cheapHeritability(vegIndices[trainTest2 == 0], names_C[trainTest2 == 0], envirement[trainTest2 == 0])
                    heritList_train = heritList_train.data.numpy()
                    heritList_test = cheapHeritability(vegIndices[trainTest2 == 1], names_C[trainTest2 == 1], envirement[trainTest2 == 1])
                    heritList_test = heritList_test.data.numpy()
                    heritArray_vegIndices[0, dataIndex, splitIndex, predNum] = heritList_train
                    heritArray_vegIndices[1, dataIndex, splitIndex, predNum] = heritList_test

                
                #np.random.seed(0)
                #np.random.seed(1)

                #trainTest_C = rowSplitter(plotRow)
                #trainTest_C[trainTest_C<=6] = 0
                #trainTest_C[trainTest_C>=7] = 1

                #names_unique_C, names_inverse_C = np.unique(names_C, return_inverse=True)
                #trainTest_C = np.random.randint(3, size=names_C.shape[0])
                #trainTest_C[trainTest_C == 2] = 0


                #trainTest_C[:] = 0 #TODO UNDO

                


                imageData_C = imageData_C.reshape((imageData_C.shape[0] , imageData_C.shape[1]*imageData_C.shape[2]*imageData_C.shape[3] ))


                imageData_C = torch.tensor(imageData_C).float()
                


                if False:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=10)
                    pca.fit(imageData_C[trainTest2 == 0])
                    Y_PCA = pca.transform(imageData_C)

                    np.savez_compressed('./data/miscPlant/eval/pred_PCA_' + namePart + '_' + str(predNum) + '.npz', Y_PCA)



                    Y = torch.tensor(Y_PCA).float()
                    
                    #heritList_train = rowHerit(plotRow[trainTest_C == 0], Y[trainTest_C == 0], names_C[trainTest_C == 0], envirement[trainTest_C == 0])
                    heritList_train = cheapHeritability(Y[trainTest2 == 0], names_C[trainTest2 == 0], envirement[trainTest2 == 0])
                    heritList_train = heritList_train.data.numpy()
                    #quit()
                    
                    #heritList_test = rowHerit(plotRow[trainTest_C == 1], Y[trainTest_C == 1], names_C[trainTest_C == 1], envirement[trainTest_C == 1])
                    heritList_test = cheapHeritability(Y[trainTest2 == 1], names_C[trainTest2 == 1], envirement[trainTest2 == 1])
                    heritList_test = heritList_test.data.numpy()
                    #print (heritList_train[0], heritList_test[0])
                    heritArray_PCA[0, dataIndex, splitIndex, predNum] = heritList_train
                    heritArray_PCA[1, dataIndex, splitIndex, predNum] = heritList_test


                #print (envirement.shape)

                if False:
                    Y = maxWaveMethod(imageData_C, synthUsed, names_C, envirement, trainTest2)

                    np.savez_compressed('./data/miscPlant/eval/pred_maxTrait_' + namePart + '_' + str(predNum) + '.npz', Y)

                    Y = torch.tensor(Y).float()
                    #heritList_train = rowHerit(plotRow[trainTest_C == 0], Y[trainTest_C == 0], names_C[trainTest_C == 0], envirement[trainTest_C == 0])
                    heritList_train = cheapHeritability(Y[trainTest2 == 0], names_C[trainTest2 == 0], envirement[trainTest2 == 0])
                    heritList_train = heritList_train.data.numpy()
                    #heritList_test = rowHerit(plotRow[trainTest_C == 1], Y[trainTest_C == 1], names_C[trainTest_C == 1], envirement[trainTest_C == 1])
                    heritList_test = cheapHeritability(Y[trainTest2 == 1], names_C[trainTest2 == 1], envirement[trainTest2 == 1])
                    heritList_test = heritList_test.data.numpy()
                    print (heritList_train[0], heritList_test[0])
                    heritArray_trait[0, dataIndex, splitIndex, predNum] = heritList_train
                    heritArray_trait[1, dataIndex, splitIndex, predNum] = heritList_test


                

                if True:

                    Y_wave = maxWaveMethod(waveSum, 2, names_C, envirement, trainTest2)

                    #print (np.mean(Y_wave))

                    print ('./data/miscPlant/eval/pred_maxWave_' + namePart + '_' + str(predNum) + '.npz')

                    np.savez_compressed('./data/miscPlant/eval/pred_maxWave_' + namePart + '_' + str(predNum) + '.npz', Y_wave)

                    #print (np.max(Y_wave))
                    Y_wave = torch.tensor(Y_wave).float()
                    #heritList_train = rowHerit(plotRow[trainTest_C == 0], Y_wave[trainTest_C == 0], names_C[trainTest_C == 0], envirement[trainTest_C == 0])
                    heritList_train = cheapHeritability(Y_wave[trainTest2 == 0], names_C[trainTest2 == 0], envirement[trainTest2 == 0])
                    heritList_train = heritList_train.data.numpy()
                    #heritList_test = rowHerit(plotRow[trainTest_C == 1], Y_wave[trainTest_C == 1], names_C[trainTest_C == 1], envirement[trainTest_C == 1])
                    heritList_test = cheapHeritability(Y_wave[trainTest2 == 1], names_C[trainTest2 == 1], envirement[trainTest2 == 1])
                    heritList_test = heritList_test.data.numpy()
                    print (heritList_train[0], heritList_test[0])
                    #quit()
                    heritArray_wave[0, dataIndex, splitIndex, predNum, :2] = heritList_train
                    heritArray_wave[1, dataIndex, splitIndex, predNum, :2] = heritList_test

    

    

    #np.savez_compressed('./data/miscPlant/eval/herits_maxTrait.npz', heritArray_trait)
    #np.savez_compressed('./data/miscPlant/eval/herits_maxWave.npz', heritArray_wave)
    #np.savez_compressed('./data/miscPlant/eval/herits_PCA.npz', heritArray_PCA)

    print ("Save")

    np.savez_compressed('./data/miscPlant/eval/herits_vegIndex.npz', heritArray_vegIndices)


#topSingleTrait()
#quit()









def evaluateHerit():


    heritArray = np.zeros((2, 2, 5, 14, 10))

    synthUsed = 10

    for dataIndex in [1]:# range(2):

        if dataIndex == 0:
            imageAll_C, names_C = MSIC_loader(doSouth=False)
            plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=False)
            namePart = 'central'
        else:
            imageAll_C, names_C = MSIC_loader(doSouth=True)
            plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=True)
            namePart = 'south'


        np.random.seed(1)
        crossValid = np.random.randint(5, size=names_C.shape[0])

        rowGroup = plotRow.astype(int) % 130

        for predNum in range(5):# range(14):
            
            #for predNum in [10]:
            #for predNum in [5]:

            #for a in range(5):
            #    print ('')

            #print ("#####################################")
            print ('time point: ', predNum)
            #print ("#####################################")
            
            for splitIndex in range(5):

                trainTest2 = np.zeros(crossValid.shape[0], dtype=int)
                trainTest2[crossValid == splitIndex] = 1

                #imageData_C = imageAll_C[:, predNum, :, :, :]
                #means1 = np.mean(imageData_C, axis=(0, 2, 3))
                #means1 = means1.reshape((1, -1, 1, 1)) * 3
                #imageData_C = imageData_C / means1

                #np.random.seed(0)
                #np.random.seed(1)

                #trainTest_C = rowSplitter(plotRow)
                #trainTest_C[trainTest_C<=6] = 0
                #trainTest_C[trainTest_C>=7] = 1

                #modelName = './data/miscPlant/models/MSI_' + namePart + '_singlepoint_' + str(predNum) + '_row13.pt'
                predName = './data/miscPlant/GWAS/MSI_' + namePart + '_singlepoint_' + str(predNum) + '_split_' + str(splitIndex) + '.npz' 
                Y = loadnpz(predName)
                Y = torch.tensor(Y).float()

                #imageData = torch.tensor(imageData_C).float()
                #model = torch.load(modelName).to('cpu')
                #Y = model(imageData, np.arange(1))
                #Y = normalizeIndependent(Y)
 
                
                envirement = np.array([plotRep_C, rowGroup]).T


                
                heritList_train = cheapHeritability(Y[trainTest2 == 0], names_C[trainTest2 == 0], envirement[trainTest2 == 0])
                heritList_train = heritList_train.data.numpy()
                #quit()
                
                #heritList_test = rowHerit(plotRow[trainTest_C == 1], Y[trainTest_C == 1], names_C[trainTest_C == 1], envirement[trainTest_C == 1])
                heritList_test = cheapHeritability(Y[trainTest2 == 1], names_C[trainTest2 == 1], envirement[trainTest2 == 1])
                heritList_test = heritList_test.data.numpy()
                print (heritList_train[0], heritList_test[0])
                heritArray[0, dataIndex, splitIndex, predNum, :heritList_train.shape[0] ] = heritList_train
                heritArray[1, dataIndex, splitIndex, predNum, :heritList_test.shape[0]] = heritList_test

            
    #np.savez_compressed('./data/miscPlant/eval/herits_H2Opt.npz', heritArray)



#evaluateHerit()
#quit()


def interpreteH2Opt():

    for dataIndex in [0]:

        if dataIndex == 0:
            imageAll_C, names_C = MSIC_loader(doSouth=False)
            namePart = 'central'
        else:
            imageAll_C, names_C = MSIC_loader(doSouth=True)
            namePart = 'south'


        dateNum = 2
        predName = './data/miscPlant/GWAS/MSI_' + namePart + '_singlepoint_' + str(dateNum) + '_split_' + str(0) + '.npz' 
        Y = loadnpz(predName)
        imageAll_C = imageAll_C[:, dateNum]

        imageAll_C = imageAll_C / np.mean(imageAll_C, axis=(0, 2, 3)).reshape((1, -1, 1, 1))

        argLow = np.argsort(Y[:, 0])[:10]
        argHigh = np.argsort(Y[:, 0])[-10:]

        image_low = imageAll_C[argLow]
        image_high = imageAll_C[argHigh]

        image_low = np.swapaxes(image_low, 1, 2)
        image_high = np.swapaxes(image_high, 1, 2)

        image_both = np.concatenate( (  image_low[:, :, 1], image_high[:, :, 1]  ) , axis=0)
        image_both = image_both.reshape(( 2, image_both.shape[0] // 2, image_both.shape[1], image_both.shape[2]  ))
        image_both = np.swapaxes(image_both, 1, 2)
        image_both = image_both.reshape(( image_both.shape[0]*image_both.shape[1], image_both.shape[2]*image_both.shape[3]  ))


        plt.imshow(image_both)
        plt.show()
        quit()

        image_low = image_low.reshape(( image_low.shape[0]*image_low.shape[1], image_low.shape[2]*image_low.shape[3]  ))
        image_high = image_high.reshape(( image_high.shape[0]*image_high.shape[1], image_high.shape[2]*image_high.shape[3]  ))

        #image_cat = np.concatenate((  image_low[:, ]  , image_high ))


        plt.imshow(image_low)
        plt.show()
        plt.imshow(image_high)
        plt.show()


        print (image_low.shape)
        print (image_high.shape)
        quit()


#interpreteH2Opt()
#quit()



def predictOutput():

    #doSouth = True
    doSouth = False


    namePart = 'central'
    if doSouth:
        namePart = 'south'


    #plotNum_C, halfsib_C, plotRow_C, plantName_C = loadMSInames()
    plotNum_C, halfsib_C, plotRow_C, plantName_C, plotRep_C = loadMSInames(doSouth=doSouth)

    both_name = []
    for a in range(plotRow_C.shape[0]):
        both_name.append( plotRow_C[a] + '_' +  plantName_C[a]  )
    both_name = np.array(both_name)

    if doSouth:
        #np.savez_compressed('./data/miscPlant/GWAS/names_MSI_S.npz', both_name)
        True
    else:
        #np.savez_compressed('./data/miscPlant/GWAS/names_MSI_C.npz', both_name)
        True
    #quit()

    #predNum = 13
    #predNum = 2

    #np.savez_compressed('./data/miscPlant/eval/herits_PCA.npz', heritArray_PCA)
    heritArray_trait = np.zeros((2, 2, 14, 10))


    np.random.seed(1)
    crossValid = np.random.randint(5, size=plotRow_C.shape[0])
    

    #for predNum in range(14):
    for predNum in [6]:

        for splitIndex in range(5):# ['all']:#

            if splitIndex != 'all':
                trainTest2 = np.zeros(crossValid.shape[0], dtype=int)
                trainTest2[crossValid == splitIndex] = 1


            imageData, names = MSIC_loader(doSouth=doSouth)
            plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=doSouth)

            

            #print (imageData.shape)
            #quit()


            #imageData, names = MSIC_loader()
            
            #imageData = imageData[:, predNum:predNum+1, :, :, :]

            imageData = imageData[:, predNum:predNum+1, :, :, :]
            
            
            #imageData = imageData[:, :, :1, :, :]

            imageData = imageData.reshape((imageData.shape[0], 1, imageData.shape[1]*imageData.shape[2],  imageData.shape[3], imageData.shape[4]  ))

            means1 = np.mean(imageData, axis=(0, 1, 3, 4))
            imageData = imageData / means1.reshape((1, 1, -1, 1, 1))

            imageData = imageData / 3

            np.random.seed(1)


            rowGroup = plotRow.astype(int) % 130
            
            #envirement = np.zeros((imageData.shape[0], 0))
            #envirement = plotRep_C.reshape((-1, 1))
            envirement = np.array([plotRep_C, rowGroup]).T

            #trainTest2 = np.random.randint(3, size=envirement.shape[0])
            #trainTest2[trainTest2 == 2] = 0

            #trainTest2 = rowSplitter(plotRow)
            #trainTest2[trainTest2<=6] = 0
            #trainTest2[trainTest2>=7] = 1

            #modelName = './data/miscPlant/models/MSI_singlepoint_' + str(predNum) + '_split2.pt'
            #modelName = './data/miscPlant/models/MSI_all_singlepoint_' + str(predNum) + '_2.pt' #_split5
            #modelName = './data/miscPlant/models/MSI_' + 'south' + '_singlepoint_' + str(predNum) + '_row1_copy.pt'
            #modelName = './data/miscPlant/models/MSI_' + namePart + '_singlepoint_' + str(predNum) + '_row13.pt'

            if splitIndex == 'all':
                modelName = './data/miscPlant/models/MSI_' + namePart + '_singlepoint_' + str(predNum) + '_noSplit.pt' 
            else:
                modelName = './data/miscPlant/models/MSI_' + namePart + '_singlepoint_' + str(predNum) + '_split_' + str(splitIndex) + '.pt' 

            #modelName = './data/miscPlant/models/MSI_' + namePart + '_singlepoint_' + str(predNum) + '_split_' + str(splitIndex) + '_CSM.pt' 

            
            #modelName = './data/miscPlant/models/MSI_singlepoint_skip2.pt'
            
            
            #model = torch.load(modelName)

            imageData = torch.tensor(imageData[:, 0]).float()
            #imageData = imageData[:, :, 4:4+55, 4:4+55]
            model = torch.load(modelName).to('cpu')
            Y = model(imageData, np.arange(5))

            #print (Y[:10])
            #print (torch.max(torch.abs(Y)))
            #quit()
            #quit()

            #print (Y[:10])
            #quit()

            Y = normalizeIndependent(Y)

            print (Y[:10])

            #Y[Y<-2] = -2
            #Y[Y>2] = 2

            Y = Y.data.numpy()


            #predName = './data/miscPlant/GWAS/MSI_' + namePart + '_singlepoint_' + str(predNum) + '_row13.npz'
            #predName = './data/miscPlant/GWAS/MSI_singlepoint_skip2.npz'

            if splitIndex == 'all':
                predName = './data/miscPlant/GWAS/MSI_' + namePart + '_singlepoint_' + str(predNum) + '_noSplit.npz'  #Was "models" instead of "GWAS" folder by mistake
            else:
                predName = './data/miscPlant/GWAS/MSI_' + namePart + '_singlepoint_' + str(predNum) + '_split_' + str(splitIndex) + '.npz' 


            #predName = './data/miscPlant/GWAS/MSI_' + namePart + '_singlepoint_' + str(predNum) + '_split_' + str(splitIndex) + '_CSM.npz' 

            print ('paused')
            quit()

            np.savez_compressed(predName, Y)



            #heritList_train = cheapHeritability(torch.tensor(Y[trainTest2==0]).float(), halfsib_C[trainTest2==0], envirement[trainTest2==0])
            #print (heritList_train)
            #heritList_test = cheapHeritability(torch.tensor(Y[trainTest2==1]).float(), halfsib_C[trainTest2==1], envirement[trainTest2==1])
            #print (heritList_test)


            #heritList_train = rowHerit(plotRow, torch.tensor(Y).float(), halfsib_C, envirement)
            #print (heritList_train)
            #heritList_test = rowHerit(plotRow[trainTest2==1], torch.tensor(Y[trainTest2==1]).float(), halfsib_C[trainTest2==1], envirement[trainTest2==1])
            #print (heritList_test)




#predictOutput()
#quit()







def simpleCoheritTrain():


    for timePoint in range(14):


        data_trait = np.loadtxt('./data/miscPlant/inputs/data_process/Msi_C_Japan_2019-URB-Plant_Data.tsv', delimiter='\t', dtype=str, skiprows=1)
        #data_trait = np.loadtxt('./data/miscPlant/inputs/data_process/Msi_S_Japan_2019-URB-Plant_Data.tsv', delimiter='\t', dtype=str, skiprows=1)

        argBiomass = np.argwhere(data_trait[0] == '268.7')[0, 0]
        argAIL = np.argwhere(data_trait[0] == '16.2')[0, 0]
        #argCULM = np.argwhere(data_trait[0] == '194.6')[0, 0] #3.17, 194.6
        argCULM = np.argwhere(data_trait[0] == '100')[0, 0] #3.17, 194.6

        
        biomass = data_trait[:, argBiomass]
        AIL = data_trait[:, argAIL]
        culm = data_trait[:, argCULM]


        #trait1 = biomass
        trait1 = culm

        


        
        

        argGood = np.argwhere(trait1 != '')[:, 0]

        #print (trait1.shape)
        #print (argGood.shape)
        #quit()


        imageData, names = MSIC_loader()

        assert trait1.shape[0] == imageData.shape[0]

        _, counts1 = np.unique(names[argGood], return_counts=True)
        _, inverse1 =  np.unique(names[argGood], return_inverse=True)
        argGood = argGood[ counts1[inverse1] >= 2 ]

        imageData, names, trait1 = imageData[argGood], names[argGood], trait1[argGood]

        
        
        trait1 = trait1.astype(float)
        _, trait1 = np.unique(trait1, return_inverse=True) #
        trait1 = trait1 / np.mean(np.abs(trait1))
        #print (np.mean(trait1))


        #plt.hist(trait1, bins=100)
        #plt.show()
        #quit()
        

        envirement = np.zeros((imageData.shape[0], 0))


        

        trait1 = torch.tensor(trait1.astype(float)  ).float()
        trait1 = trait1.reshape((-1, 1))

        #print (torch.mean(trait1))
        #quit()
        

        #heritability_train = cheapHeritability(trait1, names, envirement )
        #print (heritability_train)
        #quit()

        
        
        imageData = imageData[:, timePoint:timePoint+1, :, :, :]

        #imageData = imageData + trait1.reshape((-1, 1, 1, 1, 1)).data.numpy() #TODO JUST TESTING UNDO!!!!
        

        imageData = imageData.reshape((imageData.shape[0], 1, imageData.shape[1]*imageData.shape[2],  imageData.shape[3], imageData.shape[4]  ))


        #print (np.max(imageData))


        means1 = np.mean(imageData, axis=(0, 1, 3, 4))
        imageData = imageData / means1.reshape((1, 1, -1, 1, 1))

        imageData = imageData / 3

        
        #np.random.seed(0)
        np.random.seed(1)
        #names_unique, names_inverse = np.unique(names, return_inverse=True)
        #trainTest2 = np.random.randint(3, size=names_unique.shape[0])
        #trainTest2[trainTest2 == 2] = 0
        #trainTest2 = trainTest2[names_inverse]


        trainTest2 = np.random.randint(3, size=names.shape[0])
        trainTest2[trainTest2 == 2] = 0


        #heritability_test_mod, _, _ = coherit(trait1[trainTest2 == 1],  trait1[trainTest2 == 1], names[trainTest2 == 1], envirement[trainTest2 == 1].reshape((-1, 1)) )
        #print (heritability_test_mod)
        #quit()

        #modelName = './data/miscPlant/models/MSI_6.pt'
        #modelName = './data/miscPlant/models/MSI_singlepoint_13.pt'
        #modelName = './data/miscPlant/models/MSI_singlepoint_' + str(timePoint) + '_coBiomass2.pt'
        modelName = './data/miscPlant/models/MSI_singlepoint_' + str(timePoint) + '_coHardy.pt'
        #loadModel = './data/miscPlant/models/MSI_singlepoint_' + str(timePoint) + '_matchBiomass.pt'

        regScale = 1e-20

        #regScale = 1e-4


        #learningRate=1e-3
        learningRate=1e-4
        #learningRate=1e-5
        #learningRate=5e-6
        #learningRate=1e-6
        #learningRate=1e-7
        #learningRate=1e-8

        Nphen = 1
        #Niter = 10000
        Niter = 100
        #Niter = 1000


        #print (torch.mean(trait1[trainTest2==0]))
        #print (torch.mean(trait1[trainTest2==1]))
        #quit()

        model = multiConv(Nphen, 6, convModel)

        trainModel(model, imageData, names, envirement, trainTest2, modelName, Niter = Niter, doPrint=True, regScale=regScale, learningRate=learningRate, Nphen=Nphen, corVar=trait1)


    
    quit()

    #0
    #[0.12793887 0.07193341 0.01645868 0.00587667 0.01844362]
    #[0.11331183 0.08526807 0.02000231 0.02440868 0.00873814]

    #2
    #[0.23275156 0.12234952 0.06279992 0.04856788 0.03429307]
    #[0.21625131 0.11829969 0.11321502 0.07261565 0.05629748]


    imageData = torch.tensor(imageData[:, 0]).float()
    imageData = imageData[:, :, 4:4+55, 4:4+55]
    model = torch.load(modelName).to('cpu')
    Y = model(imageData)

    Y = normalizeIndependent(Y)

    Y[Y<-2] = -2
    Y[Y>2] = 2

    #plt.hist(Y[:, 1].data.numpy())
    #plt.show()


    heritability_train = cheapHeritability(Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0])
    heritability_train = heritability_train.detach().data.numpy()
    print (heritability_train)
    heritability_test = cheapHeritability(Y[trainTest2 == 1], names[trainTest2 == 1], envirement[trainTest2 == 1])
    heritability_test = heritability_test.detach().data.numpy()
    print (heritability_test)


    plt.plot(heritability_train * 4)
    plt.plot(heritability_test * 4)
    plt.xlabel('trait')
    plt.ylabel('heritability')
    plt.legend(['training set', 'test set'])
    plt.show()

    quit()
    

    quit()


#simpleCoheritTrain()
#quit()




def OLD_showTimesHerit():


    #timeChecks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    #timeChecks = np.arange(14)

    #dates = ['MSI_09042020_processed_ALLstack.tif', 'MSI_05062020_processed_ALLstack.tif', 'MSI_05222020_processed_ALLstack.tif', 'MSI_09192020_processed_ALLstack.tif', 'MSI_07022020_processed_ALLstack.tif', 'MSI_11282020_processed_ALLstack.tif', 'MSI_07242020_processed_ALLstack.tif', 'MSI_10052020_processed_ALLstack.tif', 'MSI_08182020_processed_ALLstack.tif', 'MSI_06182020_processed_ALLstack.tif', 'MSI_07102020_processed_ALLstack.tif', 'MSI_06082020_processed_ALLstack.tif', 'MSI_08082020_processed_ALLstack.tif', 'MSI_11062020_processed_ALLstack.tif']
    #dates = np.array(dates)
    #dates = np.sort(dates)
    #_, dates_inverse = np.unique(dates, return_inverse=True)

    #for a in range(dates.shape[0]):
    #    date1 = dates[a]
    #    date1 = date1.split('_')[1]
    #    dates[a] = date1

    #print (dates_inverse[10])
    #quit()
    #print (dates_inverse)
    #print (np.argsort(dates))
    #print (np.sort(dates))
    #quit()


    heritArray = np.zeros((2, 2, 14, 10))

    for dataIndex in range(2):

        if dataIndex == 0:
            imageData_full, names = MSIC_loader(doSouth=False)
            #plotNum_C, halfsib_C, plotRow, plantName_C = loadMSInames(doSouth=False)
            plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=False)
            namePart = 'central'
        else:
            imageData_full, names = MSIC_loader(doSouth=True)
            plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=False)
            #plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=True)
            namePart = 'south'

        #heritTrains = []
        #heritTests = []

        traitNum1 = 0

        

        for a in range( 14):#  len(timeChecks)):
            time1 = a #timeChecks[a]


            rowGroup = plotRow.astype(int) % 130
            envirement = np.array([plotRep_C, rowGroup]).T 
            #envirement = np.zeros((imageData_full.shape[0], 0))

            #np.random.seed(0)
            #trainTest2 = np.random.randint(3, size=names.shape[0])
            #trainTest2[trainTest2 == 2] = 0

            np.random.seed(1)
            trainTest2 = rowSplitter(plotRow)
            trainTest2[trainTest2<=6] = 0
            trainTest2[trainTest2>=7] = 1


            #names_unique, names_inverse = np.unique(names, return_inverse=True)
            #trainTest2 = np.random.randint(3, size=names_unique.shape[0])
            #trainTest2[trainTest2 == 2] = 0
            #trainTest2 = trainTest2[names_inverse]



            #predName = './data/miscPlant/GWAS/MSI_singlepoint_' + str(time1) + '_split2.npz'
            #predName = './data/miscPlant/GWAS/MSI_all_singlepoint_' + str(time1) + '_2.npz'
            #predName = './data/miscPlant/GWAS/MSI_south_singlepoint_' + str(time1) + '_split2.npz'
            predName = './data/miscPlant/models/MSI_' + namePart + '_singlepoint_' + str(time1) + '_row13.npz'

            Y = loadnpz(predName)
            Y = torch.tensor(Y).float()
            Y = normalizeIndependent(Y)

            #print (Y.shape)
            #quit()

            

            heritability_train = cheapHeritability(Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0])
            heritability_train = heritability_train.detach().data.numpy()
            #print (heritability_train)
            #heritTrains.append(heritability_train[traitNum1])
            heritability_test = cheapHeritability(Y[trainTest2 == 1], names[trainTest2 == 1], envirement[trainTest2 == 1])
            heritability_test = heritability_test.detach().data.numpy()
            #print (heritability_test)
            #heritTests.append(heritability_test[traitNum1])


            heritArray[0, dataIndex, time1] = heritability_train
            heritArray[1, dataIndex, time1] = heritability_test

            #quit()


        #print (heritTests)

        #heritTrains = np.array(heritTrains) * 4
        #heritTests = np.array(heritTests) * 4

    print (heritArray)
    #predName = './data/miscPlant/GWAS/MSI_all_singlepoint_' + str(time1) + '_2.npz'
    np.savez_compressed('./data/miscPlant/eval/herits_all.npz', heritArray)
    quit()


    #date_checks = dates_inverse[np.array(timeChecks )]

    #plt.plot(date_checks, heritTrains)
    #plt.plot(date_checks, heritTests)
    plt.scatter(np.arange(heritTrains.shape[0]), heritTrains)
    plt.scatter(np.arange(heritTests.shape[0]), heritTests)
    plt.xlabel('time point')
    plt.ylabel('heritability')
    plt.title('trait number ' + str(traitNum1))
    #plt.title('two times')
    #plt.xticks(np.arange(dates.shape[0]), dates)
    plt.legend(['training set', 'test set'])
    plt.show()








def days_since_initial(date_str, initial_date_str="01-Jan-20"):
    # Convert the input date and initial date to datetime objects
    date_format = "%d-%b-%y"
    date = datetime.strptime(date_str, date_format)
    initial_date = datetime.strptime(initial_date_str, date_format)
    delta = date - initial_date
    return delta.days


def convertDateNumeric(dormancyDate):

    for a in range(dormancyDate.shape[0]):
        example_date = dormancyDate[a]#example_date = "17-Oct-20"
        if len(example_date.split('-')) == 3:
            if len(example_date.split('-')[-1]) == 4:
                example_date = example_date.split('-')
                example_date[-1] = example_date[-1][2:]
                example_date = '-'.join(example_date)


            days = days_since_initial(example_date)
            dormancyDate[a] = days
        else:
            dormancyDate[a] = ''
    dormancyDate_valid = np.copy(dormancyDate[dormancyDate!=''])
    dormancyDate_valid = dormancyDate_valid.astype(int)
    dormancyDate_valid = dormancyDate_valid - np.min(dormancyDate_valid)
    dormancyDate[dormancyDate!=''] = dormancyDate_valid

    return dormancyDate



def timeCor():

    plotNum_C, halfsib, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=False)


    corMatrix = np.zeros((2, 14, 14, 5))
    for dataIndex in range(2):

        if dataIndex == 0:
            namePart = 'central'
        if dataIndex == 1:
            namePart = 'south'

    

        #Y1 = loadnpz('./data/miscPlant/GWAS/MSI_' + namePart + '_singlepoint_' + str(0) + '_split_' + str(0) + '.npz')
        #Y1 = loadnpz('./data/miscPlant/GWAS/MSI_' + namePart + '_singlepoint_' + str(0) + '_noSplit.npz' )
        Y1 = loadnpz('./data/miscPlant/GWAS/MSI_' + namePart + '_singlepoint_' + str(0) + '_split_' + str(0) + '.npz')
        Y_all = np.zeros( (14, 5, Y1.shape[0]) )
        for predNum in range(14):
            for splitIndex in range(5):
                Y1 = loadnpz('./data/miscPlant/GWAS/MSI_' + namePart + '_singlepoint_' + str(predNum) + '_split_' + str(splitIndex) + '.npz')
                #Y1 = loadnpz('./data/miscPlant/GWAS/MSI_' + namePart + '_singlepoint_' + str(predNum) + '_noSplit.npz' )
                Y_all[predNum, splitIndex] = Y1[:, 0]

        
        for predNum1 in range(14):
            for predNum2 in range(14):
                for splitIndex1 in range(5):
                    #for splitIndex2 in range(5):
                    #    if splitIndex1 != splitIndex2:

                    #corMatrix[predNum1, predNum2, splitIndex1, splitIndex2] = scipy.stats.pearsonr(Y_all[predNum1, splitIndex1], Y_all[predNum2, splitIndex2]  )[0]

                    Y1 = Y_all[predNum1, splitIndex1]
                    Y2 = Y_all[predNum2, splitIndex1]

                    #corMatrix[dataIndex, predNum1, predNum2, splitIndex1] = scipy.stats.pearsonr(Y1, Y2  )[0]

                    unique1 = np.unique(halfsib)
                    Y1_mean = np.zeros(unique1.shape[0])
                    Y2_mean = np.zeros(unique1.shape[0])
                    for a in range(unique1.shape[0]):
                        args1 = np.argwhere(halfsib == unique1[a])[:, 0]
                        Y1_mean[a] = np.mean(Y1[args1])
                        Y2_mean[a] = np.mean(Y2[args1])

                    corMatrix[dataIndex, predNum1, predNum2, splitIndex1] = scipy.stats.pearsonr(Y1_mean, Y2_mean  )[0]


        print (corMatrix.shape)

        plt.imshow(np.abs(corMatrix[dataIndex, :, :, 0]))
        plt.show()

    #np.savez_compressed('./data/miscPlant/eval/timeCor_synth.npz', corMatrix)
    np.savez_compressed('./data/miscPlant/eval/timeHalfSibCor_synth.npz', corMatrix)
    quit()


#timeCor()
#quit()



def timeCorCoherit():


    corMatrix = np.zeros((2, 14, 14, 5))
    for dataIndex in range(2):
        if dataIndex == 0:
            plotNum_C, accession_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=False)
        else:
            plotNum_C, accession_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=True)

        

        if dataIndex == 0:
            namePart = 'central'
        if dataIndex == 1:
            namePart = 'south'

        for predNum1 in range(14):
            print (predNum1)
            for predNum2 in range(14):
                for splitIndex in range(5):
                    #Y1 = loadnpz('./data/miscPlant/models/MSI_' + namePart + '_singlepoint_' + str(predNum1) + '_noSplit.npz' )[:, :1]
                    #Y2 = loadnpz('./data/miscPlant/models/MSI_' + namePart + '_singlepoint_' + str(predNum2) + '_noSplit.npz' )[:, :1]

                    Y1 = loadnpz('./data/miscPlant/GWAS/MSI_' + namePart + '_singlepoint_' + str(predNum1) + '_split_' + str(splitIndex) + '.npz')[:, :1]
                    Y2 = loadnpz('./data/miscPlant/GWAS/MSI_' + namePart + '_singlepoint_' + str(predNum2) + '_split_' + str(splitIndex) + '.npz')[:, :1]

                    Y1 = torch.tensor(Y1).float()
                    Y2 = torch.tensor(Y2).float()

                    rowGroup = plotRow.astype(int) % 130
                    envirement = np.array([plotRep_C, rowGroup]).T
                    

                    geneticCor, _, _ = coherit(Y1, Y2, accession_C, envirement, geneticCor=True)
                    geneticCor = geneticCor[0].data.numpy()

                    corMatrix[dataIndex, predNum1, predNum2, splitIndex] = geneticCor

        print (corMatrix.shape)

        corMatrix = np.abs(corMatrix)

        plt.imshow(np.mean(corMatrix[dataIndex, :, :], axis=2))
        plt.show()

    np.savez_compressed('./data/miscPlant/eval/timeGeneticCor_synth.npz', corMatrix)
    quit()


#timeCorCoherit()
#quit()


def splitCor():

    splitIndex = 0

    #corMatrix = np.zeros((14, 14, 5))

    for predNum1 in range(14):

        corMatrix = np.zeros((5, 5))

        for splitIndex1 in range(5):
            for splitIndex2 in range(5):
                Y1 = loadnpz('./data/miscPlant/GWAS/MSI_' + 'central' + '_singlepoint_' + str(predNum1) + '_split_' + str(splitIndex1) + '.npz')
                Y2 = loadnpz('./data/miscPlant/GWAS/MSI_' + 'central' + '_singlepoint_' + str(predNum1) + '_split_' + str(splitIndex2) + '.npz')

                corMatrix[splitIndex1, splitIndex2] = scipy.stats.pearsonr(Y1[:, 0], Y2[:, 0])[0]
        corMatrix = np.abs(corMatrix)

        #plt.imshow(corMatrix)
        #plt.show()
    

    print (corMatrix)

    #corMatrix = np.abs(corMatrix)
    #corMatrix = np.mean(corMatrix, axis=2)

    

    #plt.imshow(corMatrix)
    #plt.show()

#splitCor()
#quit()


def compareTrait():


    file1 = './data/miscPlant/inputs/data_process/Msi_C_Japan_2019-URB-Plant_Data.tsv'

    data_trait = np.loadtxt(file1, delimiter='\t', dtype=str, skiprows=1)

    with open(file1, 'r') as file:
        lines = file.readlines()
    row1 = np.array(lines[0].split('\t'))


    plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=False)


    #print (data_trait[0])
    argBiomass = np.argwhere(data_trait[0] == '268.7')[0, 0]
    argAIL = np.argwhere(data_trait[0] == '16.2')[0, 0]
    argCulmDensity = np.argwhere(data_trait[0] == '0.0055')[0, 0]
    argCulmDryWeight = np.argwhere(data_trait[0] == '18.5')[0, 0]
    argCulmNodeNum = np.argwhere(data_trait[0] == '12')[0, 0]
    argDiamBasal = np.argwhere(data_trait[0] == '6.08')[0, 0]
    argDiamInter = np.argwhere(data_trait[0] == '3.17')[0, 0]
    argCULM = np.argwhere(data_trait[0] == '194.6')[0, 0] #3.17, 194.6
    argBasalCirc = np.argwhere(data_trait[0] == '118.0')[0, 0] 
    argDormancy = np.argwhere(data_trait[0] == '17-Oct-20')[0, 0] 
    argHeadDate = np.argwhere(data_trait[0] == '25-Aug-2020')[0, 0] 
    arg50HeadDate = np.argwhere(data_trait[0] == '1-Sep-2020')[0, 0] 
    argHardiness = np.argwhere(data_trait[0] == '100')[0, 0] 
    argSpringRegrowth = np.argwhere(data_trait[0] == '17-Apr-2020')[0, 0] 

    argSurvival = np.argwhere(row1 == 'Survival (Oct __07-_ 2019)')[0, 0] 
    argLeftCollect = np.argwhere(row1 == 'Leaf Sample collected(__Oct 2019)')[0, 0] 
    argDNAextract = np.argwhere(row1 == 'Select for DNA Extraction ')[0, 0] 
    argSurvivalSpring = np.argwhere(row1 == 'Survival (spring 2020)')[0, 0] 
    

    
    traitList = ['biomass', 'Average Internode Length', 'Culm Density', 'Culm Dry Weight', 'Culm node num', 'Culm Outer Diameter at the Basal Internode', 'Calm Outer Diameter at last internode',
                 'Height', 'Basal circumference', 'Dormancy date', 'First Heading Date', '50% Heading Date', 'Hardiness score', 'Spring Regrowth Date', 'Survival October', 'Survival Spring']


    traitList = ['biomass']

    realTrait = 'biomass'
    realTrait = 'Average Internode Length'
    realTrait = 'Culm Density'
    realTrait = 'Culm Dry Weight'
    realTrait = 'Culm node num'
    realTrait = 'Culm Outer Diameter at the Basal Internode'
    realTrait = 'Calm Outer Diameter at last internode'
    realTrait = 'Height'
    realTrait = 'Basal circumference'
    realTrait = 'Dormancy date'
    realTrait = 'First Heading Date'
    realTrait = '50% Heading Date'
    realTrait = 'Hardiness score'
    realTrait = 'Spring Regrowth Date'
    realTrait = 'Survival October'
    #realTrait = 'Leaf collected'
    #realTrait = 'DNA extracted'
    #realTrait = 'Survival Spring'

    
    

    biomass = data_trait[:, argBiomass]
    AIL = data_trait[:, argAIL]
    culm = data_trait[:, argCULM]
    dormancyDate = data_trait[:, argDormancy]
    headDate  = data_trait[:, argHeadDate]
    halfHeadDate  = data_trait[:, arg50HeadDate]
    springRegrowthDate = data_trait[:, argSpringRegrowth]

    

    dormancyDate = convertDateNumeric(dormancyDate)
    headDate = convertDateNumeric(headDate)
    halfHeadDate = convertDateNumeric(halfHeadDate)
    springRegrowthDate = convertDateNumeric(springRegrowthDate)
    


    #corList1 = []
    #corList2 = []

    #corList1 = np.zeros(( len(traitList),  14, 5))
    #corList2 = np.zeros(( len(traitList),  14, 5))

    corList_all = np.zeros(( 3, len(traitList),  14, 5))

    #timeList = [0, 2, 4, 6, 8, 10, 12]

    timeList = np.arange(14)

    traitNum = 0

    traitCounts = np.zeros(len(traitList), dtype=int)

    np.random.seed(1)
    crossValid = np.random.randint(5, size=plotRow.shape[0])

    for predNum in [6]:# [0]:#range(14):# timeList:

        print ('time: ', predNum)
        for splitIndex in range(5):

            print (predNum)

            trainTest2 = np.zeros(crossValid.shape[0], dtype=int)
            trainTest2[crossValid == splitIndex] = 1

            
            #predName = './data/miscPlant/GWAS/MSI_' + 'central' + '_singlepoint_' + str(predNum) + '_row13.npz'

            #predName = './data/miscPlant/eval/pred_maxWave_' + 'central' + '_' + str(predNum) + '.npz'
            predName = './data/miscPlant/GWAS/MSI_' + 'central' + '_singlepoint_' + str(predNum) + '_split_' + str(splitIndex) + '.npz' 
            #predName = './data/miscPlant/models/MSI_' + 'central' + '_singlepoint_' + str(predNum) + '_noSplit.npz' 
            

            Y0 = loadnpz(predName)

            

            #Y1 = loadnpz('./data/miscPlant/GWAS/MSI_' + 'central' + '_singlepoint_' + str(2) + '_split_' + str(splitIndex) + '.npz')
            #Y2 = loadnpz('./data/miscPlant/GWAS/MSI_' + 'central' + '_singlepoint_' + str(3) + '_split_' + str(splitIndex) + '.npz' )
            #Y3 = loadnpz('./data/miscPlant/GWAS/MSI_' + 'central' + '_singlepoint_' + str(4) + '_split_' + str(splitIndex) + '.npz' )

            #print (np.mean(np.abs(Y1[:, 0])))
            #print (np.mean(np.abs(Y2[:, 0])))
            #print (np.mean(np.abs(Y3[:, 0])))
            #quit()

            #Y0 = loadnpz(predName)
            #Y0 = Y1 + Y2 + Y3
            #Y0 = Y2

            #np.random.seed(1)

            #trainTest2 = rowSplitter(plotRow)
            #trainTest2[trainTest2<=6] = 0
            #trainTest2[trainTest2>=7] = 1



            for measureTraitIndex in range(len(traitList)):
                realTrait = traitList[measureTraitIndex]

                #print (realTrait)

                if realTrait == 'biomass':
                    trait1 = biomass
                if realTrait == 'Average Internode Length':
                    trait1 = AIL 
                if realTrait == 'Culm Density':
                    trait1 = data_trait[:, argCulmDensity]
                if realTrait == 'Culm Dry Weight':
                    trait1 = data_trait[:, argCulmDryWeight]
                if realTrait == 'Culm node num':
                    trait1 = data_trait[:, argCulmNodeNum]
                if realTrait == 'Culm Outer Diameter at the Basal Internode':
                    trait1 = data_trait[:, argDiamBasal]
                if realTrait == 'Calm Outer Diameter at last internode':
                    trait1 = data_trait[:, argDiamInter]
                if realTrait == 'realTrait':
                    trait1 = data_trait[:, argDiamInter]
                if realTrait == 'Height':
                    trait1 = culm
                if realTrait == 'Basal circumference':
                    trait1 = data_trait[:, argBasalCirc] 
                if realTrait == 'Dormancy date':
                    trait1 = dormancyDate

                if realTrait == 'First Heading Date':
                    trait1 = headDate
                if realTrait == '50% Heading Date':
                    trait1 = halfHeadDate
                
                if realTrait == 'Hardiness score':
                    trait1 = data_trait[:, argHardiness]
                if realTrait == 'Spring Regrowth Date':
                    trait1 = springRegrowthDate

                if realTrait == 'Survival October':
                    trait1 = data_trait[:, argSurvival]
                if realTrait == 'Leaf collected':
                    trait1 = data_trait[:, argLeftCollect]
                if realTrait == 'DNA extracted':
                    trait1 = data_trait[:, argDNAextract]
                if realTrait == 'Survival Spring':
                    trait1 = data_trait[:, argSurvivalSpring]


                
                #trait1 = AIL
                #trait1 = culm

                argGood = np.argwhere(trait1 != '')[:, 0]
                badOptions = np.array(['Missing', 'no stem', 'no stems', 'NA'])
                argGood = argGood[np.isin(trait1[argGood], badOptions) == False]

                plotNum_C, accession_C, plotNum_S, accession_S, _ = loadMSInames()
                accession_C = accession_C[argGood]

                #_, inverse1, count1 = np.unique(accession_C, return_counts=True, return_inverse=True)
                #count_inverse = count1[inverse1]
                #argGood = argGood[count_inverse[argGood] >= 2]


                #if True:
                #    argGood = argGood[trainTest2[argGood] == 1]

                #if True:
                #    argGood1 = argGood[trainTest2[argGood] == 0]
                #    argGood2 = argGood[trainTest2[argGood] == 1]

                Y = np.copy(Y0[argGood])
                trait1 = trait1[argGood].astype(float)
                trainTest3 = trainTest2[argGood]

                print (argGood.shape)


                #Y = np.copy(Y0[argGood1])
                #trait1 = trait1[argGood2].astype(float)

                traitCounts[measureTraitIndex] = trait1.shape[0]



                


                #accession_C1 = accession_C[argGood1]
                #accession_C2 = accession_C[argGood2]

                cor1 = scipy.stats.pearsonr(Y[:, traitNum], trait1)[0]
                #corList1[measureTraitIndex, predNum, splitIndex] = cor1
                corList_all[0, measureTraitIndex, predNum, splitIndex] = cor1


                rowGroup = plotRow.astype(int) % 130
                envirement = np.array([plotRep_C, rowGroup]).T
                envirement = envirement[argGood]

                trait1_tensor = torch.tensor(trait1).float().reshape((-1, 1))
                Y_tensor = torch.tensor(Y[:, traitNum:traitNum+1]).float()

                #print (Y.shape)
                #print (trait1.shape)
                #print (accession_C.shape)
                #print (envirement.shape)



                herit1 = cheapHeritability(trait1_tensor, accession_C, envirement)
                herit2 = cheapHeritability(trait1_tensor[trainTest3 == 1], accession_C[trainTest3 == 1], envirement[trainTest3 == 1])
                print (realTrait)
                #print ('herit', herit1 * 4)
                print ('herit', herit2 * 4)


                geneticCor, _, _ = coherit(Y_tensor, trait1_tensor, accession_C, envirement, geneticCor=True)
                geneticCor = geneticCor[0].data.numpy()

                corList_all[1, measureTraitIndex, predNum, splitIndex] = geneticCor#[0]

                if True:
                    #print (geneticCor)
                    #quit()


                    unique_acc = np.unique(accession_C)
                    #unique_acc = np.unique(accession_C2)
                    #print (unique_acc.shape)
                    Y_mean = np.zeros(unique_acc.shape[0])
                    trait_mean = np.zeros(unique_acc.shape[0])
                    for a in range(unique_acc.shape[0]):
                        args1 = np.argwhere(accession_C == unique_acc[a])[:, 0]
                        Y_mean[a] = np.mean(Y[args1, traitNum])
                        trait_mean[a] = np.mean(trait1[args1])


                        #args1 = np.argwhere(accession_C1 == unique_acc[a])[:, 0]
                        #Y_mean[a] = np.mean(Y[args1, traitNum])
                        #args2 = np.argwhere(accession_C2 == unique_acc[a])[:, 0]
                        #trait_mean[a] = np.mean(trait1[args2])

                    #print (trait_mean)
                    #print (Y_mean)
                    #quit()

                    #print (scipy.stats.pearsonr(Y[:, 0], trait1))
                    #print (scipy.stats.pearsonr(Y_mean, trait_mean))

                    cor1 = scipy.stats.pearsonr(Y[:, traitNum], trait1)[0]
                    cor2 = scipy.stats.pearsonr(Y_mean, trait_mean)[0]

                    print ('cor2', cor2)


                    #print ('')
                    #print (cor1)
                    #print (cor2)
                    #print (scipy.stats.pearsonr(Y_mean, trait_mean))
                    #quit()

                    #corList1[measureTraitIndex, predNum, splitIndex] = cor1
                    corList_all[2, measureTraitIndex, predNum, splitIndex] = cor2

                    #print (corList1[predNum])

    quit()
    #GS = loadnpz('./data/miscPlant/eval/GS_' + 'central' + '.npz').astype(float)
    #print (GS)
    #quit()
    #GS = loadnpz('./data/miscPlant/eval/GS_' + 'central' + '_maxWave.npz').astype(float)
    #print (GS)
    #print (corList2.shape)
    #quit()

    #print (traitCounts)
    np.savez_compressed('./data/miscPlant/eval/corTrait_traitCount.npz', traitCounts)
    #quit()
    
    #corList1 = np.abs(corList1)

    dates = ['MSI_09042020_processed_ALLstack.tif', 'MSI_05062020_processed_ALLstack.tif', 'MSI_05222020_processed_ALLstack.tif', 'MSI_09192020_processed_ALLstack.tif', 'MSI_07022020_processed_ALLstack.tif', 'MSI_11282020_processed_ALLstack.tif', 'MSI_07242020_processed_ALLstack.tif', 'MSI_10052020_processed_ALLstack.tif', 'MSI_08182020_processed_ALLstack.tif', 'MSI_06182020_processed_ALLstack.tif', 'MSI_07102020_processed_ALLstack.tif', 'MSI_06082020_processed_ALLstack.tif', 'MSI_08082020_processed_ALLstack.tif', 'MSI_11062020_processed_ALLstack.tif']
    dates = np.array(dates)
    for a in range(dates.shape[0]):
        dates[a] = dates[a].split('_')[1]
        dates[a] = dates[a][:2] + ',' + dates[a][2:4]
    dates = np.sort(dates)

    #plt.imshow(np.mean(np.abs(corList2), axis=2))
    #plt.show()

    np.savez_compressed('./data/miscPlant/eval/corTrait.npz', corList_all)
    quit()

    plt.imshow(corList2)

    plt.xticks(np.arange(corList1.shape[1]), dates , rotation=45)
    plt.yticks(np.arange(corList1.shape[0]), traitList )
    plt.colorbar()
    plt.show()
    quit()

    print (corList1)

    dates = ['MSI_09042020_processed_ALLstack.tif', 'MSI_05062020_processed_ALLstack.tif', 'MSI_05222020_processed_ALLstack.tif', 'MSI_09192020_processed_ALLstack.tif', 'MSI_07022020_processed_ALLstack.tif', 'MSI_11282020_processed_ALLstack.tif', 'MSI_07242020_processed_ALLstack.tif', 'MSI_10052020_processed_ALLstack.tif', 'MSI_08182020_processed_ALLstack.tif', 'MSI_06182020_processed_ALLstack.tif', 'MSI_07102020_processed_ALLstack.tif', 'MSI_06082020_processed_ALLstack.tif', 'MSI_08082020_processed_ALLstack.tif', 'MSI_11062020_processed_ALLstack.tif']
    dates = np.array(dates)
    #_, dates_inverse = np.unique(dates, return_inverse=True)


    #print (dates_inverse)

    #dates_inverse = dates_inverse[np.array(timeList)]

    dates_inverse = np.arange(len(corList1))

    corList1 = np.abs(np.array(corList1))
    corList2 = np.abs(np.array(corList2))

    plt.scatter(dates_inverse, corList1)
    plt.scatter(dates_inverse, corList2)
    plt.title('synthetic trait ' + str(traitNum) + ' ' + realTrait)
    plt.xlabel('time point')
    plt.ylabel('correlation with biomass')
    plt.legend(['raw value', 'half sibling group mean'])
    plt.show()




    quit()


    plt.scatter(Y[:, 0], trait1  )
    plt.show()

    for a in range(Y.shape[1]):
        print (scipy.stats.pearsonr(Y[:, a], trait1))



compareTrait()
quit()



def convertDateNumeric2(dormancyDate):

    #print (dormancyDate)

    #print (dormancyDate)

    badOptions = np.array(['', 'Missing', 'no stem', 'no stems', 'NA', 'NaN', 'NaT', 'nan', 'Missed'])
    argGood = np.argwhere(np.isin(dormancyDate.astype(str), badOptions) == False)[:, 0]

    
    dormancyDateValid = dormancyDate[argGood]

    #print (dormancyDateValid)
    dormancyDateValid = dormancyDateValid.to_numpy(dtype='datetime64[ns]')
    
    dormancyDateValid = dormancyDateValid.astype(float)
    
    dormancyDateValid = dormancyDateValid - np.min(dormancyDateValid)

    dormancyDate[argGood] = dormancyDateValid

    return dormancyDate


def south_compareTrait():


    #file1 = './data/miscPlant/inputs/data_process/Msi_C_Japan_2019-URB-Plant_Data.tsv'
    file1 = './data/miscPlant/inputs/data_process/Msi_S_Japan_2019-URB-Plant_Data.xlsx'
    data_trait = pd.read_excel(file1)



    plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=True)

    
    #traitList = ['biomass', 'Average Internode Length', 'Culm Density', 'Culm Dry Weight', 'Culm node num', 'Culm Outer Diameter at the Basal Internode', 'Calm Outer Diameter at last internode',
    #             'Height', 'Basal circumference', 'Dormancy date', 'First Heading Date', '50% Heading Date', 'Hardiness score', 'Spring Regrowth Date', 'Survival October', 'Survival Spring']

    traitList = ['Survival (Oct 04 - Oct 07 2019)', 'Survival (spring 2020)', 'Hardiness score (%) ( July 2020)', 'Basal Circumference (BCirc; cm)   2020', 
        'Autumn Dormancy Date 2020', 'Spring Re-growth Date 2020', 'First Heading Date 2020 (HD)', '50% Heading Date 2020 (HD 50%)']
    

    #spingRegrowth = data_trait['Spring Re-growth Date 2020'].to_numpy()
    #spingRegrowth = convertDateNumeric(spingRegrowth)
    #data_trait['Spring Re-growth Date 2020'] = spingRegrowth
    
    data_trait['Autumn Dormancy Date 2020'] = convertDateNumeric2(data_trait['Autumn Dormancy Date 2020'])
    data_trait['Spring Re-growth Date 2020'] = convertDateNumeric2(data_trait['Spring Re-growth Date 2020'])
    data_trait['First Heading Date 2020 (HD)'] = convertDateNumeric2(data_trait['First Heading Date 2020 (HD)'])
    data_trait['50% Heading Date 2020 (HD 50%)'] = convertDateNumeric2(data_trait['50% Heading Date 2020 (HD 50%)'])

    #headDate = convertDateNumeric(headDate)
    #halfHeadDate = convertDateNumeric(halfHeadDate)
    #springRegrowthDate = convertDateNumeric(springRegrowthDate)
    

    corList_all = np.zeros(( 3, len(traitList),  14, 5))

    #timeList = [0, 2, 4, 6, 8, 10, 12]

    timeList = np.arange(14)

    traitNum = 0

    traitCounts = np.zeros(len(traitList), dtype=int)

    np.random.seed(1)
    crossValid = np.random.randint(5, size=plotRow.shape[0])

    for predNum in range(14):# timeList:

        print ('time: ', predNum)
        for splitIndex in range(5):

            print (predNum)

            trainTest2 = np.zeros(crossValid.shape[0], dtype=int)
            trainTest2[crossValid == splitIndex] = 1


            #predName = './data/miscPlant/GWAS/MSI_' + 'central' + '_singlepoint_' + str(predNum) + '_split_' + str(splitIndex) + '.npz' 
            predName = './data/miscPlant/GWAS/MSI_' + 'south' + '_singlepoint_' + str(predNum) + '_split_' + str(splitIndex) + '.npz' 
            

            Y0 = loadnpz(predName)

            
            
            for measureTraitIndex in range(len(traitList)):
                realTrait = traitList[measureTraitIndex]

                #print (realTrait)

                trait1 = data_trait[realTrait]
                trait1 = trait1[:-1] #there's an empty row at the end
                trait1 = trait1.to_numpy().astype(str)

                print (realTrait)

                print (trait1)
                

                argGood = np.argwhere(trait1 != '')[:, 0]
                badOptions = np.array(['Missing', 'no stem', 'no stems', 'NA', 'NaN', 'NaT', 'nan', 'Missed'])
                argGood = argGood[np.isin(trait1[argGood], badOptions) == False]

                print (argGood.shape)

                plotNum_C, accession_C, plotNum_S, accession_S, _ = loadMSInames()
                accession_C = accession_C[argGood]


                Y = np.copy(Y0[argGood])
                trait1 = trait1[argGood].astype(float)
                traitCounts[measureTraitIndex] = trait1.shape[0]

                cor1 = scipy.stats.pearsonr(Y[:, traitNum], trait1)[0]
                corList_all[0, measureTraitIndex, predNum, splitIndex] = cor1


                rowGroup = plotRow.astype(int) % 130
                envirement = np.array([plotRep_C, rowGroup]).T
                envirement = envirement[argGood]

                trait1_tensor = torch.tensor(trait1).float().reshape((-1, 1))
                Y_tensor = torch.tensor(Y[:, traitNum:traitNum+1]).float()

                herit1 = cheapHeritability(trait1_tensor, accession_C, envirement)
                #print (realTrait)
                


                geneticCor, _, _ = coherit(Y_tensor, trait1_tensor, accession_C, envirement, geneticCor=True)
                geneticCor = geneticCor[0].data.numpy()

                corList_all[1, measureTraitIndex, predNum, splitIndex] = geneticCor#[0]

                if True:
                    #print (geneticCor)
                    #quit()


                    unique_acc = np.unique(accession_C)
                    #unique_acc = np.unique(accession_C2)
                    #print (unique_acc.shape)
                    Y_mean = np.zeros(unique_acc.shape[0])
                    trait_mean = np.zeros(unique_acc.shape[0])
                    for a in range(unique_acc.shape[0]):
                        args1 = np.argwhere(accession_C == unique_acc[a])[:, 0]
                        Y_mean[a] = np.mean(Y[args1, traitNum])
                        trait_mean[a] = np.mean(trait1[args1])


                    cor1 = scipy.stats.pearsonr(Y[:, traitNum], trait1)[0]
                    cor2 = scipy.stats.pearsonr(Y_mean, trait_mean)[0]

                    #print ('cor2', cor2)
                    corList_all[2, measureTraitIndex, predNum, splitIndex] = cor2

                    #print (corList1[predNum])

                print (corList_all[:,  measureTraitIndex, predNum, splitIndex ])


    np.savez_compressed('./data/miscPlant/eval/corTrait_south.npz', corList_all)

    print (corList_all.shape)
    quit()
    quit()
    



south_compareTrait()
quit()


def compareOnlyMeasure():

    file1 = './data/miscPlant/inputs/data_process/Msi_C_Japan_2019-URB-Plant_Data.tsv'

    data_trait = np.loadtxt(file1, delimiter='\t', dtype=str, skiprows=1)

    with open(file1, 'r') as file:
        lines = file.readlines()
    row1 = np.array(lines[0].split('\t'))


    plotNum_C, halfsib_C, plotRow, plantName_C, plotRep_C = loadMSInames(doSouth=False)


    #print (data_trait[0])
    argBiomass = np.argwhere(data_trait[0] == '268.7')[0, 0]
    argAIL = np.argwhere(data_trait[0] == '16.2')[0, 0]
    argCulmDensity = np.argwhere(data_trait[0] == '0.0055')[0, 0]
    argCulmDryWeight = np.argwhere(data_trait[0] == '18.5')[0, 0]
    argCulmNodeNum = np.argwhere(data_trait[0] == '12')[0, 0]
    argDiamBasal = np.argwhere(data_trait[0] == '6.08')[0, 0]
    argDiamInter = np.argwhere(data_trait[0] == '3.17')[0, 0]
    argCULM = np.argwhere(data_trait[0] == '194.6')[0, 0] #3.17, 194.6
    argBasalCirc = np.argwhere(data_trait[0] == '118.0')[0, 0] 
    argDormancy = np.argwhere(data_trait[0] == '17-Oct-20')[0, 0] 
    argHeadDate = np.argwhere(data_trait[0] == '25-Aug-2020')[0, 0] 
    arg50HeadDate = np.argwhere(data_trait[0] == '1-Sep-2020')[0, 0] 
    argHardiness = np.argwhere(data_trait[0] == '100')[0, 0] 
    argSpringRegrowth = np.argwhere(data_trait[0] == '17-Apr-2020')[0, 0] 

    argSurvival = np.argwhere(row1 == 'Survival (Oct __07-_ 2019)')[0, 0] 
    argLeftCollect = np.argwhere(row1 == 'Leaf Sample collected(__Oct 2019)')[0, 0] 
    argDNAextract = np.argwhere(row1 == 'Select for DNA Extraction ')[0, 0] 
    argSurvivalSpring = np.argwhere(row1 == 'Survival (spring 2020)')[0, 0] 
    

    
    traitList = ['biomass', 'Average Internode Length', 'Culm Density', 'Culm Dry Weight', 'Culm node num', 'Culm Outer Diameter at the Basal Internode', 'Calm Outer Diameter at last internode',
                 'Height', 'Basal circumference', 'Dormancy date', 'First Heading Date', '50% Heading Date', 'Hardiness score', 'Spring Regrowth Date', 'Survival October', 'Survival Spring']


    #traitList = ['biomass']

    realTrait = 'biomass'
    realTrait = 'Average Internode Length'
    realTrait = 'Culm Density'
    realTrait = 'Culm Dry Weight'
    realTrait = 'Culm node num'
    realTrait = 'Culm Outer Diameter at the Basal Internode'
    realTrait = 'Calm Outer Diameter at last internode'
    realTrait = 'Height'
    realTrait = 'Basal circumference'
    realTrait = 'Dormancy date'
    realTrait = 'First Heading Date'
    realTrait = '50% Heading Date'
    realTrait = 'Hardiness score'
    realTrait = 'Spring Regrowth Date'
    realTrait = 'Survival October'
    #realTrait = 'Leaf collected'
    #realTrait = 'DNA extracted'
    #realTrait = 'Survival Spring'

    
    

    biomass = data_trait[:, argBiomass]
    AIL = data_trait[:, argAIL]
    culm = data_trait[:, argCULM]
    dormancyDate = data_trait[:, argDormancy]
    headDate  = data_trait[:, argHeadDate]
    halfHeadDate  = data_trait[:, arg50HeadDate]
    springRegrowthDate = data_trait[:, argSpringRegrowth]

    

    dormancyDate = convertDateNumeric(dormancyDate)
    headDate = convertDateNumeric(headDate)
    halfHeadDate = convertDateNumeric(halfHeadDate)
    springRegrowthDate = convertDateNumeric(springRegrowthDate)
    


    #corList1 = []
    #corList2 = []

    #corList1 = np.zeros(( len(traitList),  14, 5))
    #corList2 = np.zeros(( len(traitList),  14, 5))

    corList_all = np.zeros(( len(traitList),  len(traitList) ))

    #timeList = [0, 2, 4, 6, 8, 10, 12]

    timeList = np.arange(14)

    traitNum = 0

    traitCounts = np.zeros(len(traitList), dtype=int)

    np.random.seed(1)
    crossValid = np.random.randint(5, size=plotRow.shape[0])

    

    for measureTraitIndex1 in range(len(traitList)):

        realTrait = traitList[measureTraitIndex1]
        if realTrait == 'biomass':
            trait1 = biomass
        if realTrait == 'Average Internode Length':
            trait1 = AIL 
        if realTrait == 'Culm Density':
            trait1 = data_trait[:, argCulmDensity]
        if realTrait == 'Culm Dry Weight':
            trait1 = data_trait[:, argCulmDryWeight]
        if realTrait == 'Culm node num':
            trait1 = data_trait[:, argCulmNodeNum]
        if realTrait == 'Culm Outer Diameter at the Basal Internode':
            trait1 = data_trait[:, argDiamBasal]
        if realTrait == 'Calm Outer Diameter at last internode':
            trait1 = data_trait[:, argDiamInter]
        if realTrait == 'realTrait':
            trait1 = data_trait[:, argDiamInter]
        if realTrait == 'Height':
            trait1 = culm
        if realTrait == 'Basal circumference':
            trait1 = data_trait[:, argBasalCirc] 
        if realTrait == 'Dormancy date':
            trait1 = dormancyDate

        if realTrait == 'First Heading Date':
            trait1 = headDate
        if realTrait == '50% Heading Date':
            trait1 = halfHeadDate
        
        if realTrait == 'Hardiness score':
            trait1 = data_trait[:, argHardiness]
        if realTrait == 'Spring Regrowth Date':
            trait1 = springRegrowthDate

        if realTrait == 'Survival October':
            trait1 = data_trait[:, argSurvival]
        if realTrait == 'Leaf collected':
            trait1 = data_trait[:, argLeftCollect]
        if realTrait == 'DNA extracted':
            trait1 = data_trait[:, argDNAextract]
        if realTrait == 'Survival Spring':
            trait1 = data_trait[:, argSurvivalSpring]


        realTrait1 = realTrait
        trait_1_original = np.copy(trait1)
            
        for measureTraitIndex2 in range(len(traitList)):

            trait_1 = np.copy(trait_1_original)
            
            realTrait = traitList[measureTraitIndex2]

            if realTrait == 'biomass':
                trait1 = biomass
            if realTrait == 'Average Internode Length':
                trait1 = AIL 
            if realTrait == 'Culm Density':
                trait1 = data_trait[:, argCulmDensity]
            if realTrait == 'Culm Dry Weight':
                trait1 = data_trait[:, argCulmDryWeight]
            if realTrait == 'Culm node num':
                trait1 = data_trait[:, argCulmNodeNum]
            if realTrait == 'Culm Outer Diameter at the Basal Internode':
                trait1 = data_trait[:, argDiamBasal]
            if realTrait == 'Calm Outer Diameter at last internode':
                trait1 = data_trait[:, argDiamInter]
            if realTrait == 'realTrait':
                trait1 = data_trait[:, argDiamInter]
            if realTrait == 'Height':
                trait1 = culm
            if realTrait == 'Basal circumference':
                trait1 = data_trait[:, argBasalCirc] 
            if realTrait == 'Dormancy date':
                trait1 = dormancyDate

            if realTrait == 'First Heading Date':
                trait1 = headDate
            if realTrait == '50% Heading Date':
                trait1 = halfHeadDate
            
            if realTrait == 'Hardiness score':
                trait1 = data_trait[:, argHardiness]
            if realTrait == 'Spring Regrowth Date':
                trait1 = springRegrowthDate

            if realTrait == 'Survival October':
                trait1 = data_trait[:, argSurvival]
            if realTrait == 'Leaf collected':
                trait1 = data_trait[:, argLeftCollect]
            if realTrait == 'DNA extracted':
                trait1 = data_trait[:, argDNAextract]
            if realTrait == 'Survival Spring':
                trait1 = data_trait[:, argSurvivalSpring]


            
            realTrait2 = realTrait
            trait_2 = np.copy(trait1)

            argGood = np.argwhere(trait_1 != '')[:, 0]
            badOptions = np.array(['Missing', 'no stem', 'no stems', 'NA'])
            argGood = argGood[np.isin(trait_1[argGood], badOptions) == False]
            argGood = argGood[trait_2[argGood] != '']
            argGood = argGood[np.isin(trait_2[argGood], badOptions) == False]


            trait_1 = trait_1[argGood].astype(float)
            trait_2 = trait_2[argGood].astype(float)

            
            plotNum_C, accession_C, plotNum_S, accession_S, _ = loadMSInames()
            accession_C = accession_C[argGood]

            
            

            unique_acc = np.unique(accession_C)

            print (trait_1.shape)
            print (trait_2.shape)
            print (accession_C.shape)
        
            
            trait_mean1 = np.zeros(unique_acc.shape[0])
            trait_mean2 = np.zeros(unique_acc.shape[0])
            for a in range(unique_acc.shape[0]):
                args1 = np.argwhere(accession_C == unique_acc[a])[:, 0]
                trait_mean1[a] = np.mean(trait_1[args1])
                trait_mean2[a] = np.mean(trait_2[args1])



            cor2 = scipy.stats.pearsonr(trait_mean1, trait_mean2)[0]

            print ('cor2', cor2)

            
            corList_all[measureTraitIndex1, measureTraitIndex2] = cor2



    traitList = np.array(traitList)
    traitList[traitList ==  'Culm Dry Weight'] =  'Culm\n Dry Weight'
    traitList[traitList ==  'Culm node num'] =  'Culm\n Node\n Number'
    traitList[traitList == 'Dormancy date'] = 'Dormancy\n Date'
    traitList[traitList == 'Culm Outer Diameter at the Basal Internode'] = 'Basal\n Diameter'
    traitList[traitList == 'Basal circumference'] = 'Basal\n Circumference'
    traitList[traitList == 'biomass'] = 'Biomass'
    traitList[traitList == 'Survival October'] = 'Survival\n October'
    traitList[traitList == 'Calm Outer Diameter at last internode'] = 'Calm Outer Diameter\n at last internode'
    traitList[traitList == 'Culm Outer Diameter at the Basal Internode'] = 'Culm Outer Diameter\n at the Basal Internode'
    traitList[traitList == 'Average Internode Length'] = 'Average Internode\nLength'

    


    
    #np.savez_compressed('./data/miscPlant/eval/measureBoth_corTrait.npz', corList_all)
    #plt.imshow(corList_all)
    #plt.show()
    import seaborn as sns
    corList_all = np.abs(corList_all)
    ax = sns.heatmap(corList_all, annot=True)
    ax.set_yticklabels(  traitList , rotation=0)
    ax.set_xticklabels(  traitList , rotation=45)
    
    #plt.xlabel('date')
    plt.tight_layout()
    plt.show()

    quit()
    #GS = loadnpz('./data/miscPlant/eval/GS_' + 'central' + '.npz').astype(float)
    #print (GS)
    #quit()




compareOnlyMeasure()
quit()


def compareCorTimes():


    #timeList = [0, 2, 4, 6, 8, 10, 12]
    timeList = np.arange(14)
    timeList = np.array(timeList)

    corMatrix = np.zeros((len(timeList), len(timeList)))

    for a in range(len(timeList)):

        predNum = timeList[a]

        #predName = './data/miscPlant/GWAS/MSI_6.npz'
        predName = './data/miscPlant/GWAS/MSI_singlepoint_' + str(predNum) + '.npz'
        Y = loadnpz(predName)

        for b in range(len(timeList)):
            predNum2 = timeList[b]
            predName = './data/miscPlant/GWAS/MSI_singlepoint_' + str(predNum2) + '.npz'
            Y2 = loadnpz(predName)

            cor1 = scipy.stats.pearsonr( Y[:, 0], Y2[:, 0] )[0]
            corMatrix[a, b] = cor1 

    dates = ['MSI_09042020_processed_ALLstack.tif', 'MSI_05062020_processed_ALLstack.tif', 'MSI_05222020_processed_ALLstack.tif', 'MSI_09192020_processed_ALLstack.tif', 'MSI_07022020_processed_ALLstack.tif', 'MSI_11282020_processed_ALLstack.tif', 'MSI_07242020_processed_ALLstack.tif', 'MSI_10052020_processed_ALLstack.tif', 'MSI_08182020_processed_ALLstack.tif', 'MSI_06182020_processed_ALLstack.tif', 'MSI_07102020_processed_ALLstack.tif', 'MSI_06082020_processed_ALLstack.tif', 'MSI_08082020_processed_ALLstack.tif', 'MSI_11062020_processed_ALLstack.tif']
    dates = np.array(dates)
    _, dates_inverse = np.unique(dates, return_inverse=True)

    dates_inverse = dates_inverse[timeList]
    #dates_inverse = dates_inverse.astype(str)

    argsort1 = np.argsort(dates_inverse)


    corMatrix = corMatrix[:, argsort1][argsort1]
    dates_inverse = dates_inverse[argsort1]


    #print (dates_inverse)

    corMatrix = np.abs(corMatrix)
    plt.imshow(corMatrix)
    plt.xticks( np.arange(len(dates_inverse)) , dates_inverse)
    plt.yticks( np.arange(len(dates_inverse)) , dates_inverse)
    plt.colorbar()
    plt.xlabel('time point')
    plt.ylabel('time point')
    plt.title('correlation of synthetic traits')
    plt.show()
    quit()

        
compareCorTimes()
quit()