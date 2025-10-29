



import numpy as np
import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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


#from twin import *
from shared import *

import seaborn as sns


import netCDF4# as nc
import pyopenms as oms
from pyopenms import MSExperiment, MzMLFile






def savePathwayGenes():
    pathwayNames = ['Citrate cycle (TCA cycle)', 'Glucosinolate biosynthesis',
    'Glyoxylate and dicarboxylate metabolism',
    'Phenylalanine, tyrosine and tryptophan biosynthesis',
    'Phenylpropanoid biosynthesis']

    #pathwayCodes = ['ath00020', 'ath00966', 'ath00630', 'ath00400', 'ath00940']

    pathwayCodes = ['map00020', 'map00966', 'map00630', 'map00400', 'map00940']


    #map00020
    #https://rest.kegg.jp/link/ko/path:map00966         # glucosinolate biosynthesis
    #https://rest.kegg.jp/link/ko/path:map00630         # glyoxylate & dicarboxylate
    #https://rest.kegg.jp/link/ko/path:map00400         # Phe/Tyr/Trp biosynthesis
    #https://rest.kegg.jp/link/ko/path:map00940         # phenylpropanoid biosynthesis

    import requests, time





    def get_kos(path_id):
        r = requests.get(f"https://rest.kegg.jp/link/ko/path:{path_id}")
        kos = [line.split("\t")[1].strip() for line in r.text.strip().splitlines()]
        return sorted(set(kos))

    def kos_to_ath(kos):
        genes = []
        for i in range(0, len(kos), 50):
            batch = "+".join(kos[i:i+50])
            r = requests.get(f"https://rest.kegg.jp/link/ath/{batch}")
            genes += [line.split("\t")[1].strip() for line in r.text.strip().splitlines()]
        return sorted(set(genes))


    data_info = []

    for path_index in range(len(pathwayCodes)):
        pathNow = pathwayCodes[path_index]
        pathwayName = pathwayNames[path_index]
        kos = get_kos(pathNow)        # glucosinolate biosynthesis
        genes = kos_to_ath(kos)

        for gene_index in range(len(genes)):
            geneNow = genes[gene_index]
            geneNow = geneNow.split(':')[1]
            #print (geneNow)
            #quit()
            data_info.append([pathNow, pathwayName, geneNow])
        
        #print (genes)
        #print(len(genes), "genes")
    data_info = np.array(data_info)

    np.savez_compressed('./data/metab/metabData_Fernie/pathwayGenes/gensFromPath.npz', data_info)

    quit()







    def kegg_link_genes_in_pathway(org_code: str, pathway_id: str):
        # pathway_id like 'hsa00010' or 'sbi00010'
        url = f"https://rest.kegg.jp/link/{org_code}/path:{pathway_id}"
        r = requests.get(url); r.raise_for_status()
        pairs = [line.split("\t") for line in r.text.strip().splitlines()]
        gene_ids = [p[1].strip() for p in pairs]  # e.g. 'hsa:3098'
        return sorted(set(gene_ids))

    def kegg_get_entries(ids, chunk=10, pause=0.4):
        # Fetch detailed annotations in small chunks to be polite
        entries = []
        for i in range(0, len(ids), chunk):
            batch = "+".join(ids[i:i+chunk])
            r = requests.get(f"https://rest.kegg.jp/get/{batch}")
            r.raise_for_status()
            entries.append(r.text)
            time.sleep(pause)
        return "\n".join(entries)

    def parse_kegg_flat(text):
        # Very light parser: returns list of dicts with key fields
        records = []
        cur = {}
        current_key = None
        for line in text.splitlines():
            if line.startswith("ENTRY"):
                if cur: records.append(cur); cur = {}
                cur["ENTRY"] = line[12:].strip()
                current_key = "ENTRY"
            elif line[:12].strip():
                key = line[:12].strip()
                val = line[12:].strip()
                cur[key] = (cur.get(key, "") + ("\n" if key in cur else "") + val).strip()
                current_key = key
            else:
                # continuation line
                val = line[12:].strip()
                if current_key:
                    cur[current_key] = (cur[current_key] + " " + val).strip()
        if cur: records.append(cur)
        return records


    dataList = []

    for pathway_index in range(len(pathwayCodes)):
        # ---- example usage ----
        org = "ath"          # Sorghum bicolor (use 'hsa' for human, etc.)
        #path = "ath00020"    # Glycolysis/Gluconeogenesis in sorghum
        path = pathwayCodes[pathway_index]

        gene_ids = kegg_link_genes_in_pathway(org, path)
        raw = kegg_get_entries(gene_ids, chunk=10)
        records = parse_kegg_flat(raw)

        # Example: show (ENTRY, NAME, GENES (if present), ORTHOLOGY, PATHWAY)
        summary = [
            {
                "ENTRY": r.get("ENTRY",""),
                "NAME": r.get("NAME",""),
                "ORTHOLOGY": r.get("ORTHOLOGY",""),
                "EC": r.get("ENZYME",""),
                "PATHWAY": r.get("PATHWAY","")
            }
            for r in records
        ]

        # `summary` now has structured info per KEGG gene entry
        print(f"Found {len(summary)} genes for {path} in {org}. First 3:")
        for row in summary[:3]:
            print(row)
            #print(row.keys())
            gene  =row['ENTRY'].split('         ')
            print ([gene])
            pathway = row['PATHWAY'].split('ath')
            print ([pathway])

        quit()

#savePathwayGenes()
#quit()



def processKnownMet(fileName):

    metaboliteLabels = np.loadtxt('./data/metab/metabData_Fernie/Identified_Metabolites.tsv', dtype=str, delimiter='\t')

    #print (metaboliteLabels[:10])
    #quit()

    knownMetTable = pd.read_csv(fileName, sep='\t', encoding='utf-8')
    knownMetTable = knownMetTable.to_numpy()
    knownMetTable = knownMetTable.astype(str)
    knownMetTable = knownMetTable[:, knownMetTable[0] != 'nan']

    #print (knownMetTable[0].astype(str))
    #print (type(knownMetTable[0]))

    #knownMetTable = knownMetTable[:, np.isnan(knownMetTable[0]) == False]
    #print (knownMetTable.shape)
    genotype_known = knownMetTable[1:, 0]
    metNames = knownMetTable[0, 1:]
    metValues = knownMetTable[1:, 1:].astype(float)
    for a in range(genotype_known.shape[0]):
        genotype_known[a] = genotype_known[a].split('.')[1]


    metNames_new = []
    for a in range(metNames.shape[0]):
        name1 = metNames[a]
        arg1 = np.argwhere(metaboliteLabels[:, 0] == name1)[0, 0]
        #metNames[a] = metaboliteLabels[arg1, 2]

        if False:#metaboliteLabels[arg1, 3] != '':
            metNames_new.append( metaboliteLabels[arg1, 3])
        else:
            metNames_new.append( metaboliteLabels[arg1, 2])


        #print ([metaboliteLabels[arg1, 2]])
        #print ([metaboliteLabels[arg1, 3]])

        #metNames_new.append( metaboliteLabels[arg1, 2])
    metNames_new = np.array(metNames_new)

    #print (metNames)
    #print (metaboliteLabels)
    #quit()

    return genotype_known, metNames_new, metValues



def correlateTraits():

    Y_mean = loadnpz('./data/metab/metabData_Fernie/pred/linear_secondary5.npz')#[:, :3]
    genotype = loadnpz('./data/metab/metabData_Fernie/processed/names.npz')
    genotype_unique = np.unique(genotype)

    print (genotype[:5])

    with open('./data/metab/metabData_Fernie/phenotype_published_raw.tsv', encoding='latin1') as f:
        phenotypes = np.loadtxt(f, delimiter='\t', dtype=str)
    
    ecotype1 = phenotypes[1:, 0]
    intersect1 = np.intersect1d(genotype, ecotype1)

    arg_pheno = np.zeros(intersect1.shape[0], dtype=int)
    arg_metab =  np.zeros(intersect1.shape[0], dtype=int)

    for a in range(intersect1.shape[0]):
        arg_pheno[a] = np.argwhere(ecotype1 == intersect1[a])[0, 0]
        arg_metab[a] = np.argwhere(genotype_unique == intersect1[a])[0, 0]
    
    Y_mean = Y_mean[arg_metab]
    phenotypes = phenotypes[1:, 2:][arg_pheno]


    corMatrix = np.zeros(( phenotypes.shape[1] , Y_mean.shape[1] ))
    pMatrix = np.zeros(( phenotypes.shape[1] , Y_mean.shape[1] ))
    for a in range(phenotypes.shape[1]):
        phenotype = phenotypes[:, a]
        argGood = np.argwhere(phenotype != 'NA')[:, 0]
        for b in range(Y_mean.shape[1]):

            cor1 = scipy.stats.pearsonr( phenotype[argGood].astype(float),  Y_mean[argGood, b]  )
            pvalue = cor1[1]
            corMatrix[a, b] = abs(cor1[0])
            pMatrix[a, b] = pvalue

    pMatrix = np.log10(pMatrix) * -1


    print (np.max(pMatrix))
    print (np.log10(pMatrix.shape[0]*pMatrix.shape[1]))



    print (np.argwhere(  pMatrix > np.log10(pMatrix.shape[0]*pMatrix.shape[1])  ))
    #quit()

    #plt.imshow(corMatrix)
    #plt.show()
    #quit()
    
    plt.imshow(pMatrix)
    plt.show()
    quit()


    print (Y_mean.shape)
    print (phenotypes.shape)


#correlateTraits()
#quit()


def explainSynthKnown():


    genotype = loadnpz('./data/metab/metabData_Fernie/processed/names.npz')
    genotype_unique0, genotype_inverse = np.unique(genotype, return_inverse=True)
    Nsplit = 5
    np.random.seed(2)
    trainTest = np.random.randint(Nsplit, size=genotype_unique0.shape[0])
    trainTest2 = np.zeros(trainTest.shape[0], dtype=int)
    trainTest2[trainTest == 0] = 1


    #Y_mean = loadnpz('./data/metab/metabData_Fernie/pred/linear_primary_channel0_split0.npz')[:, :10]
    #Y_mean = loadnpz('./data/metab/metabData_Fernie/pred/linear_primary_channel1_split0.npz')[:, :10]
    #Y_mean = loadnpz('./data/metab/metabData_Fernie/pred/linear_primary_many_noDecon.npz')[:, :10]

    splitIndex = 1

    Y_mean = loadnpz('./data/metab/metabData_Fernie/pred/linear_primary_many_split' + str(splitIndex) + '.npz')



    #Y_mean = loadnpz('./data/metab/metabData_Fernie/pred/linear_secondary5.npz')[:, :5]
    #Y_mean = loadnpz('./data/metab/metabData_Fernie/pred/linear_primary5.npz')[:, :5]

    #Y_mean = loadnpz('./data/metab/metabData_Fernie/pred/linear_bothEnv_many.npz')[:, :5]





    
    #Y_mean = loadnpz('./data/metab/metabData_Fernie/pred/PCA_primary.npz')

    std = np.mean(Y_mean ** 2, axis=0) ** 0.5
    Y_mean = Y_mean / std.reshape((1, -1))

    envirement = loadnpz('./data/metab/metabData_Fernie/processed/env.npz')
    
    genotype = genotype[envirement == '0d']
    #genotype = genotype[envirement == '6d']



    #print (genotype_0d.shape)
    genotype_unique = np.unique(genotype)
    trainTest2 = trainTest2[np.isin(genotype_unique0, genotype_unique)]


    assert genotype_unique.shape[0] == Y_mean.shape[0]

    
    #print (np.mean(Y_mean, axis=0))
    #print (np.mean(Y_mean**2, axis=0))
    #quit()

    

    fileName_6d = './data/metab/metabData_Fernie/6d-Table-1.tsv'
    fileName_0d = './data/metab/metabData_Fernie/0d-Table-1.tsv'
    genotype_known_6d, metNames_6d, metValues_6d = processKnownMet(fileName_6d)
    genotype_known_0d, metNames_0d, metValues_0d = processKnownMet(fileName_0d)

    #print (metNames_6d)
    #quit()

    assert np.array_equal(genotype_known_6d, genotype_known_0d)
    genotype_known = genotype_known_6d


    arg_known = np.zeros(genotype_known.shape[0], dtype=int)
    for a in range(genotype_known.shape[0]):
        arg1 = np.argwhere(genotype_unique == genotype_known[a])[0, 0]
        arg_known[a] = arg1
    Y_mean = Y_mean[arg_known]
    trainTest2 = trainTest2[arg_known]

    corList = []
    numUsed = []
    #from sklearn.linear_model import LinearRegression

    #metValues = metValues_0d
    metValues = np.concatenate((metValues_0d, metValues_6d), axis=1)
    metNames = np.concatenate((metNames_0d, metNames_6d), axis=0)
    metNamesEnv = np.concatenate((np.zeros(metNames_0d.shape[0]), np.ones(metNames_6d.shape[0])), axis=0)

    if True:
        metValues = metValues_0d
        metNames = metNames_0d
        metNamesEnv = np.zeros(metNames_0d.shape[0])

    metNames[metNames == 'alpha-D-Galacturonic acid 1-phosphate'] = 'Galacturonic acid'
    metNames[metNames == '"Trehalose, alpha,beta-"'] = 'Trehalose'


    #print ("ALL Names", metNames)
    print ('')
    print ('')
    print ('')
    print ('')
    print ('')
    #quit()
    metValues = metValues - np.mean(metValues, axis=0).reshape((1, -1))
    metValues = metValues / (np.mean(metValues**2, axis=0).reshape((1, -1)) ** 0.5)

    boolMetUse = np.zeros(( Y_mean.shape[1], metNames.shape[0]  ))

    from sklearn.linear_model import Lasso
    for phen_index in range(Y_mean.shape[1]):
        #reg = Lasso(alpha=1e-2).fit(metValues[trainTest2 == 0], Y_mean[trainTest2 == 0, phen_index] )
        #reg = Lasso(alpha=5e-2).fit(metValues[trainTest2 == 0], Y_mean[trainTest2 == 0, phen_index] )
        #reg = Lasso(alpha=1e-1).fit(metValues[trainTest2 == 0], Y_mean[trainTest2 == 0, phen_index] )
        reg = Lasso(alpha=2e-1).fit(metValues[trainTest2 == 0], Y_mean[trainTest2 == 0, phen_index] )
        pred = reg.predict(metValues)
        cor_train = scipy.stats.pearsonr(pred[trainTest2 == 0], Y_mean[trainTest2 == 0, phen_index])
        cor_test = scipy.stats.pearsonr(pred[trainTest2 == 1], Y_mean[trainTest2 == 1, phen_index])

        coef = reg.coef_ 
        #print ('')
        argUse = np.argwhere(coef != 0)[:, 0]
        boolMetUse[phen_index, argUse] = coef[argUse]
        #print (argUse.shape)
        #print (cor_test[0])
        print (metNames[argUse][np.argsort(np.abs(coef[argUse]) * -1)]  )
        print (metNamesEnv[argUse][np.argsort(np.abs(coef[argUse]) * -1)] )
        print (coef[argUse][np.argsort(np.abs(coef[argUse]) * -1)]    )
        numUseNow = np.argwhere(coef != 0).shape[0]
        numUsed.append(numUseNow)

        #print (cor_train[0], cor_test[0])
        corList.append([cor_train[0], cor_test[0]])
    corList = np.array(corList)
    numUsed = np.array(numUsed)

    print (corList)

    

    boolMetUse = boolMetUse.T
    boolMetUse = boolMetUse[:, :3]
    #argUseAll = np.argwhere(np.sum(np.abs(boolMetUse), axis=1) > 0.1)[:, 0]
    argUseAll = np.argwhere(np.sum(np.abs(boolMetUse), axis=1) > 0.0)[:, 0]


    max1 = np.max(np.abs(boolMetUse))

    #sns.clustermap(boolMetUse, col_cluster=False)
    sns.heatmap(boolMetUse[argUseAll], cmap='bwr', vmin=-max1, vmax=max1)
    plt.yticks(np.arange(argUseAll.shape[0]) + 0.5 , metNames[argUseAll] , rotation=0)
    plt.xticks( np.arange(boolMetUse.shape[1]) + 0.5 , np.arange(boolMetUse.shape[1])+1 )
    plt.xlabel("synthetic trait")
    plt.gcf().set_size_inches(3.5, 4)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig('./images/metab/knownMetab.pdf')
    plt.show()

    arangeTrait = np.arange(Y_mean.shape[1]) + 1
    plt.scatter( arangeTrait, corList[:, 1] )
    plt.xlabel('synthetic trait number')
    plt.ylabel('correlation')
    plt.xticks(arangeTrait)
    plt.show()

    plt.scatter( np.arange(Y_mean.shape[1]) + 1, numUsed )
    plt.xlabel('synthetic trait number')
    plt.ylabel('number of known metabolites used')
    plt.xticks(arangeTrait)
    plt.show()
    
    #plt.scatter(corList[:, 0], corList[:, 1])
    #plt.show()


#explainSynthKnown()
#quit()



def predictKnownMet():


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
    
    

#predictKnownMet()
#quit()


class convMeta(nn.Module):
    def __init__(self, info):
        super(convMeta, self).__init__()


        length1, Nphen = info[0], info[1]

        #self.nonlin = torch.tanh
        #self.nonlin = nn.ReLU()
        self.nonlin = nn.LeakyReLU(0.1)


        self.dropout = nn.Dropout(p=0.5)

        #self.conv1 = torch.nn.Conv1d(1, 5, 20, 20)

        inChannel = 2
        Nchannel = 1
        self.conv1 = torch.nn.Conv2d(inChannel, Nchannel, (10, 5), stride=(10, 5))


        blank1 = torch.zeros((1,  2, 1500, 40))
        x1 = self.conv1(blank1)

        #self.lin1 = torch.nn.Linear(x1.shape[1] * x1.shape[2] *  x1.shape[3], Nphen)


        self.lin1 = torch.nn.Linear(x1.shape[1] * x1.shape[2] *  x1.shape[3], 5)
        self.lin2 = torch.nn.Linear(5, Nphen)

        #1920x535



    def forward(self, x):

        shape1 = x.shape

        #print (x.shape)
        #quit()

        #x = x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))

        #print (x.shape)
        #quit()
        
        x = self.conv1(x)

        x = self.nonlin(x)

        
        x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))



        x = self.lin1(x)
        x = self.nonlin(x)
        x = self.lin2(x)

        


        return x
    



class specialLinear(nn.Module):
    def __init__(self, info):
        super(specialLinear, self).__init__()

        length1, Nphen = info[0], info[1]

        #self.nonlin = torch.tanh
        #self.nonlin = nn.ReLU()
        self.nonlin = nn.LeakyReLU(0.1)


        self.dropout = nn.Dropout(p=0.5)

        #self.weight = nn.Parameter(torch.randn(1, 101, 5) * 0.01) 
        self.weight = nn.Parameter(torch.randn(1, 10, 200) * 0.01) 



    def forward(self, x):

        shape1 = x.shape

        #print (x.shape)
        #quit()

        x = torch.sum(x * self.weight, axis= (1, 2))

        x = x.reshape((-1, 1))



        return x




def loadCDF(file_path, mzml_filename=''):
    """
    Reads a .cdf file using netCDF4 and constructs an MSExperiment.
    
    The .cdf file is expected to have:
      - 'scan_acquisition_time': retention times for each scan.
      - 'mass_values': concatenated m/z values.
      - 'intensity_values': concatenated intensities.
      - 'scan_index': indices indicating where each scan's data starts.
      
    Returns an MSExperiment with spectra populated from the file.
    """
    ds = netCDF4.Dataset(file_path, 'r')
    rt = ds.variables['scan_acquisition_time'][:]
    mass_values = ds.variables['mass_values'][:]
    intensity_values = ds.variables['intensity_values'][:]
    scan_index = ds.variables['scan_index'][:]
    ds.close()
    
    exp = oms.MSExperiment()
    for i in range(len(scan_index)):
        start = scan_index[i]
        end = scan_index[i+1] if i < len(scan_index) - 1 else len(mass_values)
        mzs = mass_values[start:end]
        intensities = intensity_values[start:end]
        
        spectrum = oms.MSSpectrum()
        spectrum.setRT(float(rt[i]))
        spectrum.set_peaks((mzs, intensities))
        exp.addSpectrum(spectrum)

    if mzml_filename != '':
        mzml_writer = oms.MzMLFile()
        mzml_writer.store(mzml_filename, exp)
    else:
        return exp 
    

#cdf_filename = "./1_2.cdf"
#mzml_filename = "./1_2.MzML"
#convertCDFtoMzML(cdf_filename, mzml_filename)
#quit()


def loadMzML(file_path):
    #.mzML

    exp = MSExperiment()
    MzMLFile().load(file_path, exp)

    numberSpectra = exp.getNrSpectra()
    print("Spectra loaded:", numberSpectra)
    print("Chromatograms loaded:", exp.getNrChromatograms())

    # Access first MS1 spectrum
    for spectra_index in range(numberSpectra):
        spec0 = exp[spectra_index]
        print("RT (min):", spec0.getRT() / 60)
        print("MS level:", spec0.getMSLevel())
        mz, inten = spec0.get_peaks()
        print("First five m/z values:", mz[:5])
        print("First five intensities:", inten[:5])



#loadMzML('./data/metab/metabData_Fernie/ecotype_378_1_6d.mzML')
#quit()


def checkDownloaded():
    fileOriginal = os.listdir('./data/metab/metabData_Fernie/unzipped')
    fileOriginal = np.array(fileOriginal)


    import requests
    zipFiles = ['12200393.zip', '12251824.zip',  '12290707.zip',  '12291819.zip', '12341588.zip', '12251706.zip',  '12251892.zip',  '12290785.zip',  '12291915.zip', '12251770.zip',  '12251936.zip', '12291751.zip',  '12291977.zip']

    fileList_all = []
    zenodoList = []

    totalFiles = 0
    for a in range(len(zipFiles)):
        record_id = zipFiles[a][:-4]
        url = f'https://zenodo.org/api/records/' + record_id
        r = requests.get(url)
        data = r.json()
        #totalFiles += len(data['files'])
        fileList_dict = data['files']
        fileList = []
        for b in range(len(fileList_dict)):
            fileNow = fileList_dict[b]['key']
            fileNow = fileNow.replace('.raw', '.mzML')
            #fileList.append(fileNow)
            fileList_all.append(fileNow)
            zenodoList.append(record_id)
        #fileList_all = fileList_all + fileList 

    fileList_all = np.array(fileList_all)
    zenodoList = np.array(zenodoList)

    argOutside = np.argwhere(np.isin(fileList_all, fileOriginal) == False)[:, 0]




def saveTimedMassMZ():


    maxMass = 2000
    maxTime = 60

    spectrumData = np.zeros((1112, 2, maxMass, maxTime))
    #namesEnvs = np.zeros((72, 2), dtype=int)
    namesEnvs = []

    filesAll = os.listdir('./data/metab/metabData_Fernie/unzipped')

    #print (len(filesAll))
    #quit()

    #print (len(filesAll)) #1244
    #quit()
    ecotypeFiles = []
    ecotypeList = []
    yearList = []
    envList = []
    for a in range(len(filesAll)):
        file1 = filesAll[a]
        if 'ecotype' in file1:
            ecotypeFiles.append(file1)
            split1 = file1.replace('ecotype_', 'ecotype.')
            split1 = split1.split('.')[1]
            split1 = split1.split('_')
            ecotypeList.append(split1[0])
            yearList.append(split1[1])
            envList.append(split1[2])
    
    ecotypeecotypeFilesList = np.array(ecotypeFiles)
    ecotypeList = np.array(ecotypeList)
    yearList = np.array(yearList)
    envList = np.array(envList)

    print (envList.shape)
    quit()

    


    if True:
        for file_index in range(ecotypeecotypeFilesList.shape[0]):
            fileNow = ecotypeecotypeFilesList[file_index]

            print (file_index, ecotypeecotypeFilesList.shape[0] )
            
            #file_string = '10_1' 
            #file_string = str(count1) + '_' + str(rep1) 
            #file_path = "./data/metab/metabolights/MTBLS528/cdf/" + file_string + ".cdf"
            #file_path = './data/metab/metabData_Fernie/ecotype_378_1_6d.mzML'
            file_path = './data/metab/metabData_Fernie/unzipped/' + fileNow

            exp = MSExperiment()
            MzMLFile().load(file_path, exp)

            numberSpectra = exp.getNrSpectra()
            # Access first MS1 spectrum
            for spectra_index in range(numberSpectra):
                spec0 = exp[spectra_index]

                # Get the polarity enum
                #IonSource.Polarity.UNKNOWN = 0
                #IonSource.Polarity.POSITIVE = 1
                #IonSource.Polarity.NEGATIVE = 2
                polarity = spec0.getInstrumentSettings().getPolarity()
                ##if spectra_index == 0:
                #    print (file_index)
                #    print ('polarity', polarity)

                RT_now = spec0.getRT()
                #print("RT (min):", spec0.getRT() / 60)

                MStype = spec0.getMSLevel()
                #print ([MStype])
                #1 means original mass spectrometry
                #2 means MS/MS tandom mass spectrometry

                if True:

                    #print("MS level:", spec0.getMSLevel())
                    mz, inten = spec0.get_peaks()

                    mz_integer = np.round(mz).astype(int)
                    RT_int = int(RT_now / 30)

                    #print ('mass', np.max(mz))
                    #print ('RT', RT_now)
                    #print("First five m/z values:", mz[:5])
                    #print("First five intensities:", inten[:5])
                    #quit()

                    for index1 in range(mz.shape[0]):
                        spectrumData[file_index, MStype-1, mz_integer[index1], RT_int ]  += inten[index1]
    

    #names = names.reshape((-1,))

    quit()
    
    np.savez_compressed('./data/metab/metabData_Fernie/processed/imageMetab.npz', spectrumData)
    np.savez_compressed('./data/metab/metabData_Fernie/processed/names.npz', ecotypeList)
    np.savez_compressed('./data/metab/metabData_Fernie/processed/env.npz', envList)
    np.savez_compressed('./data/metab/metabData_Fernie/processed/year.npz', yearList)

#saveTimedMassMZ()
#quit()



envList = loadnpz('./data/metab/metabData_Fernie/processed/env.npz')
print (np.unique(envList, return_counts=True))


ecotypeList = loadnpz('./data/metab/metabData_Fernie/processed/names.npz')

print (np.unique(ecotypeList[ envList == '0d' ]).shape)
print (np.unique(ecotypeList[ envList == '6d' ]).shape)

_, counts1 = np.unique(ecotypeList[ envList == '0d' ], return_counts=True)
_, counts2 = np.unique(ecotypeList[ envList == '6d' ], return_counts=True)

print (np.unique(counts1))
print (np.unique(counts2))

quit()
ecotypeUnique, ecotypeCount = np.unique(ecotypeList, return_counts=True)
ecotypeUnique = ecotypeUnique[ecotypeCount == 4]
for a in range(ecotypeUnique.shape[0]):
    if '-' in ecotypeUnique[a]:
        ecotypeUnique[a] = ecotypeUnique[a].split('-')[0]
print (np.unique(ecotypeUnique).shape)
quit()


print (ecotypeList.shape)
for a in range(ecotypeList.shape[0]):
    if '-' in ecotypeList[a]:
        ecotypeList[a] = ecotypeList[a].split('-')[0]
print (np.unique(ecotypeList))
print (np.unique(ecotypeList).shape)

quit()

def savePreciseMass():


    precision = 10000
    maxMass = 2000 * precision
    
    spectrumData = np.zeros((1112, 2, maxMass))
    #namesEnvs = np.zeros((72, 2), dtype=int)
    namesEnvs = []

    filesAll = os.listdir('./data/metab/metabData_Fernie/unzipped')

    #print (len(filesAll)) #1244
    #quit()
    ecotypeFiles = []
    ecotypeList = []
    yearList = []
    envList = []
    for a in range(len(filesAll)):
        file1 = filesAll[a]
        if 'ecotype' in file1:
            ecotypeFiles.append(file1)
            split1 = file1.replace('ecotype_', 'ecotype.')
            split1 = split1.split('.')[1]
            split1 = split1.split('_')
            ecotypeList.append(split1[0])
            yearList.append(split1[1])
            envList.append(split1[2])
    
    ecotypeecotypeFilesList = np.array(ecotypeFiles)
    ecotypeList = np.array(ecotypeList)
    yearList = np.array(yearList)
    envList = np.array(envList)

    


    if True:
        for file_index in range(ecotypeecotypeFilesList.shape[0]):
            fileNow = ecotypeecotypeFilesList[file_index]

            time1 = time.time()
            spectrumData_now = np.zeros((2, maxMass))

            print (file_index, ecotypeecotypeFilesList.shape[0] )
            
            #file_string = '10_1' 
            #file_string = str(count1) + '_' + str(rep1) 
            #file_path = "./data/metab/metabolights/MTBLS528/cdf/" + file_string + ".cdf"
            #file_path = './data/metab/metabData_Fernie/ecotype_378_1_6d.mzML'
            file_path = './data/metab/metabData_Fernie/unzipped/' + fileNow

            exp = MSExperiment()
            MzMLFile().load(file_path, exp)

            numberSpectra = exp.getNrSpectra()
            # Access first MS1 spectrum
            for spectra_index in range(numberSpectra):
                spec0 = exp[spectra_index]

                # Get the polarity enum
                #IonSource.Polarity.UNKNOWN = 0
                #IonSource.Polarity.POSITIVE = 1
                #IonSource.Polarity.NEGATIVE = 2
                polarity = spec0.getInstrumentSettings().getPolarity()
                ##if spectra_index == 0:
                #    print (file_index)
                #    print ('polarity', polarity)

                RT_now = spec0.getRT()
                #print("RT (min):", spec0.getRT() / 60)

                MStype = spec0.getMSLevel()
                #print ([MStype])
                #1 means original mass spectrometry
                #2 means MS/MS tandom mass spectrometry

                if True:

                    #print("MS level:", spec0.getMSLevel())
                    mz, inten = spec0.get_peaks()

                    mz = mz * precision

                    

                    mz_integer = np.round(mz).astype(int)

                    #print (np.unique(mz_integer).shape)

                    for index1 in range(mz.shape[0]):
                        spectrumData[file_index, MStype-1, mz_integer[index1] ]  += inten[index1]
                        #spectrumData_now[MStype-1, mz_integer[index1] ]  += inten[index1]

            #print ("A")
            #print (spectrumData_now.shape)
            #print (np.argwhere(spectrumData_now > 0).shape)

            print (time.time() - time1)

            if False:#file_index == 10:
                #names = names.reshape((-1,))

                mean1 = np.mean(spectrumData, axis=(0, 1))
                print (mean1.shape)
                print (np.argwhere(mean1 > 0).shape)
                print (np.argwhere(mean1 > 1e-3).shape)
                print (np.argwhere(mean1 > 1e-6).shape)
                print (np.argwhere(mean1 > 1e-10).shape)


                plt.plot(mean1)
                plt.show()


    quit()
    
    np.savez_compressed('./data/metab/metabData_Fernie/processed/massPrecise.npz', spectrumData)
    #np.savez_compressed('./data/metab/metabData_Fernie/processed/names.npz', ecotypeList)
    #np.savez_compressed('./data/metab/metabData_Fernie/processed/env.npz', envList)
    #np.savez_compressed('./data/metab/metabData_Fernie/processed/year.npz', yearList)

#savePreciseMass()
#quit()




def saveInsidePreciseMasses():


    precision = 10000

    if True:
        maxMass = 2000
        maxTime = 60

        envChoice = '0d'

        filesAll = os.listdir('./data/metab/metabData_Fernie/unzipped')

        

        #print (len(filesAll)) #1244
        #quit()
        ecotypeFiles = []
        for a in range(len(filesAll)):
            file1 = filesAll[a]
            if 'ecotype' in file1:
                if envChoice in file1:
                    ecotypeFiles.append(file1)
        
        ecotypeecotypeFilesList = np.array(ecotypeFiles)

        
        #precision = 100
        massValues = np.zeros((4, maxMass, maxTime, precision+1))
        


        if True:
            for file_index in range(ecotypeecotypeFilesList.shape[0]):
                fileNow = ecotypeecotypeFilesList[file_index]

                time1 = time.time()

                print (file_index, ecotypeecotypeFilesList.shape[0] )
                
                #file_string = '10_1' 
                #file_string = str(count1) + '_' + str(rep1) 
                #file_path = "./data/metab/metabolights/MTBLS528/cdf/" + file_string + ".cdf"
                #file_path = './data/metab/metabData_Fernie/ecotype_378_1_6d.mzML'
                file_path = './data/metab/metabData_Fernie/unzipped/' + fileNow

                exp = MSExperiment()
                MzMLFile().load(file_path, exp)

                

                
                

                numberSpectra = exp.getNrSpectra()
                # Access first MS1 spectrum
                for spectra_index in range(numberSpectra):
                    spec0 = exp[spectra_index]

                    RT_now = spec0.getRT()

                    MStype = spec0.getMSLevel()

                    if MStype == 1:

                        timeList = []
                        timeList.append(time.time())
                        mz, inten = spec0.get_peaks()

                        mz_integer = np.round(mz).astype(int)
                        RT_int = int(RT_now / 30)

                        #ar_argsort, indicesStart, indicesEnd = fastAllArgwhere(mz_integer)

                        
                        timeList.append(time.time())
                        if True:
                            mz_integer_lower = mz_integer - 0.5
                            mass_precise = np.round( (mz - mz_integer_lower) * precision ).astype(int)

                            #mass_precise = (mz - mz_integer + 0.5) * precision
                            #mz = mz_integer - 0.5 + (mass_precise * precision)

                            mult_RT = inten * RT_now

                            indices = np.array([mz_integer, mass_precise]).T 

                            timeList.append(time.time())

                            inverse1 = uniqueValMaker(indices)

                            timeList.append(time.time())

                            _, index1 = np.unique(inverse1, return_index=True)
                            mz_integer_unique, mass_precise_unique = mz_integer[index1], mass_precise[index1]
                            weight_sum = np.bincount(inverse1, weights=inten)
                            RT_sum = np.bincount(inverse1, weights=mult_RT)

                            timeList.append(time.time())

                            #massValues[0, mz_integer_unique, RT_int, mass_precise_unique ] += weight_sum
                            massValues[2, mz_integer_unique, RT_int, mass_precise_unique ] += weight_sum
                            massValues[1, mz_integer_unique, RT_int, mass_precise_unique ] += RT_sum




                            #mass_precise_unique, mass_inverse = np.unique(mass_precise, return_inverse=True)
                            #weight_sum = np.bincount(mass_inverse, weights=inten)
                            #RT_sum = np.bincount(mass_inverse, weights=mult_RT)

                            #massValues[0, mz_integer[index1], RT_int, mass_precise_unique ] += RT_sum


                            #print (np.argwhere(mass_count == 1).shape, mass_precise.shape)

                        

                            

                            

                            #for index1 in range(mz.shape[0]):
                            #    massValues[0, mz_integer[index1], RT_int, mass_precise[index1] ]  += inten[index1]
                            #    massValues[1, mz_integer[index1], RT_int, mass_precise[index1] ]  += mult_RT[index1]
                            
                            #timeList.append(time.time())

                        timeList1 = np.array(timeList)
                        #print (timeList1[1:] - timeList1[:-1] )
                

                #massValues[0, mz_integer_unique, RT_int, mass_precise_unique ] += np.copy(massValues[2, mz_integer_unique, RT_int, mass_precise_unique ])

                massValues[0] += massValues[2]
                massValues[3] += massValues[2] ** 2
                massValues[2] = 0


                
                #RT_vals = massValues[1, :1500, :40] / massValues[0, :1500, :40]
                #print (np.median(RT_vals [np.isnan(RT_vals) == False] ))
                #quit()
                #print ("A")
                #print (massValues.shape)
                #bestPreciseMass = np.argmax(massValues[0], axis=2)
                #print (bestPreciseMass.shape)

                print (time.time() - time1)
        #names = names.reshape((-1,))

        np.savez_compressed('./data/metab/metabData_Fernie/processed/massIntensitiesPrecise2_' + envChoice + '.npz', massValues)
        
        #np.savez_compressed('./data/metab/metabData_Fernie/processed/imageMetab.npz', spectrumData)
        #np.savez_compressed('./data/metab/metabData_Fernie/processed/names.npz', ecotypeList)
        #np.savez_compressed('./data/metab/metabData_Fernie/processed/env.npz', envList)
        #np.savez_compressed('./data/metab/metabData_Fernie/processed/year.npz', yearList)

        quit()

    
    if False:

        massValues = loadnpz('./data/metab/metabData_Fernie/processed/massIntensitiesPrecise.npz')

        #print (massValues.shape)

        bestPreciseMass = np.argmax(massValues[0], axis=2)

        argAll = np.argwhere( np.zeros(bestPreciseMass.shape, dtype=int) > -1 )

        massValues_RT = massValues[:, argAll[:, 0], argAll[:, 1], bestPreciseMass[argAll[:, 0], argAll[:, 1] ] ]
        massValues_RT = massValues_RT[1] / massValues_RT[0]
        massValues_RT = massValues_RT.reshape(bestPreciseMass.shape)

        trueMass = (bestPreciseMass - (precision // 2)) / precision
        trueMass = trueMass + np.arange(trueMass.shape[0]).reshape((-1, 1))

        finalValues = np.array([trueMass, massValues_RT ])

        print (finalValues.shape)

        np.savez_compressed('./data/metab/metabData_Fernie/processed/massIntensitiesPreciseFinal.npz', finalValues)


#saveInsidePreciseMasses()
#quit()





def saveZoomedMassMZ():


    maxMass = 100
    maxTime = 60

    massCheck = 739

    spectrumData = np.zeros((1112, 2, maxMass, maxTime))
    #namesEnvs = np.zeros((72, 2), dtype=int)
    namesEnvs = []

    filesAll = os.listdir('./data/metab/metabData_Fernie/unzipped')

    #print (len(filesAll)) #1244
    #quit()
    ecotypeFiles = []
    ecotypeList = []
    yearList = []
    envList = []
    for a in range(len(filesAll)):
        file1 = filesAll[a]
        if 'ecotype' in file1:
            ecotypeFiles.append(file1)
            split1 = file1.replace('ecotype_', 'ecotype.')
            split1 = split1.split('.')[1]
            split1 = split1.split('_')
            ecotypeList.append(split1[0])
            yearList.append(split1[1])
            envList.append(split1[2])
    
    ecotypeecotypeFilesList = np.array(ecotypeFiles)
    ecotypeList = np.array(ecotypeList)
    yearList = np.array(yearList)
    envList = np.array(envList)

    


    if True:
        for file_index in range(ecotypeecotypeFilesList.shape[0]):
            fileNow = ecotypeecotypeFilesList[file_index]

            print (file_index, ecotypeecotypeFilesList.shape[0] )
            
            #file_string = '10_1' 
            #file_string = str(count1) + '_' + str(rep1) 
            #file_path = "./data/metab/metabolights/MTBLS528/cdf/" + file_string + ".cdf"
            #file_path = './data/metab/metabData_Fernie/ecotype_378_1_6d.mzML'
            file_path = './data/metab/metabData_Fernie/unzipped/' + fileNow

            exp = MSExperiment()
            MzMLFile().load(file_path, exp)

            numberSpectra = exp.getNrSpectra()
            # Access first MS1 spectrum
            for spectra_index in range(numberSpectra):
                spec0 = exp[spectra_index]

                # Get the polarity enum
                #IonSource.Polarity.UNKNOWN = 0
                #IonSource.Polarity.POSITIVE = 1
                #IonSource.Polarity.NEGATIVE = 2
                polarity = spec0.getInstrumentSettings().getPolarity()
                ##if spectra_index == 0:
                #    print (file_index)
                #    print ('polarity', polarity)

                RT_now = spec0.getRT()
                #print("RT (min):", spec0.getRT() / 60)

                MStype = spec0.getMSLevel()
                #print ([MStype])
                #1 means original mass spectrometry
                #2 means MS/MS tandom mass spectrometry

                if True:

                    #print("MS level:", spec0.getMSLevel())
                    mz, inten = spec0.get_peaks()

                    #mz_integer = np.round(mz).astype(int)
                    mz_relative = mz - massCheck
                    RT_int = int(RT_now / 30)

                    argMass = np.argwhere( np.logical_and(  mz_relative > 0, mz_relative < 1 ) )[:, 0]
                    mz_integer = np.floor(mz_relative[argMass] * 100).astype(int)
                    inten = inten[argMass]


                    for index1 in range(mz_integer.shape[0]):
                        spectrumData[file_index, MStype-1, mz_integer[index1], RT_int ]  += inten[index1]
    

    
    np.savez_compressed('./data/metab/metabData_Fernie/processed/imageMetab_mass' + str(massCheck) + '.npz', spectrumData)

#saveZoomedMassMZ()
#quit()



def trainModel(model, X, names, envirement, trainTest2, modelName,  Y_background0=[], Niter = 10000, doPrint=True, regScale=1e-8, learningRate=1e-4, NphenStart=0, Nphen=1,  noiseLevel=0.1, doMod=False):

    

    X = torch.tensor(X).float()


    numWavelengths = X.shape[1]
    
    argTrain = np.argwhere(trainTest2 == 0)[:, 0]


    for phenNow in range(NphenStart, Nphen):

        print ('X shape', X.shape)


        if phenNow > 0:
            subset1 = np.arange(phenNow)

            Y_background = model(X, subset1)
            Y_background = Y_background.detach()
            if len(Y_background0) > 0:
                Y_background = torch.cat((Y_background0[0].detach(), Y_background), axis=1)
            Y_background = normalizeIndependent(Y_background)



        else:
            if len(Y_background0) > 0:
                Y_background = Y_background0[0].detach()
                Y_background = normalizeIndependent(Y_background)
                print (Y_background.shape)
            else:
                Y_background = torch.zeros((X.shape[0]), 0)

        subset_phen = np.zeros(1, dtype=int) + phenNow

        optimizer = torch.optim.RMSprop(model.parameters(), lr = learningRate)
        

        for a in range(Niter):
            
            X_train = X[trainTest2 == 0]
            
            #rand1 = torch.randn(X_train.shape) * noiseLevel
            rand1 = torch.rand(size=X_train.shape) * noiseLevel
            X_train = X_train + rand1
        
            
            Y = model(X_train, subset_phen)

            Y_abs = torch.mean(torch.abs(Y -  torch.mean(Y, axis=0).reshape((1, -1))   ))

            Y = removeIndependence(Y, Y_background[trainTest2 == 0])

            Y = normalizeIndependent(Y, cutOff=2) #Include for now

            
            heritability_now = cheapHeritability(Y, names[trainTest2 == 0], envirement[trainTest2 == 0], doMod=doMod)#, device=mps_device )
            loss = -1 * torch.mean(heritability_now)


            count1 = 0
            regLoss = 0
            for param in model.modelList[subset_phen[0]].parameters():
                #diff1 = torch.sum(  torch.abs(param[:, 1:] - param[:, :-1])  )
                #diff2 = torch.sum(  torch.abs(param[:, :, 1:] - param[:, :, :-1]) )
                #regLoss += diff1
                #regLoss += diff2

                regLoss += torch.sum(torch.abs(param))
                True
            regLoss = regLoss / Y_abs
            loss = loss + (regLoss * regScale)
            

            if a % 100 == 0:
                
                print ('iter:', a)

                with torch.no_grad():
                    Y = model(X, subset_phen)
                    Y = removeIndependence(Y, Y_background)

                    Y = normalizeIndependent(Y, cutOff=2) #Include for now

                    heritability_train = cheapHeritability(Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0] , doMod=doMod)
                    if 1 in trainTest2:
                        heritability_test = cheapHeritability(Y[trainTest2 == 1], names[trainTest2 == 1], envirement[trainTest2 == 1] , doMod=doMod)
                

                print ('subset_phen', subset_phen)
                print (heritability_train.data.numpy())
                if 1 in trainTest2:
                    print (heritability_test.data.numpy())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if a % 10 == 0:
                torch.save(model, modelName)
        



def secondaryHerit(Y, names, envirement, doMod=False):

    herit_6 = cheapHeritability(Y[envirement[:, 0] == '6d'], names[envirement[:, 0] == '6d'], envirement[envirement[:, 0] == '6d'], doMod=doMod)
    variance_0d = Y[envirement[:, 0] == '0d']
    variance_6d = Y[envirement[:, 0] == '6d']
    variance_0d = variance_0d - torch.mean(variance_0d, axis=0).reshape((1, -1))
    variance_6d = variance_6d - torch.mean(variance_6d, axis=0).reshape((1, -1))
    variance_0d = torch.mean(variance_0d ** 2, axis=0) ** 0.5
    variance_6d = torch.mean(variance_6d ** 2, axis=0) ** 0.5
    variance_ratio = variance_0d / (variance_6d + variance_0d)

    variance_ratio = variance_ratio * 0.5

    #variance_ratio = (variance_0d +  (variance_6d * 0.01)) / (variance_6d + variance_0d)


    #print (variance_ratio, herit_6)
    #quit()
    return variance_ratio, herit_6


def special_secondaryHerit(Y, names, envirement, X, doMod=False):


    Y_6d, names_6d, envirement_6d = Y[envirement[:, 0] == '6d'], names[envirement[:, 0] == '6d'], envirement[envirement[:, 0] == '6d']

    #print ("B")

    #print (torch.mean(torch.abs(Y_6d[:, 0])))

    names_6d_unique, names_6d_inverse = np.unique(names_6d, return_inverse=True)
    X_paste = torch.zeros((names_6d_unique.shape[0], X.shape[1]))
    for a in range(names_6d_unique.shape[0]):
        args1 = np.argwhere(np.logical_and(names == names_6d_unique[a] , envirement[:, 0] == '0d' )   )[:, 0]
        X_paste[a] = torch.mean(X[args1], axis=0)

    #print ("Xmean", torch.mean(torch.abs(X_paste)))
    #print ("Ymean", torch.mean(torch.abs(Y_6d)))
    X_paste = X_paste - torch.mean(X_paste, axis=0).reshape((1, -1))
    std1 = torch.mean(X_paste ** 2, axis=0).reshape((1, -1)) ** 0.5
    X_paste = X_paste / (std1 + 1e-10)
    X_paste = X_paste[names_6d_inverse]

    #NumRemove = 10
    #NumRemove = 2
    size_original = torch.mean((Y_6d[:, 0] - torch.mean(Y_6d[:, 0])) ** 2)
    NumRemove = 1
    for index1 in range(NumRemove):
        corList = torch.matmul(X_paste.T, Y_6d) / Y_6d.shape[0]
        #print ('Y_6d.shape', Y_6d.shape)
        #print ('corList', torch.mean(corList))
        max1 = np.argmax(np.abs(corList.data.numpy()))

        Y_6d[:, 0] = Y_6d[:, 0] - (corList[max1] * X_paste[:, max1])
        #print (X_paste.shape)
        X_paste = X_paste[:, np.arange(X_paste.shape[1]) != max1]
        #print (corList[max1])
    



    herit_6 = cheapHeritability(Y_6d, names_6d, envirement_6d, doMod=doMod)
    size_new = torch.mean((Y_6d[:, 0] - torch.mean(Y_6d[:, 0])) ** 2)

    herit_6 = (size_new / size_original) * herit_6
    blank = torch.zeros(herit_6.shape)


    #print ("A")
    #print (torch.mean(torch.abs(Y_6d[:, 0])), herit_6)
    #quit()


    
    return blank, herit_6



def modified_trainModel(model, X, names, envirement, trainTest2, modelName, Niter = 10000, doPrint=True, regScale=1e-8, learningRate=1e-4, NphenStart=0, Nphen=1,  noiseLevel=0.1, doMod=False):

    

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
            
            #rand1 = torch.randn(X_train.shape) * noiseLevel
            rand1 = torch.rand(size=X_train.shape) * noiseLevel
            X_train = X_train + rand1
        
            
            Y = model(X_train, subset_phen)

            Y_abs = torch.mean(torch.abs(Y -  torch.mean(Y, axis=0).reshape((1, -1))   ))

            Y = removeIndependence(Y, Y_background[trainTest2 == 0])

            #Y = normalizeIndependent(Y, cutOff=2) #Include for now

            
            
            herit_0, herit_6 = OLD_secondaryHerit(Y, names[trainTest2 == 0], envirement[trainTest2 == 0])#, X_train, doMod=doMod)#, device=mps_device )
            herit_0 = nn.ReLU()(herit_0)
            loss = -1 * torch.mean(herit_6 -   herit_0 )


            count1 = 0
            regLoss = 0
            for param in model.modelList[subset_phen[0]].parameters():
                #diff1 = torch.sum(  torch.abs(param[:, 1:] - param[:, :-1])  )
                #diff2 = torch.sum(  torch.abs(param[:, :, 1:] - param[:, :, :-1]) )
                #regLoss += diff1
                #regLoss += diff2

                regLoss += torch.sum(torch.abs(param))
                True
            regLoss = regLoss / Y_abs
            loss = loss + (regLoss * regScale)
            

            if a % 100 == 0:
                
                print ('iter:', a)

                with torch.no_grad():
                    Y = model(X, subset_phen)
                    Y = removeIndependence(Y, Y_background)

                    #Y = normalizeIndependent(Y, cutOff=2) #Include for now

                    

                    heritability_train = OLD_secondaryHerit(Y[trainTest2 == 0], names[trainTest2 == 0], envirement[trainTest2 == 0] )#, X[trainTest2 == 0], doMod=doMod)
                    if 1 in trainTest2:
                        heritability_test = OLD_secondaryHerit(Y[trainTest2 == 1], names[trainTest2 == 1], envirement[trainTest2 == 1])# , X[trainTest2 == 1], doMod=doMod)
                

                print ('subset_phen', subset_phen)
                print (heritability_train)
                if 1 in trainTest2:
                    print (heritability_test)

                #quit()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if a % 100 == 0:
                torch.save(model, modelName)

  

def removeDependent(Y, YBackground):



    YBackground = YBackground - torch.mean(YBackground, axis=0).reshape((1, -1))
    YBackground = YBackground / (torch.mean(YBackground ** 2, axis=0).reshape((1, -1)) ** 0.5)

    
    matrixCor = torch.matmul(YBackground.T, Y) / Y.shape[0]

    Y_est = torch.matmul( YBackground, matrixCor)

    
    Y = Y - Y_est
    return Y


def trainMassSpec():


    #genotype = loadnpz('./data/metab/metabolights/MTBLS528/processed/names.npz')
    #metabolites = loadnpz('./data/metab/metabolights/MTBLS528/processed/spectrum.npz')
    #metabolites = loadnpz('./data/metab/metabolights/MTBLS528/processed/spectrumTimes.npz')
    #metabolites = loadnpz('./data/metab/metabolights/MTBLS528/processed/spectrumTimes_low4.npz')
    #metabolites = loadnpz('./data/metab/metabolights/MTBLS528/processed/spectrumTimes_mass180.npz')
    #metabolites = loadnpz('./data/metab/metabolights/MTBLS528/processed/spectrumTimes_mass180_intensity.npz')


    #metabolites = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab.npz')[:, 0:1, :1500, :40]
    #metabolites = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab.npz')[:, 1:2, :1500, :40]
    metabolites = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab.npz')[:, :, :1500, :40]
    #metabolites = loadnpz('./data/metab/metabData_Fernie/processed/massPrecise_top10k.npz')


    print (metabolites.shape)
    quit()

    genotype = loadnpz('./data/metab/metabData_Fernie/processed/names.npz')
    envirement = loadnpz('./data/metab/metabData_Fernie/processed/env.npz')
    yearList = loadnpz('./data/metab/metabData_Fernie/processed/year.npz')
    #envirement = envirement.reshape((-1, 1))
    envirement = np.array([envirement, yearList]).T


    print (metabolites.shape)

    if False:
        img = np.mean(metabolites, axis=(0, 1) )
        #img = np.log(img + 1)

        sns.heatmap(img[100:], norm='log')
        plt.ylabel('mass (m/z)')
        plt.xlabel('retention time')
        plt.xticks( np.arange(20) * 2, np.arange(20)  )
        plt.yticks( np.arange(15) * 100, np.arange(16)[1:] * 100  )
        plt.show()
        quit()
    
    #print (metabolites.shape)
    #quit()
    #metabolites_mean = np.mean(metabolites, axis=0)

    #metabolites = metabolites[:, :, 1000:]
    #print (metabolites.shape)

    #metShow = np.mean(metabolites[:, 1, 609, 12], axis=0)
    #print (metShow.shape)
    #plt.plot(metShow[0])
    #plt.plot(metShow[1])
    #plt.show()
    #quit()

    #metabolites = metabolites[:, :, 209:210]
    #metabolites = metabolites[:, :, 137:138]
    #metabolites = metabolites[:, :, 521:522]
    #metabolites = metabolites[:, :, 609:610]

    

    #metabolites = metabolites[:, :, 152:153]
    #metabolites = metabolites[:, :, 739:740]
    #metabolites = metabolites[:, :, 579:580]
    

    
    

    metabolites = metabolites / np.max(metabolites)
    metabolites = torch.tensor(metabolites).float()


    #metabolites = torch.mean(metabolites, axis=1)
    #metabolites = metabolites.reshape((metabolites.shape[0], 1, metabolites.shape[1]))
    #metabolites = metabolites[:, np.zeros(10, dtype=int)]

    

    #metabolites = metabolites[:, 440:460, 1140:1240]
    #metabolites = metabolites[:, 179-2:179+3, :120]
    shape1 = metabolites.shape

    metabolites = metabolites.reshape((metabolites.shape[0], metabolites.shape[1]*metabolites.shape[2]*metabolites.shape[3]))
    #metabolites = metabolites.reshape((metabolites.shape[0], metabolites.shape[1]* metabolites.shape[2]))

    #print (metabolites.shape)


    if False:
        metaEstimates = loadnpz('./data/metab/metabolights/MTBLS528/processed/metaboliteEstimates.npz')
        metaEstimates = metaEstimates / np.mean(metaEstimates, axis=0).reshape((1, -1))
        from sklearn.decomposition import PCA
        #pca = PCA(n_components=metaEstimates.shape[1])
        pca = PCA(n_components=30)
        metaEstimates = pca.fit_transform(metaEstimates)
        metaEstimates = torch.tensor(metaEstimates).float()
        metabolites = removeDependent(metabolites, metaEstimates)


    if False:
        heritList = cheapHeritability(torch.tensor(metaEstimates).float() , genotype,  np.zeros((genotype.shape[0], 0)))
        heritList = heritList.data.numpy()
        plt.plot(heritList)
        plt.show()
        quit()



        from sklearn.decomposition import PCA
        pca = PCA(n_components=20)
        PCA1 = pca.fit_transform(metabolites[:, :1200])
        PCA2 = pca.fit_transform(metaEstimates)
        corMatrix = np.zeros((5, 5))
        for a in range(corMatrix.shape[0]):
            for b in range(corMatrix.shape[1]):
                cor1 = scipy.stats.pearsonr(PCA1[:, a], PCA2[:, b])[0]
                corMatrix[a, b] = abs(cor1)
        #plt.imshow(corMatrix)
        #plt.show()


        plt.plot(np.mean(np.abs(PCA2), axis=0))
        plt.show()
        quit()



    
    #metabolites = metabolites.reshape((metabolites.shape[0], metabolites.shape[1]*metabolites.shape[2]*metabolites.shape[3]))
    #metabolites = metabolites.reshape((metabolites.shape[0], metabolites.shape[1]*metabolites.shape[2]))

    mean_0d = torch.mean(metabolites[envirement[:, 0] == '0d'], axis=0)
    mean_6d = torch.mean(metabolites[envirement[:, 0] == '6d'], axis=0)
    ratioHigh = np.argwhere( (mean_6d - ( mean_0d * 2) ).data.numpy() > 0 )[:, 0]


    #metabolites = metabolites[:, ratioHigh]


    #'0d', '6d'
    #herit_env1 = cheapHeritability(metabolites[envirement[:, 0] == '0d'], genotype[envirement[:, 0] == '0d'], envirement[envirement[:, 0] == '0d'])#, doMod=doMod)
    #herit_env2 = cheapHeritability(metabolites[envirement[:, 0] == '6d'], genotype[envirement[:, 0] == '6d'], envirement[envirement[:, 0] == '6d'])#, doMod=doMod)
    #herit_env3 = cheapHeritability(metabolites, genotype, envirement)#, doMod=doMod)
    #for a in range(herit_env1.shape[0]):
    #    print (a, herit_env1[a])
    #    #print (torch.mean(torch.abs(metabolites[:, a]   - torch.mean(metabolites[:, a]) )))
    #plt.scatter(np.arange(herit_env1.shape[0]), herit_env1)
    #plt.scatter(np.arange(herit_env1.shape[0]), herit_env2)
    #plt.scatter(np.arange(herit_env1.shape[0]), herit_env3)


    #plt.scatter(herit_env1, herit_env2)
    #plt.show()


    
    genotype_unique, genotype_inverse = np.unique(genotype, return_inverse=True)


    #metabolites = np.log(metabolites + (np.mean(metabolites) * 0.1))

    

    Nsplit = 5
    np.random.seed(2)
    trainTest = np.random.randint(Nsplit, size=genotype_unique.shape[0])
    trainTest = trainTest[genotype_inverse]
    #trainTest[trainTest == 2] = 0

    #trainTest = trainTest[envirement[:, 0] == '6d']
    #metabolites = metabolites[envirement[:, 0] == '6d']
    #genotype = genotype[envirement[:, 0] == '6d']
    #envirement = envirement[envirement[:, 0] == '6d']
    


    


    for split_index in range(1, Nsplit):
        trainTest2 = np.zeros(trainTest.shape[0], dtype=int)
        trainTest2[trainTest == split_index] = 1



        Nphen = 100
        args = [metabolites.shape[1], 1, 10]
        #args = [metabolites.shape[1], 1, 500]
        #model = multiConv(Nphen, args, convMeta)
        #model = multiConv(Nphen, args, neuralNet)
        model = multiConv(Nphen, args, simpleModel)
        #model = multiConv(Nphen, args, specialLinear)
        #model = torch.load('./data/metab/metabData/model/1_split' + str(split_index) + '.pt')
        


        #Niter = 2000#0
        Niter = 2000
        #Niter = 500
        #Niter = 10000
        #Niter = 1000
        #Niter = 50000

        Nphen = 30
        NphenStart = 0
        
        #Nphen = 1


        #Good for Linear
        #learningRate = 1e-3
        #noiseLevel = 0.01
        #regScale = 5e-5

        #learningRate = 1e-4
        #noiseLevel = 0.01
        #regScale = 1e-4

        #learningRate = 1e-4
        #noiseLevel = 0.0001
        #regScale = 1e-6

        learningRate = 1e-4
        #learningRate = 1e-3
        #noiseLevel = 0.0001
        noiseLevel = 0.01 #good
        #noiseLevel = 0.05 #Extra noise
        
        #regScale = 1e-20
        #regScale = 1e-3
        regScale = 1e-4 #good
        #regScale = 1e-5
        #regScale = 1e-6
        #regScale = 0
        

        #modelName = './data/metab/metabData_241/model/spectrum_split' + str(split_index) + '_linear.pt'
        
        
        #modelName = './data/metab/metabData_Fernie/models/linear_mass152.pt'
        #modelName = './data/metab/metabData_Fernie/models/linear_mass739.pt'
        #modelName = './data/metab/metabData_Fernie/models/linear_mass579_spec.pt'
        #modelName = './data/metab/metabData_Fernie/models/linear_5.pt'
        #modelName = './data/metab/metabData_Fernie/models/linear_secondary5.pt'
        #modelName = './data/metab/metabData_Fernie/models/linear_sparse.pt'
        #modelName = './data/metab/metabData_Fernie/models/linear_1Met.pt'


        #useEnv = '0d'
        #if useEnv == '6d':
        #    otherEnv = '0d'
        #else:
        #    otherEnv = '6d'


        #modelBackground = torch.load('./data/metab/metabData_Fernie/models/linear_5.pt')
        #Y_background_pre = modelBackground(metabolites[envirement[:, 0] == otherEnv], np.arange(10))
        #genoBackground = genotype[envirement[:, 0] == otherEnv]

        #argGood = np.argwhere( np.logical_and(  envirement[:, 0] == useEnv ,  np.isin(genotype, genoBackground ) )  )[:, 0]

        #argGood = np.argwhere( np.isin(  genotype,   np.intersect1d(  genotype[ envirement[:, 0] == '0d' ] , genotype[ envirement[:, 0] == '6d' ] )   ) )[:, 0]
        #argGood = np.argwhere(envirement[:, 0] == '0d')[:, 0]
        argGood = np.argwhere(envirement[:, 0] == '6d')[:, 0]
        metabolites, genotype, envirement, trainTest2 = metabolites[argGood], genotype[argGood], envirement[argGood], trainTest2[argGood]
        #Y_background = torch.zeros(( genotype.shape[0], Y_background_pre.shape[1] ))
        #for geno_index in range(genotype.shape[0]):
        #    geno_now = genotype[geno_index]
        #    argNow = np.argwhere(genoBackground == geno_now)[:, 0]
        #    Y_background[geno_index] = torch.mean(Y_background_pre[argNow], axis=0)


        #modelName = './data/metab/metabData_Fernie/models/linear_primary6-3.pt'

        #modelName = './data/metab/metabData_Fernie/models/linear_secondary17.pt'
        #modelName = './data/metab/metabData_Fernie/models/linear_primary17.pt'

        #modelName = './data/metab/metabData_Fernie/models/neural_primary1.pt'
        #modelName = './data/metab/metabData_Fernie/models/neural_secondary1.pt'

        #modelName = './data/metab/metabData_Fernie/models/linear_primary_many2.pt'
        #modelName = './data/metab/metabData_Fernie/models/linear_bothEnv_many.pt'
        #modelName = './data/metab/metabData_Fernie/models/linear_primary_lowReg2.pt'

        #modelName = './data/metab/metabData_Fernie/models/linear_primary_many_split' + str(split_index) + '.pt'
        modelName = './data/metab/metabData_Fernie/models/linear_secondary_many_split' + str(split_index) + '.pt'
        #modelName = './data/metab/metabData_Fernie/models/linear_primary_preciseMass_split' + str(split_index) + '.pt'
        


        #envirement[envirement[:, 0] == '0d', 0] = '6'
        #envirement[envirement[:, 0] == '6d', 0] = '0d'
        #envirement[envirement[:, 0] == '6', 0] = '6d'


        #print (np.unique(envirement[:, 0], return_counts=True))
        #quit()

        
        doMod = False


        trainModel(model, metabolites, genotype, envirement, trainTest2, modelName, Niter = Niter, doPrint=True, regScale=regScale, learningRate=learningRate, NphenStart=NphenStart, Nphen=Nphen,  noiseLevel=noiseLevel, doMod=doMod)
        #modified_trainModel(model, metabolites, genotype, envirement, trainTest2, modelName, Niter=Niter, doPrint=True, regScale=regScale, NphenStart=NphenStart, Nphen=Nphen, learningRate=learningRate, noiseLevel=noiseLevel, doMod=doMod)
        #quit()

        if False:
            model = torch.load(modelName)
            Nsynth = 20
            Y = model(metabolites, np.arange(Nsynth))
            Y = normalizeIndependent(Y)

            baseLineHeritList = []
            Y_rand = torch.randn((Y.shape[0], 100))
            for a in range(Y.shape[1]):
                Y_rand_mod = torch.zeros(Y_rand.shape)
                print (Y[:, :a].shape)
                for b in range(Y_rand_mod.shape[1]):
                    Y_rand_mod[:, b:b+1] = removeIndependence(Y_rand[:, b:b+1], Y[:, :a])
                heritList_test = cheapHeritability(Y_rand_mod[trainTest2 == 1], genotype[trainTest2 == 1], envirement[trainTest2 == 1], doMod=doMod)
                heritList_test = heritList_test.data.numpy()
                baseLineHeritList.append(np.copy(heritList_test))
            baseLineHeritList = np.array(baseLineHeritList)

            print (baseLineHeritList)

            #plt.plot(np.median(baseLineHeritList, axis=1))
            #plt.plot(np.zeros(30))
            #)
            #print (baseLineHeritList.shape)
            #quit()



            heritList_train = cheapHeritability(Y[trainTest2 == 0], genotype[trainTest2 == 0], envirement[trainTest2 == 0], doMod=doMod)
            heritList_test = cheapHeritability(Y[trainTest2 == 1], genotype[trainTest2 == 1], envirement[trainTest2 == 1], doMod=doMod)
            #heritList_train[heritList_train<0] = 0
            #heritList_test[heritList_test<0] = 0
            plt.scatter(np.arange(Nsynth)+1, heritList_train.data.numpy())
            plt.scatter(np.arange(Nsynth)+1, heritList_test.data.numpy())
            #plt.title('0 day darkness')
            plt.xlabel('synthetic trait')
            plt.ylabel('heritability')
            plt.legend(['training set', 'test set'])
            plt.show()
            #quit()

            _, heritList_train = secondaryHerit(Y[trainTest2 == 0], genotype[trainTest2 == 0], envirement[trainTest2 == 0], doMod=doMod)
            _, heritList_test = secondaryHerit(Y[trainTest2 == 1], genotype[trainTest2 == 1], envirement[trainTest2 == 1], doMod=doMod)
            #heritList_train[heritList_train<0] = 0
            #heritList_test[heritList_test<0] = 0
            plt.scatter(np.arange(Nsynth)+1, heritList_train.data.numpy())
            plt.scatter(np.arange(Nsynth)+1, heritList_test.data.numpy())
            #plt.title('0 day darkness')
            plt.xlabel('synthetic trait')
            plt.ylabel('6 day darkness heritability')
            plt.legend(['training set', 'test set'])
            plt.show()
            quit()



        if False:

            model = torch.load(modelName)
            coef_all = getMultiModelCoef(model, multi=True)

            Y = model(metabolites, np.arange(5))
            Y = normalizeIndependent(Y)
            heritList_test = cheapHeritability(Y[trainTest2 == 1], genotype[trainTest2 == 1], envirement[trainTest2 == 1], doMod=doMod)
            print ('heritList_test')
            print (heritList_test)


            coef_sum = np.sum(coef_all, axis=1)

            print (coef_sum.shape)

            plt.plot(coef_sum[:5].T)
            plt.show()
            quit()


            #heritList_train = cheapHeritability(Y[trainTest2 == 0], genotype[trainTest2 == 0], envirement[trainTest2 == 0], doMod=doMod)

            for a in range(5):
                print (a)
                coef = coef_all[a]

                #print (coef.shape)
                #quit()

                maxVals = np.max(np.abs(coef), axis=1)
                argTop = np.argsort(maxVals * -1)
                #nonInt = argTop % 10 
                argTop = (argTop/10) + 100
                print (argTop[:10])

                #coef[200:] = 0
                #coef[:300] = 0
                #coef[400:] = 0
                #coef[:, :100] = 0

                sns.heatmap(coef, cmap='bwr')
                #plt.yticks(np.arange(coef.shape[0] // 10) * 10, 100+(10 * np.arange(coef.shape[0] // 10)) )
                #plt.yticks(np.arange(coef.shape[0] ) , 100+( np.arange(coef.shape[0] )) )
                #plt.xticks(np.arange(coef.shape[1]+1),  ['30', '60', '90', '120', '150', '180']  )
                plt.xlabel('time')
                plt.ylabel('mass')
                plt.show()
            quit()

            for a in range(coef.shape[1]):
                plt.scatter( np.arange(coef.shape[0]) , coef[:, a])
            plt.show()
            quit()
            coef = coef.reshape((1, coef.shape[0], coef.shape[1]))

            #Y = torch.sum(metabolites * coef, axis=(1, 2)).reshape((-1, 1))
            print (Y.shape)


            heritList_train = cheapHeritability(Y[trainTest2 == 0], genotype[trainTest2 == 0], envirement[trainTest2 == 0], doMod=doMod)
            heritList_test = cheapHeritability(Y[trainTest2 == 1], genotype[trainTest2 == 1], envirement[trainTest2 == 1], doMod=doMod)


            print (heritList_test)
            print (heritList_train)
            

            
            
            quit()

        #quit()
        if False:


            
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)
            metabolites_flat = metabolites#.reshape((metabolites.shape[0], metabolites.shape[1]*metabolites.shape[2]))
            #metabolites_flat = metabolites.reshape((metabolites.shape[0], metabolites.shape[1]*metabolites.shape[2]*metabolites.shape[3]))
            metabolites_flat = metabolites_flat[:, torch.mean(metabolites_flat, axis=0) > 0]

            #print (trainTest2.shape)
            #print (metabolites_flat.shape)
            

            print (envirement.shape)
            print (trainTest2.shape)

            heritList_train = cheapHeritability(metabolites_flat[trainTest2 == 0], genotype[trainTest2 == 0], envirement[trainTest2 == 0], doMod=doMod)
            heritList_test = cheapHeritability(metabolites_flat[trainTest2 == 1], genotype[trainTest2 == 1], envirement[trainTest2 == 1], doMod=doMod)
            heritList_train, heritList_test = heritList_train.data.numpy(), heritList_test.data.numpy()

            print (torch.mean(metabolites_flat))
            print (metabolites_flat.shape)
            synthUsed = 5
            Y_wave = maxWaveMethod(metabolites_flat, synthUsed, genotype, envirement, trainTest2)
            Y_wave = torch.tensor(Y_wave).float()
            print (torch.mean(Y_wave))
            print (Y_wave.shape)
            heritList_train = cheapHeritability(Y_wave[trainTest2 == 0], genotype[trainTest2 == 0], envirement[trainTest2 == 0], doMod=doMod)
            heritList_test = cheapHeritability(Y_wave[trainTest2 == 1], genotype[trainTest2 == 1], envirement[trainTest2 == 1], doMod=doMod)
            heritList_train, heritList_test = heritList_train.data.numpy(), heritList_test.data.numpy()
            print (heritList_test)
            quit()


            #heritList_train_reshape = heritList_train.reshape(( 2, heritList_train.shape[0] // 2 ))
            #plt.scatter(np.arange(heritList_train_reshape.shape[1]), heritList_train_reshape[0])
            #plt.scatter(np.arange(heritList_train_reshape.shape[1]), heritList_train_reshape[1])
            #plt.show()


            #plt.plot(np.mean(metabolites.data.numpy(), axis=0))
            #plt.plot(heritList_train)
            #plt.plot(heritList_test)
            #plt.show()
            #quit()

            #plt.scatter(heritList_train, heritList_test)
            #plt.show()

            argGood = np.argwhere(np.logical_and( np.isnan(heritList_train) == False, np.isnan(heritList_test) == False ))[:, 0]
            
            heritList_train, heritList_test = heritList_train[argGood], heritList_test[argGood]

            argMax1 = argGood[np.argmax(heritList_train)]

            #plt.plot(metabolites[:, argMax1].data.numpy())
            #plt.plot(genotype % 2)
            #plt.show()


            plt.scatter(heritList_train, heritList_test)
            plt.show()

            #print (heritList_train)
            #print (heritList_test)
            print (heritList_test[np.argmax(heritList_train)])

            quit()
            pca.fit(metabolites_flat[trainTest2 == 0].data.numpy())
            metabolites_PCA = pca.transform(metabolites_flat.data.numpy())
            metabolites_PCA = torch.tensor(metabolites_PCA)
            quit()



#trainMassSpec()
#quit()




def savePhenotypesMassSpec():


    #metabolites = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab.npz')[:, :, :1500, :40]
    metabolites = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab.npz')[:, :, :1500, :40]
    #metabolites = loadnpz('./data/metab/metabData_Fernie/processed/massPrecise_top10k.npz')
    #metabolites = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab_mass' + str(739) + '.npz')[:, :, :, :40]
    genotype = loadnpz('./data/metab/metabData_Fernie/processed/names.npz')
    envirement = loadnpz('./data/metab/metabData_Fernie/processed/env.npz')
    yearList = loadnpz('./data/metab/metabData_Fernie/processed/year.npz')
    #envirement = envirement.reshape((-1, 1))
    envirement = np.array([envirement, yearList]).T


    #metabolites = metabolites[:, :, 209:210]
    #metabolites = metabolites[:, :, 521:522]
    #metabolites = metabolites[:, :, 609:610]
    #metabolites = metabolites[:, :, 152:153]
    #metabolites = metabolites[:, :, 739:740]
    #metabolites = metabolites[:, :, 579:580]

    #linear_mass579
    
    #metabolites = metabolites[:, 1, 609, 12:13]

    metabolites = metabolites / np.max(metabolites)
    metabolites = torch.tensor(metabolites).float()




    metabolites = metabolites.reshape((metabolites.shape[0], metabolites.shape[1]*metabolites.shape[2]*metabolites.shape[3]))
    #metabolites = metabolites.reshape((metabolites.shape[0], metabolites.shape[1]*metabolites.shape[2]))

    envChoice = '0d'
    #envChoice = '6d'
    metabolites = metabolites[envirement[:, 0] == envChoice]
    genotype = genotype[envirement[:, 0] == envChoice]
    envirement = envirement[envirement[:, 0] == envChoice]
    
    genotype_unique, genotype_inverse = np.unique(genotype, return_inverse=True)

    #print (genotype_unique.shape)
    #quit()

    splitIndex = 0

    #modelName = './data/metab/metabData_Fernie/models/linear_5.pt'
    #modelName = './data/metab/metabData_Fernie/models/linear_mass152.pt'
    #modelName = './data/metab/metabData_Fernie/models/linear_mass739_spec.pt'
    #modelName = './data/metab/metabData_Fernie/models/linear_mass579.pt'
    #modelName = './data/metab/metabData_Fernie/models/linear_sparse.pt'
    #modelName = './data/metab/metabData_Fernie/models/linear_prim6.pt'

    #modelName = './data/metab/metabData_Fernie/models/linear_secondary6.pt'
    #modelName = './data/metab/metabData_Fernie/models/linear_primary_many.pt'
    #modelName = './data/metab/metabData_Fernie/models/linear_secondary10.pt'
    #modelName = './data/metab/metabData_Fernie/models/linear_primary_channel0_split1.pt'


    modelName = './data/metab/metabData_Fernie/models/linear_primary_many_split' + str(splitIndex) + '.pt'
    #modelName = './data/metab/metabData_Fernie/models/linear_secondary_many_split' + str(splitIndex) + '.pt'
    #modelName = './data/metab/metabData_Fernie/models/linear_primary_preciseMass_split' + str(splitIndex) + '.pt'

    
    

    

    if False:
        model1 = torch.load('./data/metab/metabData_Fernie/models/linear_primary6-3.pt')
        model2 = torch.load('./data/metab/metabData_Fernie/models/linear_secondary6.pt')
        Y1 = model1(metabolites, np.arange(1))
        Y2 = model2(metabolites, np.arange(1))
        Y1 = Y1.detach().numpy()
        Y2 = Y2.detach().numpy()
        Y1[envirement[:, 0] == '0d'] = Y1[envirement[:, 0] == '0d'] - np.mean(Y1[envirement[:, 0] == '0d'])
        Y1[envirement[:, 0] == '6d'] = Y1[envirement[:, 0] == '6d'] - np.mean(Y1[envirement[:, 0] == '6d'])
        Y2[envirement[:, 0] == '0d'] = Y2[envirement[:, 0] == '0d'] - np.mean(Y2[envirement[:, 0] == '0d'])
        Y2[envirement[:, 0] == '6d'] = Y2[envirement[:, 0] == '6d'] - np.mean(Y2[envirement[:, 0] == '6d'])

        print (scipy.stats.pearsonr(  Y1[:, 0], Y2[:, 0] ))

        Y1_mean = np.zeros(( genotype_unique.shape[0], Y1.shape[1] ))
        Y2_mean = np.zeros(( genotype_unique.shape[0], Y2.shape[1] ))
        for a in range(genotype_unique.shape[0]):
            args1 = np.argwhere(genotype == genotype_unique[a])[:, 0]
            Y1_mean[a] = np.mean(Y1[args1], axis=0)
            Y2_mean[a] = np.mean(Y2[args1], axis=0)
        
        print (scipy.stats.pearsonr(  Y1_mean[:, 0] , Y2_mean[:, 0] ))
        
        quit()
    
    

    #np.savez_compressed('./data/metab/metabData_Fernie/pred/linear_6.npz', metabolites.data.numpy())
    #quit()

    model = torch.load(modelName)
    Y = model(metabolites, np.arange(30))

    #modelBackground = torch.load('./data/metab/metabData_Fernie/models/linear_5.pt')
    #YBackground = modelBackground(metabolites, np.arange(20))
    #Y = removeDependent(Y, YBackground)


    if True: #TODO false REMOVE!!!
        Y = normalizeIndependent(Y) 
    Y = Y.detach().numpy()


    #Y = Y - np.mean(Y, axis=0).reshape((1, -1))
    #variance = np.mean(Y ** 2, axis=0)
    #print (variance)
    #quit()

    #print (genotype_unique.shape)
    #quit()
    
    #cheapHeritability(Y, )
    #cheapHeritability(Y, genotype, envirement[trainTest2 == 0], doMod=doMod)
    

    Y_mean = np.zeros(( genotype_unique.shape[0], Y.shape[1] ))
    for a in range(genotype_unique.shape[0]):
        args1 = np.argwhere(genotype == genotype_unique[a])[:, 0]
        Y_mean[a] = np.mean(Y[args1], axis=0)
    
    #print (Y_mean.shape)
    #quit()

    #Y_0d = loadnpz('./data/metab/metabData_Fernie/pred/linear_primary_many_split' + str(0) + '.npz')
    #print (Y_0d.shape)
    #print (Y_mean.shape)
    #quit()

    #np.savez_compressed('./data/metab/metabData_Fernie/pred/linear_mass739_spec.npz', Y_mean)
    #np.savez_compressed('./data/metab/metabData_Fernie/pred/linear_mass579.npz', Y_mean)
    #np.savez_compressed('./data/metab/metabData_Fernie/pred/linear_secondary10.npz', Y_mean)
    #np.savez_compressed('./data/metab/metabData_Fernie/pred/linear_primary_many.npz', Y_mean)
    #np.savez_compressed('./data/metab/metabData_Fernie/pred/linear_sparseDep.npz', Y_mean)
    #np.savez_compressed('./data/metab/metabData_Fernie/pred/ecotypeNames.npz', genotype_unique)

    np.savez_compressed('./data/metab/metabData_Fernie/pred/linear_primary_many_split' + str(splitIndex) + '.npz', Y_mean)
    #np.savez_compressed('./data/metab/metabData_Fernie/pred/linear_secondary_many_split' + str(splitIndex) + '.npz', Y_mean)
    #np.savez_compressed('./data/metab/metabData_Fernie/pred/linear_primary_preciseMass_split' + str(splitIndex) + '.npz', Y_mean)



#savePhenotypesMassSpec()
#quit()


def savePCA():


    metabolites = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab.npz')[:, :, :1500, :40]
    genotype = loadnpz('./data/metab/metabData_Fernie/processed/names.npz')
    envirement = loadnpz('./data/metab/metabData_Fernie/processed/env.npz')
    yearList = loadnpz('./data/metab/metabData_Fernie/processed/year.npz')
    envirement = np.array([envirement, yearList]).T


    metabolites = metabolites / np.max(metabolites)
    metabolites = metabolites.reshape((metabolites.shape[0], metabolites.shape[1]*metabolites.shape[2]*metabolites.shape[3]))

    envChoice = '0d'
    metabolites = metabolites[envirement[:, 0] == envChoice]
    genotype = genotype[envirement[:, 0] == envChoice]
    envirement = envirement[envirement[:, 0] == envChoice]
    
    genotype_unique, genotype_inverse = np.unique(genotype, return_inverse=True)


    from sklearn.decomposition import PCA
    pca = PCA(n_components=100)
    pca.fit(metabolites.T)
    Y = pca.components_
    Y = Y.T

    Y_mean = np.zeros(( genotype_unique.shape[0], Y.shape[1] ))
    for a in range(genotype_unique.shape[0]):
        args1 = np.argwhere(genotype == genotype_unique[a])[:, 0]
        Y_mean[a] = np.mean(Y[args1], axis=0)
    
    np.savez_compressed('./data/metab/metabData_Fernie/pred/PCA_primary.npz', Y_mean)


#savePCA()
#quit()




def showSecPrimCor():

    metabolites = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab.npz')[:, :, :1500, :40]
    #metabolites = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab_mass' + str(739) + '.npz')[:, :, :, :40]
    genotype = loadnpz('./data/metab/metabData_Fernie/processed/names.npz')
    envirement = loadnpz('./data/metab/metabData_Fernie/processed/env.npz')
    yearList = loadnpz('./data/metab/metabData_Fernie/processed/year.npz')
    #envirement = envirement.reshape((-1, 1))
    envirement = np.array([envirement, yearList]).T


    #genotype_unique = np.unique()

    
    metabolites = metabolites / np.max(metabolites)
    metabolites = torch.tensor(metabolites).float()
    metabolites = metabolites.reshape((metabolites.shape[0], metabolites.shape[1]*metabolites.shape[2]*metabolites.shape[3]))


    genoBoth = np.intersect1d(genotype[envirement[:, 0] == '0d'], genotype[envirement[:, 0] == '6d'])

    argGood = np.argwhere(np.isin(genotype, genoBoth))[:, 0]
    envirement = envirement[argGood]
    metabolites = metabolites[argGood]
    genotype = genotype[argGood]
    
    genotype_unique = np.unique(genotype)

    #model1 = torch.load('./data/metab/metabData_Fernie/models/linear_primary14_copy.pt')
    #model2 = torch.load('./data/metab/metabData_Fernie/models/linear_secondary14_copy.pt')
    model1 = torch.load('./data/metab/metabData_Fernie/models/linear_primary17.pt')
    model2 = torch.load('./data/metab/metabData_Fernie/models/linear_secondary17.pt')
    #model1 = torch.load('./data/metab/metabData_Fernie/models/neural_primary1.pt')
    #model2 = torch.load('./data/metab/metabData_Fernie/models/neural_secondary1.pt')

    pred1 = model1(metabolites, np.arange(5))
    pred2 = model2(metabolites, np.arange(5))

    pred1 = normalizeIndependent(pred1).data.numpy()
    pred2 = normalizeIndependent(pred2).data.numpy()


                              
    #indexPrint = 0

    for indexPrint in [0]:# range(5):

        #print (scipy.stats.pearsonr( pred1[:, 0], pred2[:, 0] ))
        print (scipy.stats.pearsonr( pred1[envirement[:, 0] == '0d', indexPrint], pred2[envirement[:, 0] == '0d', indexPrint] ))
        print (scipy.stats.pearsonr( pred1[envirement[:, 0] == '6d', indexPrint], pred2[envirement[:, 0] == '6d', indexPrint] ))
   # quit()
    pred1_mean1 = np.zeros(( genotype_unique.shape[0], pred1.shape[1] ))
    pred1_mean2 = np.zeros(( genotype_unique.shape[0], pred1.shape[1] ))
    pred2_mean1 = np.zeros(( genotype_unique.shape[0], pred1.shape[1] ))
    pred2_mean2 = np.zeros(( genotype_unique.shape[0], pred1.shape[1] ))

    for a in range(genotype_unique.shape[0]):
        args_both = np.argwhere(genotype == genotype_unique[a])[:, 0]
        args1 = args_both[ envirement[args_both, 0] == '0d' ]
        args2 = args_both[ envirement[args_both, 0] == '6d' ]
        
        pred1_mean1[a] = np.mean(pred1[args1], axis=0)
        pred1_mean2[a] = np.mean(pred1[args2], axis=0)
        pred2_mean1[a] = np.mean(pred2[args1], axis=0)
        pred2_mean2[a] = np.mean(pred2[args2], axis=0)

    print ('')
    print (scipy.stats.pearsonr( pred1_mean1[:, indexPrint], pred2_mean2[:, indexPrint] ))

    plt.scatter(  pred1_mean1[:, indexPrint], pred2_mean2[:, indexPrint]  )
    plt.xlabel('primary genotype mean')
    plt.ylabel('secondary genotype mean')
    plt.show()

    #print (scipy.stats.pearsonr( pred1_mean1[:, indexPrint], pred2_mean1[:, indexPrint] ))
    #print (scipy.stats.pearsonr( pred1_mean2[:, indexPrint], pred2_mean2[:, indexPrint] ))

    #print (scipy.stats.pearsonr( pred1_mean2[:, indexPrint], pred1_mean1[:, indexPrint] ))
    #print (scipy.stats.pearsonr( pred2_mean2[:, indexPrint], pred2_mean1[:, indexPrint] ))



    from sklearn.linear_model import LinearRegression
    
    print ('')
    for a in range(5):
        reg = LinearRegression().fit(pred1_mean1, pred2_mean2[:, a] )
        pred = reg.predict(pred1_mean1)
        print (scipy.stats.pearsonr(pred, pred2_mean2[:, a]))

    print ('')
    for a in range(5):
        reg = LinearRegression().fit(pred2_mean2, pred1_mean1[:, a] )
        pred = reg.predict(pred2_mean2)
        print (scipy.stats.pearsonr(pred, pred1_mean1[:, a]))


    
    
    
    quit()


    
    fileName_6d = './data/metab/metabData_Fernie/6d-Table-1.tsv'
    fileName_0d = './data/metab/metabData_Fernie/0d-Table-1.tsv'
    genotype_known_6d, metNames_6d, metValues_6d = processKnownMet(fileName_6d)
    genotype_known_0d, metNames_0d, metValues_0d = processKnownMet(fileName_0d)

    
    metValues_6d = metValues_6d[:,  np.isin(metNames_6d, metNames_0d) == False ]
    metValues_0d = metValues_0d[:,  np.isin(metNames_0d, metNames_6d) == False ]
    metValues_all = np.concatenate(( metValues_0d,  metValues_6d), axis=1)
    
    assert np.array_equal(genotype_known_6d, genotype_known_0d)
    genotype_known = genotype_known_6d
    
    arg_known = np.zeros(genotype_known.shape[0], dtype=int)
    for a in range(genotype_known.shape[0]):
        arg1 = np.argwhere(genotype_unique == genotype_known[a])[0, 0]
        arg_known[a] = arg1
    
    pred1_mean1_known = pred1_mean1[arg_known]
    pred1_mean2_known = pred1_mean2[arg_known]
    pred2_mean1_known = pred2_mean1[arg_known]
    pred2_mean2_known = pred2_mean2[arg_known]

    from sklearn.linear_model import LinearRegression
    #from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso


    coef_lists = []
    original_Y = [pred1_mean1_known, pred1_mean2_known, pred2_mean1_known, pred2_mean2_known]
    for a in range(len(original_Y)):
        corList = []
        for b in range(metValues_all.shape[1]):
            #print (metValues_all[:, b].shape, original_Y[a].shape)
            cor1 = scipy.stats.pearsonr( metValues_all[:, b], original_Y[a][:, 0] )
            #corList.append(cor1[0])
            corList.append(np.log10(cor1[1]))
        corList = np.array(corList)
        coef_lists.append(np.copy(corList))
    
    print (scipy.stats.pearsonr( coef_lists[0], coef_lists[3] ))
    quit()


    projected = []
    original_Y = [pred1_mean1_known, pred1_mean2_known, pred2_mean1_known, pred2_mean2_known]
    for a in range(len(original_Y)):
        #reg = LinearRegression().fit(metValues_all, original_Y[a] )
        reg = Lasso(alpha=2e-2).fit(metValues_all, original_Y[a] )
        print (reg.coef_)
        pred = reg.predict(metValues_all)
        projected.append(np.copy(pred))
    

    proj1_mean1, proj1_mean2, proj2_mean1, proj2_mean2 = projected[0], projected[1], projected[2], projected[3]

    print (scipy.stats.pearsonr(  proj1_mean1[:, 0], proj2_mean2[:, 0] ))

    
    


#showSecPrimCor()
#quit()




def showCoef():

    #modelName = './data/metab/metabData_Fernie/models/linear_secondary5.pt'
    modelName = './data/metab/metabData_Fernie/models/linear_0Met.pt'
    model = torch.load(modelName)
    coef = getMultiModelCoef(model, multi=True)
    coef = coef[:20]

    

    #coef[np.abs(coef) < 0.05] = 0.0

    coef_copy = np.copy(coef)



    #plt.hist(coef_copy.reshape((-1,)), bins=100)
    #plt.show()
    #quit()

    corMatrix_abs = np.zeros((  coef.shape[0], coef.shape[0] ))
    for a in range(coef.shape[0]):
        for b in range(coef.shape[0]):
            cor1 = scipy.stats.pearsonr( coef[:, a], coef[:, b]  )
            corMatrix_abs[a, b] = cor1[0]
            #print (cor1)
            #print (np.mean(np.abs(coef[:, a])))
            #print (np.mean(np.abs(coef[:, b])))
            #quit()
    
    #plt.imshow(corMatrix_abs)
    #plt.show()
    #quit()

    coef_copy = coef_copy.reshape((coef_copy.shape[0], 1500, 40))

    

    

    for a in range(20):
        print (a)

        vmax = np.max(np.abs(coef_copy[a]))
        vmin = -1 * vmax


        sns.heatmap(coef_copy[a], cmap='bwr', vmin=vmin, vmax=vmax)
        plt.xlabel('retention time')
        plt.ylabel('mass')
        plt.show()

    quit()

    size_original = coef.shape
    coef_sum = np.sum(np.abs(coef), axis=0)

    print (coef.shape)

    arg_sum = np.argwhere(coef_sum > 0)[:, 0]

    #print (coef.shape)
    #print (arg_sum.shape)
    #quit()
    

    coef = coef[:, arg_sum]

    print (coef.shape)


    #metabolites = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab.npz')[:, :, :1500, :40]

    #coef_reshape = coef.reshape((coef.shape[0], 2, 1500, 40))

    #plt.hist(coef.reshape((-1,)), bins=100)
    #plt.show()
    #quit()

    #sns.heatmap(coef_reshape[0, 0], cmap='bwr')
    #plt.show()

    #plt.plot(coef[:, :1000].T)
    #plt.show()
    #quit()

    if False:
        from sklearn.decomposition import DictionaryLearning
        
        L = 50  # example value; tune depending on desired compression/sparsity
        #model = DictionaryLearning(n_components=L, transform_algorithm='omp',  fit_algorithm='cd', random_state=0, alpha=1e-2)#1e-3)
        #model = DictionaryLearning(n_components=L, transform_algorithm='omp',  fit_algorithm='cd', random_state=0, alpha=1e-3)#1e-3)
        model = DictionaryLearning(n_components=L, random_state=0, alpha=1e-3)
        B = model.fit_transform(coef.T).T  # shape will be (L, 10000)
        # Get the dictionary
        A = model.components_.T  # shape will be (M, L)


        np.savez_compressed('./data/temp/A.npz', A)
        np.savez_compressed('./data/temp/B.npz', B)

    else:
        A = loadnpz('./data/temp/A.npz')
        B = loadnpz('./data/temp/B.npz')

    print ("saved")
    

    # Now X ≈ A @ B

    coef_approx = A @ B 

    diff1 = np.mean(np.abs(coef - coef_approx))
    print (diff1)
    print (np.mean(np.abs(coef)))

    coef_paste = np.zeros((B.shape[0], size_original[1] ))
    coef_paste[:, arg_sum] = B
    coef_paste = coef_paste.reshape((coef_paste.shape[0], 1500, 40))

    coef_copy = coef_copy.reshape((coef_copy.shape[0], 1500, 40))

    print (np.argwhere(np.abs(coef_paste[:20]) > 0).shape)
    print (np.argwhere(np.abs(coef_copy) > 0).shape)


    for a in range(20):
        maxSize = np.max(np.abs(coef_paste[a, 0]))
        print (a)

        sns.heatmap(coef_copy[a], cmap='bwr', vmin=maxSize*-1, vmax=maxSize)
        plt.show()

        #sns.heatmap(coef_paste[a], cmap='bwr', vmin=maxSize*-1, vmax=maxSize)
        #plt.show()



    quit()


    B = B.reshape((B.shape[0], 2, 1500, 40))



    for a in range(B.shape[0]):
        print (np.mean(np.abs(B[a, 0])))
        #sns.heatmap(B[a, 0], cmap='bwr')
        #plt.show()
    
    #print (A.shape)
    #print (B.shape)
    quit()




    #print (coef.shape)

    #import numpy as np
    #X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    from sklearn.decomposition import NMF
    model = NMF(n_components=10, init='random', random_state=0)
    W = model.fit_transform(X)
    H = model.components_

    for a in range(10):
        sns.heatmap(coef[a, 0], cmap='bwr')
        plt.show()


#showCoef()
#quit()


def saveKnownMetab():

    #knownMetTable = np.loadtxt('./data/metab/metabData_Fernie/6d-Table-1.tsv', delimiter='\t', dtype=str)
    knownMetTable = pd.read_csv('./data/metab/metabData_Fernie/6d-Table-1.tsv', sep='\t', encoding='utf-8')
    knownMetTable = knownMetTable.to_numpy()

    genotypes = knownMetTable[1:, 0]
    metNames = knownMetTable[0, 1:]
    metValues = knownMetTable[1:, 1:].astype(float)

    for a in range(genotypes.shape[0]):
        genotypes[a] = genotypes[a].split('.')[1]
    
    #print (genotypes[:10])
    #print (metNames[:10])
    #print (knownMetTable.shape)

    argMet = np.argwhere(metNames == 'Met.821')[0, 0]

    #print (metValues.shape)
    #quit()

    #32
    #45

    #print (metNames[45])
    #quit()

    metValues_now = metValues[:, argMet]


    file_path = "./data/metab/metabData_Alex/call_method_75/call_method_75_TAIR8.csv" 
    df = pd.read_csv(file_path, sep=",", dtype=str, skiprows=1)

    chrome = df['Chromosome'].to_numpy().astype(int)
    pos = df['Positions'].to_numpy().astype(int)
    pos2 = np.copy(pos)
    for a in range(2, 6):
        args1 = np.argwhere(chrome == a)[:, 0]
        pos2[args1] = pos2[args1] + pos2[args1[0] - 1]


    

    arAll = []
    for a in range(genotypes.shape[0]):
        val1 = df[genotypes[a]].to_numpy()
        arAll.append(np.copy(val1))
    arAll = np.array(arAll)
    arAll[arAll == 'A'] = '0'
    arAll[arAll == 'T'] = '1'
    arAll[arAll == 'G'] = '2'
    arAll[arAll == 'C'] = '3'
    arAll = arAll.astype(float)

    chromeList = df['Chromosome']

    corList = []
    for a in range(arAll.shape[1]):
        cor1 = scipy.stats.pearsonr(metValues_now, arAll[:, a])
        corList.append([cor1[0], cor1[1]])

    corList = np.array(corList)

    pValLog = np.log10(corList[:, 1]) * -1

    for a in range(1, 6):
        args1 = np.argwhere(chrome == a)[:, 0]
        color1 = ['red', 'blue'][a%2]
        plt.scatter(pos2[args1], pValLog[args1], c=color1)
    plt.xlabel("genomic bin")
    plt.ylabel('-log10(p)')
    plt.show()
    quit()


    plt.plot(np.log10(corList[:, 1]) * -1)
    plt.show()

    quit()
    

    #metValues_now = metValues_now.reshape((-1, 1))

    print (metValues_now.shape)
    print (genotypes.shape)

    print (type(metValues_now))
    print (type(genotypes))



    np.savez_compressed('./data/metab/metabData_Fernie/pred/known_2.npz', metValues)
    np.savez_compressed('./data/metab/metabData_Fernie/pred/ecotypeNames_known.npz', genotypes)


#saveKnownMetab()
#quit()




def saveRelatednessMatrix():

    #awk '{$6=1}1' OFS="\t" ./data/metab/SNP/1001genomes_snp-short-indel_only_ACGTN.fam > ./data/metab/SNP/1001genomes_snp-short-indel_only_ACGTN_fixed.fam
    
    SNP_file = './data/metab/SNP/1001genomes_snp-short-indel_only_ACGTN'

    command1 = './data/software/gemma.macosx -bfile ' + SNP_file + ' -gk 1 -o ./relatedness_matrix'

    os.system(command1)

    command2 = 'mv ./output/relatedness_matrix.cXX.txt ./data/metab/SNP/relatedness_matrix.cXX.txt'
    os.system(command2)


#saveSorRelatednessMatrix()
#quit()




def topSingleTrait():



    metabolites = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab.npz')[:, :, :1500, :40]
    genotype = loadnpz('./data/metab/metabData_Fernie/processed/names.npz')
    envirement = loadnpz('./data/metab/metabData_Fernie/processed/env.npz')
    yearList = loadnpz('./data/metab/metabData_Fernie/processed/year.npz')
    envirement = np.array([envirement, yearList]).T

    metabolites = metabolites / np.max(metabolites)
    metabolites = torch.tensor(metabolites).float()


    

    metabolites = metabolites.reshape((metabolites.shape[0], metabolites.shape[1]*metabolites.shape[2]*metabolites.shape[3]))

    envChoice = '6d'
    #envChoice = '0d'
    metabolites = metabolites[envirement[:, 0] == envChoice]
    genotype = genotype[envirement[:, 0] == envChoice]
    envirement = envirement[envirement[:, 0] == envChoice]
    
    genotype_unique, genotype_inverse = np.unique(genotype, return_inverse=True)


    metabolites_use = metabolites.data.numpy()
    metabolites_use = metabolites_use - np.mean(metabolites_use, axis=0).reshape((1, -1))
    metabolites_use = metabolites_use[:, np.mean(np.abs(metabolites_use), axis=0) > 1e-10]

    

    #synthUsed = 30
    synthUsed = 20

    heritArray_trait = np.zeros((2, 5, synthUsed))
    heritArray_PCA = np.zeros((2, 5, synthUsed))
    heritArray = np.zeros((2, 5, synthUsed))


    Nsplit = 5
    np.random.seed(2)
    trainTest = np.random.randint(Nsplit, size=genotype_unique.shape[0])
    trainTest = trainTest[genotype_inverse]

    runMethods = ['H2Opt', 'PCA', 'trait']
            
            
    for split_index in range(5):

        print ('split_index', split_index)

        trainTest2 = np.zeros(trainTest.shape[0], dtype=int)
        trainTest2[trainTest == split_index] = 1
        

        #imageData_C = imageAll_C[:, predNum, :, :, :]
        #imageData_S = imageAll_S[:, predNum, :, :, :]

        #means1 = np.mean(imageData_C, axis=(0, 2, 3))
                
        #means1 = means1.reshape((1, -1, 1, 1)) * 3

        
        

        if 'H2Opt'in runMethods:
            if envChoice == '0d':
                modelName = './data/metab/metabData_Fernie/models/linear_primary_many_split' + str(split_index) + '.pt'
            else:
                modelName = './data/metab/metabData_Fernie/models/linear_secondary_many_split' + str(split_index) + '.pt'
            model = torch.load(modelName)
            Y = model(metabolites, np.arange(synthUsed))
            Y = normalizeIndependent(Y) 
            #Y = Y.detach().numpy()

            heritList_train = cheapHeritability(Y[trainTest2 == 0], genotype[trainTest2 == 0], envirement[trainTest2 == 0])
            heritList_train = heritList_train.data.numpy()
            #quit()
            
            #heritList_test = rowHerit(plotRow[trainTest_C == 1], Y[trainTest_C == 1], names_C[trainTest_C == 1], envirement[trainTest_C == 1])
            heritList_test = cheapHeritability(Y[trainTest2 == 1], genotype[trainTest2 == 1], envirement[trainTest2 == 1])
            heritList_test = heritList_test.data.numpy()

            heritArray[0, split_index] = heritList_train
            heritArray[1, split_index] = heritList_test

            
        

        if 'PCA' in runMethods:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=synthUsed)
            pca.fit(metabolites[trainTest2 == 0].data.numpy())
            Y_PCA = pca.transform(metabolites.data.numpy())

            np.savez_compressed('./data/metab/metabData_Fernie/baselines/' + 'PCA_' + envChoice + '_' + str(split_index) + '.npz', Y_PCA)

            Y = torch.tensor(Y_PCA).float()
            
            #heritList_train = rowHerit(plotRow[trainTest_C == 0], Y[trainTest_C == 0], names_C[trainTest_C == 0], envirement[trainTest_C == 0])
            heritList_train = cheapHeritability(Y[trainTest2 == 0], genotype[trainTest2 == 0], envirement[trainTest2 == 0])
            heritList_train = heritList_train.data.numpy()
            #quit()
            
            #heritList_test = rowHerit(plotRow[trainTest_C == 1], Y[trainTest_C == 1], names_C[trainTest_C == 1], envirement[trainTest_C == 1])
            heritList_test = cheapHeritability(Y[trainTest2 == 1], genotype[trainTest2 == 1], envirement[trainTest2 == 1])
            heritList_test = heritList_test.data.numpy()
            #print (heritList_train[0], heritList_test[0])
            heritArray_PCA[0, split_index] = heritList_train
            heritArray_PCA[1, split_index] = heritList_test


        #print (envirement.shape)

        if 'trait' in runMethods:
            #print (metabolites.shape)

            
            

            #plt.plot(vary.data.numpy())
            #plt.show()


            #print ("XYZ")
            Y = maxWaveMethod(metabolites_use, synthUsed, genotype, envirement, trainTest2)

            np.savez_compressed('./data/metab/metabData_Fernie/baselines/' + 'maxTrait_' + envChoice + '_' + str(split_index) + '.npz', Y)

            Y = torch.tensor(Y).float()
            #Y = normalizeIndependent(Y)

            
            #heritList_train = rowHerit(plotRow[trainTest_C == 0], Y[trainTest_C == 0], names_C[trainTest_C == 0], envirement[trainTest_C == 0])
            heritList_train = cheapHeritability(Y[trainTest2 == 0], genotype[trainTest2 == 0], envirement[trainTest2 == 0])
            heritList_train = heritList_train.data.numpy()

            #plt.plot(heritList_train)
            #plt.show()
            #heritList_test = rowHerit(plotRow[trainTest_C == 1], Y[trainTest_C == 1], names_C[trainTest_C == 1], envirement[trainTest_C == 1])
            heritList_test = cheapHeritability(Y[trainTest2 == 1], genotype[trainTest2 == 1], envirement[trainTest2 == 1])
            heritList_test = heritList_test.data.numpy()
            print (heritList_train[0], heritList_test[0])
            heritArray_trait[0, split_index] = heritList_train
            heritArray_trait[1, split_index] = heritList_test



    
    
    quit()
    heritArray[heritArray > 1] = 1
    heritArray_PCA[heritArray_PCA > 1] = 1
    heritArray_trait[heritArray_trait > 1] = 1

    heritArray[heritArray < 0] = 0
    heritArray_PCA[heritArray_PCA < 0] = 0
    heritArray_trait[heritArray_trait < 0] = 0

    print (np.mean(heritArray[1, :, :10]  ))
    print (np.mean(heritArray_PCA[1, :, :10]  ))
    print (np.mean(heritArray_trait[1, :, :10]  ))

    if 'trait'in runMethods:
        np.savez_compressed('./data/metab/metabData_Fernie/eval/herits_maxTrait_' + envChoice + '.npz', heritArray_trait)
    if 'H2Opt'in runMethods:
        np.savez_compressed('./data/metab/metabData_Fernie/eval/herits_H2Opt_' + envChoice + '.npz', heritArray)
    if 'PCA'in runMethods:
        np.savez_compressed('./data/metab/metabData_Fernie/eval/herits_PCA_' + envChoice + '.npz', heritArray_PCA)
    quit()

    traitCount = np.arange(synthUsed) + 1
    plt.plot(traitCount, np.mean(heritArray[1, :, :], axis=0), color='tab:blue')
    plt.plot(traitCount, np.mean(heritArray[0, :, :], axis=0) , color='tab:blue' , linestyle='dashed')
    plt.plot(traitCount, np.mean(heritArray_PCA[1, :, :], axis=0), color='tab:orange')
    plt.plot(traitCount, np.mean(heritArray_PCA[0, :, :], axis=0) , color='tab:orange', linestyle='dashed')
    plt.plot(traitCount, np.mean(heritArray_trait[1, :, :], axis=0), color='tab:green')
    plt.plot(traitCount, np.mean(heritArray_trait[0, :, :], axis=0) , color='tab:green', linestyle='dashed')
    plt.xlabel('trait number')
    plt.ylabel('heritability')
    plt.gcf().set_size_inches(3.5, 4)
    plt.tight_layout()
    plt.savefig('./images/metab/herit_' + envChoice + '.pdf')
    #plt.legend(['H2Opt: test', 'H2Opt: train', 'PCA: test', 'PCA: train',  'single trait: test', 'single trait: train'  ])

    plt.show()

    

    

    print ("Save")

    #np.savez_compressed('./data/miscPlant/eval/herits_vegIndex.npz', heritArray_vegIndices)


#topSingleTrait()
#quit()



def saveForMetabEnrich():


    #Idea: for original model, find the maximum peak within the time mass window 



    if False:
        spectrumData = loadnpz('./data/metab/metabData_Fernie/processed/massPrecise.npz')

        sum1 = np.sum(spectrumData, axis=(0, 1))

        

        top10K = np.argsort(sum1)[-10000:]
        print (top10K.shape)
        print (np.max(top10K))
        spectrumData = spectrumData[:, :, top10K]
        top10K_masses = top10K.astype(float) / 10000.0
        np.savez_compressed('./data/metab/metabData_Fernie/processed/massPrecise_top10k.npz', spectrumData)
        np.savez_compressed('./data/metab/metabData_Fernie/processed/massPrecise_top10k_masses.npz', top10K)
        print ("Done")
        quit()




    for phen_index in range(0, 10):


        splitIndex = 0

        #envChoice = '0d'
        envChoice = '6d'

        metabolites = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab.npz')[:, :, :1500, :40]

        #sns.heatmap(  np.mean(metabolites[:, 0], axis=0) )# , norm=LogNorm() )
        #plt.show()
        #quit()

        #massValues = loadnpz('./data/metab/metabData_Fernie/processed/massIntensitiesPrecise.npz')
        massValues = loadnpz('./data/metab/metabData_Fernie/processed/massIntensitiesPrecise2_' + envChoice + '.npz')

        
        

        #print (np.mean(massValues[2] ))

        #print (np.mean(variance ))
        #print (np.mean(massValues[0] ))

        #print (massValues.shape)
        #quit()
        #print (massValues.shape)
        #print (metabolites.shape)
        #quit()
        #massTimeLabels = loadnpz('./data/metab/metabData_Fernie/processed/massIntensitiesPreciseFinal.npz')

        #model = torch.load( './data/metab/metabData_Fernie/models/linear_primary_preciseMass_split' + str(splitIndex) + '_copy.pt')
        if envChoice == '6d':
            model = torch.load( './data/metab/metabData_Fernie/models/linear_secondary_many_split' + str(splitIndex) + '.pt')
        else:
            model = torch.load( './data/metab/metabData_Fernie/models/linear_primary_many_split' + str(splitIndex) + '.pt')
        

        coef = getMultiModelCoef(model, multi=True)



        
        Y = model(torch.tensor(metabolites.reshape((metabolites.shape[0], metabolites.shape[1]*metabolites.shape[2]* metabolites.shape[3] ))).float(), np.arange(20))
        coef = coef[:20]
        Y, trackComputation = normalizeIndependent(Y, trackCompute=True)
        print (coef.shape, trackComputation.shape)
        coef = np.matmul( coef.T, trackComputation.data.numpy() ).T




        #phen_index = 5
        #phen_index = -1
        #print (coef.shape)
        #print (metabolites.shape)
        #quit()
        #coef_now = coef[phen_index]

        #print (coef.shape)
        #quit()
        coef_now = coef[phen_index]


        #print (massValues.shape)
        #print (coef_now.shape)
        #quit()
        #coef_now = np.mean(np.abs(coef[:10]), axis=0)

        coef_now = coef_now.reshape(( metabolites.shape[1], metabolites.shape[2], metabolites.shape[3] ))
        coef_now = coef_now[0]

        massValues = massValues[:, :coef_now.shape[0], :coef_now.shape[1]]

        RT_vals = massValues[1] / massValues[0]

        #print (np.median(RT_vals[np.isnan(RT_vals) == False]  ))


        #print (RT_vals.shape)

        #print (RT_vals[:10, :10])
        #quit()

        massValues[0] = massValues[0] / metabolites.shape[0]
        massValues[3] = massValues[3] / metabolites.shape[0]

        variance = (massValues[3] - (massValues[0] ** 2))
        #print (np.min(variance))
        variance = variance ** 0.5


        
        
        
        massValues = massValues[0]
        massValues = massValues / np.max(massValues)

        #massValues = massValues * coef_now.reshape((coef_now.shape[0], coef_now.shape[1], 1))
        massValues = variance * coef_now.reshape((coef_now.shape[0], coef_now.shape[1], 1))
        massValues = massValues / np.max(massValues)

        valuesInclude = 10000
        cutoff = np.sort(np.abs(massValues.reshape((-1,))))[-valuesInclude]
        print (cutoff)
        argRelevant = np.argwhere( np.abs(massValues) >= cutoff )
        argRelevant = argRelevant[np.argsort(  -1 * np.abs(massValues[argRelevant[:, 0], argRelevant[:, 1], argRelevant[:, 2]]   ) )]
        print (argRelevant.shape)

        precise = 10000

        print (massValues.shape)

        RT_now = RT_vals[argRelevant[:, 0], argRelevant[:, 1], argRelevant[:, 2] ]
        mass_now = argRelevant[:, 0] - 0.5 + (argRelevant[:, 2].astype(float) / precise)
        massTimeLabels_sort = np.array([mass_now, RT_now]).T

        dataheader = np.array([['"m.z"', '"rt"',  '"p.value"' ]])[:, :2]
        print (dataheader.shape)
        data = np.concatenate(( dataheader, massTimeLabels_sort[:, :2]) , axis=0)
        for a in range(data.shape[0]):
            data[a, 1] = data[a, 1][:12]
        #np.savetxt('./data/metab/metabEnrich/resave_massOnly_split' + str(splitIndex) + '_phen' + str(phen_index) + '_' + envChoice + '.txt', data[:, :2], delimiter=' ', fmt='%s')
        np.savetxt('./data/metab/metabEnrich/seperated_split' + str(splitIndex) + '_phen' + str(phen_index) + '_' + envChoice + '.txt', data[:, :2], delimiter=' ', fmt='%s')



    quit()





    #sort1 = np.sort(np.abs(massValues.reshape((-1,))))

    

    print (sort1.shape)
    print (np.argwhere(sort1 > 1e-3).shape)

    plt.plot(sort1[0::1000])
    plt.show()



    print (coef_now.shape)
    print (massValues.shape)
    quit()

    massTimeLabels = massTimeLabels[:, :coef_now.shape[0], :coef_now.shape[1] ]


    coef_flat = coef_now.reshape((-1,))
    massTimeLabels_sort = massTimeLabels.reshape((2, -1,))

    coef_argsort = np.argsort(  np.abs(coef_flat) * -1 )
    massTimeLabels_sort = massTimeLabels_sort[:, coef_argsort].T 
    coef_sort = coef_flat[coef_argsort]

    #plt.plot(np.abs(coef_sort))
    #plt.show()

    #cutOff = 1e-3
    cutOff = 5e-4
    pValList = np.ones(coef_sort.shape[0])
    argStrongCoef = np.argwhere(np.abs(coef_sort) >  cutOff)[:, 0]

    print (argStrongCoef.shape)

    massTimeLabels_sort = massTimeLabels_sort[ : argStrongCoef.shape[0] * 10 ]

    #pValList[argStrongCoef] = 1e-5 * (cutOff / np.abs(coef_sort[argStrongCoef]))
    #pValList = pValList.astype(str)
    #massTimeLabels_sort = np.concatenate(( massTimeLabels_sort, pValList.reshape((-1, 1)) ), axis=1)



    

    #print (massTimeLabels_sort.shape)
    #quit()

    #massTimeLabels_sort = massTimeLabels_sort[:1000]
    #coef_sort = coef_sort[:1000]

    
    data = np.loadtxt('./data/metab/metabEnrich/mummichog_ibd.txt', delimiter=' ', dtype=str)
    #print (data[:1, :1])
    dataheader = np.array([['"m.z"', '"rt"',  '"p.value"' ]])[:, :2]
    print (dataheader.shape)
    data = np.concatenate(( dataheader, massTimeLabels_sort[:, :2]) , axis=0)
    np.savetxt('./data/metab/metabEnrich/resave_massOnly_phen' + str(phen_index) + '.txt', data[:, :2], delimiter=' ', fmt='%s')


    

    quit()



    #WIth mass based model. 
    splitIndex = 0

    #spectrumData = loadnpz('./data/metab/metabData_Fernie/processed/massPrecise_top10k.npz')
    top10K = loadnpz('./data/metab/metabData_Fernie/processed/massPrecise_top10k_masses.npz')

    model = torch.load( './data/metab/metabData_Fernie/models/linear_primary_preciseMass_split' + str(splitIndex) + '_copy.pt')

    top10K_masses = top10K.astype(float) / 10000.0



    coef = getMultiModelCoef(model, multi=True)
    #print (coef.shape)
    #quit()
    coef = coef.reshape(( coef.shape[0], 2, coef.shape[1] // 2 ))

    phen_index = 4
    coef_now = coef[phen_index]
    

    top10K_masses_sorted = top10K_masses[np.argsort( np.abs(coef_now[0])  * -1 )]


    data = np.loadtxt('./data/metab/metabEnrich/mummichog_ibd.txt', delimiter=' ', dtype=str)
    data = np.concatenate(( data[:1, :1], top10K_masses_sorted.reshape((-1, 1)) ), axis=0)
    np.savetxt('./data/metab/metabEnrich/resave_massOnly_phen' + str(phen_index) + '.txt', data[:, :1], delimiter=' ', fmt='%s')






    quit()

    #phenotypes = loadnpz('./data/metab/metabData_Fernie/pred/linear_primary_many_split' + str(0) + '.npz')
    phenotypes = loadnpz('./data/metab/metabData_Fernie/pred/linear_primary_preciseMass_split' + str(0) + '.npz')

    genotype = loadnpz('./data/metab/metabData_Fernie/processed/names.npz')
    envirement = loadnpz('./data/metab/metabData_Fernie/processed/env.npz')
    yearList = loadnpz('./data/metab/metabData_Fernie/processed/year.npz')
    envirement = np.array([envirement, yearList]).T
    envChoice = '0d'
    spectrumData = spectrumData[envirement[:, 0] == envChoice]
    genotype = genotype[envirement[:, 0] == envChoice]
    envirement = envirement[envirement[:, 0] == envChoice]
    
    genotype_unique, genotype_inverse = np.unique(genotype, return_inverse=True)

    #print (genotype_unique)
    #quit()

    print (spectrumData.shape)

    spectrumData_means = np.zeros(( genotype_unique.shape[0], 2, spectrumData.shape[2]  ))
    for a in range(genotype_unique.shape[0]):
        arg1 = np.argwhere(genotype == genotype_unique[a])[:, 0]
        spectrumData_means[a] = np.mean(spectrumData[arg1], axis=0)

    spectrumData_means = spectrumData_means[:, 0]

    synthTraitNum = 0

    argsort_phen = np.argsort(phenotypes[:, synthTraitNum])
    #argsort_phen = argsort_phen[np.random.permutation(argsort_phen.shape[0])] #TODO REMOVE!! TESTING RANDOM!!
    low_arg = argsort_phen[:argsort_phen.shape[0]//2]
    high_arg = argsort_phen[argsort_phen.shape[0]//2:]

    tList = np.zeros(top10K.shape[0])
    pList = np.zeros(top10K.shape[0])
    for a in range(top10K.shape[0]):
        t_result = scipy.stats.ttest_ind(spectrumData_means[low_arg, a], spectrumData_means[high_arg, a])
        t_now, p_now = t_result[0], t_result[1]
        tList[a], pList[a] = t_now, p_now

    top10K_masses = top10K.astype(float) / 10000.0


    data_values = np.array([top10K_masses, pList, tList]).T.astype(str)

    data_values = data_values[np.argsort(pList * -1)]


    data_values = data_values[np.isnan(data_values[:, 1].astype(float)) == False]
    data_values = data_values[np.isnan(data_values[:, 2].astype(float)) == False]


    #data_values[:, 2] = data_values[:, 2].astype(float) * -1


    data = np.loadtxt('./data/metab/metabEnrich/mummichog_ibd.txt', delimiter=' ', dtype=str)

    data = np.concatenate(( data[:1], data_values ), axis=0)

    

    

    np.savetxt('./data/metab/metabEnrich/resave_mummichog_ibd.txt', data, delimiter=' ', fmt='%s')

    np.savetxt('./data/metab/metabEnrich/resave_massOnly.txt', data[:, :1], delimiter=' ', fmt='%s')

    quit()

    spectrumData = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab.npz')

    print (spectrumData.shape)

    



#saveForMetabEnrich()
#quit()





def saveForLMER():


    metabolites = loadnpz('./data/metab/metabData_Fernie/processed/imageMetab.npz')[:, :, :1500, :40]
    #metabolites = loadnpz('./data/metab/metabData_Fernie/processed/massPrecise_top10k.npz')
    genotype = loadnpz('./data/metab/metabData_Fernie/processed/names.npz')
    envirement = loadnpz('./data/metab/metabData_Fernie/processed/env.npz')
    #print (envirement[:10])
    #quit()

    metabolites_sum = np.sum(metabolites, axis=3)
    metabolites_sum = metabolites_sum.reshape((metabolites_sum.shape[0], metabolites_sum.shape[1]*metabolites_sum.shape[2]))
    
    genotype = genotype.reshape((-1, 1))
   

    values = np.concatenate(( genotype,  metabolites_sum.astype(str)  ), axis=1)
    namePart = ['name2']
    for a in range(metabolites_sum.shape[1]):
        namePart.append('mass_' + str(a))
    namePart = np.array(namePart)
    namePart = namePart.reshape((1, -1))
    print (namePart.shape, values.shape)

    values_0d = values[envirement == '0d']
    values_6d = values[envirement == '6d']

    values_0d = np.concatenate(( namePart, values_0d ), axis=0)
    values_6d = np.concatenate(( namePart, values_6d ), axis=0)

    
    

    

    np.savetxt("./data/software/lmeHerit/input_metab_0d.csv", values_0d, delimiter=',', fmt='%s')
    np.savetxt("./data/software/lmeHerit/input_metab_6d.csv", values_6d, delimiter=',', fmt='%s')

    


    print (metabolites.shape)
    quit()

    



saveForLMER()
quit()