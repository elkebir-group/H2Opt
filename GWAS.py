import numpy as np
import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import time
import os

import torch

from sklearn.decomposition import PCA

from shared import *


import seaborn as sns

from Bio import Entrez
Entrez.email = "stefan4@illinois.edu"



def loadnpz(name, allow_pickle=False):

    #This simple function more easily loads in compressed numpy files.

    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data





def saveSNPs():


    import pysam

    #file1 = './data/SNP/GBS836_2mi_MAC20.vcf'
    #file1 = './data/SNP/WEST_mac20.vcf'
    #file1 = '../software/data/WEST_original.vcf'
    #file1 = '../software/modifiedData/pruned_dataset.vcf'
    #file1 = '../software/modifiedData/pruned_dataset2.vcf'

    #MsiLine = 'MsiC'
    #MsiLine = 'MsiS'

    #file1 = './data/miscPlant/SNP/vcf/' + MsiLine + '_annotated.vcf'
    file1 = './data/metab/metabData_Alex/call_method_75/call_method_75_TAIR8.vcf'


    data = []


    with open(file1) as f:
        lines = [line for line in f]

    data_info = []

    #headLine = []


    #seperator = '|'
    seperator = '/'

    a = -1
    #with open(file1, 'r') as file:
    # Read each line in the file
    for line in lines:
        #print (line)
        #quit()
        a += 1
        
        line1 = line.strip()

        print ('')
        print (line1)
        if a == 4:
            quit()
        
        if ( seperator in line1) and not ('##INFO' in line1):
            #print (line1)

            line1 = line1.split('\t')
            

            data_info.append(line1[:9])

            line2 = line1[9:]
            line2 = seperator.join(line2)
            line2 = line2.split(seperator)
            line2 = np.array(line2).reshape(( len(line2)  // 2, 2 ))
            line2 = line2.astype(int)

            data.append(line2)


            #quit()

        elif '#CHROM' in line1:

            headLine = line1.split('\t')
            
            #print (line1)

        #if a == 50:
        #    quit()
    

    headLine = np.array(headLine)
    #file_save = './data/miscPlant/SNP/npz/' + MsiLine + '_annotated_head.npz'
    file_save = './data/metab/metabData_Alex/call_method_75/npz/call_method_75_TAIR8_head.npz'
    np.savez_compressed(file_save, headLine)


    print ("Z")

    data_info = np.array(data_info)

    print ("Z2")
    
    data = np.array(data)

    
    #file_save = './data/miscPlant/SNP/npz/' + MsiLine + '_annotated_info.npz'
    file_save = './data/metab/metabData_Alex/call_method_75/npz/call_method_75_TAIR8_info.npz'
    np.savez_compressed(file_save, data_info)


    #file_save = './data/plant/SNP/allChr.npz'
    #file_save = './data/plant/SNP/pruned2.npz'
    #file_save = './data/miscPlant/SNP/npz/' + MsiLine + '_annotated.npz'
    file_save = './data/metab/metabData_Alex/call_method_75/npz/call_method_75_TAIR8.npz'

    print (data.shape)

    np.savez_compressed(file_save, data)

    

    quit()





#saveSNPs()
#quit()



def saveBedSNPs():


    #from pyplink import PyPlink
    from plinkio import plinkfile

    # Load the .bed file (it automatically looks for .bim and .fam files in the same directory)
    #bed_file = "path/to/your/file.bed"
    bed_file = './data/miscPlant/SNP/bed/MsiC_annotated'
   
    # Open the .bed file
    plink_file = plinkfile.open(bed_file)

    # Get SNP and sample information
    samples = plink_file.get_samples()
    loci = plink_file.get_loci()

    print("Samples:", len(samples))
    print("SNPs:", len(loci))

    # Read genotype data
    genotypes = []
    for row in plink_file:
        genotypes.append(row)

    # Convert to a DataFrame
    import pandas as pd
    genotype_df = pd.DataFrame(genotypes, columns=[l.name for l in loci])
    print(genotype_df)


#saveBedSNPs()
#quit()


def saveSNPsubsetSimulation(subset1, outputFile):

    #          snp allele chr     pos cm 4226 4722 33-16 38-11 A188
    #1 ss196422159    A/G   1  379844 NA   -1   -1    -1    -1   -1
    #2 ss196422171    G/A   1  613257 NA   -1   -1    -1    -1   -1

    head1 = np.loadtxt('./data/plant/SNP/WEST_original.txt', delimiter='\t', dtype=str)
    head_names = head1[9:]
    for a in range(head_names.shape[0]):
        head_names[a] = head_names[a].split('_')[0]


    file_SNP = './data/plant/SNP/allChr.npz'

    data = loadnpz(file_SNP)
    data = np.sum(data, axis=2) - 1# -1 for simulation version since it uses -1, 0, 1
    data = data.astype(int)


    data = data[subset1]

    

    #data = data.T

    data = data.astype(str)
    data = np.concatenate(( head_names.reshape((1, -1)),  data ), axis=0)

    #print (head_names.shape)
    #print (data.shape)
    #quit()


    data_info = loadnpz('./data/plant/SNP/allChr_info.npz')
    data_info = data_info[subset1]

    #print (data_info.shape)
    #quit()

    chr1 = data_info[:, 0]
    pos1 = data_info[:, 1]

    SNPletters = data_info[:, 3:5]
    SNPletters_both = []
    cmArray = []
    snpNameArray = []
    for a in range(SNPletters.shape[0]):
        letter_both = SNPletters[a, 0] + '/' + SNPletters[a, 1]
        SNPletters_both.append(letter_both)
        cmArray.append("NA")
        snpNameArray.append('SNP' + str(a+1))
    SNPletters_both = np.array(SNPletters_both)
    cmArray = np.array(cmArray)
    snpNameArray = np.array(snpNameArray)

    header1 = ['snp', 'allele', 'chr', 'pos', 'cm']
    header1 = np.array(header1)

    data_info2 = np.concatenate(( snpNameArray.reshape((-1, 1)) , SNPletters_both.reshape((-1, 1)),  chr1.reshape((-1, 1)), pos1.reshape((-1, 1)), cmArray.reshape((-1, 1))  ), axis=1)

    data_info2 = np.concatenate((header1.reshape((1, -1)), data_info2), axis=0)


    data = np.concatenate((data_info2, data), axis=1)

    #print (data_info2.shape)
    print (data.shape)

    print (data[:10, :10])

    #np.savetxt('./data/plant/SNP/pruned_forSimulation2.csv', data, fmt='%s', delimiter=',')
    np.savetxt(outputFile,  data, fmt='%s', delimiter=',')



#saveSNPforSimulation()
#quit()






def findUncorrelatedSNPs():


    #data_info = loadnpz('./data/plant/SNP/allChr_info.npz')
    #IDlist = data_info[:, 2]

    #for a in range(20):
    #    print (data_info[a])
    #quit()
    
    #NumSNPS = 3
    NumSNPS = 300
    #corUse = 0.5
    corUse = 0.0
    #corError = 

    SNPs = loadnpz('./data/plant/SNP/allChr.npz')
    SNPs = np.sum(SNPs, axis=2)
    
    instancesNeeded = 10

    count1 = 0
    tripletList = []
    while len(tripletList) < instancesNeeded:
        count1 += 1
        if count1 % 100000 == 0:
            print (count1 // 100000)
            print (tripletList)
        rand1 = np.random.randint(SNPs.shape[0], size= NumSNPS )
        if np.unique(rand1).shape[0] == rand1.shape[0]:
            cor1 = scipy.stats.pearsonr(SNPs[rand1[0]], SNPs[rand1[1]])[0]
            if True:# abs(cor1) < 0.02:
                cor2 = scipy.stats.pearsonr(SNPs[rand1[0]], SNPs[rand1[2]])[0]
                if True:# abs(cor2) < 0.02:
                    cor3 = scipy.stats.pearsonr(SNPs[rand1[1]], SNPs[rand1[2]])[0]
                    if True:# abs(cor3) < 0.02:
                        tripletList.append(np.copy(rand1))


    for a in range(instancesNeeded):
    
        SNPnums = tripletList[a]
        #outputFile = './data/plant/SNP/simulation/specialSNP/uncorrelatedSNP_' + str(a) + '.csv'
        #outputFile = './data/plant/SNP/simulation/specialSNP/randomSNP_' + str(a) + '.csv'
        #outputFile = './data/plant/SNP/simulation/specialSNP/random100SNP_' + str(a) + '.csv'
        outputFile = './data/plant/SNP/simulation/specialSNP/random300SNP_' + str(a) + '.csv'

        #outputFile = './data/plant/SNP/simulation/specialSNP/correlatedSNP.csv'

        saveSNPsubsetSimulation(SNPnums, outputFile)
    quit()
    
    quit()


    print (SNPs.shape)

    corList = []
    randList = []
    for a in range(1000):
        rand1 = np.random.randint(SNPs.shape[0], size=2)
        randList.append(np.copy(rand1))
        vec1, vec2 = SNPs[rand1[0]], SNPs[rand1[1]]
        cor1 = scipy.stats.pearsonr(vec1, vec2)[0]
        corList.append(cor1)

    corList = np.array(corList)
    corList = np.abs(corList)
    #arg1 = np.argmin(corList)
    arg1 = np.argwhere( np.abs(corList - corUse) < 0.1 )[0, 0]

    SNPnums = list(randList[arg1])
    print (SNPnums)
    
    for a in range(NumSNPS - 2):

        continue1 = True 
        while continue1: 
            rand1 = np.random.randint(SNPs.shape[0])
            if not rand1 in SNPnums:
                vec1 = SNPs[rand1]
                biggestCor = 0.0
                smallestCor = 1.0
                for b in range(len(SNPnums)):
                    vec2 = SNPs[SNPnums[b]]
                    cor1 = scipy.stats.pearsonr(vec1, vec2)[0]
                    biggestCor = max(biggestCor, abs(cor1))
                    smallestCor = min(smallestCor, abs(cor1))

                print (biggestCor, smallestCor)

                #if biggestCor < 0.02:
                if smallestCor > 0.4:
                    if biggestCor < 0.6:
                        #if biggestCor < 0.01:
                        SNPnums.append(rand1)
                        continue1 = False
                        print (SNPnums)

            

    #np.savetxt('')

    corMatrix = np.zeros((len(SNPnums), len(SNPnums)))
    for a in range(len(SNPnums)):
        for b in range(len(SNPnums)):
            vec1, vec2 = SNPs[SNPnums[a]], SNPs[SNPnums[b]]
            cor1 = scipy.stats.pearsonr(vec1, vec2)[0]
            corMatrix[a, b] = cor1


    print (corMatrix)
    
    #plt.hist(corList, bins=100)
    #plt.show()

    SNPnums = np.array(SNPnums).astype(int)
    #outputFile = './data/plant/SNP/simulation/specialSNP/uncorrelatedSNP.csv'
    outputFile = './data/plant/SNP/simulation/specialSNP/correlatedSNP.csv'

    saveSNPsubsetSimulation(SNPnums, outputFile)
    quit()





#findUncorrelatedSNPs()
#quit()





def predictSimPhenotypes(modelName, predName, predName2, dataFolder):


    #modelName = './data/plant/simulations/models/simA3_linear.pt'
    model = torch.load(modelName)

    #X = np.loadtxt('./data/plant/simulations/simPhen/sim16/Simulated_Data_10_Reps_Herit_0.1...0.1.txt', dtype=str)
    #dataFolder = './data/plant/simulations/simPhen/simA3'
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

    np.savez_compressed(predName, Y)
    np.savez_compressed(predName2, X)

    #np.savez_compressed('./data/plant/syntheticTraits/simA3_originalTraits.npz', X)


#predictPhenotypes()
#quit()




def savePCAforGemma():

    

    #file_fam = './data/software/data/WEST_original_copy.fam'
    #file_fam = './data/miscPlant/SNP/bed/MsiC_annotated_copy.fam'
    #file_fam = './data/miscPlant/SNP/bed/MsiS_annotated_copy.fam'
    file_fam = './data/metab/metabData_Alex/call_method_75/call_method_75_TAIR8_copy.fam'
    try:
        data = np.loadtxt(file_fam, delimiter='\t', dtype=str)
        name_fam = data[:, 0]
    except:
        data = np.loadtxt(file_fam, delimiter=' ', dtype=str)
        name_fam = data[:, 0]

    #print (data.shape)
    #print (data[:, 0])
    #print (data[:, 1])
    #quit()


    #file_head = './data/miscPlant/SNP/npz/MsiC_annotated_head.npz'
    #file_head = './data/miscPlant/SNP/npz/MsiS_annotated_head.npz'
    file_head = './data/metab/metabData_Alex/call_method_75/npz/call_method_75_TAIR8_head.npz'
    headLine = loadnpz(file_head)

    #print (headLine[:10])
    headLine = headLine[9:]
    

    #print (name_fam.shape)
    #print (headLine.shape)
    #quit() 
    for a in range(name_fam.shape[0]):
        #print (name_fam[a] , headLine[a])
        assert name_fam[a] == headLine[a]

    #print (np.intersect1d(name_fam, headLine).shape)
    #quit()

    



    #file_SNP = './data/plant/SNP/allChr.npz'
    #file_SNP = './data/miscPlant/SNP/npz/MsiC_annotated.npz'
    #file_SNP = './data/miscPlant/SNP/npz/MsiS_annotated.npz'
    file_SNP = './data/metab/metabData_Alex/call_method_75/npz/call_method_75_TAIR8.npz'

    data = loadnpz(file_SNP)

    data = np.sum(data, axis=2).T

    #print (data.shape)
    #quit()




    #names = loadnpz('./data/miscPlant/GWAS/names_MSI_C.npz')
    #names = loadnpz('./data/miscPlant/GWAS/names_MSI_S.npz')
    #names = loadnpz('./data/metab/metabData_Fernie/processed/names.npz')

    #envChoice = '0d'
    envChoice = '6d'


    genotype = loadnpz('./data/metab/metabData_Fernie/processed/names.npz')
    envirement = loadnpz('./data/metab/metabData_Fernie/processed/env.npz')
    genotype_6d = genotype[envirement == envChoice]
    names = np.unique(genotype_6d)
    
    #print (names)
    #print (np.unique(names).shape)
   

    for a in range(name_fam.shape[0]):
        name_fam1 = name_fam[a] #TODO Edit
        if 'Msi' in file_fam:
            name_fam1 = name_fam1.split('_')[2:]
            name_fam1 = '_'.join(name_fam1)
            name_fam[a] = name_fam1

    #print (name_fam[:10])
    #quit()
    #Correct name subset issue

    data = data[np.isin(name_fam, names)]

    print (np.intersect1d(name_fam, names).shape)
    

    

    n_components = 5
    pca = PCA(n_components=n_components)  # Choose n based on how many components you want
    pca_components = pca.fit_transform(data)

    

    pca_components = pca_components / (np.mean(pca_components ** 2, axis=0) ** 0.5)


    #np.savez_compressed('../software/data/WEST_original_PCA5.npz', pca_components)
    #np.savez_compressed('./data/software/data/MsiC_annotated_PCA5.npz', pca_components)
    #np.savez_compressed('./data/software/data/MsiS_annotated_PCA5.npz', pca_components)
    np.savez_compressed('./data/software/data/call_method_75_TAIR8_' + envChoice + '_PCA5.npz', pca_components)

    
    #quit()

    #pca_components = np.random.random(pca_components.shape[0]) - 0.5

    # Convert to DataFrame for easier manipulation
    pca_df = pd.DataFrame(data=pca_components)#, index=name_fam)

    # Save PCA components to a file
    #pca_df.to_csv('../software/data/WEST_original_PCA5.txt', sep='\t', header=False, index=True)
    #print ('a')
    #pca_df.to_csv('./data/software/data/MsiC_annotated_PCA5.txt', sep='\t', header=False, index=True)
    #pca_df.to_csv('./data/software/data/MsiS_annotated_PCA5.txt', sep='\t', header=False, index=True)
    pca_df.to_csv('./data/software/data/call_method_75_TAIR8_' + envChoice + '_PCA5.txt', sep='\t', header=False, index=True)


    #check1 = np.loadtxt('./data/software/data/MsiC_annotated_PCA5.txt')
    check1 = np.loadtxt('./data/software/data/call_method_75_TAIR8_' + envChoice + '_PCA5.txt')

    #279

    print (check1.shape)

    #print (check1)



#savePCAforGemma()
#quit()





def modifyFam():

    file_fam = '../software/data/WEST_original_copy.fam'

    #f = open(file_fam, "r")
    #Lines = f.readlines()
    data0 = np.loadtxt(file_fam, delimiter='\t', dtype=str)

    data0 = data0[:, :6]

    #print (data0[:10])
    #print (data0.shape)
    #quit()

    

    

    new_length = 20
    data = np.char.ljust(data0, new_length, ' ')
    data[:] = data0

    #data = np.concatenate((data, data, data), axis=1)
    #data = data[:, :data0.shape[1] + 9]

    

    Y = loadnpz('./data/plant/syntheticTraits/sor_H_3.npz')
    names = loadnpz('./data/plant/processed/sor/names.npz')


    for a in range(Y.shape[1]):
        Y[:, a] = Y[:, a] / (np.mean(Y[:, a]**2) ** 0.5)

    Y = Y[:, -1]

    
    name_fam = data[:, 0]
    for a in range(data.shape[0]):

        args1 = np.argwhere(names == name_fam[a])
        print (args1.shape)
        if args1.shape[0] > 0:
            args1 = args1[:, 0]
            #data[a, -10:] = np.mean(Y[args1], axis=0)
            data[a, -1] = np.mean(Y[args1])


    #print (name_fam.shape)
    #print (np.unique(name_fam).shape)
    #quit()

    _, counts1 = np.unique(name_fam, return_counts=True)

    #print (counts1)
    #quit()

    print (np.max(data[:, -1].astype(float)  ))

    #print (data[:, -1])
    #quit()

    #print (data[:5])

    #phenotype = np.random.random(size=data.shape[0])

    #data[:, -1] = phenotype.astype(str)


    #print (data[:10])
    #quit()

    file_fam_new = '../software/data/WEST_original.fam'

    np.savetxt(file_fam_new, data, delimiter=' ', fmt='%s')


    #print (phenotype.shape)

    #print (Lines[:10])


#modifyFam()
#quit()



def reformatGenes():

    #1   1682   2713   ID=gene-LOC8081570
    #1   11174  14887  ID=gene-LOC8059226
    #1   22450  24163  ID=gene-LOC110434520
    #1   24326  42458  ID=gene-LOC8059546

    #gff_file = './data/plant/GWAS/ncbi_dataset/ncbi_dataset/data/GCF_000003195.3/genomic.gff'
    gff_file = './data/plant/GWAS/ncbi_dataset/ncbi_dataset/data/GCF_000003195.3/genomic.bed'

    file1 = open(gff_file, 'r')
    Lines = file1.readlines()

    print (Lines[:10])




#reformatGenes()
#quit()




def findLDblocks():

    #./plink --bfile your_data_file --blocks --blocks-max-kb 200 --out output_filename

    #SNP_file = './data/software/data/WEST_original'
    #out_file = './data/software/data/WEST_LD_mod'

    #SNP_file = './data/miscPlant/SNP/bed/MsiC_annotated'
    #out_file = './data/software/data/MsiC_annotated_LD'


    SNP_file = './data/metab/SNP/1001genomes_snp-short-indel_only_ACGTN'
    out_file = './data/software/data/arabi_LD'


    plinkLoc = './data/software/plink'

    #command1 = plinkLoc + ' --bfile ' + SNP_file + ' --blocks no-pheno-req --blocks-max-kb 200 --out ' + out_file
    #command1 = plinkLoc + ' --bfile ' + SNP_file + ' --blocks no-pheno-req --allow-extra-chr --blocks-max-kb 200 --out ' + out_file
    
    #command1 = plinkLoc + ' --bfile ' + SNP_file + ' --blocks no-pheno-req --allow-extra-chr --blocks-min-maf 0.001 --blocks-max-kb 1000  --blocks-strong-lowci 0.51 --blocks-strong-highci 0.98  --out ' + out_file
    command1 = plinkLoc + ' --bfile ' + SNP_file + ' --blocks no-pheno-req --allow-extra-chr --blocks-max-kb 200 --blocks-strong-highci 0.999 --out ' + out_file #Standard  
    #command1 = plinkLoc + ' --bfile ' + SNP_file + ' --blocks no-pheno-req --allow-extra-chr --blocks-max-kb 100 --blocks-strong-lowci 0.60 --blocks-strong-highci 0.98 --out ' + out_file #New Arabi

    

    
    #    ./data/software/plink \
    #--bfile ./data/software/data/arabi_QC \
    #--allow-extra-chr \
    #--blocks \
    #--blocks-max-kb 100 \
    #--blocks-strong-highci 0.98 \
    #--blocks-strong-lowci 0.60 \
    #--out ./data/software/data/arabi_LD
    

    #Measure LD with your peak associated SNP. 


    print (command1)

    os.system(command1)


#findLDblocks()
#quit()




   # your email

#genes = ["AT3G57670", "AT1G01010"]


def getGeneDescriptions(gene_list):


    desc_list = []
    
    species_id = 3702  # Arabidopsis thaliana
    for g in gene_list:
        print (g)
        time1 = time.time()
        search = Entrez.esearch(db="gene", term=f"{g}[sym] AND txid{species_id}[Organism:exp]")
        record = Entrez.read(search)
        if not record["IdList"]:
            print(f"{g}: no NCBI Gene record found")
            continue

        gid = record["IdList"][0]
        rec = Entrez.read(Entrez.efetch(db="gene", id=gid, rettype="xml"))[0]

        gene_ref = rec["Entrezgene_gene"]["Gene-ref"]
        symbol   = gene_ref.get("Gene-ref_locus") or gene_ref.get("Gene-ref_locus-tag")

        # Prefer Gene-ref_desc if present; otherwise use the formal-name field
        desc = gene_ref.get("Gene-ref_desc")
        if desc is None:
            try:
                desc = gene_ref["Gene-ref_formal-name"]["Gene-nomenclature"]["Gene-nomenclature_name"]
            except KeyError:
                desc = "No description available"

        #print(f"{symbol} ({gid}): {desc}")
        desc_list.append(desc)

        time2 = time.time() - time1 
        #delay = 2.0 - time2 
        delay = 3.0 - time2 
        print (delay)
        if delay > 0:
            time.sleep(delay)

    return desc_list




def geneOntology():


    def convertSORBI(genes1):

        #nameMapping = np.loadtxt('./data/plant/GWAS/gene_result.txt', delimiter='\t', dtype=str)
        file1 = open('./data/plant/GWAS/gene_result.txt', 'r')
        nameMapping = file1.readlines()
        newMapping = []
        for a in range(len(nameMapping)):
            line1 = nameMapping[a]
            if 'LOC' in line1:
                if 'SORBI_' in line1:

                    locPart = line1.split('LOC')[1].split('\t')[0]
                    locPart = 'LOC' + locPart
                    #print (locPart)
                    #quit()

                    sorbPart = line1.split('SORBI_')[1].split(',')[0].split('\t')[0]
                    sorbPart = 'SORBI_' + sorbPart

                    newMapping.append([locPart, sorbPart])
                    
        newMapping = np.array(newMapping)

        for a in range(genes1.shape[0]):
            arg1 = np.argwhere(newMapping[:, 0] == genes1[a])
            
            if arg1.shape[0] > 0:
                genes1[a] = newMapping[arg1[0, 0], 1]
            else:
                genes1[a] = ''

        
        genes1 = np.unique(genes1)
        genes1 = genes1[genes1!='']

        return genes1


    

    dates = ['MSI_09042020_processed_ALLstack.tif', 'MSI_05062020_processed_ALLstack.tif', 'MSI_05222020_processed_ALLstack.tif', 'MSI_09192020_processed_ALLstack.tif', 'MSI_07022020_processed_ALLstack.tif', 'MSI_11282020_processed_ALLstack.tif', 'MSI_07242020_processed_ALLstack.tif', 'MSI_10052020_processed_ALLstack.tif', 'MSI_08182020_processed_ALLstack.tif', 'MSI_06182020_processed_ALLstack.tif', 'MSI_07102020_processed_ALLstack.tif', 'MSI_06082020_processed_ALLstack.tif', 'MSI_08082020_processed_ALLstack.tif', 'MSI_11062020_processed_ALLstack.tif']
    dates = np.array(dates)
    for a in range(dates.shape[0]):
        dates[a] = dates[a].split('_')[1]
        dates[a] = dates[a][:2] + ',' + dates[a][2:4]
    dates = np.sort(dates)

    
    #TODO: Calculate pairwise distances of consecutive genes in the genome 
    #Show gene lengths 


    plantName = 'MSI_C'
    #plantName = 'sor'
    #plantName = 'arabi'


    allPvals = []

    numSNPs = []
    numGenes = []

    dates = ['05-06', '05-22', '06-08', '06-18', '07-02']

    #starts at 0 

    #0d trait 3, 4, 5, 7, 8, 9
    #6d trait 0, 1, 2, 3, 4, 5, 6, 

    #AT1G14510


    #Secondary 0, and 2 interesting 'Citrate cycle (TCA cycle)', 'Phenylpropanoid biosynthesis'
    
    for traitIndex in [0]:##range(1, 2):

        for a in range(3):
            print ('')
        print ("trait: ", str(traitIndex))

        if plantName == 'MSI_C':
            #methodName = 'MSI_6'
            #methodName = 'MSI_singlepoint_skip2'
            #methodName = 'MSI_central_3_row13'
            #methodName = 'MSI_south_singlepoint_3_row13'
            #methodName = 'MSI_' + 'central' + '_singlepoint_' + str(3) + '_split_' + str(splitIndex)
            #methodName = 'MSI_' + 'south' + '_singlepoint_' + str(3) + '_noSplit'

            #print (dates[8])

            #namePart = 'central'
            namePart = 'south'
            namePart_copy = namePart
            timePoint = traitIndex
            methodName = 'MSI_' + namePart + '_singlepoint_' + str(timePoint) + '_split_' + str(0)
            traitIndex = 0


            #outputFile = './data/plant/GWAS/Gemma/' + methodName + '_permute:' + str(permute) + '_'


        elif plantName == 'sor':
            #methodName = 'sor_A_4'
            #methodName = 'sor_conv_1'
            #methodName = 'PCA0_sor_simple_1'
            #methodName = 'PCA0_sor_conv_1'
            #methodName = 'real_linearAug'
            #methodName = 'sor_simple_1'
            methodName = 'linear_crossVal_reg4_' + str(0) + '_mod10'
            True

        elif plantName == 'arabi':

            #methodName = '1R_T6_split0'
            #methodName = './data/metab/metabData_Alex2/GWAS/' + str(0) + '_cluster_' + str(26) + '_'
            #methodName = './data/metab/metabData_Fernie/GWAS/linear_mass579_'
            #methodName = './data/metab/metabData_Fernie/GWAS/linear_5_'
            #methodName = './data/metab/metabData_Fernie/GWAS/linear_mass739_'


            #methodName = './data/metab/metabData_Fernie/GWAS/linear_mass739_'
            methodName = './data/metab/metabData_Fernie/GWAS/linear_primary_many_split0_'
            #methodName = './data/metab/metabData_Fernie/GWAS/linear_secondary_many_split0_'


        

        if '/' in methodName:
            outputFile = methodName
        else:
            outputFile = './data/plant/GWAS/Gemma/' + methodName + '_'

        SNPvals = loadnpz(outputFile + str(traitIndex + 1) + '.npz')


        argGood = np.argwhere(np.isin( SNPvals[:, 0], np.arange(100).astype(str) ))[:, 0]
        argGood = np.concatenate((np.zeros(1, dtype=int), argGood), axis=0)
        SNPvals = SNPvals[argGood]


        cutOff = (1.0 / SNPvals.shape[0])  * 0.05
        cutOff_easy = (1.0 / SNPvals.shape[0])

        chr1 = SNPvals[1:, 0].astype(int)
        pos1 = SNPvals[1:, 2].astype(int)

        #print (SNPvals[0::1000][:10])
        #quit()

        SNP_name = SNPvals[1:, 1]

        #cutOffs = []
        if False:#plantName == 'MSI_C':
            print ("HI")

            #pCut = '0.05'
            pCut = '0.5'

            for permute1 in range(100):
                #methodName = 'MSI_' + namePart + '_singlepoint_' + str(2) + '_noSplit' + '_permute:' + str(permute1) + '_'
                methodName = 'MSI_' + namePart + '_singlepoint_' + str(timePoint) + '_split_' + str(0) + '_permute:' + str(permute1) + '_'
                outputFile = './data/plant/GWAS/Gemma/' + methodName 
                SNPvals_perm = loadnpz(outputFile + str(traitIndex + 1) + '.npz')
                SNPvals_perm = SNPvals_perm[1:, -1].astype(float)
                min1 = np.min(SNPvals_perm)
                #print (np.min(SNPvals_perm) * SNPvals_perm.shape[0])
                cutOffs.append(min1)

            cutOffs = np.array(cutOffs)
            #cutOff = np.sort(np.array(cutOffs))[(len(cutOffs) // 20)-1]
            if pCut == '0.05':
                cutOff = np.percentile(cutOff, 5)
            if pCut == '0.5':
                cutOff = np.median(cutOffs)
            pvals = SNPvals[1:, -1].astype(float)
            argSig = np.argwhere(pvals < cutOff)[:, 0]
            

        else:
            True
            #print ('hi')
            #quit()

        #print (argSig.shape)
        #quit()

        

        pvals = SNPvals[1:, -1].astype(float)

        if plantName == 'MSI_C':
            #argSig = np.argsort(pvals)[:pvals.shape[0] // 100] 
            argSig = np.argwhere( pvals < cutOff  )[:, 0]

        elif plantName == 'sor':
            #argSig = np.argsort(pvals)[:pvals.shape[0] // 1000] 
            argSig = np.argwhere( pvals < cutOff  )[:, 0]
        elif plantName == 'arabi':
            #argSig = np.argsort(pvals)[:pvals.shape[0] // 10000] 

            argSig = np.argwhere( pvals < cutOff  )[:, 0]
            #argSig = np.argsort(pvals)[:pvals.shape[0] // 1000] 
        #argSig = np.argsort(pvals)[:pvals.shape[0] // 10000]

        print (argSig.shape)
        #quit()

        numSNPs.append(argSig.shape[0])
        
        
        #argSig = np.random.permutation(pvals.shape[0])[:argSig.shape[0]] #TODO REMOVE!!


        #useBlock = True
        useBlock = False


        if useBlock:

            if plantName == 'MSI_C':
                #blockFile = open('./data/software/data/MsiC_annotated_LD.blocks', 'r')
                blockFile = open('./data/software/data/MsiC_annotated_LD.blocks.det', 'r')
            elif plantName == 'sor':
                #blockFile = open('./data/software/data/WEST_LD.blocks', 'r') #'./data/software/data/MsiC_annotated_LD'
                blockFile = open('./data/software/data/WEST_LD.blocks.det', 'r')
            elif plantName == 'arabi':
                #blockFile = open('./data/software/data/arabi_LD.blocks', 'r')
                blockFile = open('./data/software/data/arabi_LD.blocks.det', 'r')

            chr_sig = chr1[argSig]
            pos_sig = pos1[argSig]


            bedSNP = []
            blockLines = blockFile.readlines()
            for a in range(len(blockLines)):
                blockLine = blockLines[a]
                blockLine = blockLine.split(' ')
                blockLine = np.array(blockLine)
                blockLine = blockLine[blockLine!='']
                #print (blockLine)
                if a >= 1:
                    SNP_name_now = ''
                    chr_name, pos_now1, pos_now2 = blockLine[0], int(blockLine[1]), int(blockLine[2])
                    print (pos_now2 - pos_now1)

                    hasSNP = False
                    for b in range(len(argSig)):
                        chr_now = chr1[argSig[b]]
                        if str(chr_now) == chr_name:
                            pos_now = pos1[argSig[b]]
                            #print (pos_now)
                            #print (blockLine)
                            #quit()
                            if pos_now >= pos_now1:
                                if pos_now <= pos_now2:
                                    hasSNP = True 

                        pos_now = SNP_name[argSig[b]]
                        #print ()
                    

                    if hasSNP:
                        bedSNP.append([chr_name, pos_now1, pos_now2, SNP_name_now])
            

            #quit()
            #bedSNP.append([chr_name, pos_now1, pos_now2, SNP_name_now])
            bedSNP = np.array(bedSNP)
        
        else:

            bedFile = []
            for a in range(len(argSig)):
                a1 = argSig[a]

                SNP_name = SNP_name



                #chr_name = 'chr' + str(chr1[a1])
                chr_name =  str(chr1[a1])
                pos_now1 = pos1[a1].astype(str)
                pos_now2 = (pos1[a1]+1).astype(str)            


                SNP_name_now = SNP_name[a1]
                bedFile.append([chr_name, pos_now1, pos_now2, SNP_name_now])
            bedSNP = np.array(bedFile)

        print (bedSNP[:])
        #quit()

        SNP_bed_file = './data/plant/GWAS/Gemma/temp/SNP_now.bed'

        np.savetxt(SNP_bed_file, bedSNP, delimiter='\t', fmt='%s')

        #geneloc_file1 = './data/plant/GWAS/ncbi_dataset/genes_manual.bed' #TODO CHANGE THIS IS SORGUM

        if True:
            if plantName == 'MSI_C':
                geneloc_file1 = './data/plant/GWAS/genes/misc/genes_manual.bed'
            elif plantName == 'sor':
                geneloc_file1 = './data/plant/GWAS/genes/sor/genes_manual.bed'
            elif plantName == 'arabi':
                geneloc_file1 = './data/plant/GWAS/genes/arabi/genes_manual.bed'


        
        geneLoc = np.loadtxt(geneloc_file1, delimiter='\t', dtype=str)

        #print (geneLoc[:5])
        #print (bedSNP[:10])
        #quit()

        #print (geneLoc[geneLoc[:, 1] == '17127344' ])

        
        
        geneLoc1 = np.copy(geneLoc[:, 1:3]).astype(int)
        #windowSize = 200000
        #windowSize = 0

        
        #quit()

        if useBlock:
            windowSize = 0
        else:
            #windowSize = 10000
            windowSize = 200000
            #windowSize = 5000


        #windowSize = 0 

        #windowSize = 5000
        geneLoc1[:, 0] = geneLoc1[:, 0] - windowSize
        geneLoc1[:, 1] = geneLoc1[:, 1] + windowSize
        geneLoc1[geneLoc1<1] = 1
        geneLoc[:, 1:3] = geneLoc1.astype(str)

        #geneLoc[:, 1] = geneLoc[:, 1].astype(int) - 1000000
        #geneLoc[:, 2] = geneLoc[:, 2].astype(int) + 1000000
        geneLocMod_file = './data/plant/GWAS/Gemma/temp/genes_now.bed'

        #print (geneLoc[:5])
        #quit()

        np.savetxt(geneLocMod_file, geneLoc, delimiter='\t', fmt='%s')

        #print (geneLoc[0::1000])
        #quit()

        



        command2 = 'bedtools intersect -a '  + SNP_bed_file +  ' -b ' + geneLocMod_file + ' -wa -wb > ./data/plant/GWAS/Gemma/temp/snp_gene_mapping.txt'


        os.system(command2)


        #gene_result.txt

        geneList = np.loadtxt('./data/plant/GWAS/Gemma/temp/snp_gene_mapping.txt', delimiter='\t', dtype=str)

        print (geneList)

        if len(geneList.shape) == 1:
            geneList = np.array([geneList])

        #print (geneList)
        #print (geneList.shape)
        #quit()

        numGenes.append(geneList.shape[0])

        if geneList.shape[0] > 0:
            

            #data_info = loadnpz('./data/metab/metabData_Fernie/pathwayGenes/gensFromPath.npz')
            geneList_names = geneList[:, 7]

            if plantName == 'arabi':
                envName = '0d'
                if 'secondary' in methodName:
                    envName = '6d'
                
                saveFile = './data/metab/metabData_Fernie/GWAS/genes/trait' + str(traitIndex) + '_' + envName + '.txt'
                saveOnlyGenes = './data/metab/metabData_Fernie/GWAS/genesOnly/trait' + str(traitIndex) + '_' + envName + '.txt'
                np.savetxt(saveOnlyGenes, geneList_names, fmt='%s')
            

            if plantName == 'sor':
                saveOnlyGenes = './data/plant/GWAS/Gemma/genes/trait' + str(traitIndex) + '.txt'
                np.savetxt(saveOnlyGenes, geneList_names, fmt='%s')

            
            #quit()
            #descriptions = getGeneDescriptions(geneList_names)
            #descriptions = np.array(descriptions)
            #geneWithDesc = np.array( [geneList_names, descriptions ] ).T
            #np.savetxt(saveFile, geneWithDesc, fmt='%s')
            #quit()

            genes1 = geneList[:, -1]
            for a in range(genes1.shape[0]):
                print (genes1[a])
                if 'LOC' in genes1[a]:
                    genes1[a] = genes1[a].split('-')[1]
            
            for a in range(len(genes1)):
                print (genes1[a])


            if True:
                #Miscanthus orthoologs:
                file1 = './data/plant/GWAS/genes/misc/Msinensis_497_v7.1.annotation_info.txt'
                with open(file1, "r") as file:
                    lines = file.readlines()
                
                miscOrth = []

                for a in range(len(lines)):
                    line1 = lines[a]
                    line1 = line1.split('\t')
                    
                    miscName = line1[1]
                    arabiName = line1[10]
                    riceName = line1[13]
                    #print (line1)

                    #print (miscName, arabiName, riceName)
                    
                    #print (arabiName)
                    #print (riceName)

                    miscOrth.append([miscName, arabiName, riceName])
                #quit()
                miscOrth = np.array(miscOrth)

                #print (genes1[:10])
                #print ('Misin19G094600' in miscOrth)
                #print (genes1[2] in miscOrth[:, 0])
                #print (miscOrth[:20, 0])
                #quit()

                for a in range(genes1.shape[0]):
                    gene1 = genes1[a]
                    #print (gene1 in miscOrth[:, 0])
                    arg1 = np.argwhere(miscOrth[:, 0] == gene1)
                    gene1 = ''
                    if arg1.shape[0] > 0:
                        arg1 = arg1[0, 0]
                        gene1 = miscOrth[arg1, 1]
                        #gene1 = miscOrth[arg1, 2]

                        gene1 = gene1.replace('LOC_', '')

                    genes1[a] = gene1
                        

            #print (genes1.shape)
            genes1 = np.unique(genes1)
            genes1 = genes1[genes1!='']
            print (genes1.shape)

            #np.savetxt('./data/plant/GWAS/outputs/Miscanthus_Central_' + dates[timePoint] + '_pval_' + pCut + '.tsv', genes1, fmt='%s')

            #Os01g0104100
            for a in range(len(genes1)):
                print (genes1[a])

            if 'MSI' in plantName:
                saveOnlyGenes = './data/miscPlant/GWAS/genes/trait' + str(timePoint) + '_' + namePart_copy + '.txt'
                np.savetxt(saveOnlyGenes, genes1, fmt='%s')

            

            quit()


            if False: #This is for converting to "SORBI"
                genes1 = convertSORBI(genes1)
                
                
                genes1 = list(genes1)

                #for a in range(len(genes1)):
                #    print (genes1[a])

                #quit()
                
                orthoList = np.loadtxt('./data/plant/GWAS/Arabidopsis_orthologs.txt', dtype=str, delimiter='\t')

                orthList = []
                for a in range(len(genes1)):
                    gene1 = genes1[a]
                    arg1 = np.argwhere(orthoList[:, 2] == gene1)
                    if arg1.shape[0] > 0:
                        arg1 = arg1[0, 0]
                        newName = orthoList[arg1, 0]
                        orthList.append(newName)
                
                orthList = np.array(orthList)

                #quit()

                #print (orthList.shape)
                #quit()

                for a in range(len(orthList)):
                    print (orthList[a])

                quit()


            

            genes1 = np.unique(genes1)
            genes1 = genes1[genes1!='']
            
            genes1 = list(genes1)

            #print (genes1)




            #go analysis:
            from goatools.obo_parser import GODag
            from goatools.goea.go_enrichment_ns import GOEnrichmentStudy
            from goatools.anno.gaf_reader import GafReader
            from collections import defaultdict

            # Load the GO hierarchy
            godag = GODag("go-basic.obo")

            # Load GO annotations (substitute 'gene_association_file.gaf' with your downloaded file)
            gaf = GafReader("GCF_000003195.3_Sorghum_bicolor_NCBIv3_gene_ontology.gaf")


            #print (gaf)
            #quit()



            # Convert the list of associations to a dictionary where each gene ID is a key
            # and its value is a set of associated GO terms
            associations_dict = defaultdict(set)
            for association in gaf.associations:
                #print (association.DB_Symbol)
                #print (association.Symbol)
                #quit()
                #associations_dict[association.DB_ID].add(association.GO_ID)
                #associations_dict[association.DB_ID].add(association.DB_Symbol)
                associations_dict[association.DB_Symbol].add(association.GO_ID)

            # Define the background gene set as all genes in the GAF file
            background_genes = set(associations_dict.keys())

            #print (associations_dict)
            #print (list(background_genes)[:10])
            #quit()

            

            # Define your target genes (genes of interest)
            #target_genes = set(["gene1", "gene2", "gene3"])  # Replace with your genes of interest
            target_genes = set( list(genes1) )

            # Initialize the GO enrichment study with the dictionary of associations
            goea = GOEnrichmentStudy(
                background_genes,
                associations_dict,
                godag,
                propagate_counts=True,
                alpha=0.05,       # significance cut-off
                methods=['fdr_bh']  # use FDR correction for multiple testing
            )

            # Run the enrichment analysis
            results = goea.run_study(target_genes)

            for a in range(3):
                print ('')

            pvalsList = []

            # Display significant GO terms
            for res in results:
                pvalsList.append(res.p_fdr_bh)
                if res.p_fdr_bh < 0.05:  # Adjust threshold as needed
                    print(res.GO, res.name, res.p_fdr_bh)
                    

            
            minP = np.min(np.array(pvalsList))

            allPvals.append(minP)

            #print (allPvals)
    
    arange1 = np.arange(10) + 1
    plt.title('regularized neural net')
    plt.plot(arange1, numSNPs)
    plt.plot(arange1, numGenes)
    plt.xlabel('sythetic trait number')
    plt.legend(['number of significant SNPs', 'number of genes'])
    plt.show()



    #GO:0008033 tRNA processing 0.006517868051363838

    #GO:0006749 glutathione metabolic process 2.9314778224399423e-10
    #GO:0006575 cellular modified amino acid metabolic process 3.534873784008524e-08
    #GO:0043603 amide metabolic process 2.0863519417948948e-06
    #GO:0006790 sulfur compound metabolic process 2.0863519417948948e-06
    #GO:0004364 glutathione transferase activity 1.54928693103622e-09
    #GO:0016765 transferase activity, transferring alkyl or aryl (other than methyl) groups 2.0863519417948948e-06




    #Conv:
    #GO:1900150 regulation of defense response to fungus 0.027470543440313425
    #GO:0002831 regulation of response to biotic stimulus 0.027470543440313425
    #GO:0032101 regulation of response to external stimulus 0.027470543440313425



    #simple, but PC0




#geneOntology()
#quit()




def checkMISC():

    a = 0
    fileName = './data/miscPlant/SNP/vcf/MsiC_annotated.vcf'
    with open(fileName, 'r') as file:
        # Read each line in the file
        for line in file:
            #print (line)
            a += 1

            if a == 14:

                plotName = []
                plantName = []

                genotypes = line
                genotypes = genotypes.replace('\n', '')
                genotypes = genotypes.split('\t')
                genotypes = np.array(genotypes)
                genotypes = genotypes[9:]

                for a in range(genotypes.shape[0]):
                    genotypes1 = genotypes[a]

                    if not 'MsiCONTROL' in genotypes1:
                        genotypes1 = genotypes1.split('_')

                        plotName.append(genotypes1[2])
                        plantName.append(genotypes1[3])

                plotName = np.array(plotName).astype(int)
                plantName = np.array(plantName).astype(int)

                print (plotName.shape)

                print (np.unique(plotName))
                print (np.unique(plantName))



                quit()


#checkMISC()
#quit()



def doAnalyzeGoodPhen():

    Y = loadnpz('./data/plant/syntheticTraits/sor_H_4.npz')
    names = loadnpz('./data/plant/processed/sor/names.npz')

    _, count1 = np.unique(names, return_counts=True)
    _, inverse1 = np.unique(names, return_inverse=True)
    count_inverse = count1[inverse1]
    names = names[count_inverse == 2]
    Y = Y[count_inverse == 2]

    Y = Y[np.argsort(names)]
    names = names[np.argsort(names)]

    _, index1 = np.unique(names, return_index=True)

    for a in range(Y.shape[1]):
        print (a)
        plt.scatter( Y[index1, a], Y[index1+1, a] )
        plt.show()
    quit()


    print (np.mean( Y ** 2 , axis=0))
    print (np.sum( Y ** 2 , axis=0))

    Y = torch.tensor(Y).float()
    Y = normalizeIndependent(Y)
    Y = Y.data.numpy()




    #print (np.mean( Y ** 2 , axis=0))
    #print (np.sum( Y ** 2 , axis=0))
    #quit()


    good = [1, 4, 9, 12, 14]
    #good = np.array(good) - 1

    #plt.scatter(Y[:, 0], Y[:, 12])
    #plt.show()
    #quit()

    plt.hist(Y[:, 0], bins=100, alpha=0.5, range=(-5, 5))
    plt.hist(Y[:, 12], bins=100, alpha=0.5, range=(-5, 5))
    plt.show()



#doAnalyzeGoodPhen()
#quit()



def compareVCFref():

    data_info = loadnpz('./data/plant/SNP/allChr_info.npz')


    print (data_info[:10])
    quit()

#compareVCFref()
#quit()


def saveGenes():


    #plantName = 'sor'
    #plantName = 'misc'
    plantName = 'arabi'

    if plantName == 'sor':
        file1 = './data/plant/GWAS/genes/sor/Sbicolor_454_v3.1.1.gene.gff3'
    elif plantName == 'misc':
        file1 = './data/plant/GWAS/genes/misc/Msinensis_497_v7.1.gene.gff3'
    elif plantName == 'arabi':
        file1 = './data/plant/GWAS/genes/arabi/TAIR10_GFF3_genes.gff'



    data = np.loadtxt(file1, delimiter='\t', dtype=str)



    data = data[np.isin(data[:, 2], np.array(['region', 'gene']) )]

    

    #data = data[np.isin(data[:, 2], np.array(['region']) )]
    print (data[:5])
    #quit()
    #AT5G57090.1



    #bedLoad = np.loadtxt('./data/plant/GWAS/ncbi_dataset/genes_manual.bed', delimiter='\t', dtype=str)

    #print (bedLoad[:10])
    #quit()


    annotation = data[:, -1]

    chr1 = data[:, 0]
    start1 = data[:, 3]
    end1 = data[:, 4]
    geneNames = np.copy(annotation)
    geneNames_mod = np.copy(annotation)
    exclude = np.ones(data.shape[0], dtype=int)
    for a in range(annotation.shape[0]):
        chrName = chr1[a]
        chrName = chrName.replace('Chr', '')
        intChr = False
        try:
            int(chrName)
            intChr = True 
        except:
            True 

        #print (chrName)
        #quit()

        if intChr:
            exclude[a] = 0
            chr1[a] = str(int(chrName))
            

            
            list1 = annotation[a].split(';')

            #print (list1)
            #quit()

            #SORBI_3001G401300
            #   Sobic.001G000100.v3.2

            if plantName == 'arabi':
                name1 = list1[2]
            else:
                name1 = list1[1]
            

            name1 = name1.split('=')[1]

            print (name1)

            

            geneNames[a] = name1

            if plantName == 'misc':
                True

            elif plantName == 'sor':
                #print (name1)

                name1 = name1.split('.')[1]
                
                name1 = 'SORBI_3' + name1

            
            geneNames_mod[a] = name1
        

    #print (geneNames_mod.shape)
    #print (np.unique(geneNames_mod).shape)
    #quit()
    #print (geneNames[:10])

    #print (chr1[:10])
    #quit()

    print (np.unique(exclude, return_counts=True))
    #quit()

    bedArray = np.array([chr1, start1, end1, geneNames_mod]).T
    bedArray = bedArray[exclude == 0]

    print (bedArray[:5])
    #quit()

    #np.savetxt('./data/plant/GWAS/ncbi_dataset/genes_manual.bed', bedArray, delimiter='\t', fmt='%s')
    
    if plantName == 'sor':
        np.savetxt('./data/plant/GWAS/genes/sor/genes_manual.bed', bedArray, delimiter='\t', fmt='%s')
    elif plantName == 'misc':
        np.savetxt('./data/plant/GWAS/genes/misc/genes_manual.bed', bedArray, delimiter='\t', fmt='%s')
    elif plantName == 'arabi':
        np.savetxt('./data/plant/GWAS/genes/arabi/genes_manual.bed', bedArray, delimiter='\t', fmt='%s')
        #misc

    quit()


    array = np.array([chr1, start1, end1, geneNames]).T
    
    np.savez_compressed('./data/plant/GWAS/ncbi_dataset/genes_mod.npz', array)


    geneNames = loadnpz('./data/plant/GWAS/ncbi_dataset/genes_mod.npz')
    max1 = np.max(geneNames[:, 2].astype(int)) + 1
    max1 = int(max1)
    pastNames = np.zeros((10, max1), dtype=str)
    print (pastNames.shape)

    new_length = 20
    pastNames = np.char.ljust(pastNames, new_length, ' ')
    pastNames[:] = ''

    for a in range(10):
        print (a)
        names1 = []
        #print (geneNames[:10])
        #quit()
        geneNames_chr = geneNames[geneNames[:, 0] == str(a+1)]
        for b in range(geneNames_chr.shape[0]):
            pastNames[a, int(geneNames_chr[b, 1]):int(geneNames_chr[b, 2])+1] = geneNames_chr[b, 3]


    np.savez_compressed('./data/plant/GWAS/ncbi_dataset/genes_paste.npz', pastNames)



#saveGenes()
#quit()





def analyzeGWAS():

    doSouth = False 

    namePart = 'central'
    if doSouth:
        namePart = 'south'

    

    
    for splitIndex in range(0, 74):# 36):
        phenIndex = splitIndex
        #phenIndex = 0
        timePoint = splitIndex
        #clusterNum = splitIndex
        clusterNum = 26

        print (splitIndex)
        #print (timePoint)


        #data = loadnpz('./data/software/output/GWAS_phen_linear_overfit_' + str(phenIndex+1) + '.npz')

        #simAndMethod = 'simA3_linear'
        #simAndMethod = 'simA6_linear'
        #simAndMethod = 'simA3_originalTraits'
        
        #data = loadnpz('./data/plant/simulations/Gemma/' + simAndMethod + '_' + str(phenIndex+1) + '.npz')
        

        #methodName = 'sor_conv_1'
        #methodName = 'sor_A_4'
        #methodName = 'A_4b'
        #methodName = 'real_linearOther'
        #methodName = 'PCA0_sor_simple_1'
        #methodName = 'sor_simple_1'
        #methodName = 'MSI_6'
        #methodName = 'MSI_singlepoint_' + str(timePoint) + '_split2'
        #methodName = 'MSI_' + namePart + '_singlepoint_' + str(timePoint) + '_row13'




        #methodName = 'linear_crossVal_reg4_' + str(splitIndex) + '_mod10'

        

        #methodName = 'MSI_' + namePart + '_singlepoint_' + str(2) + '_split_' + str(splitIndex)
        #methodName = 'MSI_' + namePart + '_singlepoint_' + str(timePoint) + '_noSplit'

        #data = loadnpz('./data/plant/simulations/encoded/random100SNP/0/Gemma_H2Opt_' + str(phenIndex+1) + '.npz')
        #data = loadnpz('./data/plant/simulations/encoded/uncorSims/1/Gemma_PCA_' + str(phenIndex+1) + '.npz')
        #data = loadnpz('./data/plant/simulations/encoded/random100SNP/0/Gemma_H2Opt_' + str(phenIndex+1) + '.npz')
        #data = loadnpz('./data/plant/GWAS/Gemma/' + methodName + '_' + str(phenIndex+1) + '.npz' )
        


        #data = loadnpz('./data/plant/GWAS/Gemma/' + methodName + '_1.npz' )
        #data = loadnpz('./data/metab/GWAS/little_trait_' + str(phenIndex+1) +  '.npz')
        #methodName = 'linear_crossVal_reg4_' + str(0) + '_mod10'

        #data = loadnpz('./data/plant/GWAS/Gemma/' + methodName + '_' + str(phenIndex+1) + '.npz')
        
        
       
        
        
        #data = loadnpz('./data/metab/metabData_Fernie/GWAS/linear_5_'  + str(phenIndex+1) + '.npz' )
        #data = loadnpz('./data/metab/metabData_Fernie/GWAS/known_2_'  + str(phenIndex+1) + '.npz' )
        #data = loadnpz('./data/metab/metabData_Fernie/GWAS/linear_secondary5_'  + str(phenIndex+1) + '.npz' )
        #data = loadnpz('./data/metab/metabData_Fernie/GWAS/linear_secondary10_'  + str(phenIndex+1) + '.npz' )
        #data = loadnpz('./data/metab/metabData_Fernie/GWAS/linear_primary_many_split0_'  + str(phenIndex+1) + '.npz' )


        #data = loadnpz('./data/metab/metabData_Fernie/GWAS/kPCA5_linear_primary_many_split0_'  + str(phenIndex+1) + '.npz' )
        #data = loadnpz('./data/metab/metabData_Fernie/GWAS/kPCA5_linear_secondary_many_split0_'  + str(phenIndex+1) + '.npz' )

        data = loadnpz('./data/HEreg/GWAS/linear_'+ str(phenIndex+1) +'.npz' )



        #print (data[0])
        #print (data[1])
        #print (data.shape[0] - 1)
        #quit()

        


        #SNPfile = './data/plant/simulations/simPhen/' + simAndMethod.split('_')[0] + '/snp_info.npy'
        #SNPs_true = np.load(SNPfile)

        #assert  np.intersect1d(  data[:, 1], SNPs_true[:, 0] ).shape[0] == 0



        #data = loadnpz('../software/output/GWAS_phen_' + str(phenIndex+1) + '_A4_noOutlier.npz')
        #data = loadnpz('../software/output/GWAS_phen_' + str(phenIndex+1) + '_conv_noOutlier.npz')
        #data = loadnpz('../software/output/GWAS_phen_' + str(phenIndex+1) + '_sor_simple_1.npz')
        #data = loadnpz('../software/output/GWAS_phen_' + str(phenIndex+1) + '_sor_traits.npz') #N, SLA, PN, PS
        #data = loadnpz('../software/output/GWAS_phen_' + str(phenIndex+1) + '_related_7.npz')


        #data = data[np.isin(  data )]

        #print (np.unique(data[1:, 0]))
        #print (data[0, 0])
        #quit()


        argGood = np.argwhere(np.isin( data[:, 0], np.arange(20).astype(str)  ))[:, 0]
        argGood = np.concatenate(( np.zeros(1, dtype=int), argGood ), axis=0)
        data = data[argGood]

        #print (data[:, 0][:10])
        #data[1:, 0].astype(int)
        #quit()


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

        print (np.min(pvals))

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

        #18
        

        

        if True:
            max1 = min( np.max(observed_p_values), np.max(expected_p_values) )
            # Create the Q-Q plot
            print (observed_p_values.shape, expected_p_values.shape)
            print (max1)
            plt.figure(figsize=(8, 8))
            plt.plot(expected_p_values, observed_p_values, marker='o', linestyle='none')
            plt.plot([0, max1], [0, max1], color='red', linestyle='--')  # Reference line
            plt.xlabel('Expected p-values')
            plt.ylabel('Observed p-values')
            plt.title('Q-Q Plot for GWAS p-values')
            plt.title('synthetic trait ' + str(timePoint + 1))
            plt.gcf().set_size_inches(5, 4.5)
            plt.tight_layout()
            #plt.xlim(0, 1)
            #plt.ylim(0, 1)
            plt.grid()
            plt.show()

        #quit()







        

        

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
            plt.plot( line1, np.zeros(2) + cutOff  )
            #plt.plot( line1, np.zeros(2) + cutOff + np.log10(10)  )
            #plt.title('phenotype ' + str(phenIndex+1))
            plt.title('phenotype ' + str(timePoint))
            plt.xlabel("genomic bin")
            plt.ylabel('-log10(p)')
            plt.gcf().set_size_inches(5, 4.5)
            plt.tight_layout()
            plt.show()






analyzeGWAS()
quit()




def saveSorRelatednessMatrix():


    SNP_file = './data/software/data/WEST_original'

    command1 = './data/software/gemma.macosx -bfile ' + SNP_file + ' -gk 1 -o ./relatedness_matrix'

    os.system(command1)

    command2 = 'mv ./output/relatedness_matrix.cXX.txt ./data/plant/RNA/relatedness_matrix.cXX.txt'
    os.system(command2)


#saveSorRelatednessMatrix()
#quit()


def runGWAS(SNPfile = './data/software/data/WEST_original', nPCA=5, doGS=False):

    def moveClearOutput():
        
        command2 = 'mv ./output/* ./data/software/output/'
        os.system(command2)

        command3 = 'rm -r ./output'
        os.system(command3)

    #./plink --vcf ./data/all.vcf --make-bed --out ./data/all  --geno 0.1 --maf 0.05
    # 
    # # --hwe 1e-6

    #./gemma.macosx -bfile ./data/all -gk 1 -o relatedness_matrix

    #./gemma.macosx -bfile ./data/all  -k ./output/relatedness_matrix.cXX.txt -lmm 1 -o gwas_results


    command1 = './data/software/gemma.macosx -bfile ' + SNPfile + ' -gk 1 -o ./relatedness_matrix'

    os.system(command1)

    #quit()

    moveClearOutput()
    

    #'../software/data/WEST_original_PCA5.txt'

    if nPCA > 0:
        #'./data/software/data/MsiC_annotated_PCA5.txt'
        if 'WEST_original' in SNPfile:
            PCAline = '-c ./data/software/data/WEST_original_PCA' + str(nPCA) + '.txt '
        if 'MsiC_annotated' in SNPfile:
            PCAline = '-c ./data/software/data/MsiC_annotated_PCA' + str(nPCA) + '.txt '
            #('./data/software/data/MsiC_annotated_PCA5.txt'
        if 'TAIR' in SNPfile:
            famInfo = np.loadtxt(SNPfile + '.fam', delimiter=' ', dtype=str)
            if famInfo.shape[0] == 276:
                PCAline = '-c ./data/software/data/call_method_75_TAIR8_' + '6d' + '_PCA' + str(nPCA) + '.txt '
            if famInfo.shape[0] == 262:
                PCAline = '-c ./data/software/data/call_method_75_TAIR8_' + '0d' + '_PCA' + str(nPCA) + '.txt '
            #PCAline = '-c ./data/software/data/call_method_75_TAIR8_PCA' + str(nPCA) + '.txt '


        #check1 = np.loadtxt('./data/software/data/call_method_75_TAIR8_PCA5.txt')
        #print ('check1', check1.shape)


    else:
        PCAline = ''
    
    #SNPfile = './data/software/data/WEST_original' 

    #check1 = np.loadtxt('./data/software/data/MsiC_annotated_PCA5.txt')
    #check1 = np.loadtxt('./data/software/data/WEST_original_PCA5.txt')
    #print (check1[:5])
    #quit()
    #print ([PCAline[3:-1]])
    #np.loadtxt(PCAline[3:-1])

    #PCAline = ''


    #./data/software/output/phenotype.txt

    if doGS:
        famInfo = np.loadtxt(SNPfile + '.fam', delimiter=' ', dtype=str)

        #print (SNPfile)
        #quit()

        phenotypes = famInfo[:, -1]

        #phenotypes = np.where(phenotypes == '-9', '-9', phenotypes).astype(str)

        #print (phenotypes.shape)
        #quit()
        
        np.savetxt('./data/software/output/phenotype.txt', phenotypes, delimiter='\t', fmt='%s')
        #quit()
        
        #command4 = './data/software/gemma.macosx -bfile '  + SNPfile + ' -k ./data/software/output/relatedness_matrix.cXX.txt -p ./data/software/output/phenotype.txt ' + ' -predict 1 -o gs_results'


        command4 =  './data/software/gemma.macosx -bfile ' + SNPfile + '  -k ./data/software/output/relatedness_matrix.cXX.txt -p ./data/software/output/phenotype.txt -bslmm 1 -o bslmm_results'
    


        #print (command4)
        #os.system(command4)

        command5 = './data/software/gemma.macosx -bfile ' + SNPfile + ' -epm ./output/bslmm_results.param.txt -emu ./output/bslmm_results.log.txt -ebv ./output/bslmm_results.bv.txt -k ./data/software/output/relatedness_matrix.cXX.txt -p ./data/software/output/phenotype.txt -predict 1 -o gs_results'

        print (command5)

        os.system(command5)
        quit()

    else:

        

        command4 = './data/software/gemma.macosx -bfile ' + SNPfile + '  -k ./data/software/output/relatedness_matrix.cXX.txt '+  PCAline + '-lmm 1 -o gwas_results'

        os.system(command4)

    


    moveClearOutput()

    #quit()


#runGWAS()
#quit()



def sor_RunGemma(Y, names, argTraits, outputFile, SNPprefix = './data/software/data/WEST_original',nPCA=5, doGS=False):


    file_fam = './data/software/data/WEST_original_copy.fam' #Original 

    data_fam0 = np.loadtxt(file_fam, delimiter='\t', dtype=str)

    data_fam0 = data_fam0[:, :6]

    new_length = 20
    data_fam = np.char.ljust(data_fam0, new_length, ' ')
    data_fam[:] = data_fam0

    

    nameKey = np.loadtxt('./data/software/data/nameKey_corrected.tsv', delimiter='\t', dtype=str)

    names_unique = np.unique(names)
    name_fam = np.copy(data_fam[:, 0])

    if True:
        for a in range(nameKey.shape[0]):
            nameKey[a, 2] = nameKey[a, 2].replace(' ', '')
        
        for a in range(data_fam.shape[0]):
            name1 = name_fam[a]
            arg1 = np.argwhere(nameKey[:, 0] == name1)[0, 0]
            name2 = nameKey[arg1, 2]
            name_fam[a] = name2
    
    

    head1 = np.loadtxt('./data/plant/SNP/WEST_original.txt', delimiter='\t', dtype=str)
    head_names = head1[9:]
    for a in range(head_names.shape[0]):
        head_names[a] = head_names[a].split('_')[0]



    file_SNP = './data/plant/SNP/allChr.npz'
    data_SNP = loadnpz(file_SNP)
    data_SNP = np.sum(data_SNP, axis=2)
    data_SNP = data_SNP.astype(int).T
    data_SNP = data_SNP[:, :10]#

    

    for a in argTraits:# Y.shape[1]):

        print ("Phenotype number: ", a)

        Y_now = Y[:, a]

        Y_now = Y_now - np.mean(Y_now)
        Y_now = Y_now / (np.mean(Y_now ** 2) ** 0.5)

        Y_now[Y_now > 2] = 2
        Y_now[Y_now < -2] = -2


        if True:
            data_fam[:, -1] = 'NA'
            for b in range(data_fam.shape[0]):
                #args1 = np.argwhere(names == name_fam[b])
                args1 = np.argwhere(names == name_fam[b])
                if args1.shape[0] > 0:
                    args1 = args1[:, 0]
                    str_value = np.mean(Y_now[args1])
                    str_value = str(str_value)[:6]
                    data_fam[b, -1] = str_value

            data_fam[data_fam[:, -1] == 'NA', -1] = '0.0'

        else:

            data_fam[:, -1] = 'NA'
            for b in range(data_fam.shape[0]):
                print (name_fam[b],head_names[b] )
                data_fam[b, -1] = data_SNP[b, a] + (np.random.random() * 0.2) + 0.01

            data_fam[data_fam[:, -1] == 'NA', -1] = '0.0'

            print (np.mean(data_fam[:, -1].astype(float)))

            
        #quit()


        

        file_fam_new = SNPprefix + '.fam'
        #file_fam_new = './data/software/data/WEST_original.fam'
        

        np.savetxt(file_fam_new, data_fam, delimiter=' ', fmt='%s')

        

        runGWAS(SNPfile=SNPprefix, nPCA=nPCA, doGS=doGS)

        file1 = './data/software/output/gwas_results.assoc.txt'
        #file1 = '../software/output/original/gwas_results.assoc.txt'

        data_GWAS = np.loadtxt(file1, dtype=str, delimiter='\t')


        np.savez_compressed(outputFile +  str(a + 1) +  '.npz', data_GWAS)


#Y = loadnpz('./data/plant/syntheticTraits/linear_overfit.npz')
#names = loadnpz('./data/plant/processed/sor/names.npz')
#argTraits = np.arange(1)
#outputFile = './data/software/output/GWAS_phen_linear_overfit_'

#fullRunGemma(Y, names, argTraits, outputFile)
#quit()






def littleArabi_RunGemma(Y, names, argTraits, outputFile, SNPprefix = './data/metab/metabData_Alex/call_method_75/call_method_75_TAIR8',nPCA=0, doGS=False):


    file_fam = './data/metab/metabData_Alex/call_method_75/call_method_75_TAIR8_copy.fam' #Original 


    #info = np.loadtxt('./data/metab/metabData_Alex/call_method_75/call_method_75_TAIR8.bim', dtype=str)
    #print (info.shape)
    #quit()
    
    

    data_fam0 = np.loadtxt(file_fam, delimiter=' ', dtype=str)

    #print ([data_fam0[0]])
    #quit()

    data_fam0 = data_fam0[:, :6]

    new_length = 20
    data_fam = np.char.ljust(data_fam0, new_length, ' ')
    data_fam[:] = data_fam0


    

    for a in argTraits:# Y.shape[1]):

        name_fam = np.copy(data_fam[:, 0])

        print ("Phenotype number: ", a)

        Y_now = Y[:, a]

        Y_now = Y_now - np.mean(Y_now)
        Y_now = Y_now / (np.mean(Y_now ** 2) ** 0.5)

        #Y_now[Y_now > 2] = 2
        #Y_now[Y_now < -2] = -2


        if True:
            data_fam[:, -1] = 'NA'
            for b in range(data_fam.shape[0]):
                #args1 = np.argwhere(names == name_fam[b])
                args1 = np.argwhere(names == name_fam[b])
                if args1.shape[0] > 0:
                    args1 = args1[:, 0]
                    str_value = np.mean(Y_now[args1])
                    str_value = str(str_value)[:6]
                    data_fam[b, -1] = str_value

                    #print (str_value)

            data_fam = data_fam[data_fam[:, -1] != 'NA']

            #data_fam[data_fam[:, -1] == 'NA', -1] = '0.0'


        #print (data_fam.shape)
        #quit()


        file_fam_new = SNPprefix + '.fam'
        #file_fam_new = './data/software/data/WEST_original.fam'
        

        np.savetxt(file_fam_new, data_fam, delimiter=' ', fmt='%s')

        

        runGWAS(SNPfile=SNPprefix, nPCA=nPCA, doGS=doGS)

        file1 = './data/software/output/gwas_results.assoc.txt'
        #file1 = '../software/output/original/gwas_results.assoc.txt'

        data_GWAS = np.loadtxt(file1, dtype=str, delimiter='\t')


        np.savez_compressed(outputFile +  str(a + 1) +  '.npz', data_GWAS)






def doFernieGWAS():

    envChoice = '0d'
    #envChoice = '6d'

    #np.savez_compressed('./data/metab/metabData_Fernie/pred/known_1.npz', metValues_now)
    #np.savez_compressed('./data/metab/metabData_Fernie/pred/ecotypeNames_known.npz', genotypes)
    
    #names = loadnpz('./data/metab/metabData_Fernie/pred/ecotypeNames.npz')
    #names = loadnpz('./data/metab/metabData_Fernie/pred/ecotypeNames_known.npz', allow_pickle=True)

    genotype = loadnpz('./data/metab/metabData_Fernie/processed/names.npz')
    envirement = loadnpz('./data/metab/metabData_Fernie/processed/env.npz')
    genotype_6d = genotype[envirement == envChoice]
    names = np.unique(genotype_6d)


    #argTraits = np.arange(30)
    argTraits = np.arange(10)
    #argTraits = np.arange(1) + 1

    #Y =  loadnpz('./data/metab/metabData_Fernie/pred/linear_6.npz')
    #outputFile = './data/metab/metabData_Fernie/GWAS/linear_6_' 
    #Y =  loadnpz('./data/metab/metabData_Fernie/pred/known_2.npz')
    #outputFile = './data/metab/metabData_Fernie/GWAS/known_2_' 

    if False:
        if envChoice == '0d':
            Y =  loadnpz('./data/metab/metabData_Fernie/pred/linear_primary_many_split0.npz')
            outputFile = './data/metab/metabData_Fernie/GWAS/kPCA5_linear_primary_many_split0_' 
        if envChoice == '6d':
            Y =  loadnpz('./data/metab/metabData_Fernie/pred/linear_secondary_many_split0.npz')
            outputFile = './data/metab/metabData_Fernie/GWAS/kPCA5_linear_secondary_many_split0_' 

    else:
        #methodName = 'PCA'
        methodName = 'maxTrait'
        Y = loadnpz('./data/metab/metabData_Fernie/baselines/' + methodName + '_' + envChoice + '_' + str(0) + '.npz')
        outputFile = './data/metab/metabData_Fernie/GWAS/kPCA5_'  + methodName + '_' + envChoice + '_'
        

    #print (Y.shape, names.shape)
    #quit()

    littleArabi_RunGemma(Y, names, argTraits, outputFile, SNPprefix = './data/metab/metabData_Alex/call_method_75/call_method_75_TAIR8',nPCA=5, doGS=False)

#doFernieGWAS()
#quit()



def runGenomeSelection(names, SNP_file, famInfo, name_fam, Y,  phenIndex):



    grm_file = './data/software/output/data_grm'
    gcta_file = './data/software/gcta64'
    pheno_file = './data/software/output/phenotype.txt'




    #famInfo = np.loadtxt('./data/miscPlant/SNP/bed/MsiC_annotated' + '.fam', delimiter=' ', dtype=str)
    #famInfo = np.loadtxt(famFile, delimiter=' ', dtype=str)
    phenotypes = famInfo[:,  np.array([0, 1, -1]) ]
    phenotypes[:, -1] = 0.0

    #print (phenotypes[:5])
    #quit()

    #print (phenotypes[:5])
    #quit()
    #name_fam = np.copy(famInfo[:, 0])
    bool1 = np.zeros(famInfo.shape[0], dtype=int)
    for a in range(name_fam.shape[0]):
        name_fam1 = name_fam[a]
        if name_fam1 in names:
            bool1[a] = 1
            arg1 = np.argwhere(names == name_fam1)[0, 0]
            phenotypes[a, -1] = Y[arg1, phenIndex]
    phenotypes = phenotypes[bool1 == 1]
    #print (np.mean(bool1))
    #quit()

    
    #print (phenotypes.shape)
    #quit()

    
    np.savetxt(pheno_file, phenotypes, delimiter='\t', fmt='%s')

    command1 = gcta_file + ' --bfile ' + SNP_file + ' --make-grm --out ' + grm_file

    print (command1)
    os.system(command1)
    
    
   


    command2 = gcta_file + ' --grm ' + grm_file + ' --pheno ' + pheno_file + ' --reml --out ./data/software/output/gblup_results'
    #command2 = gcta_file + ' --mlma-loco' +  ' --bfile ' + SNP_file +  ' --grm ' +  grm_file + ' --pheno ' + pheno_file + ' --out ./data/software/output/gblup_results'

    
    #./gcta64 --mlma-loco \ --bfile yourdata \ --grm yourdata_grm \ --pheno phenotype.txt \ --out gwas_results_loco


    print (command2)
    os.system(command2)

    #quit()

    with open('./data/software/output/gblup_results.hsq', 'r') as file:
        # Read each line in the file
        for line in file:
            if 'V(G)/Vp' in line:
                HeritValue = line.split('\t')[1]
                #print ([value[1]])

                #heritValues.append(value)
                #print ([line])
                #quit()

    return HeritValue




def getHeadMisc():

    a = 0
    fileName = './data/miscPlant/SNP/vcf/MsiC_annotated.vcf'
    with open(fileName, 'r') as file:
        # Read each line in the file
        for line in file:
            #print (line)
            a += 1

            if a == 14:

                plotName = []
                plantName = []

                genotypes = line
                genotypes = genotypes.replace('\n', '')
                genotypes = genotypes.split('\t')
                genotypes = np.array(genotypes)
                genotypes = genotypes[9:]

                mixed_name = []

                for a in range(genotypes.shape[0]):
                    genotypes1 = genotypes[a]

                    if not 'MsiCONTROL' in genotypes1:
                        genotypes1 = genotypes1.split('_')

                        plotName.append(genotypes1[2])
                        plantName.append(genotypes1[3])

                        mixed_name.append( genotypes1[2] + '_' + genotypes1[3]  )

                    else:
                        plotName.append('control')
                        plantName.append('control')

                        mixed_name.append('control_control')

                plotName = np.array(plotName)#.astype(int)
                plantName = np.array(plantName)#.astype(int)
                mixed_name = np.array(mixed_name)

                for a in range(20):
                    print (mixed_name[a])
                quit()

    return mixed_name


def misc_RunGemma(Y, names, argTraits, outputFile, SNPprefix = './data/miscPlant/SNP/bed/MsiC_annotated',nPCA=5, doGS=False, doSouth=False):


    #newSNP = './data/miscPlant/SNP/bed/MsiC_annotated'
    

    if not doSouth:
        file_fam = './data/miscPlant/SNP/bed/MsiC_annotated_copy.fam'
    else:
        file_fam = './data/miscPlant/SNP/bed/MsiS_annotated_copy.fam'
        print ('south')
    #else:
    #    print ('issue, not sure which misc to use central vs south')
    #    quit()
    #file_fam = './data/software/data/WEST_original_copy.fam' #Original 

    try:
        data_fam0 = np.loadtxt(file_fam, delimiter='\t', dtype=str)
        data_fam0 = data_fam0[:, :6]
    except:
        data_fam0 = np.loadtxt(file_fam, delimiter=' ', dtype=str)
        data_fam0 = data_fam0[:, :6]


    new_length = 20
    data_fam = np.char.ljust(data_fam0, new_length, ' ')
    data_fam[:] = data_fam0

    
    control_count = 0
    name_fam = np.copy(data_fam[:, 0])
    for a in range(name_fam.shape[0]):
        name_fam1 = name_fam[a]

        if 'MsiCONTROL' in name_fam1:
            control_count += 1

        name_fam1 = name_fam1.split('_')[2:]
        name_fam1 = '_'.join(name_fam1)

        #print (name_fam[a], name_fam1)
        name_fam[a] = name_fam1

    #print (control_count)

    #quit()
        




    for a in argTraits:# Y.shape[1]):

        print ("Phenotype number: ", a)

        Y_now = Y[:, a]

        Y_now = Y_now - np.mean(Y_now)
        Y_now = Y_now / (np.mean(Y_now ** 2) ** 0.5)

        Y_now[Y_now > 2] = 2
        Y_now[Y_now < -2] = -2


        #print (names[:10])
        #print (name_fam[:10])
        #quit()

        if True:
            data_fam[:, -1] = 'NA'
            for b in range(data_fam.shape[0]):
                #args1 = np.argwhere(names == name_fam[b])
                args1 = np.argwhere(names == name_fam[b])
                if args1.shape[0] > 0:
                    args1 = args1[:, 0]
                    str_value = np.mean(Y_now[args1])
                    str_value = str(str_value)[:6]
                    data_fam[b, -1] = str_value

                else:
                    True
                    #print (name_fam[b])

            #data_fam[data_fam[:, -1] == 'NA', -1] = '0.0'

            if doGS:
                data_fam[data_fam[:, -1] == 'NA', -1] = '-9'
            else:
                name_fam = name_fam[data_fam[:, -1] != 'NA']
                data_fam = data_fam[data_fam[:, -1] != 'NA']


        #quit()

        #print (name_fam.shape)
        #quit()
        #plt.hist(data_fam[:, -1].astype(float), bins=100)
        #plt.show()

        data_fam[:, -2] = '-9' #Sex column
        #print (data_fam[:10])
        #np.random.seed(0)
        #data_fam[:, -1] = np.random.random(data_fam[:, -1].shape[0]) #TODO REMOVE!!!!
        #data_fam[:, -1] = np.random.randint(2, size=data_fam[:, -1].shape[0]) #TODO REMOVE!!!!

        data_fam[data_fam[:, 0] == '', 0] = "control"

        #data_fam[:, -1] = data_fam[:, -1].astype(float) - np.min(data_fam[:, -1].astype(float))

        file_fam_new = SNPprefix + '.fam'
        #file_fam_new = './data/software/data/WEST_original.fam'
        

        np.savetxt(file_fam_new, data_fam, delimiter=' ', fmt='%s')


        #quit()

        

        runGWAS(SNPfile=SNPprefix, nPCA=nPCA, doGS=doGS)

        file1 = './data/software/output/gwas_results.assoc.txt'
        #file1 = '../software/output/original/gwas_results.assoc.txt'

        data_GWAS = np.loadtxt(file1, dtype=str, delimiter='\t')

        #print (data_GWAS[:10])
        #quit()


        np.savez_compressed(outputFile +  str(a + 1) +  '.npz', data_GWAS)




def south_RunGemma(Y, names, argTraits, outputFile, SNPprefix = './data/miscPlant/SNP/bed/MsiC_annotated',nPCA=5, doGS=False):


    #newSNP = './data/miscPlant/SNP/bed/MsiC_annotated'


    file_fam = './data/miscPlant/SNP/bed/MsiC_annotated_copy.fam'
    #file_fam = './data/software/data/WEST_original_copy.fam' #Original 

    data_fam0 = np.loadtxt(file_fam, delimiter='\t', dtype=str)

    data_fam0 = data_fam0[:, :6]

    new_length = 20
    data_fam = np.char.ljust(data_fam0, new_length, ' ')
    data_fam[:] = data_fam0

    
    control_count = 0
    name_fam = np.copy(data_fam[:, 0])
    for a in range(name_fam.shape[0]):
        name_fam1 = name_fam[a]

        if 'MsiCONTROL' in name_fam1:
            control_count += 1

        name_fam1 = name_fam1.split('_')[2:]
        name_fam1 = '_'.join(name_fam1)

        #print (name_fam[a], name_fam1)
        name_fam[a] = name_fam1

    #print (control_count)

    #quit()
        




    for a in argTraits:# Y.shape[1]):

        print ("Phenotype number: ", a)

        Y_now = Y[:, a]

        Y_now = Y_now - np.mean(Y_now)
        Y_now = Y_now / (np.mean(Y_now ** 2) ** 0.5)

        Y_now[Y_now > 2] = 2
        Y_now[Y_now < -2] = -2


        #print (names[:10])
        #print (name_fam[:10])
        #quit()

        if True:
            data_fam[:, -1] = 'NA'
            for b in range(data_fam.shape[0]):
                #args1 = np.argwhere(names == name_fam[b])
                args1 = np.argwhere(names == name_fam[b])
                if args1.shape[0] > 0:
                    args1 = args1[:, 0]
                    str_value = np.mean(Y_now[args1])
                    str_value = str(str_value)[:6]
                    data_fam[b, -1] = str_value

                else:
                    True
                    #print (name_fam[b])

            #data_fam[data_fam[:, -1] == 'NA', -1] = '0.0'

            if doGS:
                data_fam[data_fam[:, -1] == 'NA', -1] = '-9'
            else:
                name_fam = name_fam[data_fam[:, -1] != 'NA']
                data_fam = data_fam[data_fam[:, -1] != 'NA']


        #quit()

        #print (name_fam.shape)
        #quit()
        #plt.hist(data_fam[:, -1].astype(float), bins=100)
        #plt.show()

        data_fam[:, -2] = '-9' #Sex column
        #print (data_fam[:10])
        #np.random.seed(0)
        #data_fam[:, -1] = np.random.random(data_fam[:, -1].shape[0]) #TODO REMOVE!!!!
        #data_fam[:, -1] = np.random.randint(2, size=data_fam[:, -1].shape[0]) #TODO REMOVE!!!!

        data_fam[data_fam[:, 0] == '', 0] = "control"

        #data_fam[:, -1] = data_fam[:, -1].astype(float) - np.min(data_fam[:, -1].astype(float))

        file_fam_new = SNPprefix + '.fam'
        #file_fam_new = './data/software/data/WEST_original.fam'
        

        np.savetxt(file_fam_new, data_fam, delimiter=' ', fmt='%s')


        #quit()

        

        runGWAS(SNPfile=SNPprefix, nPCA=nPCA, doGS=doGS)

        file1 = './data/software/output/gwas_results.assoc.txt'
        #file1 = '../software/output/original/gwas_results.assoc.txt'

        data_GWAS = np.loadtxt(file1, dtype=str, delimiter='\t')

        #print (data_GWAS[:10])
        #quit()


        np.savez_compressed(outputFile +  str(a + 1) +  '.npz', data_GWAS)

def predictSimPhenotypes(modelName, predName, predName2, dataFolder):


    #modelName = './data/plant/simulations/models/simA3_linear.pt'
    model = torch.load(modelName)

    #X = np.loadtxt('./data/plant/simulations/simPhen/sim16/Simulated_Data_10_Reps_Herit_0.1...0.1.txt', dtype=str)
    #dataFolder = './data/plant/simulations/simPhen/simA3'
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

    np.savez_compressed(predName, Y)
    np.savez_compressed(predName2, X)






def saveSimulationExclusionSNPs(simulationFolder, outputFile):

    #./plink --bfile your_data --exclude exclude_snps.txt --make-bed --out filtered_data

    #simName = 'simA3'

    originalSNP = './data/software/data/WEST_original'
    plinkLoc = './data/software/plink'
    #simulationFolder = './data/plant/simulations/simPhen/' + simName + '/'

    #First get the SNP files
    causalSNP_files = []
    files1 = os.listdir(simulationFolder)
    print (files1)
    for a in range(len(files1)):
        if 'Selected_QTNs.txt' in files1[a]:
            causalSNP_files.append(files1[a])

    SNP_list = []
    SNP_info = []

    for a in range(len(causalSNP_files)):
        file1 = open( simulationFolder +  causalSNP_files[a], 'r')
        print (file1)
        Lines = file1.readlines()

        #print (Lines)

        line1 = np.array(Lines[0].split('\t'))

        #ids: Chr01_928237
        
        #argSNP = np.argwhere(line1 == 'snp')[0, 0]
        argChr = np.argwhere(line1 == 'chr')[0, 0]
        argPos = np.argwhere(line1 == 'pos')[0, 0]

        for b in range(1, len(Lines)):
            line2 = Lines[b].split('\t')
            chrNum = line2[argChr]
            posNum = line2[argPos]
            chrName = chrNum
            if len(chrName) == 1:
                chrName = '0' + chrName
            idName = 'Chr' + chrName + '_' + posNum
            SNP_list.append(idName)

            SNP_info.append([ idName, line2[argChr], line2[argPos] ])
    
    SNP_info = np.array(SNP_info)
    
    #print (SNP_info)
    #quit()
    SNPfile = simulationFolder + 'snp_info.npy'
    np.save(SNPfile, SNP_info)
    
    SNP_list = np.array(SNP_list)

    #print (SNP_list)

    SNPfile = simulationFolder + 'snp_list.txt'
    np.savetxt(SNPfile, SNP_list, fmt='%s')

    #outputFile = simulationFolder + 'WEST_original'

    #./plink --bfile your_data --exclude exclude_snps.txt --make-bed --out filtered_data
    command1 = plinkLoc + ' --bfile ' + originalSNP + ' --exclude ' + SNPfile + ' --make-bed --out ' +  outputFile

    os.system(command1)







    



    True 

#saveSimulationExclusionSNPs()
#quit()



def runSimulations():

    np.random.seed(0)

    #simulationNames = ['uncorSims', 'random100SNP']
    simulationNames = ['seperate100SNP']
    #methodNames = ['H2Opt', 'maxWave', 'PCA', 'groundTruth']
    #methodNames = ['groundTruth']
    methodNames = ['H2Opt', 'maxWave', 'PCA']

    synthUsed = 5

    for simIndex in range(0, 10):

        for a in range(10):
            print ('')
        print ('simIndex: ', simIndex)
        for a in range(10):
            print ('')

        for simulationName in simulationNames:

            for methodName in methodNames:
            
                print ('simulationName: ', simulationName)
                print ('methodName: ', methodName)


                folder0 = './data/plant/simulations/encoded/' + simulationName + '/'



                folder1 = folder0 + str(simIndex)


                SNPprefix = './data/plant/simulations/temp/WEST_original'
                #simulationFolder = './data/plant/simulations/encoded/' + simulationName + '/' + str(simIndex)
                simulationFolder = './data/plant/simulations/simPhen/' + simulationName + '/' + str(simIndex) + '/'

                #SNPfile = simulationFolder + 'snp_info.npy'


                saveSimulationExclusionSNPs(simulationFolder, SNPprefix)

                #quit()

                if methodName == 'groundTruth':
                    dataFolder =  './data/plant/simulations/simPhen/' + simulationName + '/' + str(simIndex) 
                    dataFile = dataFolder + '/'
                    files1 = os.listdir(dataFolder)
                    #print (files1)
                    #quit()
                    for file1 in files1:
                        if 'Simulated_Data_' in file1:
                            if '0.6_0.4_0.2' in file1:
                                dataFile = dataFile + file1
                    X_original = np.loadtxt(dataFile, dtype=str)
                    Y = X_original[1:, 1:-1].astype(float)

            
                if methodName == 'H2Opt':
                    Yfile = folder1 + '/H2Opt_predValues.npz'
                if methodName == 'PCA':
                    Yfile = folder1 + '/PCA_predValues.npz'
                if methodName == 'maxWave':
                    Yfile = folder1 + '/maxWave_predValues.npz'
                
                
                outputFile = folder1 + '/Gemma_'  + methodName + '_'              
                names = loadnpz(folder1 + '/names.npz')
                if methodName != 'groundTruth':
                    Y = loadnpz(Yfile)
                #argTraits = np.arange(5)
                
                #argTraits = np.arange(1) + 1 #For the second trait ATM
                argTraits = np.arange(1) + 2


                
                sor_RunGemma(Y, names, argTraits, outputFile, SNPprefix = SNPprefix)



#runSimulations()
#quit()







def binned_compareGWASTrue():


    def binned_getBoolClose(chr_sig, pos_sig, chr_true, pos_true, distanceCutoff, chrSizes):


        totalCount = np.sum((chrSizes // distanceCutoff) + 1)

        

        pos_sig = pos_sig //distanceCutoff
        pos_true = pos_true // distanceCutoff

        fullName_sig = []
        for a in range(pos_sig.shape[0]):
            fullName_sig.append(str(  chr_sig[a]) + '_' + str(pos_sig[a]))
        fullName_true = []
        for a in range(pos_true.shape[0]):
            fullName_true.append(str(  chr_true[a]) + '_' + str(pos_true[a]))
        fullName_sig = np.array(fullName_sig)
        fullName_true = np.array(fullName_true)

        fullName_sig = np.unique(fullName_sig)
        fullName_true = np.unique(fullName_true)

        #print (fullName_true.shape)

        TP = np.intersect1d(fullName_sig, fullName_true).shape[0]
        FP = fullName_sig[np.isin(fullName_sig, fullName_true) == False].shape[0]
        FN = fullName_true[np.isin(fullName_true, fullName_sig) == False].shape[0]
        TN = totalCount - TP - FP - FN

        #print (TP, FP, FN )

        return TP, FP, FN, TN 
    

    
    simulationNames = ['seperate100SNP']
    methodNames = ['H2Opt', 'maxWave', 'PCA', 'groundTruth']
    #methodNames = ['H2Opt']

    synthUsed = 5


    blockFile = open('./data/software/data/WEST_LD.blocks', 'r') #'./data/software/data/MsiC_annotated_LD'
    #blockFile = open('./data/software/data/WEST_LD_mod.blocks', 'r')
    blockLines = blockFile.readlines()
    for a in range(len(blockLines)):
        line1 =  blockLines[a]
        line1 = line1.replace('* ', '')
        line1 = line1.replace('\n', '')
        blockLines[a] = line1.split(' ')


    count1 = 0
    blockKeys = {}
    for a in range(len(blockLines)):
        for b in range(len(blockLines[a])):
            blockKeys[blockLines[a][b]] = a 
            count1 += 1

    #print (count1)
    #quit()




    for simulationName in simulationNames:

        #cutOffList = [-5, -3, -2, -1, 0, 1, 2, 3, 4]
        #cutOffList = [-5, -4, -3, -2, -1, 0, 1, 2]
        #cutOffList = [-5, -4, -3, -2, -1, 0]

        #cutOffList = [-5, -4.5, -4, -3.5, -3, -2, -1, 0]


        Nsim = 10
        Ntrait = 3

        #traitIndex = 0


        #TPR_ar = np.zeros((Nsim, 4, len(cutOffList)))
        #FPR_ar = np.zeros((Nsim, 4, len(cutOffList)))

        
        #probValues = [np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int)]
        #trueValues = [np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int)]
        arrayExist = False

        folder0 = './data/plant/simulations/encoded/' + simulationName + '/'

        for simIndex in range(0, Nsim):

            folder1 = folder0 + str(simIndex)

            

            for methodIndex in range(len(methodNames)):
                methodName = methodNames[methodIndex]

                print ('simulationName: ', simulationName)
                print ('methodName: ', methodName)

                

                for a in range(3):
                    print ('')
                print ('simIndex: ', simIndex)
                for a in range(3):
                    print ('')

                for traitIndex in range(Ntrait):
                

                    simulationFolder = './data/plant/simulations/simPhen/' + simulationName + '/' + str(simIndex) + '/'
                    SNPfile = simulationFolder + 'snp_info.npy'
                    SNPs_true = np.load(SNPfile)

                    #print (SNPs_true.shape)
                    #quit()

                    if simulationName == 'seperate100SNP':
                        if traitIndex == 0:
                            SNPs_true = SNPs_true[:100]
                        if traitIndex == 1:
                            SNPs_true = SNPs_true[100:200]
                        if traitIndex == 2:
                            SNPs_true = SNPs_true[200:300]
                        #SNPs_true = SNPs_true[100:200]
                    
                    #for phenIndex in range(1):# [0]:#range(1):

                    phenIndex = traitIndex

                    outputFile = folder1 + '/Gemma_'  + methodName + '_'   
                    data_mini = loadnpz(outputFile + str(phenIndex+1) + '.npz')


                    pvals = data_mini[1:, -1].astype(float)

                    #if phenIndex == 0:
                    data = np.copy(data_mini)
                    #else:
                    #    data = np.concatenate((data, data_mini[1:]), axis=0)

                    

                    #distanceCutoff = 10000
                    distanceCutoff = 100000

                    chr1 = data[1:, 0].astype(int)
                    pos1 = data[1:, 2].astype(int)
                    pvals = data[1:, -1]
                    pvals_log = np.log10(pvals.astype(float)) * -1

                    chr_true = SNPs_true[:, 1].astype(int)
                    pos_true = SNPs_true[:, 2].astype(int)

                    valuesList_full = np.zeros(0)
                    truthList_full = np.zeros(0)

                    
                    probValues_cat = np.zeros(0)
                    trueValues_cat = np.zeros(0)

                    chr_unique = np.unique(chr1)
                    for chrIndex in range(chr_unique.shape[0]):
                        args1 = np.argwhere(chr1 == chr_unique[chrIndex])[:, 0]
                        max_pos = np.max(pos1[args1])
                        Nbins = (max_pos // distanceCutoff) + 1

                        valuesList = np.zeros(Nbins)
                        for index1 in range(args1.shape[0]):
                            pos_now = pos1[args1[index1]] // distanceCutoff
                            valuesList[pos_now] = max(valuesList[pos_now], pvals_log[args1[index1]])

                        truthList = np.zeros(Nbins)
                        args2 = np.argwhere(chr_true == chr_unique[chrIndex])[:, 0]
                        for index2 in range(args2.shape[0]):
                            pos_now = pos_true[args2[index2]] // distanceCutoff
                            truthList[pos_now] = 1


                        #valuesList_full = np.concatenate((valuesList_full, valuesList), axis=0)
                        #truthList_full = np.concatenate((truthList_full, truthList), axis=0)

                        probValues_cat = np.concatenate((probValues_cat, valuesList), axis=0)
                        trueValues_cat = np.concatenate((trueValues_cat, truthList), axis=0)
                        

                        #probValues_now[methodIndex] = np.concatenate((probValues_now[methodIndex], valuesList), axis=0)
                        #trueValues_now[methodIndex] = np.concatenate((trueValues_now[methodIndex], truthList), axis=0)
                        #print (probValues[methodIndex].shape)
                        #quit()

                    

                    if not arrayExist:
                        probValues = np.zeros(( Ntrait, len(methodNames), Nsim, probValues_cat.shape[0] ))
                        trueValues = np.zeros( probValues.shape ) 
                        arrayExist = True

                    print ('indices', traitIndex, methodIndex, simIndex)

                    probValues[traitIndex, methodIndex, simIndex] = np.copy(probValues_cat)
                    trueValues[traitIndex, methodIndex, simIndex] = np.copy(trueValues_cat)
                    
                    
                    
                    if False:
                        TP, FP, FN, TN  = binned_getBoolClose(chr_sig, pos_sig, chr_true, pos_true, distanceCutoff, chrSizes)

                        TPR = TP / (TP + FN)
                        if FP + TP == 0:
                            FPR = 1.0
                        else:
                            FPR = FP / (FP + TN)

                        TPR_ar[simIndex, methodIndex, cutOffIndex] = TPR
                        FPR_ar[simIndex, methodIndex, cutOffIndex] = FPR



                    if methodName == 'H2Opt':
                        color = 'tab:blue'
                    if methodName == 'PCA':
                        color = 'tab:orange'
                    if methodName == 'maxWave':
                        color = 'tab:green'

            

            #print (probValues_now[0].shape, probValues_now[1].shape, probValues_now[2].shape)
            #print (trueValues_now[0].shape, trueValues_now[1].shape, trueValues_now[2].shape)
        
        #quit()

        

        #print (np.max(trueValues[0], axis=(1, 2)))
        #quit()
        

        #probValues = np.array(probValues)
        #trueValues = np.array(trueValues)

        #probValues = [np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int)]
        #trueValues = [np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int)]

        #np.savez_compressed('./data/plant/eval/GWAS/probValues.npz', probValues)
        #np.savez_compressed('./data/plant/eval/GWAS/trueValues.npz', trueValues)


        np.savez_compressed(folder0 + 'GWAS_probValues_new.npz', probValues)
        np.savez_compressed(folder0 + 'GWAS_trueValues_new.npz', trueValues)


        quit()


        


#binned_compareGWASTrue()
#quit()




def runRealGWAS():

    #methodName = 'sor_A_4'
    #methodName = 'A_4b'
    #methodName = 'real_linearOther'
    
    #methodName = 'sor_conv_1'
    #methodName = 'real_linearAug'
    #methodName = 'sor_linear_7'

    if False:
        methodName = 'linear_trainAll2'
        X = loadnpz('./data/plant/processed/sor/X.npz')
        X = torch.tensor(X).float()
        model  = torch.load('./data/plant/models/' + methodName + '.pt').to(X.device)
        Y = model(X, np.arange(10))
        Y = normalizeIndependent(Y).data.numpy()


    for splitIndex in range(1):
        #Y = loadnpz('./data/plant/syntheticTraits/linear_crossVal_reg4_' + str(splitIndex) + '_mod10.npz')
        #methodName = 'linear_crossVal_reg4_' + str(splitIndex) + '_mod10'

        #methodName = 'linear_crossVal_reg4_' + str(splitIndex) + '_mod10'

        #methodName = 'maxWave'
        methodName = 'PCA'

        Y = loadnpz('./data/plant/syntheticTraits_baseline/' + methodName + '_' + str(splitIndex) + '.npz')
        #Y = loadnpz('./data/plant/syntheticTraits/' + methodName + '.npz')
        
        
        #Y = loadnpz('./data/plant/syntheticTraits/deconv_0.npz')
        #names = loadnpz('./data/plant/processed/sor/names.npz')


        #Y = loadnpz('./data/plant/syntheticTraits/deconv_0.npz')
        file_fam = './data/software/data/WEST_original_copy.fam'
        data = np.loadtxt(file_fam, delimiter='\t', dtype=str)
        names = data[:, 0]

        #print (Y.shape, names.shape)

        #methodName = methodName + '_deconvolve'
        #argTraits = np.arange(10)
        #argTraits = np.arange(1) + 2 #+ 6# + 1

        #A = loadnpz('./data/temp/deconv.npz')
        
        #Y = (A @ Y.T).T #Y @ A.T
        

        argTraits = np.arange(10)
        #argTraits = np.array([0, 1, 15])

        outputFile = './data/plant/GWAS/Gemma/' + methodName + '_'
        



        sor_RunGemma(Y, names, argTraits, outputFile, nPCA=5, doGS=False)

        quit()


#runRealGWAS()
#quit()



#np.savez_compressed('./data/HEreg/pred/linear.npz', Y)
def runHE_GWAS():


    Y = loadnpz('./data/HEreg/pred/linear.npz')
    
    #file_fam = './data/software/data/WEST_original_copy.fam'
    #data = np.loadtxt(file_fam, delimiter='\t', dtype=str)
    #names = data[:, 0]

    names = loadnpz('./data/plant/processed/sor/names.npz')
    

    argTraits = np.arange(10)

    outputFile = './data/HEreg/GWAS/linear_'
    
    sor_RunGemma(Y, names, argTraits, outputFile, nPCA=5, doGS=False)

runHE_GWAS()
quit()



def permute_runRealSorGWAS():

    
    for permute in range(100):
        print ('permute', permute)
        for splitIndex in range(1):
            methodName = 'linear_crossVal_reg4_' + str(splitIndex) + '_mod10'

            Y = loadnpz('./data/plant/syntheticTraits/' + methodName + '.npz')
            names = loadnpz('./data/plant/processed/sor/names.npz')

            np.random.seed(permute)
            perm1 = np.random.permutation(Y.shape[0])
            Y = Y[perm1]
            

            argTraits = np.arange(2)

            outputFile = './data/plant/GWAS/Gemma/' + methodName + '_permute:' + str(permute) + '_'

            sor_RunGemma(Y, names, argTraits, outputFile, nPCA=5, doGS=False)


#permute_runRealSorGWAS()
#quit()



def runAllPhenotypes():


    file_fam = './data/software/data/WEST_original_copy.fam'

    data_fam0 = np.loadtxt(file_fam, delimiter='\t', dtype=str)

    data_fam0 = data_fam0[:, :6]

    new_length = 20
    data_fam = np.char.ljust(data_fam0, new_length, ' ')
    data_fam[:] = data_fam0

    #Y = loadnpz('./data/plant/syntheticTraits/sor_simple_1.npz')
    #Y = loadnpz('./data/plant/syntheticTraits/sor_traits.npz')

    #Y = loadnpz('./data/plant/syntheticTraits/sor_cheat.npz')

    #Y = loadnpz('./data/plant/syntheticTraits/sor_cheat_indep.npz')


    #Y = loadnpz('./data/plant/syntheticTraits/sor_A_4.npz')

    #Y = loadnpz('./data/plant/syntheticTraits/sor_conv_1.npz')
    #Y = loadnpz('./data/plant/syntheticTraits/sor_cheat.npz')
    
    #Y = loadnpz('./data/plant/syntheticTraits/sor_singleWave_1.npz')
    #Y = loadnpz('./data/plant/syntheticTraits/sor_singleWave_2.npz')
    #Y = loadnpz('./data/plant/syntheticTraits/sor_PCA_1.npz')
    #Y = loadnpz('./data/plant/syntheticTraits/sor_mixed_3.npz')

    Y = loadnpz('./data/plant/syntheticTraits/linear_overfit.npz')

    



    #print (Y.shape)
    #quit()

    

    '''
    traitAll = loadnpz('./data/plant/processed/sor/traits.npz')
    for a in range(traitAll.shape[1]):
        traitAll[np.isnan(traitAll[:, a]), a] = np.mean(traitAll[np.isnan(traitAll[:, a]) == False, a])
    Y = traitAll[:,  np.array([0, 2, 3, 5]) ]

    Y2 = loadnpz('./data/plant/syntheticTraits/sor_A_5.npz')
    for a in range(10):
        print ("A", a)
        print (scipy.stats.pearsonr(Y2[:, a], Y[:, 0]))
        print (scipy.stats.pearsonr(Y2[:, a], Y[:, 1]))
        print (scipy.stats.pearsonr(Y2[:, a], Y[:, 2]))
        print (scipy.stats.pearsonr(Y2[:, a], Y[:, 3]))

    quit()
    '''

    

    

    names = loadnpz('./data/plant/processed/sor/names.npz')

    #print (names.shape)
    #print (Y.shape)
    #quit()

    nameKey = np.loadtxt('./data/software/data/nameKey_corrected.tsv', delimiter='\t', dtype=str)

    names_unique = np.unique(names)
    name_fam = data_fam[:, 0]

    if True:
        for a in range(nameKey.shape[0]):
            nameKey[a, 2] = nameKey[a, 2].replace(' ', '')
        
        for a in range(data_fam.shape[0]):
            name1 = name_fam[a]
            arg1 = np.argwhere(nameKey[:, 0] == name1)[0, 0]
            name2 = nameKey[arg1, 2]
            name_fam[a] = name2
    

    
    
    file_SNP = './data/plant/SNP/allChr.npz'

    data_SNP = loadnpz(file_SNP)
    data_SNP = np.sum(data_SNP, axis=2)
    data_SNP = data_SNP.astype(int).T
    data_SNP = data_SNP[:, :10]# np.random.permutation(data_SNP.shape[1])[:30] ]

    head1 = np.loadtxt('./data/plant/SNP/WEST_original.txt', delimiter='\t', dtype=str)
    head_names = head1[9:]
    for a in range(head_names.shape[0]):
        head_names[a] = head_names[a].split('_')[0]

    #print (data_SNP.shape)
    #print (head_names.shape)
    #quit()


    for a in range(0, 10):# Y.shape[1]):

        print ("Phenotype number: ", a)

        Y_now = Y[:, a]

        Y_now = Y_now - np.mean(Y_now)
        Y_now = Y_now / (np.mean(Y_now ** 2) ** 0.5)

        Y_now[Y_now > 2] = 2
        Y_now[Y_now < -2] = -2

        #Y_now = data_SNP[:, a]
        #Y_now = Y[:, 2]#[np.random.permutation(Y.shape[0])]

        #print (np.argsort(Y_now[:20]))

        #_, Y_now = np.unique(Y_now, return_inverse=True)
        #Y_now = Y_now / np.max(Y_now)

        

        if True:
            data_fam[:, -1] = 'NA'
            for b in range(data_fam.shape[0]):
                #args1 = np.argwhere(names == name_fam[b])
                args1 = np.argwhere(names == name_fam[b])
                if args1.shape[0] > 0:
                    args1 = args1[:, 0]

                    #print (Y_now[args1])
                    #quit()

                    str_value = np.mean(Y_now[args1])
                    str_value = str(str_value)[:6]
                    data_fam[b, -1] = str_value


            #print (data_fam[:10, -1])
            #print (data_SNP[:10, 0])

            print (data_fam[:, -1].shape)
            print ('NA', np.argwhere(data_fam[:, -1] == 'NA' ).shape)



            #for b in range(data_fam.shape[0]):
            #    if data_fam[b, -1] != 'NA':
            #        assert int(float(data_fam[b, -1])) == int(data_SNP[b, a])
            

            data_fam[data_fam[:, -1] == 'NA', -1] = '0.0'

        else:
            data_fam[:, -1] = 'NA'



            #print (data_fam[:10, 0])
            #print (head_names[:10])
            #quit()

            for b in range(data_fam.shape[0]):
                print (name_fam[b],head_names[b] )
                #args1 = np.argwhere(head_names == name_fam[b])
                #if args1.shape[0] > 0:
                #    args1 = args1[0, 0]
                #    data_fam[b, -1] = data_SNP[args1, a]
                data_fam[b, -1] = data_SNP[b, a]

            #data_fam[:, -1] = data_SNP[:, a]
        

        file_fam_new = './data/software/data/WEST_original.fam'

        np.savetxt(file_fam_new, data_fam, delimiter=' ', fmt='%s')

        runGWAS()

        file1 = './data/software/output/gwas_results.assoc.txt'
        #file1 = '../software/output/original/gwas_results.assoc.txt'

        data_GWAS = np.loadtxt(file1, dtype=str, delimiter='\t')



        #np.savez_compressed('../software/output/GWAS_phen_' + str(a+1) + '_related3_noOutlier.npz', data_GWAS)
        np.savez_compressed('./data/software/output/GWAS_phen_' + str(a+1) + '_linear_overfit.npz', data_GWAS)



#runAllPhenotypes()
#quit()


def runRealMiscGWAS():


    doSouth = True

    namePart = 'central'
    if doSouth:
        namePart = 'south'


    
    for timePoint1 in range(14):
        for splitIndex in [0]:#range(5):
    
            
            #methodName = 'MSI_' + namePart + '_singlepoint_' + str(timePoint1) + '_split_' + str(splitIndex)
            #methodName = 'MSI_' + namePart + '_singlepoint_' + str(timePoint1) + '_noSplit'

            #methodName = 'PCA'
            methodName = 'maxTrait'
            #methodName = 'maxWave'

            methodName = methodName + '_' + namePart + '_' + str(timePoint1)
            Y = loadnpz('./data/miscPlant/eval/pred_' + methodName + '.npz')
            #Y = loadnpz('./data/miscPlant/GWAS/' + methodName + '.npz')
            if doSouth:
                names = loadnpz('./data/miscPlant/GWAS/names_MSI_S.npz')
                SNPprefix = './data/miscPlant/SNP/bed/MsiS_annotated'
            else:
                names = loadnpz('./data/miscPlant/GWAS/names_MSI_C.npz')
                SNPprefix = './data/miscPlant/SNP/bed/MsiC_annotated'
            #argTraits = np.arange(10)

            argTraits = np.arange(1)

            outputFile = './data/plant/GWAS/Gemma/' + methodName + '_'

            


            
            misc_RunGemma(Y, names, argTraits, outputFile, nPCA=5, doGS=False, SNPprefix=SNPprefix, doSouth=doSouth)



#runRealMiscGWAS()
#quit()


def permute_runRealMiscGWAS():


    doSouth = False

    namePart = 'central'
    if doSouth:
        namePart = 'south'


    
    for timePoint1 in range(2, 5):# range(14):
        for splitIndex in [0]:#range(5):

            print ('timepoint', timePoint1)

            for permute in range(20, 100):

                print ('permute', permute)
    
                #methodName = 'MSI_6'
                #methodName = 'MSI_singlepoint_5'
                #methodName = 'MSI_singlepoint_skip2'
                #methodName = 'MSI_singlepoint_' + str(timePoint1) + '_split2'

                #methodName = 'MSI_' + namePart + '_singlepoint_' + str(timePoint1) + '_row13'
                methodName = 'MSI_' + namePart + '_singlepoint_' + str(timePoint1) + '_split_' + str(splitIndex)
                #methodName = 'MSI_' + namePart + '_singlepoint_' + str(timePoint1) + '_noSplit'


                Y = loadnpz('./data/miscPlant/GWAS/' + methodName + '.npz')
                np.random.seed(permute)
                permute1 = np.random.permutation(Y.shape[0])
                Y = Y[permute1]

                #Y = loadnpz( './data/miscPlant/GWAS/MSI_' + 'central' + '_singlepoint_' + str(timePoint1) + '_row1_copy.npz')
                if doSouth:
                    names = loadnpz('./data/miscPlant/GWAS/names_MSI_S.npz')
                    SNPprefix = './data/miscPlant/SNP/bed/MsiS_annotated'
                else:
                    names = loadnpz('./data/miscPlant/GWAS/names_MSI_C.npz')
                    SNPprefix = './data/miscPlant/SNP/bed/MsiC_annotated'
                #argTraits = np.arange(10)

                argTraits = np.arange(1)
                #argTraits = np.arange(1) + 2
                #argTraits = np.array([0, 1, 15])

                
                #print (names[:3])
                #quit()
                

                #outputFile = './data/plant/GWAS/Gemma/' + methodName + '_PCA0_'
                #outputFile = './data/plant/GWAS/Gemma/' + methodName + '_'
                outputFile = './data/plant/GWAS/Gemma/' + methodName + '_permute:' + str(permute) + '_'

                


                
                misc_RunGemma(Y, names, argTraits, outputFile, nPCA=5, doGS=False, SNPprefix=SNPprefix, doSouth=doSouth)



#permute_runRealMiscGWAS()
#quit()


def convertBED():


    


    #originalSNP = './data/miscPlant/SNP/vcf/MsiS_annotated.vcf'
    #newSNP = './data/miscPlant/SNP/bed/MsiS_annotated'

    originalSNP = './data/miscPlant/SNP/vcf/MsiS_filtered.vcf'
    newSNP = './data/miscPlant/SNP/bed/MsiS_filtered'
    plinkLoc = './data/software/plink2'
    


    #command1 = 'bcftools view -r 1-30 ./data/miscPlant/SNP/vcf/MsiS_annotated.vcf -o ./data/miscPlant/SNP/vcf/MsiS_filtered.vcf'
    #os.system(command1)
    #quit()

    #chrString = '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26'

    #command1 = plinkLoc + ' --vcf ' + originalSNP + ' --make-bed  --max-alleles 2 --out ' + newSNP + ' --double-id'
    #command1 = plinkLoc + ' --vcf ' + originalSNP + ' --make-bed  --max-alleles 2 --out ' + newSNP_temp + ' --double-id'
    command1 = plinkLoc + ' --vcf ' + originalSNP + ' --make-bed  --max-alleles 2 --allow-extra-chr --chr 1-20 --out ' + newSNP + ' --double-id'

    os.system(command1)

    #plinkLoc = './data/software/plink'

    #command2 = plinkLoc + ' --bfile ' + newSNP_temp + ' --make-bed  --allow-extra-chr -chr 1-20 --out ' + newSNP + ' --double-id'

    #os.system(command2)



#convertBED()
#quit()



def sec_convert_CSV_VCF():


    

    vcf_output_path = './data/metab/metabData_Alex/call_method_75/call_method_75_TAIR8.vcf'
    #vcf_output_path = './data/metab/metabData_Alex/call_method_32/call_method_32_TAIR8.vcf'

    if True:
        # Load the CSV
        import pandas as pd
        from collections import Counter

        file_path = (
            "./data/metab/metabData_Alex/call_method_75/call_method_75_TAIR8.csv"
        )
        #vcf_output_path = "./call_method_75_TAIR8.vcf"

        ###############################################################################
        # READ – skip first metadata row, strip spaces in headers
        ###############################################################################
        df = pd.read_csv(file_path, dtype=str, skiprows=1)
        df.columns = df.columns.str.strip()          # <- remove sneaky spaces

        # Keep real column names for later access
        CHR_COL, POS_COL = df.columns[:2]
        sample_ids = list(df.columns[2:])            # after stripping

        ###############################################################################
        # HELPERS
        ###############################################################################
        CANON = {"A", "C", "G", "T"}

        def ref_alt(allele_list):
            """Return (REF, alt_list) after ignoring missing/ambiguous calls."""
            calls = [a.upper() for a in allele_list if a.upper() in CANON]
            if not calls:
                return None, None
            ref = Counter(calls).most_common(1)[0][0]
            alt = sorted({a for a in calls if a != ref})
            return ref, alt                       # alt is a **list**

        def encode(g, ref, alt):
            """
            Encode one‐letter genotype as VCF GT field.
            * haploid or homozygous calls only *
            """
            g = g.upper()
            if g == ref:
                return "0/0"
            elif g in alt:
                return f"{alt.index(g)+1}/{alt.index(g)+1}"
            else:
                return "./."

        ###############################################################################
        # BUILD VCF RECORDS
        ###############################################################################
        records = []
        for index_print, row in df.iterrows():
            print (index_print)
            chrom, pos = row[CHR_COL], row[POS_COL]
            alleles = row[sample_ids].tolist()

            print (alleles)

            ref, alt_list = ref_alt(alleles)
            if ref is None or not alt_list:        # skip monomorphic / no valid base
                continue

            alt_field = ",".join(alt_list)
            gts = [encode(a, ref, alt_list) for a in alleles]

            print (gts)
            quit()

            records.append(
                "\t".join(
                    [chrom, pos, f"{chrom}_{pos}", ref, alt_field,
                    ".", "PASS", ".", "GT"] + gts
                )
            )

        ###############################################################################
        # WRITE VCF
        ###############################################################################
        header = (
            "##fileformat=VCFv4.2\n"
            "##source=csv_to_vcf_clean\n"
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(sample_ids)
        )

        with open(vcf_output_path, "w") as f:
            f.write(header + "\n")
            f.write("\n".join(records))

        print(f"VCF saved → {vcf_output_path}")



    else:
        plinkLoc = './data/software/plink'
        #plinkLoc = './data/software/plink2'
        
        bedLoc = './data/metab/metabData_Alex/call_method_75/call_method_75_TAIR8'
        #bedLoc = './data/metab/metabData_Alex/call_method_32/call_method_32_TAIR8'
        #command1 = plinkLoc + ' --vcf ' + vcf_output_path + ' --make-bed --out ' + bedLoc
        command1 = plinkLoc + ' --vcf ' + vcf_output_path + ' --make-bed --allow-extra-chr --chr 1-20 --out ' + bedLoc + ' --double-id'
        os.system(command1)



#sec_convert_CSV_VCF()
#quit()



def OLD_compareGWASTrue():


    def getBoolClose(chr_sig, pos_sig, chr_true, pos_true, distanceCutoff):

        boolClose = np.zeros(chr_sig.shape[0], dtype=int) - 1

        #distanceCutoff = 50000


        for SNP_index in range(chr_true.shape[0]):
            chr_now = chr_true[SNP_index]
            pos_now = pos_true[SNP_index]
            argSameChrome = np.argwhere(  chr_sig == chr_now)[:, 0]
            
            argClose = argSameChrome[ np.abs(pos_sig[argSameChrome] - pos_now) < distanceCutoff ]
            boolClose[argClose] = SNP_index 

        return boolClose
    

    #simulationNames = ['uncorSims', 'random100SNP']
    #simulationNames = ['random100SNP']
    simulationNames = ['seperate100SNP']
    #simulationNames = ['uncorSims']
    methodNames = ['H2Opt', 'maxWave', 'PCA', 'groundTruth']
    #methodNames = ['groundTruth']

    synthUsed = 5


    blockFile = open('./data/software/data/WEST_LD.blocks', 'r') #'./data/software/data/MsiC_annotated_LD'
    #blockFile = open('./data/software/data/WEST_LD_mod.blocks', 'r')
    blockLines = blockFile.readlines()
    for a in range(len(blockLines)):
        line1 =  blockLines[a]
        line1 = line1.replace('* ', '')
        line1 = line1.replace('\n', '')
        blockLines[a] = line1.split(' ')


    count1 = 0
    blockKeys = {}
    for a in range(len(blockLines)):
        for b in range(len(blockLines[a])):
            blockKeys[blockLines[a][b]] = a 
            count1 += 1

    #print (count1)
    #quit()




    for simulationName in simulationNames:

        #cutOffList = [-5, -3, -2, -1, 0, 1, 2, 3, 4]
        #cutOffList = [-5, -4, -3, -2, -1, 0, 1, 2]
        cutOffList = [-5, -4, -3, -2, -1, 0]

        Nsim = 10

        precision_ar = np.zeros((Nsim, 4, len(cutOffList)))
        truePos_ar = np.zeros((Nsim, 4, len(cutOffList)))


        folder0 = './data/plant/simulations/encoded/' + simulationName + '/'

        for simIndex in range(0, Nsim):

            folder1 = folder0 + str(simIndex)
            

            for methodIndex in range(len(methodNames)):
                methodName = methodNames[methodIndex]

                print ('simulationName: ', simulationName)
                print ('methodName: ', methodName)

                

                totalSNP = np.zeros(5)
                validSNP = np.zeros(5)
                validSNP_100k = np.zeros(5)
                validSNP_m = np.zeros(5)

                for a in range(3):
                    print ('')
                print ('simIndex: ', simIndex)
                for a in range(3):
                    print ('')
                

                simulationFolder = './data/plant/simulations/simPhen/' + simulationName + '/' + str(simIndex) + '/'
                SNPfile = simulationFolder + 'snp_info.npy'
                SNPs_true = np.load(SNPfile)

                if simulationName == 'seperate100SNP':
                    #SNPs_true = SNPs_true[:100]
                    SNPs_true = SNPs_true[100:200]
                
                for phenIndex in [0]:#range(1):

                    outputFile = folder1 + '/Gemma_'  + methodName + '_'   

                    #print (outputFile)
                    #quit()  

                    data_mini = loadnpz(outputFile + str(phenIndex+1) + '.npz')


                    pvals = data_mini[1:, -1].astype(float)
                    print ('min1, ', np.min( pvals  ))

                    if phenIndex == 0:
                        data = np.copy(data_mini)
                    else:
                        data = np.concatenate((data, data_mini[1:]), axis=0)

                    

                #print (SNPs_true)
                #quit()
                validBlocks = []
                for a in range(SNPs_true.shape[0]):
                    if SNPs_true[a, 0] in blockKeys:
                        validBlocks.append( blockKeys[SNPs_true[a, 0]]  )
                    else:
                        #validBlocks.append(SNPs_true[a, 0])
                        False
                
                validBlocks = np.unique(np.array(validBlocks))

                #print (validBlocks.shape)
                #quit()

                

                chr1 = data[1:, 0].astype(int)


                pos1 = data[1:, 2].astype(int)


                pvals = data[1:, -1]
                pvals = pvals.astype(float)
                pvals = pvals + (1e-300)
                pvals_log = -1 * np.log(pvals) / np.log(10)

                cutOff_original = np.log(data.shape[0]) / np.log(10)
                #cutOff = cutOff - (np.log(0.05) / np.log(10))

                #cutOffAdjustment 

                #cutOffList = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
                
                #cutOffList = [-2, -1, 0, 1, 2, 3, 4]
                #cutOffList = [0, 5, 10, 15, 20, 40, 50]

                #cutOffList = [0]

                precision_list = []
                recall_list = []
                truePos_list = []

                for cutOffIndex in range(len(cutOffList)):

                    cutOff = cutOff_original + cutOffList[cutOffIndex] 


                    argHigh = np.argwhere(pvals_log > cutOff)[:, 0]

                    print ('argHigh', argHigh.shape)

                    #argHigh = argHigh[np.argsort(pvals_log[argHigh] *-1)] #[:10] #TEMPORARY top 10 SNPS
                    data_sigSNPS = data[argHigh+1] #+1 to remove header

                    chr_sig = data_sigSNPS[:, 0].astype(int)
                    pos_sig = data_sigSNPS[:, 2].astype(int)



                    chr_true = SNPs_true[:, 1].astype(int)
                    pos_true = SNPs_true[:, 2].astype(int)


                    if True:

                        if pos_sig.shape[0] == 0:
                            truePos = 0
                            precision_now = 0
                        else:

                            #distanceCutoff = 10000
                            distanceCutoff = 100000
                            boolClose = getBoolClose(chr_sig, pos_sig, chr_true, pos_true, distanceCutoff)
                            
                            truePos = np.unique(boolClose[boolClose!=-1]).shape[0]

                            precision_now = boolClose[boolClose!=-1].shape[0] / boolClose.shape[0]


                        


                        #precision_list.append(precision_now)
                        #recall_list.append(recall_now)
                        #truePos_list.append(truePos)

                        precision_ar[simIndex, methodIndex, cutOffIndex] = precision_now
                        truePos_ar[simIndex, methodIndex, cutOffIndex] = truePos




                    if False:
                        
                        print (data_sigSNPS.shape)

                        includedBlocks = []
                        for a in range(data_sigSNPS.shape[0]):
                            if data_sigSNPS[a, 1] in blockKeys:
                                includedBlocks.append( blockKeys[data_sigSNPS[a, 1]]  )
                            else:
                                #includedBlocks.append(data_sigSNPS[a, 1]  )
                                False

                        print (len(includedBlocks))
                        includedBlocks = np.unique(np.array(includedBlocks))

                        

                        #print ('')
                        #print (includedBlocks)
                        #print (validBlocks)
                        #quit()

                        #plt.scatter(includedBlocks, np.zeros(includedBlocks.shape[0]))
                        #plt.scatter(validBlocks, np.ones(validBlocks.shape[0]))
                        #plt.show()


                        truePos = np.intersect1d(includedBlocks, validBlocks).shape[0]
                        
                        includedBlocks_size = includedBlocks.shape[0]
                        if includedBlocks_size == 0:
                            includedBlocks_size = 1

                        precision_now = float(truePos) / includedBlocks_size
                        recall_now = float(truePos) / float(validBlocks.shape[0])


                        #print (includedBlocks_size, validBlocks.shape[0] , truePos)

                        precision_list.append(precision_now)
                        recall_list.append(recall_now)

                        truePos_list.append(truePos)

                        #print (precision_now, recall_now )
                


                

                

                if False:
                

                    totalSNP[phenIndex] = pos_sig.shape[0]
                    
                    distanceCutoff = 10000
                    boolClose = getBoolClose(chr_sig, pos_sig, chr_true, pos_true, distanceCutoff)
                    print (boolClose)

                    validSNP[phenIndex] = np.argwhere(boolClose >=0).shape[0]


                    #for a in range(8):
                    #    snpsNearEach[phenIndex, a] = np.argwhere(boolClose == a).shape[0]


                    distanceCutoff = 100000
                    boolClose = getBoolClose(chr_sig, pos_sig, chr_true, pos_true, distanceCutoff)
                    print (boolClose)

                    validSNP_100k[phenIndex] = np.argwhere(boolClose >=0).shape[0]

                    distanceCutoff = 1000000
                    boolClose = getBoolClose(chr_sig, pos_sig, chr_true, pos_true, distanceCutoff)
                    print (boolClose)

                    #for a in range(8):
                    #    snpsNearEach_1M[phenIndex, a] = np.argwhere(boolClose == a).shape[0]

                    validSNP_m[phenIndex] = np.argwhere(boolClose >=0).shape[0]


                if methodName == 'H2Opt':
                    color = 'tab:blue'
                if methodName == 'PCA':
                    color = 'tab:orange'
                if methodName == 'maxWave':
                    color = 'tab:green'

        


        np.savez_compressed(folder0 + 'GWAS_precision.npz', precision_ar)
        np.savez_compressed(folder0 + 'GWAS_truePos.npz', truePos_ar)

        #np.savez_compressed(folder0 + 'GWAS_precision_p1.npz', precision_ar)
        #np.savez_compressed(folder0 + 'GWAS_truePos_p1.npz', truePos_ar)

        print ("Done")

        quit()

        
        for methodIndex2 in range(3):
            precision_list = np.median(precision_ar[:, methodIndex2], axis=0)
            truePos_list = np.median(truePos_ar[:, methodIndex2], axis=0)
            plt.plot(precision_list, truePos_list)
            plt.scatter(precision_list, truePos_list)
        plt.xlabel('precision')
        plt.ylabel('detected causal SNPs')
        #plt.yscale('log')
        plt.show()
        #sns.heatmap(snpsNearEach_1M.T, annot=True)
        #plt.xlabel('trait number')
        #plt.ylabel('SNP number')
        #plt.title('number of causal SNPs within 1Mb of significant SNPs for each trait')
        #plt.show()
        #quit()

        #plt.plot(snpsNearEach)
        #plt.plot(np.zeros(8), color='grey')
        #plt.title('number of significant SNPs within 10kb on each causal SNP')
        #plt.xlabel('causal SNP number')
        #plt.ylabel('number of signficiant SNPs')
        #plt.show()
        #quit()

        if False:
            plt.plot(totalSNP)
            plt.plot(validSNP)
            plt.plot(validSNP_100k)
            plt.plot(validSNP_m)
            plt.legend(['total SNPs', 'within 10kb of causal SNP',  'within 100kb of causal SNP', 'within 1Mb of causal SNP'])
            plt.xlabel('Sythetic trait number')
            plt.ylabel('number of SNPs')
            
            plt.show()

    
    

#compareGWASTrue()
#quit()
