# H2Opt
H2Opt: A novel self-supervised algorithm to mine high-throughput phenotyping data for genetically-driven traits

## Running H2Opt

### Calculating ANOVA heritability 
Let n be the number of individuals. Define "traits" as an n by k PyTorch tensor of phenotypes. Define "groups" as a length n integer array representing genetically related groups such as clones. Define "environments" as an n by g matrix of categorical variables, where g is the number of environmental variables (and can be zero). Then, the heritability can be calculated as follows in Python. 

```python
from shared import cheapHeritability
H = cheapHeritability(traits, groups, envirements)
```
Specifically, if groups represent clones, this directly gives the broad-sense heritability. If groups have genetic relatedness Gamma, then narrow-sense heritability is H / Gamma. 

### Optimizing heritability 
Let n be the number of individuals. Define "groups" as a length n integer array representing genetically related groups such as clones. Define "environments" as an n by g matrix of categorical variables, where g is the number of environmental variables (and can be zero). Define "model" as the PyTorch model that determines the synthetic traits and will be trained. Define "HTP" as the HTP measurement data tensor (with the first axis having length n). Define "trainTest" as a numpy array of length n, with values 0 indicating individuals in the training set and values 1 indicating individuals in the test set. Define "modelFile" as the location for the trained model to be saved. The minimal usage of H2Opt heritability optimization is as below in Python. 
```python
from shared import trainModel
trainModel(model, HTP, groups, envirement, trainTest, modelFile)
```
Additional optional parameters include the following. ``Nphen`` is the number of phenotypes to extract, denoted by k in kH2Opt formulas. By default Nphen = 1. "learningRate" is the Pytorch learning rate with a default of 1e-4. noiseLevel is data augmentation-based regularization level with a default value of 0.1. The below code sets these values. 
```python
from shared import trainModel
trainModel(model, HTP, groups, envirement, trainTest, modelFile, Nphen=Nphen, learningRate=learningRate, noiseLevel=noiseLevel)
```
Below is a full example of training a linear model to extract 10 synthetic traits on our sorghum hyperspectral measurement dataset. 
```python
import numpy as np
from h2opt import loadnpz, trainModel, multiConv, simpleModel

X = loadnpz('./data/examples/measurements.npz')
genotype = loadnpz('./data/examples/genotypes.npz')
envirement = loadnpz('./data/examples/envirement.npz')
trainTest = np.zeros(genotype.shape[0], dtype=int)
Nphen = 10
model = multiConv(Nphen, [X.shape[1], 1], simpleModel)

trainModel(model, X, genotype, envirement, trainTest, './model.pt', Niter=10000, doPrint=True, Nphen=Nphen, learningRate=1e-5, noiseLevel=0.005)
```





