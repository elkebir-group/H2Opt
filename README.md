# H2Opt
H2Opt: A novel self-supervised algorithm to mine high-throughput phenotyping data for genetically-driven traits

## Running H2Opt

### Calculating ANOVA heritability 
Let n be the number of individuals. Define Y as an n by k PyTorch tensor of phenotypes. Define "groups" as a length n integer array representing genetically related groups such as clones. Define environment as an n by g matrix of categorical variables, where g is the number of environmental variables (and can be zero). Then, the heritability can be calculated as 

```python
from shared import cheapHeritability
H = cheapHeritability(Y, names, envirement)
```

