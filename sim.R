library(simplePHENOTYPES)
#data("SNP55K_maize282_maf04")
#SNP55K_maize282_maf04[1:8, 1:10]

#write.csv(SNP55K_maize282_maf04, './data/plant/simulations/SNP/maize.csv')


#print (SNP55K_maize282_maf04[1:8, 1:10])


#sorgumSNPs <-read.csv('./data/plant/SNP/pruned_forSimulation2.csv')
#sorgumSNPs <-read.csv('./data/plant/SNP/allChr_full.csv')
#uncorrelatedSNP <-read.csv('./data/plant/SNP/simulation/uncorrelatedSNP.csv')
#uncorrelatedSNP <-read.csv('./data/plant/SNP/simulation/correlatedSNP.csv')


for (a in 0:9) {

  #generateSim = 'uncorrelated'
  #generateSim = 'randomSNPs'
  #generateSim = 'manySNPs'
  #generateSim = 'seperate100SNP'
  generateSim = 'sameHeritSep100'


  if (generateSim == 'uncorrelated') {
    fileName <- paste('./data/plant/SNP/simulation/specialSNP/uncorrelatedSNP_', toString(a) ,'.csv', sep="")
    uncorrelatedSNP <-read.csv(fileName)
    saveFolder =  paste('./data/plant/simulations/simPhen/uncorSims/', toString(a), sep="") 
    numSNPs = 3
    
  }

  if (generateSim == 'randomSNPs') {
    fileName <- paste('./data/plant/SNP/simulation/specialSNP/random3SNP_', toString(a) ,'.csv', sep="")
    uncorrelatedSNP <-read.csv(fileName)
    saveFolder =  paste('./data/plant/simulations/simPhen/random3SNP/', toString(a), sep="") 
    numSNPs = 3
  }

  if (generateSim == 'manySNPs') {
    fileName <- paste('./data/plant/SNP/simulation/specialSNP/random100SNP_', toString(a) ,'.csv', sep="")
    uncorrelatedSNP <-read.csv(fileName)
    saveFolder =  paste('./data/plant/simulations/simPhen/random100SNP/', toString(a), sep="") 
    numSNPs = 100
  }

  if (generateSim == 'seperate100SNP') {
    fileName <- paste('./data/plant/SNP/simulation/specialSNP/random300SNP_', toString(a) ,'.csv', sep="")
    uncorrelatedSNP <-read.csv(fileName)
    saveFolder =  paste('./data/plant/simulations/simPhen/seperate100SNP/', toString(a), sep="") 
    numSNPs = 300
  }

  if (generateSim == 'sameHeritSep100') {
    fileName <- paste('./data/plant/SNP/simulation/specialSNP/random300SNP_', toString(a) ,'.csv', sep="")
    uncorrelatedSNP <-read.csv(fileName)
    saveFolder =  paste('./data/plant/simulations/simPhen/sameHeritSep100/', toString(a), sep="") 
    numSNPs = 300
  }
  
  #print (saveFolder)
  #quit()
  
  

  #manySNP <-read.csv('./data/plant/SNP/simulation/specialSNP/random100SNP_0.csv')
  #randomSNP <-read.csv('./data/plant/SNP/simulation/specialSNP/randomSNP_0.csv')


  if (numSNPs == 3) {


  custom_a <- list(trait_1 = c(0.1, 0.0, 0.0),
                          trait_2 = c(0.0, 0.1, 0.0),
                          trait_3 = c(0.0, 0.0, 0.1))

  test2 <-  create_phenotypes(
    geno_obj = uncorrelatedSNP,
    add_QTN_num = 3, #dom_QTN_num = 4,
    h2 = c(0.6,0.4, 0.2),
    add_effect = custom_a, #dom_effect = custom_geometric_d,
    ntraits = 3,#QTN_list = QTN_list,
    rep = 2,
    vary_QTN = FALSE, #output_format = "multi-file",
    architecture = "pleiotropic",
    to_r = T,
    sim_method = "custom",
    seed = 10,
    model = "A", #"AD"
    home_dir = saveFolder
  )
  }


  if (numSNPs == 100) {

    min1 = -1
    max1 = 1
    n = 100
    num_traits = 3

    # Initialize an empty list
    custom_a <- list()

    herit_list <- c()

    # Use a for loop to generate the traits
    for (i in 1:num_traits) {
      custom_a[[paste0("trait_", i)]] <- runif(n, min=min1, max=max1)
      herit_list <- c(herit_list, 0.1)
    }

  test2 <-  create_phenotypes(
    geno_obj = uncorrelatedSNP,
    add_QTN_num = n, #dom_QTN_num = 4,
    h2 = c(0.6, 0.4, 0.2),
    add_effect = custom_a, #dom_effect = custom_geometric_d,
    ntraits = 3,#QTN_list = QTN_list,
    rep = 2,
    vary_QTN = FALSE, #output_format = "multi-file",
    architecture = "pleiotropic",
    to_r = T,
    sim_method = "custom",
    seed = 10,
    model = "A", #"AD"
    home_dir = saveFolder
  )
  }


  if (numSNPs == 300) {

    min1 = -1
    max1 = 1
    n = 300
    num_traits = 3

    # Initialize an empty list
    custom_a <- list()

    zeros100 = rep(0, 100)

    # Use a for loop to generate the traits
    for (i in 1:num_traits) {
      if (i == 1) {
        custom_a[[paste0("trait_", i)]] <- c( runif(100, min=min1, max=max1), zeros100, zeros100)
      }
      if (i == 2) {
        custom_a[[paste0("trait_", i)]] <- c( zeros100, runif(100, min=min1, max=max1), zeros100)
      }
      if (i == 3) {
        custom_a[[paste0("trait_", i)]] <- c( zeros100, zeros100, runif(100, min=min1, max=max1))
      }
    }

  heritList = c(0.6, 0.4, 0.2)

  if (generateSim == 'sameHeritSep100'){
    heritList = c(0.4, 0.4, 0.4)
  }
    

    test2 <-  create_phenotypes(
      geno_obj = uncorrelatedSNP,
      add_QTN_num = n, #dom_QTN_num = 4,
      h2 = heritList,# c(0.6, 0.4, 0.2),
      add_effect = custom_a, #dom_effect = custom_geometric_d,
      ntraits = 3,#QTN_list = QTN_list,
      rep = 2,
      vary_QTN = FALSE, #output_format = "multi-file",
      architecture = "pleiotropic",
      to_r = T,
      sim_method = "custom",
      seed = 10,
      model = "A", #"AD"
      home_dir = saveFolder
    )
  }

}









#uncorSims



quit()





if (FALSE){
min1 = -1
max1 = 1
n = 100
num_traits = 50

# Initialize an empty list
custom_a <- list()

herit_list <- c()

# Use a for loop to generate the traits
for (i in 1:num_traits) {
  custom_a[[paste0("trait_", i)]] <- runif(n, min=min1, max=max1)
  herit_list <- c(herit_list, 0.1)
}

 test2 <-  create_phenotypes(
   geno_obj = sorgumSNPs,
   add_QTN_num = n, 
   h2 = herit_list,
   add_effect = custom_a, 
   ntraits = num_traits,
   rep = 10,
   vary_QTN = FALSE,
   architecture = "pleiotropic",
   to_r = T,
   sim_method = "custom",
   seed = 10,
   model = "A", #"AD", output_dir = 'sim14',
  home_dir = './data/plant/simulations/simPhen/simA3'
 )
}



if (FALSE) {


custom_a <- list(trait_1 = c(-0.3, 0.2, 0.1),
                         trait_2 = c(0.3, -0.2, 0.1),
                         trait_3 = c(0.3, 0.2, -0.1))

 test2 <-  create_phenotypes(
   geno_obj = uncorrelatedSNP,
   add_QTN_num = 3, #dom_QTN_num = 4,
   h2 = c(0.5,0.5, 0.5),
   add_effect = custom_a, #dom_effect = custom_geometric_d,
   ntraits = 3,#QTN_list = QTN_list,
   rep = 2,
   vary_QTN = FALSE, #output_format = "multi-file",
   architecture = "pleiotropic",
   to_r = T,
   sim_method = "custom",
   seed = 10,
   model = "A", #"AD"
  home_dir = './data/plant/simulations/simPhen/uncor2'
 )
}







#

if (FALSE) {
test1 <-  create_phenotypes(
    geno_obj = SNP55K_maize282_maf04,
    add_QTN_num = 3,
    dom_QTN_num = 4,
    big_add_QTN_effect = c(0.3, 0.3, 0.3),
    h2 = c(0.2, 0.4, 0.4),
    add_effect = c(0.04,0.2,0.1),
    dom_effect = c(0.04,0.2,0.1),
    ntraits = 3,
    rep = 1,
    vary_QTN = FALSE,
    output_format = "multi-file",
    architecture = "pleiotropic", #output_dir = "Results_Pleiotropic",
    to_r = TRUE,
    seed = 10,
    model = "AD",
    sim_method = "geometric",
  home_dir = './data/plant/simulations/sim2'
  )
}




if (FALSE) {
 custom_geometric_a <- list(trait_1 = c(0.3, 0.2, 0.1),
                         trait_2 = c(0.3, 0.2, 0.1),
                         trait_3 = c(0.3, 0.2, 0.1))

 test2 <-  create_phenotypes(
   geno_obj = SNP55K_maize282_maf04,
   add_QTN_num = 3, #dom_QTN_num = 4,
   h2 = c(0.2,0.4, 0.4),
   add_effect = custom_geometric_a, #dom_effect = custom_geometric_d,
   ntraits = 3,
   rep = 10,
   vary_QTN = FALSE, #output_format = "multi-file",
   architecture = "pleiotropic",
   to_r = T,
   sim_method = "custom",
   seed = 10,
   model = "A", #"AD"
  home_dir = './data/plant/simulations/sim3'
 )
}
 


if (FALSE) {
custom_a <- list(trait_1 = c(0.3, 0.3),
                         trait_2 = c(0.3, -0.3),
                         trait_3 = c(0.3, 0.3),
                         trait_4 = c(0.3, -0.3))

 test2 <-  create_phenotypes(
   geno_obj = SNP55K_maize282_maf04,
   add_QTN_num = 2, #dom_QTN_num = 4,
   h2 = c(0.4,0.4, 0.4, 0.4),
   add_effect = custom_a, #dom_effect = custom_geometric_d,
   ntraits = 4,
   rep = 10,
   vary_QTN = FALSE, #output_format = "multi-file",
   architecture = "pleiotropic",
   to_r = T,
   sim_method = "custom",
   seed = 10,
   model = "A", #"AD"
  home_dir = './data/plant/simulations/sim4'
 )
}


if (FALSE) {
custom_a <- list(trait_1 = c(0.3, 0.3, 0.0 , 0.0),
                         trait_2 = c(0.3, 0.3, 0.0 , 0.0),
                         trait_3 = c(0.0, 0.0, 0.3, 0.3),
                         trait_4 = c(0.0, 0.0, 0.3, 0.3))

 test2 <-  create_phenotypes(
   geno_obj = SNP55K_maize282_maf04,
   add_QTN_num = 4, 
   h2 = c(0.4,0.4, 0.4, 0.4),
   add_effect = custom_a, 
   ntraits = 4,
   rep = 10,
   vary_QTN = FALSE,
   architecture = "pleiotropic",
   to_r = T,
   sim_method = "custom",
   seed = 10,
   model = "A", #"AD"
  home_dir = './data/plant/simulations/sim5'
 )
}



if (FALSE) {
 custom_a <- list(trait_1 = c(0.3, 0.3, 0.0 , 0.0),
                         trait_2 = c(0.3, 0.3, 0.0 , 0.0),
                         trait_3 = c(0.0, 0.0, 0.3, 0.3),
                         trait_4 = c(0.0, 0.0, 0.3, 0.3))

 test2 <-  create_phenotypes(
   geno_obj = SNP55K_maize282_maf04,
   add_QTN_num = 4, 
   h2 = c(0.5,0.4, 0.3, 0.2),
   add_effect = custom_a, 
   ntraits = 4,
   rep = 10,
   vary_QTN = FALSE,
   architecture = "pleiotropic",
   to_r = T,
   sim_method = "custom",
   seed = 10,
   model = "A", #"AD"
  home_dir = './data/plant/simulations/sim6'
 )
}

 #What impacts the standard deviation of each trait? 
 #variances for this simulation are: [0.33110754 0.42806546 0.49783171 0.78119534]

#0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1 , 0.1, 0.1, 0.1, 0.1, 0.1 , 0.1, 0.1
#c(0.1, 0.1, 0.1 , 0.1, 0.1, 0.1, 0.1, 0.1 , 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 )

if (FALSE) {
list_zeros = rep(0, 50)
list_effect = rep(0.1, 50)
effect1 = c(list_effect, list_zeros)
effect2 = c(list_zeros, list_effect)

 custom_a <- list(trait_1 = effect1,
                         trait_2 = effect1,
                         trait_3 = effect2,
                         trait_4 = effect2)

 test2 <-  create_phenotypes(
   geno_obj = SNP55K_maize282_maf04,
   add_QTN_num = 100, 
   h2 = c(0.4, 0.4, 0.4, 0.4),
   add_effect = custom_a, 
   ntraits = 4,
   rep = 10,
   vary_QTN = FALSE,
   architecture = "pleiotropic",
   to_r = T,
   sim_method = "custom",
   seed = 10,
   model = "A", #"AD"
  home_dir = './data/plant/simulations/sim7'
 )
}





if (FALSE) {
 custom_a <- list(trait_1 = c(0.5, 0.0, 0.0 , 0.0, 0.0, 0.0, 0.0 , 0.0),
                         trait_2 = c(0.0, 0.5, 0.0 , 0.0, 0.0, 0.0, 0.0 , 0.0),
                         trait_3 = c(0.0, 0.0, 0.5 , 0.0, 0.0, 0.0, 0.0 , 0.0),
                         trait_4 = c(0.0, 0.0, 0.0 , 0.5, 0.0, 0.0, 0.0 , 0.0),
                         trait_5 = c(0.0, 0.0, 0.0 , 0.0, 0.5, 0.0, 0.0 , 0.0),
                         trait_6 = c(0.0, 0.0, 0.0 , 0.0, 0.0, 0.5, 0.0 , 0.0),
                         trait_7 = c(0.0, 0.0, 0.0 , 0.0, 0.0, 0.0, 0.5 , 0.0),
                         trait_8 = c(0.0, 0.0, 0.0 , 0.0, 0.0, 0.0, 0.0 , 0.5))

 test2 <-  create_phenotypes(
   geno_obj = sorgumSNPs,
   add_QTN_num = 8, 
   h2 = c(0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4),
   add_effect = custom_a, 
   ntraits = 8,
   rep = 10,
   vary_QTN = FALSE,
   architecture = "pleiotropic",
   to_r = T,
   sim_method = "custom",
   seed = 10,
   model = "A", #"AD"
  home_dir = './data/plant/simulations/simPhen/simA1'
 )
}



if (FALSE){
#Simple phenotypes glitch. If all herit under 0.1, then it makes all herit equal to 1.0

 custom_a <- list(trait_1 = c(0.01, 0.01, 0.01 , 0.01, 0.01, 0.01, 0.01 , 0.01, 0.01, 0.01),
                         trait_2 = c(0.0, 0.01, 0.0 , 0.0, 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0),
                         trait_3 = c(0.0, 0.0, 0.01 , 0.0, 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0),
                         trait_4 = c(0.0, 0.0, 0.0 , 0.01, 0.0, 0.0, 0.0 , 0.0, 0.0, 0.0),
                         trait_5 = c(0.0, 0.0, 0.0 , 0.0, 0.01, 0.0, 0.0 , 0.0, 0.0, 0.0),
                         trait_6 = c(0.0, 0.0, 0.0 , 0.0, 0.0, 0.01, 0.0 , 0.0, 0.0, 0.0),
                         trait_7 = c(0.0, 0.0, 0.0 , 0.0, 0.0, 0.0, 0.01 , 0.0, 0.0, 0.0),
                         trait_8 = c(0.0, 0.0, 0.0 , 0.0, 0.0, 0.0, 0.0 , 0.01, 0.0, 0.0),
                         trait_9 = c(0.0, 0.0, 0.0 , 0.0, 0.0, 0.0, 0.0 , 0.0, 0.01, 0.0),
                         trait_10 = c(0.0, 0.0, 0.0 , 0.0, 0.0, 0.0, 0.0 , 0.0, 0.0, 0.01))

 test2 <-  create_phenotypes(
   geno_obj = sorgumSNPs,
   add_QTN_num = 10, 
   h2 = c(0.1, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
   add_effect = custom_a, 
   ntraits = 10,
   rep = 10,
   vary_QTN = FALSE,
   architecture = "pleiotropic",
   to_r = T,
   sim_method = "custom",
   seed = 10,
   model = "A", #"AD"
  home_dir = './data/plant/simulations/sim11'
 )
}







#quit()



if (FALSE){
min1 = -1
max1 = 1
n = 5
num_traits = 50

# Initialize an empty list
custom_a <- list()

herit_list <- c()

# Use a for loop to generate the traits
for (i in 1:num_traits) {
  custom_a[[paste0("trait_", i)]] <- runif(n, min=min1, max=max1)
  herit_list <- c(herit_list, 0.1)
}

 test2 <-  create_phenotypes(
   geno_obj = sorgumSNPs,
   dom_QTN_num = n,
   h2 = herit_list,
   dom_effect = custom_a, #dom_effect = custom_geometric_d,
   ntraits = num_traits,
   rep = 10,
   vary_QTN = FALSE, #output_format = "multi-file",
   architecture = "pleiotropic",
   to_r = T,
   sim_method = "custom",
   seed = 10,
   model = "D", #"AD"
  home_dir = './data/plant/simulations/simPhen/simA4'
 )
}


if (TRUE){
min1 = -1
max1 = 1
n = 1000
num_traits = 50

# Initialize an empty list
custom_a <- list()

herit_list <- c()

# Use a for loop to generate the traits
for (i in 1:num_traits) {
  custom_a[[paste0("trait_", i)]] <- runif(n, min=min1, max=max1)
  herit_list <- c(herit_list, 0.1)
}

 test2 <-  create_phenotypes(
   geno_obj = sorgumSNPs,
   add_QTN_num = n, 
   h2 = herit_list,
   add_effect = custom_a, 
   ntraits = num_traits,
   rep = 10,
   vary_QTN = FALSE,
   architecture = "pleiotropic",
   to_r = T,
   sim_method = "custom",
   seed = 10,
   model = "A", #"AD", output_dir = 'sim14',
  home_dir = './data/plant/simulations/simPhen/simA5'
 )
}
quit()






min1 = -1
max1 = 1
n = 5
num_traits = 50

# Initialize an empty list
custom_a <- list()

herit_list <- c()

# Use a for loop to generate the traits
for (i in 1:num_traits) {
  custom_a[[paste0("trait_", i)]] <- runif(n, min=min1, max=max1)
  herit_list <- c(herit_list, 0.1)
}

 test2 <-  create_phenotypes(
   geno_obj = sorgumSNPs,
   dom_QTN_num = n,
   h2 = herit_list,
   dom_effect = custom_a, #dom_effect = custom_geometric_d,
   ntraits = num_traits,
   rep = 10,
   vary_QTN = FALSE, #output_format = "multi-file",
   architecture = "pleiotropic",
   to_r = T,
   sim_method = "custom",
   seed = 10,
   model = "D", #"AD"
  home_dir = './data/plant/simulations/simPhen/sim17'
 )