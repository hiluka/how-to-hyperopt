#--------------------------------------------------------------------------------
# The Role of Hyperparameters in Machine Learning Models and How to Tune Them (PSRM, 2023)
# Christian Arnold, Luka Biedebach, Andreas Küpfer, and Marcel Neunhoeffer
#--------------------------------------------------------------------------------

#install.packages("pacman")
pacman::p_load(here,
               lme4,
               car)

setwd(here::here())

# Table 1

# -- Data ---------------------------------------------------------------------- 
dat.temp <- read.csv("replication/data/annotations_march22_2023.csv", sep = ',', header = TRUE)
dat <- subset(dat.temp, 
  subset = dat.temp$Does.the.paper.use.machine.learning.in.the.sense.of.our.definition. == TRUE)

# Recoding some data 
dat$journal <- car::recode(dat$journal, "
            'American Political Science Review' = 'APSR';
            'Political Analysis' = 'PA';
            'Political Science Research and Methods' = 'PSRM'
            ")

# Table for Main Paper
table(dat$model.replicability, dat$tuning.replicability)
round(prop.table(table(dat$model.replicability, dat$tuning.replicability)), 4)*100

