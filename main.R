# clean slate
rm(list = ls())
invisible(gc())


#--- directory
mpath <- "/Users/bryanbalajadia/DataScience/Github_repos"
spath <- "McKinseyAOHack_HealthcareAnalytics"
dpath <- "data"

setwd(paste(mpath, spath, sep = "/"))

#--- Libraries
library(dplyr)
library(keras)

#--------------------------------------------------
#--- 1. Loading the datasets
#--------------------------------------------------
train_df <- read.csv(paste(mpath, spath, dpath, "train.csv", sep = "/"), header = TRUE, stringsAsFactors = FALSE)
test_df <- read.csv(paste(mpath, spath, dpath, "test.csv", sep = "/"), header = TRUE, stringsAsFactors = FALSE)

#--------------------------------------------------
#--- 2. Preparing the data
#--------------------------------------------------

#--------------------------------------------------
#--- 3. Building the network
#--------------------------------------------------

#--- 3.1 Defining the model
#--- 3.2 Compiling the model
#--- 3.3 Configuring the optimizer
#--- 3.1 Defining loss function and metrics

#--------------------------------------------------
#--- 4. Model Build and Approach Validation
#--------------------------------------------------

#--- 4.1 Setting aside a validation set
#--- 4.2 First model training to identify epochs
#--- 4.3 Re-training model with explicit number of epochs (output from 4.2)

#--------------------------------------------------
#--- 5. Generate predictions on new data (test set)
#--------------------------------------------------

