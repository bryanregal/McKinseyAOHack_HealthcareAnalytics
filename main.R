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
library(lime)
library(tidyquant)
library(rsample)
library(recipes)
library(yardstick)
library(corrr)
library(caret)

#--------------------------------------------------
#--- 1. Loading the datasets
#--------------------------------------------------
train_df <- read.csv(paste(mpath, spath, dpath, "train.csv", sep = "/"), header = TRUE, stringsAsFactors = FALSE)
test_df <- read.csv(paste(mpath, spath, dpath, "test.csv", sep = "/"), header = TRUE, stringsAsFactors = FALSE)

#--------------------------------------------------
#--- 2. Preparing the data
#--------------------------------------------------

# [1] "id"                "gender"            "age"               "hypertension"      "heart_disease"     "ever_married"     
# [7] "work_type"         "Residence_type"    "avg_glucose_level" "bmi"               "smoking_status"    "stroke" 

# Create recipe
rec_obj <- recipe(stroke ~ .,
                  data = train_df %>%
                    select(-id)) %>%
  add_role(stroke, new_role = "outcome") %>%
  step_discretize(age, options = list(cuts = 5)) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep(data = train_df)

# Predictors
x_train <- bake(rec_obj, newdata = train_df)
x_train[is.na(x_train)] <- 0
x_test  <- bake(rec_obj, newdata = test_df)
x_test[is.na(x_test)] <- 0

# Response
y_train_vec <- x_train$stroke
x_train$stroke <- NULL

#--------------------------------------------------
#--- 3. Building the network
#--------------------------------------------------

#--- 3.1 Defining the model
model_keras <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(ncol(x_train))) %>% 
  # layer_dropout(rate = 0.1) %>%
  layer_dense(units = 16, activation = "relu") %>% 
  # layer_dropout(rate = 0.1) %>%
  layer_dense(units = 16, activation = "relu") %>% 
  # layer_dropout(rate = 0.1) %>%
  layer_dense(units = 1, activation = "sigmoid")

model_keras

#--- 3.2 Compiling the model
model <- model_keras %>% 
  compile(optimizer = "rmsprop",
          loss = "binary_crossentropy",
          metrics = c("accuracy"))

#--------------------------------------------------
#--- 4. Model Build and Approach Validation
#--------------------------------------------------

#--- 4.1 Setting aside a 10% validation set
set.seed(2018)
val_indices <-sample(1:nrow(x_train), ceiling(nrow(x_train)*.2))
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train_vec[val_indices]
partial_y_train <- y_train_vec[-val_indices]

#--- 4.2 First model training to identify epochs
history <- model %>% 
  fit(as.matrix(partial_x_train), as.matrix(partial_y_train), 
      epochs = 30, batch_size = 1000,
      # class_weight = list("0" = 55,"1" = 1),
      validation_data = list(as.matrix(x_val), as.matrix(y_val)))

#--- 4.3 Re-training model with explicit number of epochs (output from 4.2)
results <- model %>% evaluate(as.matrix(x_val), as.matrix(y_val))


k <- 10
set.seed(2018)
indices <- sample(1:nrow(x_train))
folds <- cut(indices, breaks = k, labels = FALSE)
num_epochs <- 100
all_scores <- c()
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- as.matrix(x_train[val_indices,])
  val_targets <- as.matrix(y_train_vec[val_indices])
  partial_train_data <- as.matrix(x_train[-val_indices,])
  partial_train_targets <- as.matrix(y_train_vec[-val_indices])
  model %>% fit(partial_train_data, partial_train_targets,
                class_weight = list("0" = 55,"1" = 1),
                epochs = num_epochs, batch_size = 1000, verbose = 0)
  
  results <- model %>% evaluate(val_data, val_targets, verbose = 0)
  all_scores <- c(all_scores, results$acc)
}
mean(all_scores)

#--------------------------------------------------
#--- 5. Generate predictions on new data (test set)
#--------------------------------------------------

# test_df

# Predicted Class Probability
test_df$stroke  <- predict_proba(object = model, x = as.matrix(x_test)) %>%
  as.vector() 

test_df %>%
  select(id, stroke) %>%
  write.csv("submission.csv", row.names = FALSE)
