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
library(recipes)

#--------------------------------------------------
#--- 1. Loading the datasets
#--------------------------------------------------
train_df <- read.csv(paste(mpath, spath, dpath, "train.csv", sep = "/"), header = TRUE, stringsAsFactors = FALSE)
test_df <- read.csv(paste(mpath, spath, dpath, "test.csv", sep = "/"), header = TRUE, stringsAsFactors = FALSE)

#--------------------------------------------------
#--- 2. Preparing the data
#--------------------------------------------------
df <- train_df %>%
  rbind(test_df %>%
          mutate(stroke = NA))

# Impute BMI via decision tree
library(rpart)
set.seed(2018)
bmi.i <- rpart(bmi ~ gender + age + hypertension + heart_disease + ever_married + work_type + Residence_type + avg_glucose_level +
                 smoking_status, data = df[!is.na(df$bmi),], method = "anova")
df$bmi[is.na(df$bmi)] <- predict(bmi.i, df[is.na(df$bmi),])

df <- df %>%
  mutate(smoking_status = ifelse(smoking_status == "", "unknown", smoking_status),
         gender = ifelse(gender == "Other", "Female", gender),
         bmi_orig = bmi,
         age_orig = age,
         avg_glucose_level_orig = avg_glucose_level,
         age_grp = cut(age_orig, breaks = seq(0,90,5)),
         bmi_grp = ifelse(bmi_orig < 18.5, 1,
                          ifelse(bmi_orig >= 18.5 & bmi_orig <= 24.9, 2, 
                                 ifelse(bmi_orig > 24.9 & bmi_orig <= 29.9, 3, 
                                        ifelse(bmi_orig > 29.9 & bmi_orig <= 34.9, 4, 5)))),
         gen_bmi = paste(gender, bmi_grp, sep = "_"),
         gen_age = paste(gender, age_grp, sep = "_"),
         bmi_age = paste(bmi_grp, age_grp, sep = "_"),
         gen_sm = paste(gender, smoking_status, sep = "_"),
         age_sm = paste(age_grp, smoking_status, sep = "_"),
         gsa = paste(gender, age_grp, smoking_status, sep = "_"),
         age_he = paste(age_grp, heart_disease, paste = "_"),
         age_hy = paste(age_grp, hypertension, paste = "_"),
         he_hy = paste(heart_disease, hypertension, paste = "_"),
         gen_he = paste(gender, heart_disease, paste = "_"),
         gen_hy = paste(gender, hypertension, paste = "_"),
         gen_age_he = paste(gender, age_grp, heart_disease, sep = "_"),
         gen_age_hy = paste(gender, age_grp, hypertension, sep = "_"),
         gen_age_he_hy = paste0(gender, age, heart_disease, hypertension, sep = "_"),
         gba = paste(bmi_grp, age_grp, sep = "_"))


train_df <- df %>%
  filter(!is.na(stroke))

test_df <- df %>%
  filter(is.na(stroke))

# # Create recipe
rec_obj <- recipe(stroke ~ .,
                  data = df %>%
                    select(-id)) %>%
  add_role(stroke, new_role = "outcome") %>%
  step_discretize(age, options = list(cuts = 8)) %>%
  step_discretize(bmi, options = list(cuts = 5)) %>%
  step_discretize(avg_glucose_level, options = list(cuts = 20)) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  prep(data = df)

# # Predictors
x_train <- bake(rec_obj, newdata = train_df)
x_train[is.na(x_train)] <- 0
x_test  <- bake(rec_obj, newdata = test_df)
x_test[is.na(x_test)] <- 0

# Response
# y_train_vec <- x_train$stroke
# x_train$stroke <- NULL
# x_test$stroke <- NULL

# x_train <- train_df
# x_test <- test_df
# x_train$id <- NULL
# x_test$id <- NULL

library(DMwR)
# SMOTE(Synthetic Minority Over-sampling Technique) Sampling
x_train$stroke <- as.factor(x_train$stroke)
x_train <- data.frame(x_train)
x_train <- SMOTE(stroke ~ ., data = x_train, perc.over = 1000, perc.under=500, seed = 1)

rm(df)
invisible(gc())

library(h2o)
h2o.init(nthreads = -1)

# Identify predictors and response
y <- "stroke"
x <- setdiff(names(x_train), y)

# For binary classification, response should be a factor
x_train$stroke <- as.factor(x_train$stroke)
x_test$stroke <- as.factor(x_test$stroke)

x_train <- as.h2o(x_train)
x_test <- as.h2o(x_test)

# 
# # --- XGBOOST PARAMETER GRID SEARCH
# # Some XGboost/GBM hyperparameters
# hyper_params <- list(ntrees = seq(10, 1000, 1),
#                      learn_rate = seq(0.0001, 0.2, 0.0001),
#                      max_depth = seq(1, 20, 1),
#                      sample_rate = seq(0.5, 1.0, 0.0001),
#                      col_sample_rate = seq(0.2, 1.0, 0.0001))
# 
# search_criteria <- list(strategy = "RandomDiscrete",
#                         max_models = 10,
#                         seed = 1)
# 
# # Train the grid
# xgb_grid <- h2o.grid(algorithm = "xgboost",
#                      x = x, y = y,
#                      training_frame = x_train,
#                      nfolds = 5,
#                      seed = 1,
#                      hyper_params = hyper_params,
#                      search_criteria = search_criteria)
# 
# # Sort the grid by CV AUC
# grid <- h2o.getGrid(grid_id = xgb_grid@grid_id, sort_by = "AUC", decreasing = TRUE)
# grid_top_model <- grid@summary_table[1, "model_ids"]
# 


#--- MODEL ENSEMBLE

# Number of CV folds
nfolds <- 5

# Train & Cross-validate best params from grid search
my_xgb1 <- h2o.xgboost(x = x,
                       y = y,
                       training_frame = x_train,
                       distribution = "bernoulli",
                       ntrees = 419,
                       max_depth = 2,
                       min_rows = 1,
                       learn_rate = 0.1102,
                       col_sample_rate = 0.3475,
                       sample_rate = 0.779,
                       nfolds = nfolds,
                       fold_assignment = "Stratified",
                       keep_cross_validation_predictions = TRUE,
                       seed = 1)
invisible(gc())
# Train & Cross-validate another XGB-GBM (2nd best)
my_xgb2 <- h2o.xgboost(x = x,
                       y = y,
                       training_frame = x_train,
                       distribution = "bernoulli",
                       ntrees = 19,
                       max_depth = 7,
                       min_rows = 1,
                       learn_rate = 0.0152,
                       sample_rate = 0.501,
                       col_sample_rate = 0.2288,
                       nfolds = nfolds,
                       fold_assignment = "Stratified",
                       keep_cross_validation_predictions = TRUE,
                       seed = 1)

invisible(gc())

# Train & Cross-validate another XGB-GBM (3rd best)
my_xgb3 <- h2o.xgboost(x = x,
                       y = y,
                       training_frame = x_train,
                       distribution = "bernoulli",
                       ntrees = 407,
                       max_depth = 2,
                       min_rows = 1,
                       learn_rate = 0.1563,
                       sample_rate = 0.7349,
                       col_sample_rate = 0.5737,
                       nfolds = nfolds,
                       fold_assignment = "Stratified",
                       keep_cross_validation_predictions = TRUE,
                       seed = 1)
invisible(gc())
# # Train & Cross-validate another (deeper) XGB-GBM
# my_xgb2 <- h2o.xgboost(x = x,
#                        y = y,
#                        training_frame = x_train,
#                        distribution = "bernoulli",
#                        ntrees = 501,
#                        max_depth = 8,
#                        min_rows = 1,
#                        learn_rate = 0.1,
#                        sample_rate = 0.7,
#                        col_sample_rate = 0.9,
#                        nfolds = nfolds,
#                        fold_assignment = "Stratified",
#                        keep_cross_validation_predictions = TRUE,
#                        seed = 1)
# 
# # Train & Cross-validate a DNN
# my_dl <- h2o.deeplearning(x = x,
#                           y = y,
#                           training_frame = x_train,
#                           hidden = c(16, 16, 16),
#                           nfolds = nfolds,
#                           fold_assignment = "Stratified",
#                           keep_cross_validation_predictions = TRUE,
#                           seed = 1)

# Train a stacked ensemble using the H2O and XGBoost models from above
base_models <- list(my_xgb1@model_id, my_xgb2@model_id, my_xgb3@model_id)

ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = x_train,
                                base_models = base_models)

# Eval ensemble performance on the full train set
perf <- h2o.performance(ensemble, newdata = x_train)

rm(x_train)
invisible(gc())

# Generate predictions on a test set
# Predicted Class Probability
pred  <- as.data.frame(h2o.predict(ensemble, newdata = x_test)) 
test_df$stroke <- pred$p1

test_df %>%
  select(id, stroke) %>%
  write.csv("submission.csv", row.names = FALSE)


# 
# #--------------------------------------------------
# #--- 3. Building the network
# #--------------------------------------------------
# 
# #--- 3.1 Defining the model
# model_keras <- keras_model_sequential() %>% 
#   layer_dense(units = 32, activation = "relu", input_shape = c(ncol(x_train))) %>% 
#   layer_dropout(rate = 0.1) %>%
#   layer_dense(units = 16, activation = "relu") %>%
#   layer_dropout(rate = 0.1) %>%
#   layer_dense(units = 16, activation = "relu") %>% 
#   layer_dropout(rate = 0.1) %>%
#   layer_dense(units = 16, activation = "relu") %>% 
#   layer_dropout(rate = 0.1) %>%
#   layer_dense(units = 1, activation = "sigmoid")
# 
# model_keras
# 
# #--- 3.2 Compiling the model
# model <- model_keras %>% 
#   compile(optimizer = "adam",
#           loss = "binary_crossentropy",
#           metrics = c("accuracy"))
# 
# #--------------------------------------------------
# #--- 4. Model Build and Approach Validation
# #--------------------------------------------------
# 
# # #--- 4.1 Setting aside a 10% validation set
# # set.seed(2018)
# # val_indices <-sample(1:nrow(x_train), ceiling(nrow(x_train)*.2))
# # x_val <- x_train[val_indices,]
# # partial_x_train <- x_train[-val_indices,]
# # y_val <- y_train_vec[val_indices]
# # partial_y_train <- y_train_vec[-val_indices]
# # 
# # #--- 4.2 First model training to identify epochs
# # history <- model %>% 
# #   fit(as.matrix(partial_x_train), as.matrix(partial_y_train), 
# #       epochs = 30, batch_size = 1000,
# #       # class_weight = list("0" = 55,"1" = 1),
# #       validation_data = list(as.matrix(x_val), as.matrix(y_val)))
# # 
# # #--- 4.3 Re-training model with explicit number of epochs (output from 4.2)
# # results <- model %>% evaluate(as.matrix(x_val), as.matrix(y_val))
# 
# k <- 5
# set.seed(2018)
# indices <- sample(1:nrow(x_train))
# folds <- cut(indices, breaks = k, labels = FALSE)
# num_epochs <- 30
# all_scores <- c()
# for (i in 1:k) {
#   cat("processing fold #", i, "\n")
#   val_indices <- which(folds == i, arr.ind = TRUE)
#   val_data <- as.matrix(x_train[val_indices,])
#   val_targets <- as.matrix(y_train_vec[val_indices])
#   partial_train_data <- as.matrix(x_train[-val_indices,])
#   partial_train_targets <- as.matrix(y_train_vec[-val_indices])
#   model %>% fit(partial_train_data, partial_train_targets,
#                 class_weight = list("0" = 50,"1" = 1),
#                 epochs = num_epochs, batch_size = 1000, verbose = 0)
#   
#   results <- model %>% evaluate(val_data, val_targets, verbose = 0)
#   all_scores <- c(all_scores, results$acc)
# }
# 
# all_scores
# mean(all_scores)
# sd(all_scores)
# 
# #--------------------------------------------------
# #--- 5. Generate predictions on new data (test set)
# #--------------------------------------------------
# 
# # test_df
# 
# # Predicted Class Probability
# test_df$stroke  <- predict_proba(object = model, x = as.matrix(x_test)) %>%
#   as.vector() 
# 
# test_df %>%
#   select(id, stroke) %>%
#   write.csv("submission.csv", row.names = FALSE)
