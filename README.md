# McKinseyAOHack_HealthcareAnalytics
Healthcare Analytics, McKinsey Analytics Online Hackathon

This repository contains my final solution to a 24-hour McKinley Analytics online hackathon organized by Analytics Vidhya on 14 April 2018. My final solution managed to get an AUC score of 84.72 on the public LB and 84.50 on the private. The complete details for this hackathon is available at: https://datahack.analyticsvidhya.com/contest/mckinsey-analytics-online-hackathon/

For this hackathon, I've used R (3.4.4) via RStudio (1.0.153) on my ever reliable 2016 Macbook Pro (2 GHz Intel Core i5, 8GB RAM).

As for my approach, I've used SMOTE sampling technique to handle heavy class imbalance in the dataset and then XGBoost with hyperparameter tuning via the grid search for model runs. Finally, the top 3 best performing models from the validation were stack ensembled to predict the class probability for each data row in the test set. Here's a list R libraries I'ved used in the process:
###### library(dplyr) - for easy data manipulation
###### library(recipes) - for its high-level functions for data preprocessing and normalization
###### library(rpart) - for missing data imputation via Decision Tree algorithm
###### library(DMwR) - for SMOTE(Synthetic Minority Over-sampling Technique) sampling to handle data imbalance 
###### library(h2o) - for running XGBOOST with parameter grid search and model stack ensembling



