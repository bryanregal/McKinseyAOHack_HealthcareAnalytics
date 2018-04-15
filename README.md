# McKinseyAOHack_HealthcareAnalytics
Healthcare Analytics, McKinsey Analytics Online Hackathon

This repository contains my solution to a 24-hour McKinley Analytics online hackathon organized by Analytics Vidhya on 14 April 2018. The complete details for this hackathon is available at: https://datahack.analyticsvidhya.com/contest/mckinsey-analytics-online-hackathon/


#### Tools and Software
For this hackathon, I've used R (3.4.4) via RStudio (1.0.153) on my ever reliable 2013 Macbook Pro (2 GHz Intel Core i5, 8GB RAM).

Here's a list R libraries I'ved used in the process:
library(dplyr) - for easy data manipulation
library(recipes) - for its high-level functions for data preprocessing and normalization
library(rpart) - for missing data imputation via Decision Tree algorithm
library(DMwR) - for SMOTE(Synthetic Minority Over-sampling Technique) sampling to handle data imbalance 
library(h2o) - for running XGBOOST with parameter grid search and model stack ensembling



