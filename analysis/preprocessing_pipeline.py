{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}

import pandas as pd
import SimpleImputer from sklearn.impute
import Pipeline from sklearn.pipeline
import StandardScaler from sklearn.preprocessing
import os
def main():
    path = os.path.join("data", "train", "housing_train.csv")
    housing_data = pd.read_csv(path)
    
    
    pipelines = Pipeline([("scaler", StandardScaler(), 
                          "imputer", SimpleImputer())])
    
    housing_pipelines = pipelines.fit_transform(housing_data)
    
    processPath = os.path.join("data", "train", "housing_train_processed.csv")
    pd.DataFrame(housing_pipelines).to_csv(processPath, index=False)
if __name__ == "main":
    main()
