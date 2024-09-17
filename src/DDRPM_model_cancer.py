import os
import json
import gc
import torch
import pandas as pd
import numpy as np
import argparse
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim 
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr
from optuna.samplers import TPESampler
import optuna
from DDRPM_model import train_and_evaluate_linear_regression, train_and_evaluate_xgboost, train_and_evaluate_mlp, run_regression_head

# Prepare data for training
def prepare_data_for_model(data_df, target_column='AUC', test_size=0.2, val_size=0.2, random_state=42):
    data_only_df = data_df.drop([target_column, 'label', 'cancer_type'], axis=1)
    X = data_only_df.values
    Y = data_df[target_column].values

    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=val_relative_size, random_state=random_state)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def main(args):
    # Load data
    data_query = pd.read_csv(args.input)
    
    # List of cancer types to iterate through
    cancer_type_list = ['Lung Cancer', 'Breast Cancer', 'Skin Cancer',
                        'Neuroblastoma', 'Leukemia', 'Acute Myeloid Leukemia', 
                        'Brain Cancer', 'Pancreatic Adenocarcinoma', 
                        'Ovarian Cancer', 'Colon/Colorectal Cancer']
    
    # Initialize an empty DataFrame to store results from all models and cancer types
    all_results_df = pd.DataFrame()

    # Loop through each cancer type
    for cancer in cancer_type_list:
        print(f'Processing cancer type: {cancer}')
        # Filter data by the current cancer type
        data_query_cancer = data_query[data_query['cancer_type'] == cancer]
        
        # Prepare data for model training
        X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data_for_model(data_query_cancer)

        # Define the list of models to run based on the input argument
        if args.model == 'all':
            models_to_run = ['linear', 'xgboost', 'mlp', 'custom']
        else:
            models_to_run = [args.model]

        # Loop through each model and run it
        for model_name in models_to_run:
            print(f'Running {model_name} for {cancer}')
            if model_name == 'linear':
                results = train_and_evaluate_linear_regression(X_train, X_val, X_test, Y_train, Y_val, Y_test)
            elif model_name == 'xgboost':
                results = train_and_evaluate_xgboost(X_train, X_val, X_test, Y_train, Y_val, Y_test)
            elif model_name == 'mlp':
                results = train_and_evaluate_mlp(X_train, X_val, X_test, Y_train, Y_val, Y_test)
            elif model_name == 'custom':
                results = run_regression_head(X_train, X_val, X_test, Y_train, Y_val, Y_test)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Add the model name and cancer type to the results
            results['Model'] = model_name
            results['Cancer_Type'] = cancer  # Add the cancer type here

            # Convert the results dictionary to a DataFrame and append to all_results_df
            results_df = pd.DataFrame([results])
            all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train regression models.")
    parser.add_argument('--input', type=str, required=True, help="Input dataset (CSV).")
    parser.add_argument('--output', type=str, required=True, help="output filename")    
    parser.add_argument('--model', type=str, choices=['linear', 'xgboost', 'mlp', 'custom', 'all'], required=True, help="Which model to run.")
    args = parser.parse_args()
    main(args)
