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
from DRPLLM_model import train_and_evaluate_linear_regression, train_and_evaluate_xgboost, train_and_evaluate_mlp, run_regression_head

# Prepare data for training
def prepare_data_for_model(data_df, target_column='AUC', test_size=0.2, val_size=0.2, random_state=42):
    meta_columns = ['AUC', 'label', 'cancer_type','cell_line_name', 'drug_name',
                    'Tissue', 'Tissue_sub_type'] #AUC,label,cancer_type,cell_line_name,drug_name                                            
    meta_data = data_df[meta_columns].copy()
    data_only_df = data_df.drop([target_column, 'label', 'cancer_type',
                                 'cell_line_name', 'drug_name', 'Tissue',
                                 'Tissue_sub_type'], axis=1)
    X = data_only_df.values
    Y = data_df[target_column].values

    X_train_val, X_test, Y_train_val, Y_test, meta_train_val, meta_test = train_test_split(X, Y, meta_data, test_size=test_size, random_state=random_state)

    val_relative_size = val_size / (1 - test_size)                   
    X_train, X_val, Y_train, Y_val, meta_train, meta_val = train_test_split(
        X_train_val, Y_train_val, meta_train_val, test_size=val_relative_size,
        random_state=random_state)
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, meta_test, meta_val


def prepare_and_save_splits(data_df, target_column='AUC', test_size=0.2,
                            val_size=0.2, random_state=42):
    meta_columns = ['AUC', 'label', 'cancer_type','cell_line_name', 'drug_name', 'Tissue', 'Tissue_sub_type'] #AUC,label,cancer_type,cell_line_name,drug_name
    meta_data = data_df[meta_columns].copy()
    data_only_df = data_df.drop(columns=meta_columns)
    X = data_only_df.values
    Y = data_df[target_column].values
    X_train_val, X_test, Y_train_val, Y_test, meta_train_val, meta_test = train_test_split(
        X, Y, meta_data, test_size=test_size, random_state=random_state
    )

    val_relative_size = val_size / (1 - test_size)  # Adjust validation size relative to train+val size
    X_train, X_val, Y_train, Y_val, meta_train, meta_val = train_test_split(
        X_train_val, Y_train_val, meta_train_val, test_size=val_relative_size, random_state=random_state
    )
#    train_file = "train_metadata.tsv"
#    val_file = "val_metadata.tsv"
#    test_file = "test_metadata.tsv"

#    meta_train.to_csv(train_file, sep='\t', index=False)
#    meta_val.to_csv(val_file, sep='\t', index=False)
#    meta_test.to_csv(test_file, sep='\t', index=False)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, meta_test, meta_val


def main(args):
    # Load data
    data_query = pd.read_csv(args.input)
    analysis_type = args.atype
    tissue_type_list = ['blood', 'skin', 'lung', 'urogenital_system',
                        'digestive_system', 'nervous_system', 'aero_digestive_tract', 'breast']
#    tissue_type_list =  ['aero_digestive_tract', 'blood', 'bone',
#                         'breast', 'digestive_system', 'kidney', 'lung',
#                         'nervous_system', 'pancreas', 'skin', 'soft_tissue',
#                         'thyroid', 'urogenital_system']
    
    all_results_df = pd.DataFrame()
    for tissue in tissue_type_list:
        analysis_type = tissue + "_" +  analysis_type
        print(f'ignoring tissue type: {tissue}')
        data_query_tissue = data_query[data_query['Tissue'] != tissue]
        print(data_query_tissue.shape)
        X_train, X_val, X_test, Y_train, Y_val, Y_test, meta_test, meta_val = prepare_data_for_model(data_query_tissue)
        if args.model == 'all':
            models_to_run = ['linear', 'xgboost', 'mlp', 'custom']
        else:
            models_to_run = [args.model]

        # Loop through each model and run it
        for model_name in models_to_run:
            print(f'Running {model_name} dropping tissue type {tissue}')
            if model_name == 'linear':
                results = train_and_evaluate_linear_regression(X_train, X_val, X_test, Y_train, Y_val, Y_test)
            elif model_name == 'xgboost':
                results = train_and_evaluate_xgboost(X_train, X_val, X_test, Y_train, Y_val, Y_test)
            elif model_name == 'mlp':
                results = train_and_evaluate_mlp(X_train, X_val, X_test, Y_train, Y_val, Y_test)
            elif model_name == 'custom':
                results = run_regression_head(X_train, X_val, X_test, Y_train, Y_val, Y_test,meta_test, meta_val, analysis_type)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            results['Model'] = model_name
            results['drop_tissue'] = tissue  # Add the cancer type here

            results_df = pd.DataFrame([results])
            all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)


        output_file = args.output
        all_results_df.to_csv(output_file, index=False)
        
        print(f"Results saved to {output_file}")
        print(all_results_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train regression models.")
    parser.add_argument('--input', type=str, required=True, help="Input dataset (CSV).")
    parser.add_argument('--output', type=str, required=True, help="output filename")
    parser.add_argument('--atype', type=str, required=True, help="type of analysis")        
    parser.add_argument('--model', type=str, choices=['linear', 'xgboost', 'mlp', 'custom', 'all'], required=True, help="Which model to run.")
    args = parser.parse_args()
    main(args)
