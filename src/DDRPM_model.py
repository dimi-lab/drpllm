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
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from optuna.samplers import TPESampler
import optuna

# Define the neural network for regression
class DeepNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(DeepNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        
        self.output_layer = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)

# Prepare data for training
def prepare_data_for_model(data_df, target_column='AUC', test_size=0.2, val_size=0.2, random_state=42):
    data_only_df = data_df.drop([target_column, 'label', 'cancer_type'], axis=1)
    X = data_only_df.values
    Y = data_df[target_column].values

    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=val_relative_size, random_state=random_state)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

# Train and evaluate a linear regression model
def train_and_evaluate_linear_regression(X_train, X_val, X_test, Y_train, Y_val, Y_test):
    linreg_model = LinearRegression()
    linreg_model.fit(X_train, Y_train)
    y_pred_val = linreg_model.predict(X_val)
    val_mse = mean_squared_error(Y_val, y_pred_val)
    val_r2 = r2_score(Y_val, y_pred_val)
    
    y_pred_test = linreg_model.predict(X_test)
    test_mse = mean_squared_error(Y_test, y_pred_test)
    test_r2 = r2_score(Y_test, y_pred_test)

    return {
        'Validation MSE': val_mse,
        'Validation R² Score': val_r2,
        'Test MSE': test_mse,
        'Test R² Score': test_r2
    }

# Train and evaluate an XGBoost model
def train_and_evaluate_xgboost(X_train, X_val, X_test, Y_train, Y_val, Y_test):
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgboost_model.fit(X_train, Y_train)
    
    y_pred_val = xgboost_model.predict(X_val)
    val_mse = mean_squared_error(Y_val, y_pred_val)
    val_r2 = r2_score(Y_val, y_pred_val)
    
    y_pred_test = xgboost_model.predict(X_test)
    test_mse = mean_squared_error(Y_test, y_pred_test)
    test_r2 = r2_score(Y_test, y_pred_test)

    return {
        'Validation MSE': val_mse,
        'Validation R² Score': val_r2,
        'Test MSE': test_mse,
        'Test R² Score': test_r2
    }

# Train and evaluate a simple MLP regressor
def train_and_evaluate_mlp(X_train, X_val, X_test, Y_train, Y_val, Y_test):
    mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    mlp_model.fit(X_train, Y_train)

    y_pred_val = mlp_model.predict(X_val)
    val_mse = mean_squared_error(Y_val, y_pred_val)
    val_r2 = r2_score(Y_val, y_pred_val)

    y_pred_test = mlp_model.predict(X_test)
    test_mse = mean_squared_error(Y_test, y_pred_test)
    test_r2 = r2_score(Y_test, y_pred_test)

    return {
        'Validation MSE': val_mse,
        'Validation R² Score': val_r2,
        'Test MSE': test_mse,
        'Test R² Score': test_r2
    }

# Train and evaluate the custom regression head
def run_regression_head(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=128, num_epochs=500, learning_rate=0.001, hidden_dims=[512, 128, 64], early_stop_patience=30):
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = DeepNN(input_dim, hidden_dims, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    patience_counter = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        model.eval()
        val_running_loss = 0.0
        all_val_preds = []
        all_val_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss = criterion(outputs.squeeze(), batch_y)
                val_running_loss += val_loss.item()
                all_val_preds.append(outputs.cpu().numpy())
                all_val_targets.append(batch_y.cpu().numpy())

        avg_val_loss = val_running_loss / len(val_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

    return {
        'Validation MSE': avg_val_loss,
        'Model': model
    }

# Main function
def main(args):
    # Load data
    data_query = pd.read_csv(args.input)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data_for_model(data_query)

    # Choose which model to run
    if args.model == 'linear':
        results = train_and_evaluate_linear_regression(X_train, X_val, X_test, Y_train, Y_val, Y_test)
    elif args.model == 'xgboost':
        results = train_and_evaluate_xgboost(X_train, X_val, X_test, Y_train, Y_val, Y_test)
    elif args.model == 'mlp':
        results = train_and_evaluate_mlp(X_train, X_val, X_test, Y_train, Y_val, Y_test)
    else:
        results = run_regression_head(X_train, X_val, X_test, Y_train, Y_val, Y_test)

    print(f"Results: {results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train regression models.")
    parser.add_argument('--input', type=str, required=True, help="Input dataset (CSV).")
    parser.add_argument('--model', type=str, choices=['linear', 'xgboost', 'mlp', 'custom'], required=True, help="Which model to run.")
    args = parser.parse_args()
    main(args)
