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

def bootstrap_spearman(y_true, y_pred, n_bootstrap=1000, alpha=0.05):
    """
    Compute bootstrap confidence interval for Spearman's rank correlation.
    
    Parameters:
    - y_true: Ground truth values.
    - y_pred: Predicted values.
    - n_bootstrap: Number of bootstrap samples.
    - alpha: Significance level (default 0.05 for 95% confidence interval).
    
    Returns:
    - spearman_corr: Spearman correlation coefficient.
    - conf_interval: Confidence interval (lower bound, upper bound).
    """
    # Calculate the original Spearman correlation
    spearman_corr, _ = spearmanr(y_true, y_pred)
    
    # Bootstrap sampling
    bootstrapped_corrs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(range(n), size=n, replace=True)
        y_true_resample = y_true[indices]
        y_pred_resample = y_pred[indices]
        
        # Calculate Spearman correlation for the resample
        boot_corr, _ = spearmanr(y_true_resample, y_pred_resample)
        bootstrapped_corrs.append(boot_corr)
    
    # Calculate the confidence interval from the bootstrap distribution
    lower_bound = np.percentile(bootstrapped_corrs, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrapped_corrs, 100 * (1 - alpha / 2))
    
    return spearman_corr, (lower_bound, upper_bound)


# Train and evaluate a linear regression model
def train_and_evaluate_linear_regression(X_train, X_val, X_test, Y_train, Y_val, Y_test, model='LinearRegression'):
    linreg_model = LinearRegression()
    linreg_model.fit(X_train, Y_train)
    y_pred_val = linreg_model.predict(X_val)
    val_mse = mean_squared_error(Y_val, y_pred_val)
    val_r2 = r2_score(Y_val, y_pred_val)
    val_spearman, val_spearman_p = spearmanr(Y_val, y_pred_val)
    val_spearman, val_spearman_ci = bootstrap_spearman(Y_val, y_pred_val)
    
    y_pred_test = linreg_model.predict(X_test)
    test_mse = mean_squared_error(Y_test, y_pred_test)
    test_r2 = r2_score(Y_test, y_pred_test)
    test_spearman, test_spearman_p  = spearmanr(Y_test, y_pred_test)
    test_spearman, test_spearman_ci = bootstrap_spearman(Y_test, y_pred_test)
    
    return {
        'Validation MSE': val_mse,
        'Validation R² Score': val_r2,
        'Validation Spearman': val_spearman,
        'Validation Spearman P-Value': val_spearman_p,
        'Validation Spearman CI': val_spearman_ci,
        'Test MSE': test_mse,
        'Test R² Score': test_r2,
        'Test Spearman':test_spearman,
        'Test Spearman P-Value': test_spearman_p,
        'Test Spearman CI': test_spearman_ci,
        'model':model
    }

# Train and evaluate an XGBoost model
def train_and_evaluate_xgboost(X_train, X_val, X_test, Y_train, Y_val, Y_test, model='XGboost'):
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgboost_model.fit(X_train, Y_train)
    
    y_pred_val = xgboost_model.predict(X_val)
    val_mse = mean_squared_error(Y_val, y_pred_val)
    val_r2 = r2_score(Y_val, y_pred_val)
    val_spearman_corr, val_spearman_p = spearmanr(Y_val, y_pred_val)
    val_spearman, val_spearman_ci = bootstrap_spearman(Y_val, y_pred_val)
    
    y_pred_test = xgboost_model.predict(X_test)
    test_mse = mean_squared_error(Y_test, y_pred_test)
    test_r2 = r2_score(Y_test, y_pred_test)
    test_spearman_corr, test_spearman_p = spearmanr(Y_test, y_pred_test)
    test_spearman, test_spearman_ci = bootstrap_spearman(Y_test, y_pred_test)
    
    return {
        'Validation MSE': val_mse,
        'Validation R² Score': val_r2,
        'Validation Spearman': val_spearman_corr,
        'Validation Spearman P-Value': val_spearman_p,
        'Validation Spearman CI': val_spearman_ci,
        'Test MSE': test_mse,
        'Test R² Score': test_r2,
        'Test Spearman': test_spearman_corr,
        'Test Spearman P-Value': test_spearman_p,
        'Test Spearman CI': test_spearman_ci,
        'model': model
    }

# Train and evaluate a simple MLP regressor
def train_and_evaluate_mlp(X_train, X_val, X_test, Y_train, Y_val, Y_test, model='MLP'):
    mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    mlp_model.fit(X_train, Y_train)

    y_pred_val = mlp_model.predict(X_val)
    val_mse = mean_squared_error(Y_val, y_pred_val)
    val_r2 = r2_score(Y_val, y_pred_val)
    val_spearman_corr, val_spearman_p = spearmanr(Y_val, y_pred_val)
    val_spearman, val_spearman_ci = bootstrap_spearman(Y_val, y_pred_val)
    
    y_pred_test = mlp_model.predict(X_test)
    test_mse = mean_squared_error(Y_test, y_pred_test)
    test_r2 = r2_score(Y_test, y_pred_test)
    test_spearman_corr, test_spearman_p = spearmanr(Y_test, y_pred_test)
    test_spearman, test_spearman_ci = bootstrap_spearman(Y_test, y_pred_test)
    
    return {
        'Validation MSE': val_mse,
        'Validation R² Score': val_r2,
        'Validation Spearman': val_spearman_corr,
        'Validation Spearman P-Value': val_spearman_p,
        'Validation Spearman CI': val_spearman_ci,
        'Test MSE': test_mse,
        'Test R² Score': test_r2,
        'Test Spearman': test_spearman_corr,
        'Test Spearman P-Value': test_spearman_p,
        'Test Spearman CI': test_spearman_ci,
        'model': model
    }



# Train and evaluate the custom regression head
def run_regression_head(X_train, X_val, X_test, y_train, y_val, y_test, 
                        batch_size=128, num_epochs=500, 
                        learning_rate=0.001, hidden_dims=[512, 128, 64],
                        early_stop_patience=30, model='DNN'):
    # Convert data to tensors
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create TensorDataset
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model initialization
    input_dim = X_train.shape[1]  
    output_dim = 1                

    nn_model = DeepNN(input_dim, hidden_dims, output_dim)
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate)

    patience = early_stop_patience
    best_loss = float('inf')
    epochs_since_best = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        nn_model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = nn_model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate on validation set
        nn_model.eval()
        with torch.no_grad():
            val_running_loss = 0.0
            all_val_preds = []
            all_val_targets = []
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = nn_model(batch_x)
                val_loss = criterion(outputs.squeeze(), batch_y)
                val_running_loss += val_loss.item()
                
                all_val_preds.append(outputs.cpu().numpy())
                all_val_targets.append(batch_y.cpu().numpy())
            
            avg_val_loss = val_running_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Compute validation R² score and MSE
            all_val_preds = np.concatenate(all_val_preds, axis=0)
            all_val_targets = np.concatenate(all_val_targets, axis=0)
            val_r2 = r2_score(all_val_targets, all_val_preds)
            val_mse = mean_squared_error(all_val_targets, all_val_preds)
            val_spearman_corr, val_spearman_p = spearmanr(all_val_targets, all_val_preds)
            val_spearman, val_spearman_ci = bootstrap_spearman(all_val_targets, all_val_preds)
            
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, "
              f"Validation R²: {val_r2:.4f}, "
              f"Validation MSE: {val_mse:.4f}"
              f"Validation Spearman Corr: {val_spearman_corr:.4f}"
              f"CI: {val_spearman_ci}")
        
        # Early Stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Final evaluation on the test set
    nn_model.eval()
    with torch.no_grad():
        all_test_preds = []
        all_test_targets = []
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = nn_model(batch_x)
            all_test_preds.append(outputs.cpu().numpy())
            all_test_targets.append(batch_y.cpu().numpy())
        
        all_test_preds = np.concatenate(all_test_preds, axis=0)
        all_test_targets = np.concatenate(all_test_targets, axis=0)

    test_r2_final = r2_score(all_test_targets, all_test_preds)
    test_mse_final = mean_squared_error(all_test_targets, all_test_preds)
    test_spearman_corr, test_spearman_p = spearmanr(all_test_targets, all_test_preds)
    test_spearman, test_spearman_ci = bootstrap_spearman(all_test_targets, all_test_preds)
    
    print(f"Final Test R² Score: {test_r2_final:.4f}, "
          f"Final Test MSE: {test_mse_final:.4f}, "
          f"Final Test Spearman Corr: {test_spearman_corr:.4f}")
    
    return {
        'Test R² Score': test_r2_final,
        'Test MSE': test_mse_final,
        'Test Spearman': test_spearman_corr,
        'Test Spearman P-Value': test_spearman_p,
        'Test Spearman CI': test_spearman_ci,
        'Validation R² Score': val_r2,
        'Validation MSE': val_mse,
        'Validation Spearman': val_spearman_corr,
        'Validation Spearman P-Value': val_spearman_p,
        'Validation Spearman CI': val_spearman_ci,
        'model': model
    }

def main(args):
    # Load data
    data_query = pd.read_csv(args.input)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data_for_model(data_query)

    # Initialize an empty DataFrame to store all results
    all_results_df = pd.DataFrame()

    # Check which models to run
    if args.model == 'all':
        models_to_run = ['linear', 'xgboost', 'mlp', 'custom']
    else:
        models_to_run = [args.model]

    # Loop over each model to train and evaluate
    for model_name in models_to_run:
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

        # Add the model name to the results
        results['Model'] = model_name

        # Convert the results dictionary to a DataFrame and concatenate it with the main results DataFrame
        results_df = pd.DataFrame([results])
        all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)

    # Write all results to a CSV file
    output_file = args.output
    all_results_df.to_csv(output_file, index=False)

    # Print the final results
    print(f"Results saved to {output_file}")
    print(all_results_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train regression models.")
    parser.add_argument('--input', type=str, required=True, help="Input dataset (CSV).")
    parser.add_argument('--output', type=str, required=True, help="output filename")    
    parser.add_argument('--model', type=str, choices=['linear', 'xgboost', 'mlp', 'custom', 'all'], required=True, help="Which model to run.")
    args = parser.parse_args()
    main(args)
