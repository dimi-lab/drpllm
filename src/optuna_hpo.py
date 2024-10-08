import json
import joblib
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim 
from DDRPM_model import run_regression_head, DeepNN  

def prepare_data_for_model(data_df, target_column='AUC', test_size=0.1, val_size=0.1, random_state=42):
    """
    Prepares the data by splitting it into training, validation, and test sets.
    """
    # Step 1: Remove the target column and 'label' column to get feature data
    data_only_df = data_df.copy()
    data_only_df = data_only_df.drop([target_column, 'label', 'cancer_type'], axis=1)
    X = data_only_df.values
    Y = data_df[target_column].values

    # Step 2: Split into training+validation and test sets
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Step 3: Split training+validation set into training and validation sets
    val_relative_size = val_size / (1 - test_size)  # Adjust val_size to be relative to the training set
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

def run_optuna_hpo(X_train, X_val, X_test, y_train, y_val, y_test, 
                   n_trials=30, model_save_path='best_model.pkl', 
                   params_save_path='best_params.json', results_save_path='HPO_results.csv'):
    """
    Hyperparameter optimization using Optuna, optimized for maximizing the Validation Spearman correlation.
    Saves the best model, best hyperparameters, and trial results.
    """
    best_model = None  # Placeholder for best model during trials
    best_val_spearman = -float('inf')  # To track the best validation Spearman correlation
    trial_results = []  # To store results for each trial
    
    def objective(trial):
        # Suggest values for the hyperparameters
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        num_epochs = 300  # Reasonable number of epochs for convergence
        early_stop_patience = 20
        
        # Suggest architecture (hidden_dims)
        n_layers = trial.suggest_int('n_layers', 2, 5)
        hidden_dims = [trial.suggest_int(f"n_units_l{i}", 64, 1024) for i in range(n_layers)]
        
        # Train the model
        regression_head_results = run_regression_head(
            X_train, X_val, X_test, y_train, y_val, y_test,
            batch_size=batch_size, num_epochs=num_epochs, 
            learning_rate=learning_rate, hidden_dims=hidden_dims,
            early_stop_patience=early_stop_patience
        )
        
        # Collect trial result
        trial_results.append({
            'trial_number': trial.number,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_dims': hidden_dims,
            'val_r2_final': regression_head_results['Validation R² Score'],
            'mse_final': regression_head_results['Validation MSE'],
            'val_spearman': regression_head_results['Validation Spearman'],
            'test_r2_final': regression_head_results['Test R² Score'],
            'Test_Spearman': regression_head_results['Test Spearman'],
        })

        # Update best model if current trial has better Spearman correlation
        nonlocal best_model, best_val_spearman
        val_spearman = regression_head_results['Validation Spearman']
        if val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            best_model = regression_head_results['model']

        return val_spearman  # Optimize for Validation Spearman correlation

    # Create a study object and set direction to "maximize" for Spearman correlation
    study = optuna.create_study(direction="maximize", sampler=TPESampler())
    study.optimize(objective, n_trials=n_trials)

    if best_model is not None:
        torch.save(best_model.state_dict(), model_save_path)
        print(f"Best model saved as {model_save_path}")

    # Save best hyperparameters to a JSON file
    with open(params_save_path, 'w') as f:
        json.dump(study.best_params, f)
    print(f"Best hyperparameters saved to {params_save_path}")

    # Save the trial results to a DataFrame and write it to a CSV file
    df_results = pd.DataFrame(trial_results)
    df_results.to_csv(results_save_path, index=False)
    print(f"Trial results saved to {results_save_path}")

    print("Best Spearman Correlation: ", study.best_value)
    print("Best hyperparameters: ", study.best_params)
    
    return study.best_params, study.best_value, df_results



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Optuna Hyperparameter Optimization.")
    parser.add_argument('--input_data', type=str, required=True, help="Input path for the full dataset (CSV).")
    parser.add_argument('--target_column', type=str, required=True, help="Target column for the dataset.")
    parser.add_argument('--n_trials', type=int, default=50, help="Number of Optuna trials for optimization.")
    parser.add_argument('--model_save_path', type=str, default='best_model.pkl', help="Path to save the best model.")
    parser.add_argument('--params_save_path', type=str, default='best_params.json', help="Path to save the best parameters.")
    parser.add_argument('--results_save_path', type=str, default='HPO_results.csv', help="Path to save the trial results.")
    parser.add_argument('--test_size', type=float, default=0.1, help="Proportion of data to use for testing.")
    parser.add_argument('--val_size', type=float, default=0.1, help="Proportion of data to use for validation.")

    args = parser.parse_args()

    # Load your dataset
    data_df = pd.read_csv(args.input_data)

    # Prepare data using the `prepare_data_for_model` function
    X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data_for_model(
        data_df, target_column=args.target_column, test_size=args.test_size, val_size=args.val_size
    )

    # Run Optuna HPO
    run_optuna_hpo(X_train, X_val, X_test, Y_train, Y_val, Y_test,
                   n_trials=args.n_trials,
                   model_save_path=args.model_save_path,
                   params_save_path=args.params_save_path,
                   results_save_path=args.results_save_path)
