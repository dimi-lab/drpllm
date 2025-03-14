import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
from tqdm import tqdm 
import os

def load_data(file_path):
    df = pd.read_feather(file_path)
    df = df.groupby(['AUC', 'CELL_LINE_NAME', 'DRUG_NAME', 'Site']).first().reset_index()
    return df


def split_by_cancer_type(data_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    train_list, val_list, test_list = [], [], []
    for cancer_type, group in data_df.groupby('cancer_type'):
        if len(group) < 10:
            continue
        train, temp = train_test_split(group, test_size=(val_ratio + test_ratio), random_state=random_state, stratify=group['cancer_type'])
        val, test = train_test_split(temp, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=random_state, stratify=temp['cancer_type'])
        train_list.append(train)
        val_list.append(val)
        test_list.append(test)
    return pd.concat(train_list), pd.concat(val_list), pd.concat(test_list)


def remove_low_variance_and_high_corr_features(X, variance_threshold=0.01, correlation_threshold=0.9):
    print("Applying feature selection...")
    selector = VarianceThreshold(threshold=variance_threshold)
    X_var_filtered = selector.fit_transform(X)
    selected_columns = X.columns[selector.get_support()]
    X_filtered_df = pd.DataFrame(X_var_filtered, columns=selected_columns)
#    correlation_matrix = X_filtered_df.corr(method='spearman').abs()
#    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
#    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > correlation_threshold)]
    return X_filtered_df


def bootstrap_spearman(y_true, y_pred, num_bootstrap_samples=1000, confidence_level=0.95):
    spearman_bootstraps = []
    n = len(y_true)

    for _ in range(num_bootstrap_samples):
        indices = np.random.choice(n, n, replace=True)  # Sample with replacement
        spearman_bootstraps.append(spearmanr(y_true[indices], y_pred[indices])[0])

    lower_bound = np.percentile(spearman_bootstraps, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(spearman_bootstraps, (1 + confidence_level) / 2 * 100)
    return lower_bound, upper_bound

def train_model(model, X_train_scaled, X_val_scaled, X_test_scaled, Y_train, Y_val, Y_test, cancer_type_test, model_name):
    print(f"Training {model_name}...")
    model.fit(X_train_scaled, Y_train)
    y_pred = model.predict(X_test_scaled)
    y_test_array = Y_test.to_numpy().flatten()
    spearman = spearmanr(y_test_array, y_pred)[0]
    lower_ci, upper_ci = bootstrap_spearman(y_test_array, y_pred)

    print(f"Spearman Correlation for {model_name}: {spearman:.4f}")
    print(f"95% Confidence Interval: ({lower_ci:.4f}, {upper_ci:.4f})")
    test_data_df = cancer_type_test.copy().reset_index(drop=True)
    test_data_df[f'{model_name}_pred'] = y_pred
    test_data_df['AUC_actual'] = Y_test.reset_index(drop=True)
    test_data_df.dropna(subset=['AUC_actual', f'{model_name}_pred'], inplace=True)
    spearman_per_cancer = (
        test_data_df.groupby('cancer_type')
        .apply(lambda df: spearmanr(df['AUC_actual'], df[f'{model_name}_pred'])[0] if len(df) > 1 else np.nan)
        .reset_index()
        .rename(columns={0: "Spearman_Correlation"})
    )
    spearman_per_cancer.to_csv(f"spearman_{model_name}_per_cancer.csv", index=False)    
    return spearman_per_cancer


class DeepNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2): 
        super(DeepNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() 
        self.dropouts = nn.ModuleList() 
        
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(in_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim)) 
            self.dropouts.append(nn.Dropout(p=dropout_rate)) 
            in_dim = hidden_dim
        
        self.output_layer = nn.Linear(in_dim, output_dim)
        self.elu = nn.ELU(alpha=1.0)

    def forward(self, x):
        for layer, batch_norm, dropout in zip(self.hidden_layers, self.batch_norms, self.dropouts):  
            x = self.elu(batch_norm(layer(x)))
            x = dropout(x)
        return self.output_layer(x)
    

def run_regression_head(X_train, X_val, X_test, y_train, y_val, y_test, cancer_type_test,
                        batch_size=128, num_epochs=500, dropout_rate=0.2, 
                        learning_rate=0.001, hidden_dims=[512, 128, 64],
                        early_stop_patience=30, model='DNN'): 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_train = y_train.values.reshape(-1, 1) if isinstance(y_train, pd.Series) else y_train.reshape(-1, 1)
    y_val = y_val.values.reshape(-1, 1) if isinstance(y_val, pd.Series) else y_val.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1) if isinstance(y_test, pd.Series) else y_test.reshape(-1, 1)
    
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
    output_dim = 1                

    nn_model = DeepNN(input_dim, hidden_dims, output_dim)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate)

    patience = early_stop_patience
    best_loss = float('inf')
    epochs_since_best = 0

    patience = early_stop_patience
    best_loss = float('inf')
    epochs_since_best = 0

    train_losses = []
    val_losses = []

    log_file = "training_log.txt"
    
    for epoch in range(num_epochs):
        nn_model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = nn_model(batch_x)

            loss = criterion(outputs.squeeze(),  batch_y.squeeze())
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        nn_model.eval()
        with torch.no_grad():
            val_running_loss = 0.0
            all_val_preds = []
            all_val_targets = []
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = nn_model(batch_x)
                val_loss = criterion(outputs.squeeze(), batch_y.squeeze())
                val_running_loss += val_loss.item()
                
                all_val_preds.append(outputs.cpu().numpy())
                all_val_targets.append(batch_y.cpu().numpy())
            
            avg_val_loss = val_running_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            all_val_preds = np.concatenate(all_val_preds, axis=0)
            all_val_targets = np.concatenate(all_val_targets, axis=0)
            val_r2 = r2_score(all_val_targets, all_val_preds)
            val_mse = mean_squared_error(all_val_targets, all_val_preds)
            val_spearman_corr, val_spearman_p = spearmanr(all_val_targets, all_val_preds)
            val_spearman, val_spearman_ci = bootstrap_spearman(all_val_targets, all_val_preds)
            
        log_message = (f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, "
              f"Validation R²: {val_r2:.4f}, "
              f"Validation MSE: {val_mse:.4f}"
              f"Validation Spearman Corr: {val_spearman_corr:.4f}"
              f"CI: {val_spearman_ci}")

        print(log_message)

        if not os.path.exists(log_file) or os.stat(log_file).st_size == 0:
            with open(log_file, "w") as f:
                f.write("Epoch\tTrain Loss\tValidation Loss\tValidation R²\tValidation MSE\tValidation Spea>rman Corr\tCI\n")

        log_message = f"{epoch+1}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\t{val_r2:.4f}\t{val_mse:.4f}\t{val_spearman_corr:.4f}\t{val_spearman_ci}\n"


        with open(log_file, "a") as f:
            f.write(log_message)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

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

    y_pred = np.concatenate(all_val_preds, axis=0)
    y_actual = np.concatenate(all_val_targets, axis=0)
    spearman_corr = spearmanr(y_actual, y_pred)[0]

    lower_ci, upper_ci = bootstrap_spearman(y_actual, y_pred)

    print(f"Spearman Correlation for DNN: {spearman_corr:.4f}")
    print(f"95% Confidence Interval: ({lower_ci:.4f}, {upper_ci:.4f})")
    
    # Restore 'cancer_type' for grouping
    test_data_df = cancer_type_test.copy().reset_index(drop=True)
    test_data_df['DNN_pred'] = all_test_preds
    test_data_df['AUC_actual'] = y_test

    # Compute Spearman per cancer type
    spearman_per_cancer = (
        test_data_df.groupby('cancer_type')
        .apply(lambda df: spearmanr(df['AUC_actual'], df['DNN_pred'])[0])
        .reset_index()
        .rename(columns={0: "Spearman_Correlation"})
    )

    spearman_per_cancer.to_csv("spearman_DNN_per_cancer.csv", index=False)
    print(f"Spearman Correlation for DNN: {spearman_corr:.4f}")
    
    return spearman_per_cancer

def main():
    data_df = load_data('CCLE_GDSCv2_metric_combined_data.feather')
    print(data_df.shape)
#    train_df, val_df, test_df = split_by_cancer_type(data_df)
    # Keep 'cancer_type' while dropping other categorical columns
    object_columns = [col for col in data_df.select_dtypes(include=['object']).columns if col != 'cancer_type']
    all_data_df = data_df.drop(columns=object_columns, axis=1)
    # Define features and target
    target_col = "AUC"
    feature_cols = [col for col in all_data_df.columns if col not in [target_col, 'cancer_type']]
    all_data_df = data_df.dropna(subset=[target_col])

    X = data_df[feature_cols]
    y = data_df[target_col]

    # Ensure 'cancer_type' is correctly aligned with X and y
    cancer_type_data = all_data_df[['cancer_type']].loc[X.index].reset_index(drop=True)
    
    X_filtered = remove_low_variance_and_high_corr_features(X)
    test_size=0.2
    val_size=0.2
    random_state=42
    X_train_val, X_test, Y_train_val, Y_test, cancer_type_train, cancer_type_test = train_test_split(
    X, y, cancer_type_data, test_size=test_size, random_state=random_state, stratify=cancer_type_data)

    val_relative_size = val_size / (1 - test_size)  # Adjust validation size relative to train+val size
    X_train, X_val, Y_train, Y_val, can_type_train, can_type_val = train_test_split(
        X_train_val, Y_train_val, cancer_type_train, test_size=val_relative_size,
        random_state=random_state, stratify=cancer_type_train
    )


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
#    run_regression_head(X_train_scaled, X_val_scaled, X_test_scaled, Y_train, Y_val, Y_test, cancer_type_test)
    
    models = {
        "XGBoost": xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
        "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500),
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)        
    }

    for name, model in models.items():
        train_model(model, X_train_scaled, X_val_scaled, X_test_scaled, Y_train, Y_val, Y_test, cancer_type_test, name)

#    run_deep_learning_model(X_train_scaled, X_test_scaled, y_train, y_test, cancer_type_test)

if __name__ == "__main__":
    main()
