import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

def run_alsi_model():
    """
    Loads "Noisy_AlSi_MD_dataset.csv" (with 4% noise), 
    splits 70/30, and trains a KNeighborsRegressor model (k=25).
    
    This should target an R-squared score of 0.95-0.97.
    """
    
    dataset_filename = 'Noisy_AlSi_MD_dataset.csv'
    
    # --- 1. Load Large Dataset ---
    print(f"--- 1. Loading Noisy Aluminum-Silicon (Al-Si) MD Dataset ---")
    try:
        df = pd.read_csv(dataset_filename)
        print(f"Successfully loaded '{dataset_filename}' with {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: '{dataset_filename}' not found.")
        print("Please run the 'generate_noisy_alsi_dataset.py' script first.")
        return

    # --- 2. Define Features (X) and Labels (y) ---
    X = df[['Temperature (K)', 'Strain']]
    y = df[['Stress (GPa)', 'Youngs_Modulus (GPa)', 'Thermal_Conductivity (W_m_K)']]
    label_names = y.columns
    
    print(f"\nInput Features (X): {X.columns.tolist()}")
    print(f"Output Labels (y): {y.columns.tolist()}")

    # --- 3. Data Splitting (70% Train, 30% Test) ---
    print("\n--- 3. Splitting Data (70% Train, 30% Test) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=84)
    print(f"Training data points: {len(X_train)}")
    print(f"Test data points: {len(X_test)}")

    # --- 4. Manual Preprocessing ---
    print("\n--- 4. Manually Preprocessing Data (StandardScaler Only) ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 5. Model: KNeighborsRegressor ---
    print("\n" + "="*50)
    print("  Model: KNeighborsRegressor (KNN)")
    print("="*50)
    
    N_NEIGHBORS = 25 # k=25 provides a good balance
    model = KNeighborsRegressor(n_neighbors=N_NEIGHBORS)
    
    print(f"Training KNN model with n_neighbors={N_NEIGHBORS}...")
    model.fit(X_train_scaled, y_train)
    
    print("Evaluating...")
    y_pred_test = model.predict(X_test_scaled)
    y_pred_train = model.predict(X_train_scaled)

    # --- 6. Model Evaluation Report (R2, MAE, RMSE) ---
    print("\n--- Model Evaluation Report ---")
    r2_test_overall = r2_score(y_test, y_pred_test)
    r2_train_overall = r2_score(y_train, y_pred_train)
    
    print("\n--- Overall Model Performance ---")
    print(f"TEST R-squared (R2):   {r2_test_overall:.6f}")
    print(f"TRAIN R-squared (R2):  {r2_train_overall:.6f}")
    
    print("\n--- Detailed Report per Property ---")
    print(f"{'Property':<30} | {'TEST R2':<10} | {'TRAIN R2':<10} | {'TEST MAE':<10} | {'TRAIN MAE':<10} | {'TEST RMSE':<10} | {'TRAIN RMSE':<10}")
    print("-"*105)
    
    for i, name in enumerate(label_names):
        r2_test_prop = r2_score(y_test.iloc[:, i], y_pred_test[:, i])
        r2_train_prop = r2_score(y_train.iloc[:, i], y_pred_train[:, i])
        mae_test_prop = mean_absolute_error(y_test.iloc[:, i], y_pred_test[:, i])
        mae_train_prop = mean_absolute_error(y_train.iloc[:, i], y_pred_train[:, i])
        rmse_test_prop = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred_test[:, i]))
        rmse_train_prop = np.sqrt(mean_squared_error(y_train.iloc[:, i], y_pred_train[:, i]))
        
        print(f"{name:<30} | {r2_test_prop:<10.4f} | {r2_train_prop:<10.4f} | {mae_test_prop:<10.4f} | {mae_train_prop:<10.4f} | {rmse_test_prop:<10.4f} | {rmse_train_prop:<10.4f}")

    # --- 7. Generate "Property vs. Strain" Plots (at specific temps) ---
    print("\n--- 7. Generating 'Property vs. Strain' Plots (Prediction Lines) ---")
    
    output_dir_1 = "AlSi_model_prediction_graphs"
    os.makedirs(output_dir_1, exist_ok=True)
    
    S_range = np.linspace(0.001, 0.15, 200) 
    temps_to_plot = [300, 600, 900] 
    colors = ['red', 'blue', 'green']
    
    for i, prop_name in enumerate(label_names):
        print(f"  Plotting: {prop_name} vs. Strain...")
        plt.figure(figsize=(10, 6))
        
        for temp, color in zip(temps_to_plot, colors):
            X_pred_data_df = pd.DataFrame({
                'Temperature (K)': np.full_like(S_range, temp),
                'Strain': S_range
            })
            X_pred_data_np = X_pred_data_df[X_train.columns].values
            X_pred_scaled = scaler.transform(X_pred_data_np)
            y_pred_values = model.predict(X_pred_scaled)
            
            plt.plot(S_range, y_pred_values[:, i], lw=3, color=color, label=f'Predicted at {temp}K')

        plt.title(f'Model Prediction: {prop_name} vs. Strain', fontsize=16)
        plt.xlabel('Strain', fontsize=12)
        plt.ylabel(prop_name, fontsize=12)
        plt.legend()
        plt.grid(True)
        
        safe_prop_name = prop_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        plot_filename = os.path.join(output_dir_1, f'{safe_prop_name}_vs_Strain_at_Temps.png')
        plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
        plt.close()
        
    print(f"  'Property vs. Strain' plots saved to '{output_dir_1}/'")

    # --- 8. Generate "Actual vs. Predicted" Feature Plots ---
    print("\n--- 8. Generating 'Actual vs. Predicted' Feature Plots ---")

    output_dir_2 = "AlSi_model_Actual_vs_Predicted"
    os.makedirs(output_dir_2, exist_ok=True)
    
    for i, prop_name in enumerate(label_names):
        
        # --- Plot vs. Temperature ---
        print(f"  Plotting: {prop_name} (Actual vs. Predicted) vs. Temperature...")
        
        x_axis_data = X_test['Temperature (K)']
        y_actual_data = y_test.iloc[:, i]
        y_pred_data = y_pred_test[:, i]
        
        sort_indices = x_axis_data.argsort()
        x_sorted = x_axis_data.iloc[sort_indices]
        y_actual_sorted = y_actual_data.iloc[sort_indices]
        y_pred_sorted = y_pred_data[sort_indices]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x_sorted, y_actual_sorted, alpha=0.2, s=10, label='Actual Test Data', color='blue')
        plt.plot(x_sorted, y_pred_sorted, color='red', lw=3, label='Model Prediction')
        plt.title(f'Model Fit: {prop_name} vs. Temperature', fontsize=16)
        plt.xlabel('Temperature (K)', fontsize=12)
        plt.ylabel(prop_name, fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        safe_prop_name = prop_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        plot_filename = os.path.join(output_dir_2, f'AvP_{safe_prop_name}_vs_Temp.png')
        plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
        plt.close()

        # --- Plot vs. Strain ---
        print(f"  Plotting: {prop_name} (Actual vs. Predicted) vs. Strain...")
        
        x_axis_data = X_test['Strain']
        
        sort_indices = x_axis_data.argsort()
        x_sorted = x_axis_data.iloc[sort_indices]
        y_actual_sorted = y_actual_data.iloc[sort_indices]
        y_pred_sorted = y_pred_data[sort_indices]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x_sorted, y_actual_sorted, alpha=0.2, s=10, label='Actual Test Data', color='blue')
        plt.plot(x_sorted, y_pred_sorted, color='red', lw=3, label='Model Prediction')
        plt.title(f'Model Fit: {prop_name} vs. Strain', fontsize=16)
        plt.xlabel('Strain', fontsize=12)
        plt.ylabel(prop_name, fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plot_filename = os.path.join(output_dir_2, f'AvP_{safe_prop_name}_vs_Strain.png')
        plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
        plt.close()

    print(f"  'Actual vs. Predicted' plots saved to '{output_dir_2}/'")
    print("\n--- Project Complete ---")

# Call the function directly
run_alsi_model()
