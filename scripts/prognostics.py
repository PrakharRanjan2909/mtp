# prognostics.py
import numpy as np
import pickle
from scripts.preprocess_data import load_and_preprocess_data
import os

def estimate_rul(model, data):
    """
    Estimate Remaining Useful Life (RUL) based on state transitions.
    """
    # if data.ndim == 1:  # If data is 1D, reshape it to 2D
    #     data = data.reshape(1, -1)
    log_likelihood = model.score(data)
    # Simple heuristic: higher log-likelihood indicates better health
    estimated_rul = 1 / (1 + np.exp(-log_likelihood)) * 100  # Normalize between 0 and 100%
    return estimated_rul

def main():
    model_path = 'models/hmm_model_1.pkl'
    
    # Load HMM model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Load actual normalized data
    # data_files = [f'data/DB{i}.txt' for i in range(1, 15)] 
    data_files = ['data/DB1.txt', 'data/DB2.txt', 'data/DB3.txt', 'data/DB4.txt', 'data/DB5.txt', 'data/DB6.txt','data/DB7.txt','data/DB8.txt','data/DB9.txt','data/DB10.txt', 'data/DB11.txt','data/DB12.txt','data/DB13.txt', 'data/DB14.txt']  # All 14 files
    all_data = load_and_preprocess_data(data_files)

    # Example dummy data (replace with actual normalized data)
    # data = np.random.rand(100, 2)
    
    # Use testing data for RUL estimation
    test_data = np.concatenate([df.values for df in all_data[10:]], axis=0)
    
    # Estimate RUL for the first 4 datasets
    rul = estimate_rul(model, test_data)
    
    # Save RUL estimation results
    os.makedirs('results', exist_ok=True)
    with open('results/rul_estimation.txt', 'w') as f:
        f.write(f"Estimated RUL: {rul:.2f}%\n")
    
    print(f"Estimated RUL: {rul:.2f}%")

if __name__ == '__main__':
    main()
