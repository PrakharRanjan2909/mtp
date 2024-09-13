# visualize_results.py
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scripts.preprocess_data import load_and_preprocess_data
import os

def plot_normalized_torque_vs_thrust(data):
    """
    Plot normalized torque vs thrust.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data[:, 0], data[:, 1], label='Torque vs Thrust')
    plt.xlabel('Normalized Thrust')
    plt.ylabel('Normalized Torque')
    plt.title('Normalized Torque vs Thrust')
    plt.legend()
    plt.savefig('results/torque_vs_thrust.png')
    plt.show()

def plot_log_likelihood(models, data):
    """
    Plot log-likelihood trajectories for different HMM models.
    """
    log_likelihoods = [model.score(data) for model in models]
    
    plt.figure(figsize=(10, 6))
    plt.plot(log_likelihoods, label='Log-Likelihood')
    plt.xlabel('Time')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood Trajectories')
    plt.legend()
    plt.savefig('results/log_likelihood_plot.png')
    plt.show()

def main():
    # Load actual normalized data
    data_files = ['data/DB1.txt', 'data/DB2.txt', 'data/DB3.txt', 'data/DB4.txt', 'data/DB5.txt', 'data/DB6.txt','data/DB7.txt','data/DB8.txt','data/DB9.txt','data/DB10.txt', 'data/DB11.txt','data/DB12.txt','data/DB13.txt', 'data/DB14.txt']   # All 14 files
    all_data = load_and_preprocess_data(data_files)
    
    # Use the first drill-bit's data for plotting
    data = all_data[0].values
    
    # Plot normalized torque vs thrust
    plot_normalized_torque_vs_thrust(data)
    
    # Load HMM models for log-likelihood plotting
    model_paths = ['models/hmm_model_1.pkl', 'models/hmm_model_2.pkl', 'models/hmm_model_3.pkl']
    models = [pickle.load(open(path, 'rb')) for path in model_paths]
    
    # Plot log-likelihood trajectories for the first 4 drill bits
    combined_test_data = np.concatenate([df.values for df in all_data[10:]], axis=0)
    plot_log_likelihood(models, combined_test_data)

if __name__ == '__main__':
    main()
