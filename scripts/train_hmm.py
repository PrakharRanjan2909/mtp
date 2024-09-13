# train_hmm.py
import numpy as np
from hmmlearn import hmm
import pickle
from scripts.preprocess_data import load_and_preprocess_data

def train_hmm(data, n_states=3):
    model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=100)
    model.fit(data)
    return model

def competitive_learning(data, n_hmms=10, n_states=3, n_best=3):
    # Step 1: Initialize 10 HMMs
    models = [train_hmm(data, n_states) for _ in range(n_hmms)]
    
    # Step 2: Compute log-likelihood for each HMM
    log_likelihoods = [model.score(data) for model in models]
    
    # Step 3: Select the top 3 HMMs with the highest log-likelihood
    best_indices = np.argsort(log_likelihoods)[-n_best:]
    best_models = [models[i] for i in best_indices]
    
    return best_models

def save_hmm_models(models):
    """
    Save trained HMM models to files.
    """
    for i, model in enumerate(models):
        with open(f'models/hmm_model_{i+1}.pkl', 'wb') as file:
            pickle.dump(model, file)

def main():
    # Load and preprocess data
    data_files = ['data/DB1.txt', 'data/DB2.txt', 'data/DB3.txt', 'data/DB4.txt', 'data/DB5.txt', 'data/DB6.txt','data/DB7.txt','data/DB8.txt','data/DB9.txt','data/DB10.txt', 'data/DB11.txt','data/DB12.txt','data/DB13.txt', 'data/DB14.txt']  # Use all 14 files in actual run
    
    data = np.concatenate([df.values for df in load_and_preprocess_data(data_files)], axis=0)
    
    # Competitive learning with 10 HMMs, select the top 3 models
    best_models = competitive_learning(data)
    
    # Save the best models
    save_hmm_models(best_models)
    print("Top 3 HMMs saved successfully!")

if __name__ == '__main__':
    main()
