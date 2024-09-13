# main.py
import os
from scripts.preprocess_data import main as preprocess_data_main
from scripts.train_hmm import main as train_hmm_main
from scripts.diagnostics import main as diagnostics_main
from scripts.prognostics import main as prognostics_main
from scripts.visualize_results import main as visualize_results_main

def create_folders():
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/log_likelihood_plots', exist_ok=True)

def run_all():
    # Create necessary folders
    create_folders()

    # Step 1: Preprocess Data
    preprocess_data_main()
    
    # Step 2: Train HMMs with Competitive Learning
    train_hmm_main()
    
    # Step 3: Run Diagnostics
    diagnostics_main()
    
    # Step 4: Run Prognostics
    # prognostics_main()
    
    # Step 5: Visualize Results
    visualize_results_main()

if __name__ == '__main__':
    run_all()
