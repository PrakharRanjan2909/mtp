# diagnostics.py
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from scripts.preprocess_data import load_and_preprocess_data


def create_train_data(data, num_arrays=10):
    """
    Create train data as a list of arrays, each with shape (x, 2).
    """
    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(data, ignore_index=True)
    
    # Convert to NumPy array
    combined_data = combined_df.to_numpy()
    
    # Determine the number of rows per array
    total_rows = combined_data.shape[0]
    num_rows_per_array = total_rows // num_arrays
    
    # Create the list of arrays
    train_data = [combined_data[i*num_rows_per_array:(i+1)*num_rows_per_array, :] for i in range(num_arrays)]
    
    # If there are remaining rows, handle them by appending to the last array
    remaining_rows = total_rows % num_arrays
    if remaining_rows > 0:
        train_data[-1] = np.vstack([train_data[-1], combined_data[-remaining_rows:]])
    
    return train_data

def create_test_data(data, num_arrays=4):
    """
    Create test data as a list of arrays, each with shape (x, 2).
    """
    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(data, ignore_index=True)
    
    # Convert to NumPy array
    combined_data = combined_df.to_numpy()
    
    # Determine the number of rows per array
    total_rows = combined_data.shape[0]
    num_rows_per_array = total_rows // num_arrays
    
    # Create the list of arrays
    test_data = [combined_data[i*num_rows_per_array:(i+1)*num_rows_per_array, :] for i in range(num_arrays)]
    
    # If there are remaining rows, handle them by appending to the last array
    remaining_rows = total_rows % num_arrays
    if remaining_rows > 0:
        test_data[-1] = np.vstack([test_data[-1], combined_data[-remaining_rows:]])
    
    return test_data

def load_hmm_models(model_paths):
    models = []
    for path in model_paths:
        with open(path, 'rb') as file:
            models.append(pickle.load(file))
    return models

# def classify_data(models, data):
#     """
#     Classify data based on the HMM model that gives the highest log-likelihood.
#     """
#     log_likelihoods = np.array([model.score(data) for model in models])
#     return np.argmax(log_likelihoods)
def classify_data(models, data):
    """
    Classify data based on the HMM model that gives the highest log-likelihood.
    Reshape data if it's a single sample to ensure it is 2D.
    """
    if data.ndim == 1:  # If data is 1D, reshape it to 2D
        data = data.reshape(1, -1)
    
    log_likelihoods = np.array([model.score(data) for model in models])
    return np.argmax(log_likelihoods)
def evaluate_models(models, train_data, test_data):
    """
    Evaluate classification accuracy of models.
    """
    # Training accuracy
    train_preds = [classify_data(models, seq) for seq in train_data]
    test_preds = [classify_data(models, seq) for seq in test_data]
    
    train_labels = [0] * len(train_data)  # Assuming single class label for now
    test_labels = [0] * len(test_data)
    
    train_acc = accuracy_score(train_labels, train_preds)
    test_acc = accuracy_score(test_labels, test_preds)
    
    return train_acc, test_acc


def main():
    model_paths = ['models/hmm_model_1.pkl', 'models/hmm_model_2.pkl', 'models/hmm_model_3.pkl']
    models = load_hmm_models(model_paths)
    
    # Example dummy data for train and test (replace with actual data)
    # Load and preprocess data
    # Load actual normalized data
    # data_files = [f'data/DB{i}.txt' for i in range(1, 15)] 
    data_files = ['data/DB1.txt', 'data/DB2.txt', 'data/DB3.txt', 'data/DB4.txt', 'data/DB5.txt', 'data/DB6.txt','data/DB7.txt','data/DB8.txt','data/DB9.txt','data/DB10.txt', 'data/DB11.txt','data/DB12.txt','data/DB13.txt', 'data/DB14.txt']  # All 14 files
    all_data = load_and_preprocess_data(data_files)
    
    
    
    # Circular cross-validation example (using 10 for training and 4 for testing)
    # Example dummy data for train and test (replace with actual data)
    train_data_demo = [np.random.rand(100, 2) for _ in range(10)]
    # print(len(train_data_demo))
    # print(len(train_data_demo[0]))
    # print(len(train_data_demo[0][0]))
    # test_data = [np.random.rand(100, 2) for _ in range(4)]
    

    # train_data = np.concatenate([df.values for df in all_data[:10]], axis=0)
    # print(len(train_data))
   
    train_data = create_train_data(all_data)
    print(len(train_data))
    print(len(train_data[0]))
    print(len(train_data[0][0]))

    # test_data = np.concatenate([df.values for df in all_data[10:]], axis=0)
    test_data = create_test_data(all_data)
    
    train_acc, test_acc = evaluate_models(models, train_data, test_data)
    
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Testing Accuracy: {test_acc * 100:.2f}%")

if __name__ == '__main__':
    main()
