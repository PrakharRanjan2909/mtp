# preprocess_data.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_paths):
    """
    Load data from file paths and preprocess it (normalize torque and thrust).
    """
    data = []
    for file in file_paths:
        df = pd.read_csv(file, delim_whitespace=True, header=None)
        df.columns = ['Thrust', 'Torque']
        
        # Normalize the data (StandardScaler)
        scaler = StandardScaler()
        df[['Thrust', 'Torque']] = scaler.fit_transform(df[['Thrust', 'Torque']])
        data.append(df)
    #print dimensions of data
    # for i in range(len(data)):
    #     print(data[i].shape)
        #type of data
        # print(type(data[i]))
    return data

def main():
    # Example file paths for 14 drill-bits data
    # data_files = [f'data/DB{i}.txt' for i in range(1, 15)]
    file_paths = ['data/DB1.txt', 'data/DB2.txt', 'data/DB3.txt', 'data/DB4.txt', 'data/DB5.txt', 'data/DB6.txt','data/DB7.txt','data/DB8.txt','data/DB9.txt','data/DB10.txt', 'data/DB11.txt','data/DB12.txt','data/DB13.txt', 'data/DB14.txt']  # Extend this to DB14
    normalized_data = load_and_preprocess_data(file_paths)
    
    # Return normalized data for use in training HMM
    return normalized_data

if __name__ == '__main__':
    main()
