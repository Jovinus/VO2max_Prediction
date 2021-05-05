import torch
from datatable import fread
import os

def load_dataset():
    DATA_PATH = "/home/lkh256/Studio/VO2max_Prediction/Data"
    df_orig = fread(os.path.join(DATA_PATH, 'general_eq.csv'),
                    encoding='utf-8-sig', 
                    na_strings=['', 'NA']).to_pandas()
    
    #### basic data preprocessing
    categorical = ['sex', 'MVPA']
    numeric = ['AGE', 'sex', 'percentage_fat', 'BMI', 'ASMI', 'rest_HR']
    
    df_orig[categorical] = df_orig[categorical].astype(float)
    df_orig[numeric] = df_orig[numeric].astype(float)
    
    #### Define columns to analysis
    column_mask = numeric + categorical
    
    x = torch.from_numpy(df_orig[column_mask].values).float()
    y = torch.from_numpy(df_orig['CRF'].values).float()
    
    return x, y

def split_data(x, y, train_ratio=0.8, device='cpu'):
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = x.size(0) - train_cnt
    
    # Shuffle dataset to split into train/valid set
    indices = torch.randperm(x.size(0)).to(device)
    
    x = torch.index_select(
        x, 
        dim=0, 
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    
    y = torch.index_select(
        y, 
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    
    return x, y


def get_hidden_sizes(input_size, output_size, n_layers):
    step_size = int((input_size - output_size) / n_layers)
    
    hidden_sizes =[]
    current_size = input_size
    
    for i in range(n_layers - 1):
        hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1]
        
    return hidden_sizes

if __name__ == '__main__':
    x, y = load_dataset()
    x, y = split_data(x, y, train_ratio=0.8)
    print(x)
    print(y)
    
    print(get_hidden_sizes(100, 1, 3))