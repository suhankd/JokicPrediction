"""
Contains a master function to return a group of processed dfs.
"""

import pandas as pd
import glob
from torch.utils.data import Dataset, DataLoader

def second_remove(s):
    s = str(s)
    return int(s[:2])

def processing(path):
    
    # Here, we made a group of dfs, each df representing the game-logs of one season.

    raw_datafiles = glob.glob(f"{path}/*.csv")
    dfs = {file.split('/')[-1].split('.')[0]: pd.read_csv(file) for file in raw_datafiles}

    features = {'MP','FG','FGA','3P','3PA','FT','FTA', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'GmSc','+/-'}
    num_features = len(features)

    for year in range(2020,2025):
        
        dfs[f'Data\\{year}']['Date'] = pd.to_datetime(dfs[f'Data\\{year}']['Date'], format = "%Y-%m-%d")

        """
        The data needs to be tidied up. Here's what we need to do :
        Use only the following stats as features : 
        {MP,FG,FGA,3P,3PA,FT,FTA,ORB,DRB,AST,STL,BLK,TOV,PF,PTS,GmSc,+/-}, 
        since the other statistics are either redundant or useless.
        """

        dfs[f'Data\\{year}'] = dfs[f'Data\\{year}'][list(features)]
        dfs[f'Data\\{year}']['MP'] = dfs[f'Data\\{year}']['MP'].apply(second_remove)
        dfs[f'Data\\{year}'] = dfs[f'Data\\{year}'].apply(pd.to_numeric, errors = 'coerce')

        dfs[f'Data\\{year}']['PtsNextGame'] = (dfs[f'Data\\{year}']['PTS'].shift(-1))
        dfs[f'Data\\{year}'] = dfs[f'Data\\{year}'].iloc[:-1]


    return dfs
