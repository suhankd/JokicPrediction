"""
Contains a master function to return a group of processed dfs.
"""

import pandas as pd
import glob

def second_remove(s):
    s = str(s)
    return int(s[:2])

def processing(path):
    
    # Here, we made a group of dfs, each df representing the game-logs of one season.

    raw_datafiles = glob.glob(f"{path}/*.csv")
    dfs = {file.split('/')[-1].split('.')[0]: pd.read_csv(file) for file in raw_datafiles}

    for year in range(2020,2025):
        dfs[f'Data\\{year}'].set_index('G',inplace=True)


    """
    The data needs to be tidied up. Here's what we need to do :
    Use only the following stats as features : 
    {MP,FG,FGA,3P,3PA,FT,FTA,ORB,DRB,AST,STL,BLK,TOV,PF,PTS,GmSc,+/-}, 
    since the other statistics are either redundant or useless.
    """

    features = {'MP','FG','FGA','3P','3PA','FT','FTA', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'GmSc','+/-'}
    num_features = len(features)

    # Removing seconds for convenience.

    for i in range(2020, 2025):

        dfs[f'Data\\{i}'] = dfs[f'Data\\{i}'][list(features)]
        dfs[f'Data\\{i}']['MP'] = dfs[f'Data\\{i}']['MP'].apply(second_remove)
        dfs[f'Data\\{i}'] = dfs[f'Data\\{i}'].apply(pd.to_numeric, errors = 'coerce')

    return dfs
