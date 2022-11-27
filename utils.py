import pandas as pd
from sklearn.model_selection import train_test_split

def read_data(path):
    df = pd.read_csv(path, sep = '\t')
    dfs = {}
    for lang in df.language.unique():
        dfs[lang] = df[df.language == lang]
    return dfs

def train_dev_split(path, test_size, seed):
    dfs = read_data(path)
    train_dfs = {}
    test_dfs = {}
    for lang in dfs:
        train_df, test_df = train_test_split(dfs[lang], test_size=test_size, random_state=seed)
        train_dfs[lang] = train_df
        test_dfs[lang] = test_df
    return train_dfs, test_dfs