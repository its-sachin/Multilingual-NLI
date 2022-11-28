from imports import *

def read_data(path):
    df = pd.read_csv(path, sep = '\t')
    df = df.fillna('')
    df.gold_label = df.gold_label.map(lambda x: params['label_map'][x])
    dfs = {}
    for lang in df.language.unique():
        dfs[lang] = df[df.language == lang]
    return dfs

def train_dev_split(dfs, test_size, seed, langs):
    train_dfs = {}
    test_dfs = {}
    for lang in langs:
        train_df, test_df = train_test_split(dfs[lang], test_size=test_size, random_state=seed)
        train_dfs[lang] = train_df
        test_dfs[lang] = test_df
    return train_dfs, test_dfs

def seed_everything(seed=69):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def print_params (params, logger):
    msg = '\n\n=== Arguments ===\n'
    cnt = 0
    for key in sorted(params):
        msg += str (key) + ' : ' + str (params[key]) + '  |  ' 
        cnt += 1
        if (cnt + 1) % 5 == 0:
            msg += '\n'
    msg += '=== End of Arguments ===\n\n'
    logger.info (msg)