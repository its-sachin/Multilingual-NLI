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
    msg += '\n=== End of Arguments ===\n\n'
    logger.info (msg)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=1)
    all_metrics = {
        'accuracy' : metrics.accuracy_score(labels, pred),
        'micro_f1' : metrics.f1_score (labels, pred, average='micro'),
        'macro_f1' : metrics.f1_score (labels, pred, average='macro'),
        'temp_acc' : (pred == labels).mean()
    }
    all_metrics['avg_f1'] = (all_metrics['micro_f1'] + all_metrics['macro_f1'])/2
    return all_metrics


def encode_batch(tokenizer):
    def encode_temp(batch):
        return tokenizer(
            batch["hypothesis"],
            batch["premise"],
            truncation=True,
            padding=True
        )
    return encode_temp

def df_to_ds(df, encode_batch):
    ds = Dataset.from_dict({l: df[l] for l in df})
    ds = ds.map(encode_batch, batched=True, batch_size = params['train_bs'], load_from_cache_file=False)
    ds = ds.rename_column("gold_label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds