from transformers import AutoTokenizer

from transformers import AutoConfig, AutoAdapterModel

config = AutoConfig.from_pretrained(
    "xlm-roberta-base",
)
model = AutoAdapterModel.from_pretrained(
    "xlm-roberta-base",
    config=config,
)

from transformers import AdapterConfig
from transformers.adapters.composition import Stack

# Load the language adapters
lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
model.load_adapter("en/wiki@ukp", config=lang_adapter_config)
model.load_adapter("zh/wiki@ukp", config=lang_adapter_config)

# Add a new task adapter
model.add_adapter("copa")

# Add a classification head for our target task
model.add_classification_head("copa", num_labels=3)

model.train_adapter(["copa"])

# Unfreeze and activate stack setup
model.active_adapters = Stack("en", "copa")

import pandas as pd
from sklearn.model_selection import train_test_split
l_map = {
    'neutral': 0,
    'contradiction' : 1,
    'entailment': 2
}
def read_data(path):
    df = pd.read_csv(path, sep = '\t')
    df = df.fillna('')
    df.gold_label = df.gold_label.map(lambda x: l_map[x])
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

dfs = read_data('./data/train.tsv')
# dfs['en'] = dfs['en'].head(1000)
train_dfs, test_dfs = train_dev_split(dfs, 0.2, 69, ['en'])


from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


def encode_batch(batch):
    # print(batch["hypothesis"])
    return tokenizer(
        batch["hypothesis"],
        batch["premise"],
        truncation=True,
        padding=True
    )
    
train_ds = Dataset.from_dict({l: train_dfs['en'][l] for l in train_dfs['en']})
train_ds = train_ds.map(encode_batch, batched=True, load_from_cache_file=False)
train_ds = train_ds.rename_column("gold_label", "labels")
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

dev_ds = Dataset.from_dict({l: test_dfs['en'][l] for l in test_dfs['en']})
dev_ds = dev_ds.map(encode_batch, batched=True, load_from_cache_file=False)
dev_ds = dev_ds.rename_column("gold_label", "labels")
dev_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


from transformers import TrainingArguments, AdapterTrainer

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=8,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=100,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)
import numpy as np

def compute_accuracy(p):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    compute_metrics=compute_accuracy,
)

trainer.train()
print(trainer.evaluate())