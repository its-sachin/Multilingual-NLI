from utils import *
from dataset import *
from mBERT import MBERT
from aBERT import ABERT
from callback import CustomCallback


if __name__ == '__main__':

    seed_everything(params['seed'])
    params['train_path'] = sys.argv[1]
    print_params(logger=logger, params=params)

    dfs = read_data(params['train_path'])
    dfs['en'] = dfs['en'].head(100)
    train_dfs, test_dfs = train_dev_split(dfs, 0.2, params['seed'], ['en'])
    dfs['en'] = test_dfs['en']


    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoAdapterModel.from_pretrained("xlm-roberta-base")
    model = ABERT(model, tokenizer).to (params['device'])

    train_ds = df_to_ds(train_dfs['en'], encode_batch(model.tokenizer))
    dev_ds = df_to_ds(test_dfs['en'], encode_batch(model.tokenizer))

    training_args = TrainingArguments(
        learning_rate=params['lr'],
        num_train_epochs=params['train_epochs'],
        per_device_train_batch_size=params['train_bs'],
        per_device_eval_batch_size=params['val_bs'],
        evaluation_strategy = "epoch",
        logging_strategy ='epoch',
        logging_steps=100,
        save_strategy='no',
        output_dir=params['model_path'],
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
    )


    trainer = AdapterTrainer(
        model=model.mbert,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=model.tokenizer,
        compute_metrics = compute_metrics,
    )
    
    trainer.add_callback(CustomCallback(trainer)) 
    trainer.train()
    
    torch.save (model, params['my_model_file'])
    logger.critical (trainer.evaluate())