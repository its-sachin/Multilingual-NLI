from utils import *
from dataset import *
from aBERT import ABERT
from callback import CustomCallback

def other_adapters():
    configs = {
        'vi': [
            "vi/wiki@ukp",
            AdapterConfig.load("pfeiffer", non_linearity="gelu", reduction_factor=2)
        ],
        'el' :[
            "el/wiki@ukp",
            AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2)
        ]
    }


if __name__ == '__main__':

    seed_everything(params['seed'])
    print_params(logger=logger, params=params)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.debug(f'device: {device}')

    to_eval = ['hi', 'sw', 'zh', 'es']

    dfs = read_data(params['train_path'])
    train_dfs, test_dfs = train_dev_split(dfs, 0.1, params['seed'], to_eval)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


    model = torch.load('models/1669726610/my_model.pkl')
    

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    model.to(device)

    for lang in to_eval:

        logger.info(F'Fine tuning {lang}......')
        train_ds = df_to_ds(train_dfs[lang], encode_batch(tokenizer))
        dev_ds = df_to_ds(test_dfs[lang], encode_batch(tokenizer))
        model.train_adapter([lang, 'nli'])
        model.active_adapters = Stack(lang, 'nli')

        training_args = TrainingArguments(
            learning_rate=params['lr'],
            num_train_epochs=params['train_epochs'],
            per_device_train_batch_size=params['train_bs'],
            per_device_eval_batch_size=params['val_bs'],
            evaluation_strategy = "epoch",
            logging_strategy ='epoch',
            logging_steps=100,
            save_strategy='epoch',
            output_dir=params['model_path'],
            overwrite_output_dir=True,
            # The next line is important to ensure the dataset labels are properly passed to the model
            remove_unused_columns=False,
        )


        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            tokenizer=tokenizer,
            compute_metrics = compute_metrics,
        )
        
        trainer.add_callback(CustomCallback(trainer)) 
        trainer.train()
        logger.debug('\n\n\n\n')
    
    torch.save (model, params['my_model_file'])
    model.save_adapter(params['my_adapter_file'], "nli")
    logger.critical (trainer.evaluate())