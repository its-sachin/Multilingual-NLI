from utils import *
from dataset import *
from aBERT import ABERT
from callback import CustomCallback

def other_adapters():
    return {
        'vi': [
            "vi/wiki@ukp",
            AdapterConfig.load("pfeiffer", non_linearity="gelu", reduction_factor=2)
        ],
        'el' :[
            "el/wiki@ukp",
            AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2)
        ],
        'de' :[
            "de/wiki@ukp",
            AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2)
        ],
        'ar': [
            "ar/wiki@ukp",
            AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2)
        ],
        'th': [
            "th/wiki@ukp",
            AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2)
        ],
        'ru':[
            "ru/wiki@ukp",
            AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2)
        ],
        'tr':[
            "tr/wiki@ukp",
            AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2)
        ],
        # 'fr': [
        #     "fr/wiki@ukp",
        #     AdapterConfig.load("houlsby", non_linearity="gelu", reduction_factor=2)
        # ]
    }


if __name__ == '__main__':

    seed_everything(params['seed'])
    params['lr'] = 1e-6
    params['train_epochs'] = 5
    print_params(logger=logger, params=params)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.debug(f'device: {device}')

    dfs = read_data(params['train_path'])

    to_eval = ['hi', 'sw', 'zh', 'es', 'en']
    to_tune = []
    for i in dfs.keys():
        if i not in to_eval:
            to_tune.append(i)

    train_dfs, test_dfs = train_dev_split(dfs, 0.1, params['seed'], to_tune)

    model = torch.load(params['my_model_file'])

    lang_adap = other_adapters()
    for lang in lang_adap:
        model.mbert.load_adapter(lang_adap[lang][0], config=lang_adap[lang][1])
    model.to(device)

    for lang in lang_adap:

        logger.info(F'Fine tuning {lang}......')
        train_ds = df_to_ds(train_dfs[lang], encode_batch(model.tokenizer))
        dev_ds = df_to_ds(test_dfs[lang], encode_batch(model.tokenizer))
        model.mbert.train_adapter(['nli'])
        model.mbert.active_adapters = Stack(lang, 'nli')

        training_args = TrainingArguments(
            learning_rate=params['lr_ft'],
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
        logger.debug('\n\n\n\n')
    
    torch.save (model, params['my_model_file_nli'])
    logger.critical (trainer.evaluate())