from utils import *
from dataset import *
from aBERT import ABERT
from callback import CustomCallback


if __name__ == '__main__':

    seed_everything(params['seed'])
    print_params(logger=logger, params=params)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.debug(f'device: {device}')

    to_eval = ['hi', 'sw', 'zh', 'es']
    dfs = read_data(params['test_path'])
    # train_dfs, test_dfs = train_dev_split(dfs, 0.2, params['seed'], to_eval)
    # dfs['en'] = test_dfs['en']
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    #only 4 lang
    # model = torch.load('models/1669729888/my_model.pkl')    # with nli and adapter fine tuning
    # model = torch.load('models/1669728712/my_model.pkl')  # with adapter fine tuning only
    # model = torch.load('models/1669726610/my_model.pkl')  # without adapter fine tuning
    model = torch.load(params['load_model'])
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    

    # TODO: remove this
    # train_dfs, test_dfs = train_dev_split(dfs, 0.2, params['seed'], ['en'])
    # to_eval = ['en']
    for lang in to_eval:

        logger.info(F'Testing on {lang}......')
        test_ds = df_to_ds(dfs[lang], encode_batch(tokenizer))
        # test_ds = df_to_ds(test_dfs[lang], encode_batch(tokenizer))
        model.active_adapters = Stack(lang, "nli")
        eval_trainer = AdapterTrainer(
            model=model,
            args=TrainingArguments(output_dir=f"{params['model_path']}/eval_output", remove_unused_columns=False,),
            eval_dataset=test_ds,
            compute_metrics=compute_metrics,
        )
        logger.critical(eval_trainer.evaluate())
        logger.debug('\n\n\n')
