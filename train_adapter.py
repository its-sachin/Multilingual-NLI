from utils import *
from dataset import *
from mBERT import MBERT
from aBERT import ABERT

def translate(data, model, tokenizer, src_lang, dest_lang):

    tokenizer.src_lang = src_lang
    data = tokenizer(data, return_tensors="pt", padding=True)
    data = model.generate(**data, forced_bos_token_id=tokenizer.get_lang_id(dest_lang), max_length=200)
    return tokenizer.batch_decode(data, skip_special_tokens=True)


def predict(
    model,
    src_lang,
    dest_lang,
    trans_model,
    trans_tokenizer,
    tokenizer,
    data_loader,
    device
):

    pred = []
    for batch in tqdm(data_loader):
        if src_lang != dest_lang :
            hyp = translate(batch['hyp'], trans_model, trans_tokenizer, src_lang, dest_lang)
            prem = translate(batch['prem'], trans_model, trans_tokenizer, src_lang, dest_lang)
        else:
            hyp = batch['hyp']
            prem = batch['prem']
        input = tokenizer(hyp, prem, return_tensors='pt', padding=True)
        input.to(device)

        with torch.no_grad():
            logits = model(
                input_ids=input['input_ids'], 
                attention_mask = input['attention_mask'], 
                token_type_ids = input['token_type_ids']
            )
        predictions = torch.argmax(logits, dim=-1)
        pred += list(predictions.cpu())
    return pred

# def compute_accuracy(p):
#   preds = np.argmax(p.predictions, axis=1)
#   micro = metrics.f1_score(p.label_ids, preds, average="micro")
#   macro = metrics.f1_score(p.label_ids, preds, average="macro")
#   return {"acc": (micro+macro)/2 }   

def compute_accuracy(p):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}

if __name__ == '__main__':

    seed_everything(params['seed'])
    print_params(logger=logger, params=params)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.debug(f'device: {device}')

    dfs = read_data(params['train_path'])
    # dfs['en'] = dfs['en'].head(10000)
    train_dfs, test_dfs = train_dev_split(dfs, 0.2, params['seed'], ['en'])
    dfs['en'] = test_dfs['en']

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
    

    model = AutoAdapterModel.from_pretrained("xlm-roberta-base")
    mbert = ABERT(model)

    mbert = mbert.mbert
    mbert.to(device)

    optimizer = AdamW(model.parameters(), lr=params['lr'])

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch: 1e-4 ** epoch)  
    scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=params['train_epochs'] * len(train_dfs['en'])
    ) 


    training_args = TrainingArguments(
        learning_rate=params['lr'],
        num_train_epochs=params['train_epochs'],
        per_device_train_batch_size=params['train_bs'],
        per_device_eval_batch_size=params['val_bs'],
        logging_strategy ='epoch',
        logging_steps=100,
        save_strategy='epoch',
        output_dir=params['model_path'],
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
    )


    trainer = AdapterTrainer(
        model=mbert,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics = compute_accuracy,
    )


    trainer.train()
    print(trainer.evaluate())

    # trans_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    # trans_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    # langs = ['hi', 'es', 'sw', 'zh']

    # for lang in langs:
    #     test_df = dfs[lang]
    #     test_ds = LangDataset (test_df)
    #     test_dl = DataLoader (test_ds, batch_size = params['train_bs'], shuffle=False, num_workers= params['num_workers'])
    #     pred = predict(
    #         model,
    #         lang,
    #         'en',
    #         trans_model,
    #         trans_tokenizer,
    #         tokenizer,
    #         test_dl,
    #         device
    #     )

    #     gold = list(dfs[lang].gold_label)
    #     micro = metrics.f1_score(gold, pred, average="micro")
    #     macro = metrics.f1_score(gold, pred, average="macro")

    #     logger.info(f'{lang}: {(micro+macro)/2}')


