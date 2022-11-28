from utils import *
from dataset import *
from mBERT import MBERT


def test(
    model,
    test_dataset
):
    model.eval()
    gold = []; pred = []
    for batch in tqdm(test_dataset):

        input = tokenizer(batch['hyp'], batch['prem'], return_tensors='pt', padding=True)
        input.to(device)

        with torch.no_grad():
            logits = model(
                input_ids=input['input_ids'], 
                attention_mask = input['attention_mask'], 
                token_type_ids = input['token_type_ids']
            )
        predictions = torch.argmax(logits, dim=-1)
        gold += batch['label']
        pred += list(predictions.cpu())
    return  (metrics.f1_score(gold, pred, average="micro") + metrics.f1_score(gold, pred, average="macro"))/2


def train(
    model,
    device,
    optimizer,
    scheduler,
    tokenizer,
    train_dataset,
    dev_dataset,
    num_epoch
):

    best = 0
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epoch):

        logger.info(f'EPOCH: {epoch}')

        model.train()

        # TODO: optimize this
        for batch in tqdm(train_dataset):

            label = batch['label'].to(device)

            input = tokenizer(batch['hyp'], batch['prem'], return_tensors='pt', padding=True)

            input.to(device)

            outputs = model(
                input_ids=input['input_ids'], 
                attention_mask = input['attention_mask'], 
                token_type_ids = input['token_type_ids']
            )
            # outputs = outputs.softmax(dim=1)
            outputs = loss(outputs, label)
            outputs.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # print('\tTrain acc: ' , test(model, train_dataset))
        score = test(model, dev_dataset)
        if score > best:
            best = score
            torch.save(model, params['model_file'])
        logger.info(f'\tDev acc: {score} [{best}]')
    logger.info(f'Best score: {best}')


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
    

if __name__ == '__main__':

    seed_everything(params['seed'])
    print_params(logger=logger, params=params)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.debug(f'device: {device}')

    dfs = read_data(params['train_path'])
    train_dfs, test_dfs = train_dev_split(dfs, 0.2, params['seed'], ['en'])
    dfs['en'] = test_dfs['en']

    # TODO: Remove 1k
    train_df_eng = train_dfs['en']
    train_ds = LangDataset (train_df_eng)
    train_dl = DataLoader (train_ds, batch_size = params['train_bs'], shuffle=True, num_workers= params['num_workers'])

    logger.debug(f'TRAIN SIZE: {len(train_df_eng)}')

    # TODO: Remove 100
    dev_df_eng = test_dfs['en']
    dev_ds = LangDataset (dev_df_eng)
    dev_dl = DataLoader (dev_ds, batch_size = params['train_bs'], shuffle=True, num_workers= params['num_workers'])

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained("bert-base-multilingual-cased")
    mbert = MBERT(model)
    mbert.to(device)

    optimizer = AdamW(model.parameters(), lr=params['lr'])

    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=params['train_epochs'] * len(train_df_eng)
    )   


    train(
        mbert,
        device,
        optimizer,
        lr_scheduler,
        tokenizer,
        train_dl,
        dev_dl,
        params['train_epochs'] 
    )

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


