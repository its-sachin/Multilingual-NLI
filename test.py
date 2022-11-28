from utils import *
from dataset import *
from mBERT import MBERT

def translate(data, model, tokenizer, src_lang, dest_lang):

    tokenizer.src_lang = src_lang
    data = tokenizer(data, return_tensors="pt", padding=True)
    with torch.no_grad():
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
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
trans_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
trans_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
model = torch.load('models/1669644626/model.pkl')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

dfs = read_data(params['train_path'])
train_dfs, test_dfs = train_dev_split(dfs, 0.2, params['seed'], ['en'])
dfs['en'] = test_dfs['en']

langs = ['hi', 'es', 'sw', 'zh']

for lang in langs:
    print(lang)
    test_df = dfs[lang].head(64)
    gold = list(dfs[lang].head(64).gold_label)
    test_ds = LangDataset (test_df)
    test_dl = DataLoader (test_ds, batch_size = params['train_bs'], shuffle=False, num_workers= params['num_workers'])
    pred = predict(
        model,
        lang,
        'en',
        trans_model,
        trans_tokenizer,
        tokenizer,
        test_dl,
        device
    )

    micro = metrics.f1_score(gold, pred, average="micro")
    macro = metrics.f1_score(gold, pred, average="macro")

    logger.info(f'{lang}: {(micro+macro)/2}')