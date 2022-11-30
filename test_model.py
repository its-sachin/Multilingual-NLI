from utils import *
from dataset import *
from imports import *
from aBERT import ABERT
from callback import CustomCallback
            
def get_pred_labels (model, test_dl, lang):
    mbert = model.mbert
    if lang not in model.present:
        mbert.active_adapters = Stack("nli")    
    else:
        mbert.active_adapters = Stack(lang, "nli")
    
    preds = []
    for batch in tqdm(test_dl):
        input = model.tokenizer(batch['hyp'], batch['prem'], return_tensors='pt', padding=True)
        input.to(params['device'])

        logits = mbert(
            input_ids=input['input_ids'], 
            attention_mask = input['attention_mask'], 
        ).logits
        
        preds.extend ([params['inv_label_map'][torch.argmax(i,axis=-1).item()] for i in logits])
    return preds


if __name__ == '__main__':

    seed_everything(params['seed'])
    # print_params(logger=logger, params=params)

    df = pd.read_csv(sys.argv[1], sep='\t')
    df.fillna ('')
    df.insert(loc=0, column='row_num', value=np.arange(len(df)))
    
    dfs = {}
    for lang in df.language.unique():
        dfs[lang] = df[df.language == lang]
    
    test_datasets = {}
    for lang in dfs.keys():
        test_datasets[lang] = LangDataset (dfs[lang])
    
    test_dls = {}
    for lang in dfs.keys():
        test_dls[lang] = DataLoader (test_datasets[lang], batch_size = params['test_bs'], shuffle=False, num_workers= params['num_workers'])
    
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = torch.load(params['load_model']).to(params['device'])
    
    for lang, test_dl in test_dls.items():
        dfs[lang]['pred_labels'] = get_pred_labels (model, test_dl, lang)

    final_df = pd.concat (dfs.values())
    final_df = final_df.sort_values(by=['row_num'], ascending=True)
    write_list (list(final_df.pred_labels), sys.argv[2])