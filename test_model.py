from utils import *
from dataset import *
from aBERT import ABERT
from callback import CustomCallback
import sys

if __name__ == '__main__':

    seed_everything(params['seed'])
    # print_params(logger=logger, params=params)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dfs = pd.read_csv(sys.argv[1], sep='\t').values
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = torch.load(params['load_model'])
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    present = ['en', 'hi', 'sw', 'es', 'zh']


    pred = []
    id2val = {}
    for i in params['label_map']:
        id2val[params['label_map'][i]] = i
    
    for data in dfs:
        encoded_input = tokenizer(data[1], data[0], return_tensors='pt', padding=True)
        encoded_input.to(device)
        if data[-1] not in present:
            model.active_adapters = Stack("nli")    
        else:
            model.active_adapters = Stack(data[-1], "nli")
        with torch.no_grad():
            logits = model(**encoded_input).logits
            pred.append(id2val[torch.argmax(logits,axis=-1).item()])

    with open(sys.argv[2],'w') as file:
        for p in pred:
            file.write(f'{p}\n')
    