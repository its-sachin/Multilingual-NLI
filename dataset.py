from imports import *

class LangDataset (Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        return {'hyp' : row['hypothesis'], 'prem' : row['premise']}



if __name__ == '__main__':
    print ('dataset running')
    
    technologies = {
        'hypothesis':["Spark","PySpark","Hadoop","Python","pandas","Oracle","Java"],
        'label' :[20000,25000,26000,22000,24000,21000,22000],
        'premise':['30days','40days','35days','40days','45days', '50days','55days'],
        'Discount':[1000,2300,1500,1200,2500,2100,2000]
    }
    
    df = pd.DataFrame(technologies)
    
    train_ds = LangDataset (df)
    train_dl = Dataloader (train_ds, batch_size = params['train_bs'], shuffle=True, num_workers= params['num_workers'])
    
    hyps = [row['hyp'] for row in batch]
    prems = [row['prems'] for row in batch]
    labels = [row['gold_label'] for row in batch]
    
    
    sent_chars = [sent['chars'] for sent in batch]
    sent_labels = [sent['labels'] for sent in batch]
    
    
