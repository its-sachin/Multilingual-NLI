from imports import *

class ABERT(nn.Module):

    def get_adapters(self):
        return {

            'hi': [ 
                "hi/wiki@ukp",
                AdapterConfig.load("houlsby", non_linearity="gelu", reduction_factor=2)
            ],

            'en': [
                "en/wiki@ukp",
                AdapterConfig.load("houlsby", non_linearity="gelu", reduction_factor=2)
            ],

            'sw': [
                "sw/wiki@ukp",
                AdapterConfig.load("pfeiffer", non_linearity="gelu", reduction_factor=2)
            ],

            'zh': [
                "zh/wiki@ukp",
                AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2)
            ],

            'es': [
                "es/wiki@ukp",
                AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2)
            ]

        }


    def __init__(self, model, tokenizer ):
        super(ABERT, self).__init__()

        lang_adap = self.get_adapters()
        for lang in lang_adap:
            model.load_adapter(lang_adap[lang][0], config=lang_adap[lang][1])

        model.add_adapter("nli")
        model.train_adapter(["nli"])
        model.active_adapters = Stack("en", "nli")
        model.add_classification_head(
            "nli",
            num_labels=3,
        )
    
        self.mbert = model
        self.tokenizer = tokenizer
        self.present = ['en', 'hi', 'sw', 'es', 'zh']
        
    def forward(self, **args):
        return self.mbert(**args)