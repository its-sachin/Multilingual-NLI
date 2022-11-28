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


    def __init__(self, model, freeze=False, ):
        super(ABERT, self).__init__()
        D_in, H, D_out = 768, 50, 3

        lang_adap = self.get_adapters()
        for lang in lang_adap:
            model.load_adapter(lang_adap[lang][0], config=lang_adap[lang][1])

        model.add_adapter("copa")
        model.train_adapter(["copa"])
        model.active_adapters = Stack("en", "copa")

        self.mbert = model

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def set_adapters(self, train, active):
        self.mbert.train_adapter(train)
        self.mbert.active_adapters = Stack(*active)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.mbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids
        )
        
        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)

        return logits