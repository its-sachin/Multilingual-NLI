from utils import *
from const import *
import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


def translate(model, tokenizer, src_lang, dest_lang):

    src_text = "जीवन एक चॉकलेट बॉक्स की तरह है।"
    
    tokenizer.src_lang = src_lang
    encoded = tokenizer(src_text, return_tensors="pt")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id[dest_lang])
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

if __name__ == '__main__':

    train_dfs, test_dfs = train_dev_split(os.path.join(DATA_PATH, TRAIN_FILE), 0.2, SEED)
    trans_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    trans_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    print(translate(trans_model, trans_tokenizer, 'hi_IN', 'en_XX'))