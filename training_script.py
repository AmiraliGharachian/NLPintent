import torch
import os, sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from seqTagger import tokenize_and_pad_text_for_train
from transformers import AutoConfig, AutoTokenizer
import tagger_trainer
from utils import load_obj
from BERT_Joint_IDSF import BertIDSF
import project_statics


def get_Data(Data):
    Data["tr_inputs"] = torch.tensor(Data["tr_inputs"])
    Data["val_inputs"] = torch.tensor(Data["val_inputs"])
    Data["test_inputs"] = torch.tensor(Data["test_inputs"])

    Data["tr_tags"] = torch.tensor(Data["tr_tags"])
    Data["val_tags"] = torch.tensor(Data["val_tags"])
    Data["test_tags"] = torch.tensor(Data["test_tags"])

    Data["tr_intents"] = torch.tensor(Data["tr_intents"])
    Data["val_intents"] = torch.tensor(Data["val_intents"])
    Data["test_intents"] = torch.tensor(Data["test_intents"])

    Data["tr_masks"] = torch.tensor(Data["tr_masks"])
    Data["val_masks"] = torch.tensor(Data["val_masks"])
    Data["test_masks"] = torch.tensor(Data["test_masks"])
    return Data


if __name__ == '__main__':
    # set to  CUDA if you have GPU o.w. cpu
    # device = torch.device("cpu")
    device = torch.device("cuda")
    save_path = 'test_IDSF_bert'
    """
    sentences is a list of sentences each sentence list of tokens: [['جمله','دوم','تست'],['جمله','اول','تست']]
    tags is list of tags: [['O','O','O'],['O','O','O']]
    dict2 is dictionary of id(1 to N) to tag_text
    ditc2_rev is dictionary of tag_text to id (1 to N) 
    inte2 is dictionary of id(1 to N) to tag_sentence
    inte2_rev is dictionary of tag_sentence to id (1 to N) 
    """
    data_path = project_statics.SFID_pickle_files
    Data = load_obj(data_path + '/Data')
    dict2 = load_obj(data_path + '/dict2')
    dict_rev2 = load_obj(data_path + '/dict_rev2')
    inte2 = load_obj(data_path + '/inte2')
    inte_rev2 = load_obj(data_path + '/inte_rev2')
    alltags = load_obj(data_path + '/alltags')

    # find best iteration number for your self
    epochs = 1
    # batch size
    bs = 128
    # learning rate
    lr = 5e-5

    xlmr_model = 'xlm-roberta-base'
    bert_model = 'bert-base-uncased'
    parsbert = "HooshvareLab/bert-base-parsbert-uncased"
    mbert = 'bert-base-multilingual-cased'

    ## choose the pretrained language model
    model_name = mbert

    tokenizer = AutoTokenizer.from_pretrained("model_name", clean_up_tokenization_spaces=True)

    ## max_len must be < 510, if too small there would be warnings about truncation and there will be truncation
    max_len = 509
    Data["tr_inputs"], Data["tr_lens"], tokenized_sentences, Data["tr_tags"], Data["tr_intents"], Data[
        "tr_starts"] = tokenize_and_pad_text_for_train(Data["tr_inputs"],
                                                       Data["tr_tags"], Data["tr_intents"], tokenizer, max_len=max_len,
                                                       dict_rev2=dict_rev2, inte_rev2=inte_rev2)
    Data["val_inputs"], Data["val_lens"], tokenized_sentences, Data["val_tags"], Data["val_intents"], Data[
        "val_starts"] = tokenize_and_pad_text_for_train(Data["val_inputs"],
                                                        Data["val_tags"], Data["val_intents"], tokenizer,
                                                        max_len=max_len, dict_rev2=dict_rev2, inte_rev2=inte_rev2)
    Data["test_inputs"], Data["test_lens"], tokenized_sentences, Data["test_tags"], Data["test_intents"], Data[
        "test_starts"] = tokenize_and_pad_text_for_train(Data["test_inputs"],
                                                         Data["test_tags"], Data["test_intents"], tokenizer,
                                                         max_len=max_len, dict_rev2=dict_rev2, inte_rev2=inte_rev2)
    print('tokenized')

    # set this to true if you want to update the whole model
    FULL_FINETUNING = False

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(dict2)
    config.model_name = model_name
    config.classifier_dropout = 0.1
    config.use_gpu = True
    config.output_attentions = True
    config.max_len = max_len

    model_class = BertIDSF

    Data["tr_masks"] = [[i < Data["tr_lens"][j] + 2 for i in range(len(ii))] for j, ii in enumerate(Data["tr_inputs"])]
    Data["tr_inputs"] = Data["tr_inputs"].astype('int64')
    Data["tr_tags"] = Data["tr_tags"].astype('int64')

    Data["val_masks"] = [[i < Data["val_lens"][j] + 2 for i in range(len(ii))] for j, ii in
                         enumerate(Data["val_inputs"])]
    Data["val_inputs"] = Data["val_inputs"].astype('int64')
    Data["val_tags"] = Data["val_tags"].astype('int64')

    Data["test_masks"] = [[i < Data["test_lens"][j] + 2 for i in range(len(ii))] for j, ii in
                          enumerate(Data["test_inputs"])]
    Data["test_inputs"] = Data["test_inputs"].astype('int64')
    Data["test_tags"] = Data["test_tags"].astype('int64')

    Data = get_Data(Data)

    config.dict2 = dict2
    config.inte2 = inte2

    model = model_class.from_pretrained(model_name, config=config, slot_label_lst=dict2.keys(),
                                        intent_label_lst=inte2.keys())

    tagger_trainer.run_transformer_trainer(Data, batch_size=bs, FULL_FINETUNING=FULL_FINETUNING, model=model,
                                           tokenizer=tokenizer,
                                           device=device,
                                           lr=lr, epochs=epochs, save_dir=save_path)
